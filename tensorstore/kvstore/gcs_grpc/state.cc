// Copyright 2024 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorstore/kvstore/gcs_grpc/state.h"

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <utility>

#include "absl/crc/crc32c.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "riegeli/bytes/cord_writer.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

// proto
#include "google/storage/v2/storage.pb.h"

namespace tensorstore {
namespace internal_gcs_grpc {
namespace {

static constexpr size_t kMaxWriteBytes =
    google::storage::v2::ServiceConstants::MAX_WRITE_CHUNK_BYTES;

/// Abseil has a convenient crc32_t type, but it doesn't handle absl::Cord.
absl::crc32c_t ComputeCrc32c(const absl::Cord& cord) {
  absl::crc32c_t crc{0};
  for (auto chunk : cord.Chunks()) {
    crc = absl::ExtendCrc32c(crc, chunk);
  }
  return crc;
}

}  // namespace

void ReadState::SetupRequest(Request& request) {
  if (!StorageGeneration::IsUnknown(options_.generation_conditions.if_equal)) {
    uint64_t gen =
        StorageGeneration::IsNoValue(options_.generation_conditions.if_equal)
            ? 0
            : StorageGeneration::ToUint64(
                  options_.generation_conditions.if_equal);
    request.set_if_generation_match(gen);
  }
  if (!StorageGeneration::IsUnknown(
          options_.generation_conditions.if_not_equal)) {
    uint64_t gen = StorageGeneration::IsNoValue(
                       options_.generation_conditions.if_not_equal)
                       ? 0
                       : StorageGeneration::ToUint64(
                             options_.generation_conditions.if_not_equal);
    request.set_if_generation_not_match(gen);
  }
  if (options_.byte_range.inclusive_min != 0) {
    request.set_read_offset(options_.byte_range.inclusive_min);
  }
  if (options_.byte_range.exclusive_max != -1) {
    auto target_size = options_.byte_range.size();
    assert(target_size >= 0);
    // read_limit == 0 reads the entire object; instead just read a single
    // byte.
    request.set_read_limit(target_size == 0 ? 1 : target_size);
  }
}

Result<kvstore::ReadResult> ReadState::HandleFinalStatus(absl::Status status) {
  if (absl::IsFailedPrecondition(status) || absl::IsAborted(status)) {
    // Failed precondition is set when either the if_generation_match or
    // the if_generation_not_match fails.
    if (!StorageGeneration::IsUnknown(
            options_.generation_conditions.if_equal)) {
      storage_generation_.generation = StorageGeneration::Unknown();
    } else {
      storage_generation_.generation =
          options_.generation_conditions.if_not_equal;
    }
    return kvstore::ReadResult::Unspecified(std::move(storage_generation_));
  } else if (absl::IsNotFound(status)) {
    return kvstore::ReadResult::Missing(storage_generation_.time);
  } else if (!status.ok()) {
    return status;
  }

  if (StorageGeneration::IsUnknown(storage_generation_.generation)) {
    // Bad metadata was returned by BlobService; this is unexpected, and
    // usually indicates a bug in our testing.
    return absl::InternalError("Object missing a valid generation");
  }

  // Validate the content checksum.
  absl::Cord combined_content;
  absl::crc32c_t combined_crc32c = absl::crc32c_t(0);
  riegeli::CordWriter writer(&combined_content);

  for (auto& [chunk, crc32c] : chunks_) {
    absl::crc32c_t content_crc32c = ComputeCrc32c(chunk);
    combined_crc32c =
        absl::ConcatCrc32c(combined_crc32c, content_crc32c, chunk.size());
    if (crc32c != absl::crc32c_t(0) && content_crc32c != crc32c) {
      return absl::DataLossError(absl::StrFormat(
          "Object fragment crc32c %08x does not match expected crc32c %08x",
          static_cast<uint32_t>(content_crc32c),
          static_cast<uint32_t>(crc32c)));
    }
    writer.Write(std::move(chunk));
  }
  if (crc32c_ && combined_crc32c != *crc32c_) {
    return absl::DataLossError(absl::StrFormat(
        "Object  crc32c %08x does not match expected crc32c %08x",
        static_cast<uint32_t>(combined_crc32c),
        static_cast<uint32_t>(*crc32c_)));
  }
  chunks_.clear();

  // Validate the content checksum.
  if (!writer.Close()) {
    return writer.status();
  }
  if (options_.byte_range.size() == 0) {
    return kvstore::ReadResult::Value({}, std::move(storage_generation_));
  }
  return kvstore::ReadResult::Value(std::move(combined_content),
                                    std::move(storage_generation_));
}

absl::Status ReadState::HandleResponse(ReadState::Response& response) {
  if (response.has_metadata()) {
    storage_generation_.generation =
        StorageGeneration::FromUint64(response.metadata().generation());
  }
  if (response.has_object_checksums() &&
      response.object_checksums().crc32c() != 0 &&
      options_.byte_range.inclusive_min == 0 &&
      !options_.byte_range.exclusive_max) {
    // Do not validate byte-range requests.
    crc32c_ = absl::crc32c_t(response.object_checksums().crc32c());
  }
  if (response.has_content_range()) {
    // The content-range request indicates the expected data. If it does not
    // satisfy the byte range request, cancel the read with an error. Allow
    // the returned size to exceed the requested size.
    auto returned_size =
        response.content_range().end() - response.content_range().start();
    if (auto size = options_.byte_range.size();
        (size > 0 && size != returned_size) ||
        (options_.byte_range.inclusive_min >= 0 &&
         response.content_range().start() !=
             options_.byte_range.inclusive_min)) {
      return absl::OutOfRangeError(
          tensorstore::StrCat("Requested byte range ", options_.byte_range,
                              " was not satisfied by GCS object with size ",
                              response.content_range().complete_length()));
    }
  }
  if (response.has_checksummed_data()) {
    chunks_.emplace_back(response.checksummed_data().content(),
                         absl::crc32c_t(response.checksummed_data().crc32c()));
  }
  return absl::OkStatus();
}

Result<TimestampedStorageGeneration> WriteState::HandleFinalStatus(
    absl::Status status, WriteState::Response& response) {
  TimestampedStorageGeneration result;
  result.time = start_time_;
  if (response.has_resource()) {
    result.generation =
        StorageGeneration::FromUint64(response.resource().generation());
  }
  if (absl::IsFailedPrecondition(status) || absl::IsAlreadyExists(status)) {
    // if_equal condition did not match.
    result.generation = StorageGeneration::Unknown();
  } else if (absl::IsNotFound(status) &&
             !StorageGeneration::IsUnknown(
                 options_.generation_conditions.if_equal)) {
    // precondition did not match.
    result.generation = StorageGeneration::Unknown();
  } else if (!status.ok()) {
    return status;
  }
  return std::move(result);
}

void WriteState::UpdateRequestForNextWrite(WriteState::Request& request) {
  if (value_offset_ == 0) {
    request.mutable_write_object_spec()->set_object_size(value_.size());
    if (!StorageGeneration::IsUnknown(
            options_.generation_conditions.if_equal)) {
      auto gen =
          StorageGeneration::ToUint64(options_.generation_conditions.if_equal);
      request.mutable_write_object_spec()->set_if_generation_match(gen);
    }
  } else {
    // After the first request, clear the spec.
    request.clear_write_object_spec();
  }

  request.set_write_offset(value_offset_);
  auto next_part = value_.Subcord(value_offset_, kMaxWriteBytes);
  auto& checksummed_data = *request.mutable_checksummed_data();
  checksummed_data.set_content(next_part);
  auto chunk_crc32c = ComputeCrc32c(checksummed_data.content());
  checksummed_data.set_crc32c(static_cast<uint32_t>(chunk_crc32c));
  crc32c_ = absl::ConcatCrc32c(crc32c_, chunk_crc32c,
                               checksummed_data.content().size());
  value_offset_ = value_offset_ + next_part.size();
  if (value_offset_ == value_.size()) {
    /// This is the last request.
    request.mutable_object_checksums()->set_crc32c(
        static_cast<uint32_t>(crc32c_));
    request.set_finish_write(true);
  }
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
