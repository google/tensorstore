// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/kvstore/zip/zip_dir_cache.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/compression/zip_details.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

// specializations
#include "tensorstore/internal/estimate_heap_usage/std_vector.h"  // IWYU pragma: keep

namespace tensorstore {
namespace internal_zip_kvstore {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag zip_logging("zip");

struct ReadDirectoryOp
    : public internal::AtomicReferenceCount<ReadDirectoryOp> {
  ZipDirectoryCache::Entry* entry_;
  std::shared_ptr<const Directory> existing_read_data_;

  kvstore::ReadOptions options_;
  internal_zip::ZipEOCD eocd_;

  void StartEOCDBlockRead() {
    auto& cache = internal::GetOwningCache(*entry_);
    ABSL_LOG_IF(INFO, zip_logging)
        << "StartEOCDBlockRead " << entry_->key() << " " << options_.byte_range;

    auto future =
        cache.kvstore_driver_->Read(std::string(entry_->key()), options_);

    future.Force();
    future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ReadDirectoryOp>(this)](
            ReadyFuture<kvstore::ReadResult> ready) {
          self->OnEOCDBlockRead(std::move(ready));
        });
  }

  void OnEOCDBlockRead(ReadyFuture<kvstore::ReadResult> ready) {
    auto& r = ready.result();
    if (!r.ok()) {
      ABSL_LOG_IF(INFO, zip_logging) << r.status();
      if (absl::IsOutOfRange(r.status())) {
        // Retry, reading the full range.
        assert(!options_.byte_range.IsFull());
        options_.byte_range = OptionalByteRangeRequest{};
        StartEOCDBlockRead();
        return;
      }
      entry_->ReadError(
          internal::ConvertInvalidArgumentToFailedPrecondition(r.status()));
      return;
    }

    auto& read_result = *r;
    if (read_result.aborted()) {
      // yield the original data.
      entry_->ReadSuccess(ZipDirectoryCache::ReadState{
          entry_->read_request_state_.read_state.data,
          std::move(read_result.stamp)});
      return;
    }
    if (read_result.not_found()) {
      entry_->ReadError(absl::NotFoundError(""));
      return;
    }

    GetOwningCache(*entry_).executor()(
        [self = internal::IntrusivePtr<ReadDirectoryOp>(this),
         ready = std::move(ready)]() {
          self->DoDecodeEOCDBlock(std::move(ready));
        });
  }

  void DoDecodeEOCDBlock(ReadyFuture<kvstore::ReadResult> ready) {
    absl::Cord* eocd_block = &ready.value().value;
    riegeli::CordReader<absl::Cord*> reader(eocd_block);

    int64_t block_offset =
        options_.byte_range.IsFull() ? 0 : options_.byte_range.inclusive_min;

    auto read_eocd_variant = TryReadFullEOCD(reader, eocd_, block_offset);
    if (auto* status = std::get_if<absl::Status>(&read_eocd_variant);
        status != nullptr && !status->ok()) {
      entry_->ReadError(std::move(*status));
      return;
    }

    if (auto* inclusive_min = std::get_if<int64_t>(&read_eocd_variant);
        inclusive_min != nullptr) {
      // Issue a retry since the initial block did not contain the entire
      // EOCD64 Directory.
      assert(!options_.byte_range.IsFull());
      options_.byte_range = OptionalByteRangeRequest::Suffix(*inclusive_min);
      StartEOCDBlockRead();
      return;
    }

    if (block_offset >= 0 && block_offset <= eocd_.cd_offset) {
      // The EOCD block contains the directory, so fall through to
      // handling directory entries without another read request.
      DoDecodeDirectory(std::move(ready), eocd_.cd_offset - block_offset);
      return;
    }

    // Issue a read of the directory.
    kvstore::ReadOptions other_options = options_;
    other_options.generation_conditions.if_equal =
        ready.value().stamp.generation;
    other_options.byte_range = OptionalByteRangeRequest::Range(
        eocd_.cd_offset, eocd_.cd_offset + eocd_.cd_size);

    auto& cache = internal::GetOwningCache(*entry_);
    auto future =
        cache.kvstore_driver_->Read(std::string(entry_->key()), other_options);
    future.Force();
    future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ReadDirectoryOp>(this)](
            ReadyFuture<kvstore::ReadResult> ready) {
          self->OnDirectoryBlockRead(std::move(ready));
        });
  }

  void OnDirectoryBlockRead(ReadyFuture<kvstore::ReadResult> ready) {
    auto& r = ready.result();
    if (!r.ok()) {
      ABSL_LOG_IF(INFO, zip_logging) << r.status();
      entry_->ReadError(
          internal::ConvertInvalidArgumentToFailedPrecondition(r.status()));
      return;
    }

    auto& read_result = *r;
    if (read_result.aborted() || read_result.not_found() ||
        !ready.value().has_value()) {
      // Any non-value is an error.
      entry_->ReadError(
          absl::InvalidArgumentError("Faild to read ZIP directory"));
      return;
    }

    GetOwningCache(*entry_).executor()(
        [self = internal::IntrusivePtr<ReadDirectoryOp>(this),
         ready = std::move(ready)]() {
          self->DoDecodeDirectory(std::move(ready), 0);
        });
  }

  void DoDecodeDirectory(ReadyFuture<kvstore::ReadResult> ready,
                         size_t seek_pos) {
    absl::Cord* cd_block = &ready.value().value;
    riegeli::CordReader<absl::Cord*> reader(cd_block);
    if (seek_pos > 0) {
      reader.Seek(seek_pos);
    }

    Directory dir{};
    dir.full_read = options_.byte_range.IsFull();
    dir.entries.reserve(eocd_.num_entries);
    for (size_t i = 0; i < eocd_.num_entries; ++i) {
      internal_zip::ZipEntry entry{};
      if (auto entry_status = ReadCentralDirectoryEntry(reader, entry);
          !entry_status.ok()) {
        entry_->ReadError(entry_status);
        return;
      }
      // Only add validated entries to the zip directory.
      if (ValidateEntryIsSupported(entry).ok()) {
        ABSL_LOG_IF(INFO, zip_logging) << "Adding " << entry;
        dir.entries.push_back(
            Directory::Entry{entry.filename, entry.crc, entry.compressed_size,
                             entry.uncompressed_size, entry.local_header_offset,
                             entry.estimated_read_size});
      } else {
        ABSL_LOG_IF(INFO, zip_logging) << "Skipping " << entry;
      }
    }

    // Sort by local header offset first, then by name, to determine
    // the estimated read size for each entry. Typically a ZIP file will
    // already be ordered like this. Subsequently, the gap between the
    // headers is used to determine how many bytes to read.
    std::sort(dir.entries.begin(), dir.entries.end(),
              [](const auto& a, const auto& b) {
                return std::tie(a.local_header_offset, a.filename) <
                       std::tie(b.local_header_offset, b.filename);
              });
    auto last_header_offset = eocd_.cd_offset;
    for (auto it = dir.entries.rbegin(); it != dir.entries.rend(); ++it) {
      it->estimated_size = last_header_offset - it->local_header_offset;
      last_header_offset = it->local_header_offset;
    }

    // Sort directory by filename.
    std::sort(dir.entries.begin(), dir.entries.end(),
              [](const auto& a, const auto& b) {
                return std::tie(a.filename, a.local_header_offset) <
                       std::tie(b.filename, a.local_header_offset);
              });

    ABSL_LOG_IF(INFO, zip_logging) << dir;

    entry_->ReadSuccess(ZipDirectoryCache::ReadState{
        std::make_shared<const Directory>(std::move(dir)),
        std::move(ready.value().stamp)});
  }
};

}  // namespace

size_t ZipDirectoryCache::Entry::ComputeReadDataSizeInBytes(
    const void* read_data) {
  return internal::EstimateHeapUsage(*static_cast<const ReadData*>(read_data));
}

void ZipDirectoryCache::Entry::DoRead(AsyncCacheReadRequest request) {
  auto state = internal::MakeIntrusivePtr<ReadDirectoryOp>();
  state->entry_ = this;
  {
    ZipDirectoryCache::ReadLock<ZipDirectoryCache::ReadData> lock(*this);
    state->existing_read_data_ = lock.shared_data();
    state->options_.generation_conditions.if_not_equal =
        lock.read_state().stamp.generation;
  }

  // Setup options.
  state->options_.staleness_bound = request.staleness_bound;
  if (state->existing_read_data_ && state->existing_read_data_->full_read) {
    state->options_.byte_range = OptionalByteRangeRequest{};
  } else {
    state->options_.byte_range =
        OptionalByteRangeRequest::SuffixLength(internal_zip::kEOCDBlockSize);
  }

  state->StartEOCDBlockRead();
}

ZipDirectoryCache::Entry* ZipDirectoryCache::DoAllocateEntry() {
  return new Entry;
}

size_t ZipDirectoryCache::DoGetSizeofEntry() { return sizeof(Entry); }

}  // namespace internal_zip_kvstore
}  // namespace tensorstore
