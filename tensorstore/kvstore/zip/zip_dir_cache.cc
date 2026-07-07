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

#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <variant>

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
#include "tensorstore/kvstore/zip/cached_dir.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_builder.h"

// specializations
#include "tensorstore/internal/estimate_heap_usage/std_vector.h"  // IWYU pragma: keep

namespace tensorstore {
namespace internal_zip_kvstore {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag zip_logging("zip");

struct ReadDirectoryOp
    : public internal::AtomicReferenceCount<ReadDirectoryOp> {
  ZipDirectoryCache::Entry* entry_;

  kvstore::ReadOptions options_;
  internal_zip::ZipEOCD eocd_;
  std::shared_ptr<const CachedDir> existing_read_data_;

  explicit ReadDirectoryOp(ZipDirectoryCache::Entry* entry) : entry_(entry) {}

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
        // File is smaller than the EOCD block; retry with full read.
        assert(!options_.byte_range.IsFull());
        options_.byte_range = OptionalByteRangeRequest{};
        StartEOCDBlockRead();
        return;
      }
      entry_->ReadError(
          StatusBuilder(std::move(r).status())
              .With(internal::ConvertInvalidArgumentToFailedPrecondition));
      return;
    }

    auto& read_result = *r;
    if (read_result.aborted()) {
      // The generation matched `if_not_equal`, indicating the cached data is
      // still valid and unchanged. Re-publish it with the updated stamp.
      entry_->ReadSuccess(ZipDirectoryCache::ReadState{
          std::move(existing_read_data_), std::move(read_result.stamp)});
      return;
    }
    if (read_result.not_found()) {
      // The base file was not found. Return a missing entry.
      entry_->ReadSuccess(
          ZipDirectoryCache::ReadState{nullptr, std::move(read_result.stamp)});
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
      // Suffix was too small for the full EOCD (e.g. large comment);
      // re-read from the position identified by TryReadFullEOCD.
      assert(!options_.byte_range.IsFull());
      options_.byte_range = OptionalByteRangeRequest::Suffix(*inclusive_min);
      StartEOCDBlockRead();
      return;
    }

    // block_offset is always >= 0 (0 for full read, or the suffix start).
    if (block_offset >= 0 && block_offset <= eocd_.cd_offset) {
      // The EOCD block contains the directory, so fall through to
      // handling directory entries without another read request.
      DoDecodeDirectory(std::move(ready), eocd_.cd_offset - block_offset);
      return;
    }

    // Central Directory is outside the EOCD block; read it separately.
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
          StatusBuilder(std::move(r).status())
              .With(internal::ConvertInvalidArgumentToFailedPrecondition));
      return;
    }

    auto& read_result = *r;
    if (read_result.aborted()) {
      // The `if_equal` condition was not satisfied, meaning that the file was
      // modified or replaced, so reading starts over from the EOCD block.
      options_.byte_range =
          OptionalByteRangeRequest::SuffixLength(internal_zip::kEOCDBlockSize);
      StartEOCDBlockRead();
      return;
    }
    if (!read_result.has_value()) {
      // no_value and not_found are equivalent here.
      entry_->ReadSuccess(
          ZipDirectoryCache::ReadState{nullptr, std::move(read_result.stamp)});
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

    auto dir_result =
        DecodeDirectoryEntries(reader, eocd_.num_entries, eocd_.cd_offset);
    if (!dir_result.ok()) {
      // Decoding the directory failed, which is a legitimate error.
      entry_->ReadError(dir_result.status());
      return;
    }
    CachedDir dir = std::move(*dir_result);
    dir.full_read = options_.byte_range.IsFull();

    ABSL_LOG_IF(INFO, zip_logging) << dir;

    entry_->ReadSuccess(ZipDirectoryCache::ReadState{
        std::make_shared<const CachedDir>(std::move(dir)),
        std::move(ready.value().stamp)});
  }
};

}  // namespace

size_t ZipDirectoryCache::Entry::ComputeReadDataSizeInBytes(
    const void* read_data) {
  return internal::EstimateHeapUsage(*static_cast<const ReadData*>(read_data));
}

void ZipDirectoryCache::Entry::DoRead(AsyncCacheReadRequest request) {
  bool do_full_read = false;
  auto state = internal::MakeIntrusivePtr<ReadDirectoryOp>(this);
  {
    ZipDirectoryCache::ReadLock<ZipDirectoryCache::ReadData> lock(*this);
    if (auto data = lock.shared_data()) {
      do_full_read = data->full_read;
    }
    state->options_.generation_conditions.if_not_equal =
        lock.read_state().stamp.generation;
    state->existing_read_data_ = lock.shared_data();
  }

  state->options_.staleness_bound = request.staleness_bound;
  if (do_full_read) {
    // The previous read required the full file (e.g., suffix was too small
    // for the EOCD), so don't regress to a suffix read.
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
