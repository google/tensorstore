// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/kvstore/tiff/tiff_dir_cache.h"

#include <memory>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/future.h"
#include "absl/status/status.h"

namespace tensorstore {
namespace internal_tiff_kvstore {

namespace {

ABSL_CONST_INIT internal_log::VerboseFlag tiff_logging("tiff");

struct ReadDirectoryOp : public internal::AtomicReferenceCount<ReadDirectoryOp> {
  TiffDirectoryCache::Entry* entry_;
  std::shared_ptr<const TiffDirectoryParseResult> existing_read_data_;
  kvstore::ReadOptions options_;
  bool is_full_read_;

  void StartRead() {
    auto& cache = internal::GetOwningCache(*entry_);
    ABSL_LOG_IF(INFO, tiff_logging)
        << "StartRead " << entry_->key();
    
    // 1.  Default to the “slice‑first” strategy -----------------------------
    is_full_read_ = false;

    // Honour any *caller‑supplied* range that is smaller than the slice.
    if (!options_.byte_range.IsFull() &&
        options_.byte_range.size() <= kInitialReadBytes) {
      // Caller already requested an explicit (small) range → keep it.
    } else {
      // Otherwise issue our standard 0‑1023 probe.
      options_.byte_range =
          OptionalByteRangeRequest::Range(0, kInitialReadBytes);
    }

    auto future = cache.kvstore_driver_->Read(std::string(entry_->key()), options_);
    future.Force();
    future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ReadDirectoryOp>(this)](
            ReadyFuture<kvstore::ReadResult> ready) {
          self->OnReadComplete(std::move(ready));
        });
  }

  void OnReadComplete(ReadyFuture<kvstore::ReadResult> ready) {
    auto& r = ready.result();
    if (!r.ok()) {
      // If the ranged request overshot the file, retry with a full read.
      if (!is_full_read_ && absl::IsOutOfRange(r.status())) {
      is_full_read_ = true;
      options_.byte_range = {};  // Full read.
      auto retry_future =
          internal::GetOwningCache(*entry_).kvstore_driver_->Read(
              std::string(entry_->key()), options_);
      retry_future.Force();
      retry_future.ExecuteWhenReady(
          [self = internal::IntrusivePtr<ReadDirectoryOp>(this)](
              ReadyFuture<kvstore::ReadResult> f) {
              self->OnReadComplete(std::move(f));
          });
      return;
      }
      entry_->ReadError(internal::ConvertInvalidArgumentToFailedPrecondition(r.status()));
      return;
    }

    auto& read_result = *r;
    if (read_result.not_found()) {
      entry_->ReadError(absl::NotFoundError(""));
      return;
    }

    if (read_result.aborted()) {
      // Return existing data if we have it
      if (existing_read_data_) {
        entry_->ReadSuccess(TiffDirectoryCache::ReadState{
            existing_read_data_,
            std::move(read_result.stamp)
        });
        return;
      }
      entry_->ReadError(absl::AbortedError("Read aborted"));
      return;
    }

    TiffDirectoryParseResult result;
    result.raw_data = std::move(read_result.value);
    // If we asked for a slice but got fewer than requested bytes,
    // we effectively have the whole file.
    if (!is_full_read_ &&
          result.raw_data.size() < internal_tiff_kvstore::kInitialReadBytes) {
        result.full_read = true;
      } else {
        result.full_read = is_full_read_;
      }
  
    entry_->ReadSuccess(TiffDirectoryCache::ReadState{
        std::make_shared<TiffDirectoryParseResult>(std::move(result)),
        std::move(read_result.stamp)
    });
  }
};

}  // namespace

size_t TiffDirectoryCache::Entry::ComputeReadDataSizeInBytes(
    const void* read_data) {
  return static_cast<const ReadData*>(read_data)->raw_data.size();
}

void TiffDirectoryCache::Entry::DoRead(AsyncCacheReadRequest request) {
  auto state = internal::MakeIntrusivePtr<ReadDirectoryOp>();
  state->entry_ = this;
  state->options_.staleness_bound = request.staleness_bound;
  {
    ReadLock<ReadData> lock(*this);
    state->existing_read_data_ = lock.shared_data();
    state->options_.generation_conditions.if_not_equal = 
        lock.read_state().stamp.generation;
  }

  state->StartRead();
}

TiffDirectoryCache::Entry* TiffDirectoryCache::DoAllocateEntry() {
  return new Entry;
}

size_t TiffDirectoryCache::DoGetSizeofEntry() {
  return sizeof(Entry);
}

}  // namespace internal_tiff_kvstore
}  // namespace tensorstore