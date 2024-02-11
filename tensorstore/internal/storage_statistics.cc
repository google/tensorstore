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

#include "tensorstore/internal/storage_statistics.h"

#include <stdint.h>

#include <atomic>
#include <utility>

#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal {

GetStorageStatisticsAsyncOperationState::
    GetStorageStatisticsAsyncOperationState(
        Future<ArrayStorageStatistics>& future,
        const GetArrayStorageStatisticsOptions& options)
    : options(options) {
  // Initialize the contained `Result` with default-constructed
  // `ArrayStorageStatistics`.  The destructor of `AsyncOperationState` checks
  // if it has been replaced with an error status, before setting the actual
  // statistics.
  auto p = PromiseFuturePair<ArrayStorageStatistics>::Make(std::in_place);
  this->promise = std::move(p.promise);
  future = std::move(p.future);
}

void GetStorageStatisticsAsyncOperationState::MaybeStopEarly() {
  if (options.mask & ArrayStorageStatistics::query_not_stored) {
    if (chunks_present.load() == 0) {
      // Don't yet know if any data is stored.
      return;
    }
  }

  if (options.mask & ArrayStorageStatistics::query_fully_stored) {
    if (chunk_missing.load() == false) {
      // Don't yet know if any data is missing.
      return;
    }
  }

  // Mark `promise` as `result_not_needed`.  The actual statistics will be set
  // by `~GetStorageStatisticsAsyncOperationState`.
  SetDeferredResult(promise, ArrayStorageStatistics{});
}

GetStorageStatisticsAsyncOperationState::
    ~GetStorageStatisticsAsyncOperationState() {
  auto& r = promise.raw_result();
  if (!r.ok()) return;
  r->mask = options.mask;
  int64_t num_present = chunks_present.load(std::memory_order_relaxed);
  if (options.mask & ArrayStorageStatistics::query_not_stored) {
    r->not_stored = (num_present == 0);
  }
  if (options.mask & ArrayStorageStatistics::query_fully_stored) {
    r->fully_stored = num_present == total_chunks;
  }
}

}  // namespace internal
}  // namespace tensorstore
