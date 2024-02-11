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

#ifndef TENSORSTORE_INTERNAL_STORAGE_STATISTICS_H_
#define TENSORSTORE_INTERNAL_STORAGE_STATISTICS_H_

#include <atomic>

#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal {

// Base class for operation state that may be used to implement
// `Driver::GetStorageStatistics` in a composable way.
struct GetStorageStatisticsAsyncOperationState
    : public internal::AtomicReferenceCount<
          GetStorageStatisticsAsyncOperationState> {
  // Constructs a new operation state.
  //
  // Args:
  //   future[out]: Set to the future representing the storage statistics result
  //     on return.
  //   options: Storage statistics options.
  explicit GetStorageStatisticsAsyncOperationState(
      Future<ArrayStorageStatistics>& future,
      const GetArrayStorageStatisticsOptions& options);

  // Number of chunks known to be stored.
  std::atomic<int64_t> chunks_present{0};

  // Total number of chunks known to be required.
  std::atomic<int64_t> total_chunks = 0;

  // Options passed to constructor.
  GetArrayStorageStatisticsOptions options;

  // Promise representing the storage statistics result.
  Promise<ArrayStorageStatistics> promise;

  // Indicates that at least one chunk is known to be missing.
  std::atomic<bool> chunk_missing{false};

  // Check if we can stop early.
  //
  // Sets a deferred result on `promise` if the result is known.
  void MaybeStopEarly();

  // Must be called when a chunk is known to be present.  Depending on
  // `options`, this may result in `promise.result_needed()` becoming `false`.
  void IncrementChunksPresent() {
    if (++chunks_present == 1) {
      MaybeStopEarly();
    }
  }

  // Must be called when a chunk is known to be missing.  Depending on
  // `options`, this may result in `promise.result_needed()` becoming `false`.
  void ChunkMissing() {
    if (chunk_missing.exchange(true) == false) {
      MaybeStopEarly();
    }
  }

  // Called to indicate that an error occurred.
  void SetError(absl::Status error) {
    SetDeferredResult(promise, std::move(error));
  }

  // Marks `promise` ready.
  virtual ~GetStorageStatisticsAsyncOperationState();
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_STORAGE_STATISTICS_H_
