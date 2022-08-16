// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_MULTI_BARRIER_H_
#define TENSORSTORE_INTERNAL_MULTI_BARRIER_H_

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

namespace tensorstore {
namespace internal {

/// A barrier which blocks threads until a prespecified number of threads
/// utilizes the barrier, then resets allowing the barrier to be used for
/// sequence points (similar to std::barrier). A thread utilizes the
/// `MultiBarrier` by calling `Block()` on the barrier, which will block that
/// thread; no call to `Block()` will return until `num_threads` threads have
/// called it.
///
/// After `num_threads` have called Block(), exactly one call will return
/// `true`, and the barrier will set to an initial state.
///
/// Example:
///
///   // Main thread creates a `Barrier`:
///   MultiBarrier barrier(kNumThreads);
///
///   // Each participating thread could then call:
///   std::atomic<int> winner[kNumThreads] = {};
///   auto thread_func = [&, index]() {
///     while (!done) {
///       if (barrier.Block()) {
///         winner[index]++;
///       }
///     }
///   }
///
class MultiBarrier {
 public:
  /// Construct a MultiBarrier.
  ///
  /// \param num_threads  Number of threads that participate in the barrier.
  explicit MultiBarrier(int num_threads);
  ~MultiBarrier();

  /// Blocks the current thread, and returns only when the `num_threads`
  /// threshold of threads utilizing this barrier has been reached. `Block()`
  /// returns `true` for precisely one caller at each sync point.
  bool Block();

 private:
  absl::Mutex lock_;
  int blocking_[2] ABSL_GUARDED_BY(lock_);
  int asleep_ ABSL_GUARDED_BY(lock_);
  int num_threads_ ABSL_GUARDED_BY(lock_);
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_MULTI_BARRIER_H_
