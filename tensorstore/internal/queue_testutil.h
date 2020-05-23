// Copyright 2020 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_QUEUE_TESTUTIL_H_
#define TENSORSTORE_INTERNAL_QUEUE_TESTUTIL_H_

#include <optional>
#include <queue>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "tensorstore/util/assert_macros.h"

namespace tensorstore {
namespace internal {

/// Thread-safe queue used for tests.
template <typename T>
class ConcurrentQueue {
 public:
  void push(T x) {
    absl::MutexLock lock(&mutex_);
    queue_.push(std::move(x));
  }

  T pop() {
    absl::MutexLock lock(&mutex_);
    // If 5 seconds isn't enough, assume the test has failed.  This avoids
    // delaying failure until the entire test times out.
    TENSORSTORE_CHECK(mutex_.AwaitWithTimeout(
        absl::Condition(
            +[](std::queue<T>* q) { return !q->empty(); }, &queue_),
        absl::Seconds(5)));
    T x = std::move(queue_.front());
    queue_.pop();
    return x;
  }

  std::optional<T> pop_nonblock() {
    std::optional<T> x;
    absl::MutexLock lock(&mutex_);
    if (!queue_.empty()) {
      x.emplace(std::move(queue_.front()));
      queue_.pop();
    }
    return x;
  }

  /// Returns the size.
  ///
  /// Requires external synchronization to ensure a meaningful result.
  std::size_t size() {
    absl::MutexLock lock(&mutex_);
    return queue_.size();
  }

  /// Returns `true` if empty.
  ///
  /// Requires external synchronization to ensure a meaningful result.
  bool empty() { return size() == 0; }

 private:
  std::queue<T> queue_;
  absl::Mutex mutex_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_QUEUE_TESTUTIL_H_
