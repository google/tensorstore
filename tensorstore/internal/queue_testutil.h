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

#include <deque>
#include <optional>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace tensorstore {
namespace internal {

/// Thread-safe queue used for tests.
template <typename T>
class ConcurrentQueue {
 public:
  void push(T x) {
    absl::MutexLock lock(&mutex_);
    queue_.push_back(std::move(x));
  }

  T pop() {
    absl::MutexLock lock(&mutex_);
    // If 5 seconds isn't enough, assume the test has failed.  This avoids
    // delaying failure until the entire test times out.
    ABSL_CHECK(mutex_.AwaitWithTimeout(
        absl::Condition(
            +[](std::deque<T>* q) { return !q->empty(); }, &queue_),
        absl::Seconds(5)));
    T x = std::move(queue_.front());
    queue_.pop_front();
    return x;
  }

  std::optional<T> pop_nonblock() {
    std::optional<T> x;
    absl::MutexLock lock(&mutex_);
    if (!queue_.empty()) {
      x.emplace(std::move(queue_.front()));
      queue_.pop_front();
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

  /// Removes all elements from the queue.
  std::deque<T> pop_all() {
    absl::MutexLock lock(&mutex_);
    return std::exchange(queue_, {});
  }

 private:
  std::deque<T> queue_;
  absl::Mutex mutex_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_QUEUE_TESTUTIL_H_
