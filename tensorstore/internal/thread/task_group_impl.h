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

#ifndef TENSORSTORE_INTERNAL_THREAD_TASK_GROUP_IMPL_H_
#define TENSORSTORE_INTERNAL_THREAD_TASK_GROUP_IMPL_H_

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <cassert>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/container/block_queue.h"
#include "tensorstore/internal/container/single_producer_queue.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/thread/pool_impl.h"
#include "tensorstore/internal/thread/task.h"
#include "tensorstore/internal/thread/task_provider.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_thread_impl {

using InFlightTaskQueue =
    internal_container::SingleProducerQueue<InFlightTask*, false>;

/// TaskGroup is TaskProvider which allows adding additional tasks to a
/// task provider, and allowing up to a specific number of threads to
/// work on the tasks concurrently.
class TaskGroup : public TaskProvider {
  struct private_t {};

 public:
  struct PerThreadData;

  static internal::IntrusivePtr<TaskGroup> Make(
      internal::IntrusivePtr<SharedThreadPool> pool, size_t thread_limit) {
    return internal::MakeIntrusivePtr<TaskGroup>(private_t{}, std::move(pool),
                                                 thread_limit);
  }

  TaskGroup(private_t, internal::IntrusivePtr<SharedThreadPool> pool,
            size_t thread_limit);

  ~TaskGroup() override;

  /// Enqueues a task.
  ///
  /// Thread safety: safe to call concurrently from multiple threads.
  void AddTask(std::unique_ptr<InFlightTask> task);

  /// Enqueues a task.
  ///
  /// Thread safety: safe to call concurrently from multiple threads.
  void BulkAddTask(tensorstore::span<std::unique_ptr<InFlightTask>> tasks);

  /// Retrieve work units available.
  int64_t EstimateThreadsRequired() override;

  /// Worker method: Assign a thread to this task provider.
  void DoWorkOnThread() override;

 private:
  /// Worker method: Acquire work from the global queue or another thread.
  std::unique_ptr<InFlightTask> AcquireTask(PerThreadData* thread_data,
                                            absl::Duration timeout);

  const internal::IntrusivePtr<SharedThreadPool> pool_;
  const size_t thread_limit_;

  // worker thread state counters; updated under lock, read without locks.
  ABSL_CACHELINE_ALIGNED std::atomic<int64_t> threads_blocked_;
  std::atomic<int64_t> threads_in_use_;

  absl::Mutex mutex_;
  internal_container::BlockQueue<std::unique_ptr<InFlightTask>> queue_
      ABSL_GUARDED_BY(mutex_);
  std::vector<PerThreadData*> thread_queues_ ABSL_GUARDED_BY(mutex_);
  size_t steal_index_ ABSL_GUARDED_BY(mutex_);
  size_t steal_count_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace internal_thread_impl
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_THREAD_TASK_GROUP_IMPL_H_
