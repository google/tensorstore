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

#ifndef TENSORSTORE_INTERNAL_THREAD_POOL_IMPL_H_
#define TENSORSTORE_INTERNAL_THREAD_POOL_IMPL_H_

#include <stddef.h>

#include <cassert>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/container/circular_queue.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/thread/task_provider.h"

namespace tensorstore {
namespace internal_thread_impl {

/// Dynamically-sized thread pool shared by multiple `TaskProvider` objects.
///
/// Worker threads are started automatically at a limited rate when needed
/// for registered TaskProviders. Threads are started by an overseer thread
/// to provide rate-limiting and fairness.
///
/// Both worker threads and the overseer thread automatically terminate after
/// they are idle for longer than `kThreadIdleBeforeExit` or
/// `kOverseerIdleBeforeExit`, respectively.
class SharedThreadPool
    : public internal::AtomicReferenceCount<SharedThreadPool> {
 public:
  SharedThreadPool();

  /// TaskProviderMethod:  Notify that there is work available.
  /// If the task provider identified by the token is not in the waiting_
  /// queue, add it.
  void NotifyWorkAvailable(internal::IntrusivePtr<TaskProvider>)
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  struct Overseer;
  struct Worker;

  // Gets the next TaskProvider with work available where the last thread
  // assignment time was before the deadline.
  internal::IntrusivePtr<TaskProvider> FindActiveTaskProvider()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  /// Starts the overseer thread.
  void StartOverseer() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  /// Starts a worker thread.
  void StartWorker(internal::IntrusivePtr<TaskProvider>, absl::Time now)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  absl::Mutex mutex_;
  size_t worker_threads_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t idle_threads_ ABSL_GUARDED_BY(mutex_) = 0;

  // Overseer state.
  absl::CondVar overseer_condvar_;
  bool overseer_running_ ABSL_GUARDED_BY(mutex_) = false;

  // Timing information for thread startup/shutdown.
  absl::Time last_thread_start_time_ ABSL_GUARDED_BY(mutex_) =
      absl::InfinitePast();
  absl::Time last_thread_exit_time_ ABSL_GUARDED_BY(mutex_) =
      absl::InfinitePast();
  absl::Time queue_assignment_time_ ABSL_GUARDED_BY(mutex_) =
      absl::InfinitePast();

  absl::flat_hash_set<TaskProvider*> in_queue_ ABSL_GUARDED_BY(mutex_);
  internal_container::CircularQueue<internal::IntrusivePtr<TaskProvider>>
      waiting_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace internal_thread_impl
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_THREAD_POOL_IMPL_H_
