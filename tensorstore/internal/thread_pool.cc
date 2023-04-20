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

#include <algorithm>
#include <cassert>
#include <queue>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/internal/thread.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal {

namespace {

constexpr absl::Duration kThreadStartDelay = absl::Milliseconds(5);
constexpr absl::Duration kThreadExitDelay = absl::Milliseconds(5);
constexpr absl::Duration kThreadIdleBeforeExit = absl::Seconds(20);
constexpr absl::Duration kOverseerIdleBeforeExit = absl::Seconds(20);

auto& thread_pool_queued_ops = internal_metrics::Gauge<int64_t>::New(
    "/tensorstore/thread_pool/queued_ops",
    "Operations in flight on the managed thread pool");

auto& thread_pool_total_queue_time_ns = internal_metrics::Counter<double>::New(
    "/tensorstore/thread_pool/total_queue_time_ns",
    "Total queue time in in all tensorstore::DetachedThreadPool instances");

auto& thread_pool_max_delay_ns = internal_metrics::MaxGauge<int64_t>::New(
    "/tensorstore/thread_pool/max_delay_ns",
    "Max queue time in all tensorstore::DetachedThreadPool instances.");

class SharedThreadPool;
class ManagedTaskQueue;

struct InFlightTask {
  absl::AnyInvocable<void() &&> callback;
  int64_t start_nanos;
  IntrusivePtr<ManagedTaskQueue> managed_queue;

  void Run() {
    int64_t delay_ns = absl::GetCurrentTimeNanos() - start_nanos;
    std::move(callback)();
    callback = {};  // Ensure the task destructor runs.
    thread_pool_total_queue_time_ns.IncrementBy(delay_ns);
    thread_pool_max_delay_ns.Set(delay_ns);
    thread_pool_queued_ops.Decrement();
  }
};

class ManagedTaskQueue : public AtomicReferenceCount<ManagedTaskQueue> {
 public:
  explicit ManagedTaskQueue(IntrusivePtr<SharedThreadPool> pool,
                            std::size_t thread_limit);

  /// Enqueues a task.  Never blocks.
  ///
  /// Thread safety: safe to call concurrently from multiple threads.
  void AddTask(InFlightTask task);

  /// Called by `SharedThreadPool` when a task previously enqueued on the
  /// `SharedThreadPool` completes.
  ///
  /// Thread safety: safe to call concurrently from multiple threads.
  void TaskDone();

 private:
  const IntrusivePtr<SharedThreadPool> pool_;
  const std::size_t thread_limit_;
  absl::Mutex mutex_;
  std::size_t num_threads_in_use_ ABSL_GUARDED_BY(mutex_);
  std::queue<InFlightTask> queue_ ABSL_GUARDED_BY(mutex_);
};

/// Dynamically-sized thread pool shared by multiple `ManagedTaskQueue` objects.
///
/// Worker threads are started automatically at a limited rate when the queue is
/// blocked.  Threads may be started either directly from the thread that
/// enqueues a task, if the rate limit has not been exceeded, or from an
/// overseer thread that is created when needed to start threads at a limited
/// rate.
///
/// Both worker threads and the overseer thread automatically terminate after
/// they are idle for longer than `kThreadIdleBeforeExit` or
/// `kOverseerIdleBeforeExit`, respectively.  Worker threads are terminated only
/// at a limited rate.
class SharedThreadPool : public AtomicReferenceCount<SharedThreadPool> {
 public:
  /// Enqueues a task on the thread pool.
  void AddTask(InFlightTask task) ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  /// Called when the task queue becomes non-empty and `idle_threads_ == 0`.
  /// Starts a worker thread if it would not exceed the maximum thread creation
  /// rate, otherwise triggers the overseer thread to start a thread after a
  /// delay.
  void HandleQueueBlocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  /// Starts another worker thread.  This may be called either from the thread
  /// that enqueues a task, or from the overseer thread.
  void StartThread() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  /// Starts the overseer thread.  Called when the queue is blocked, there is no
  /// existing overseer thread, and the thread creation limited is reached.
  void StartOverseerThread() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool queue_blocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return !queue_.empty() && idle_threads_ == 0;
  }

  absl::Mutex mutex_;
  /// Signaled when `queue_blocked()` changes from `false` to `true`.
  absl::CondVar overseer_condvar_;
  std::queue<InFlightTask> queue_ ABSL_GUARDED_BY(mutex_);
  std::size_t idle_threads_ ABSL_GUARDED_BY(mutex_);
  bool has_overseer_thread_ ABSL_GUARDED_BY(mutex_) = false;
  absl::Time last_thread_start_time_ ABSL_GUARDED_BY(mutex_) =
      absl::InfinitePast();
  absl::Time last_thread_exit_time_ ABSL_GUARDED_BY(mutex_) =
      absl::InfinitePast();
  absl::Time queue_blocked_time_ ABSL_GUARDED_BY(mutex_) =
      absl::InfiniteFuture();
};

void SharedThreadPool::AddTask(InFlightTask task) {
  absl::MutexLock lock(&mutex_);
  queue_.push(std::move(task));
  if (idle_threads_ == 0) {
    HandleQueueBlocked();
  }
}

void SharedThreadPool::HandleQueueBlocked() {
  auto now = absl::Now();
  if (now >= last_thread_start_time_ + kThreadStartDelay) {
    // Start new thread immediately.
    StartThread();
  } else {
    // Overseer thread will start new thread after delay.
    queue_blocked_time_ = now;
    // Start overseer thread if it is isn't already running.
    if (!has_overseer_thread_) {
      StartOverseerThread();
    } else if (queue_.size() == 1) {
      // Queue was previously empty.  Wake up overseer thread.
      overseer_condvar_.Signal();
    }
  }
}

void SharedThreadPool::StartThread() {
  idle_threads_++;
  tensorstore::internal::Thread::StartDetached(
      {"pool_worker"}, [self = IntrusivePtr<SharedThreadPool>(this)] {
        absl::MutexLock lock(&self->mutex_);
        while (true) {
          absl::Time idle_start_time = absl::Now();
          if (self->queue_.empty()) {
            // Block until queue is empty, or until deadline to exit may have
            // been reached.
            while (!self->mutex_.AwaitWithDeadline(
                absl::Condition(
                    +[](SharedThreadPool* self) ABSL_EXCLUSIVE_LOCKS_REQUIRED(
                         self->mutex_) { return !self->queue_.empty(); },
                    self.get()),
                std::max(idle_start_time + kThreadIdleBeforeExit,
                         self->last_thread_exit_time_ + kThreadExitDelay))) {
              auto now = absl::Now();
              if (self->last_thread_exit_time_ + kThreadExitDelay <= now) {
                self->last_thread_exit_time_ = now;
                --self->idle_threads_;
                return;
              }
            }
          }

          InFlightTask task = std::move(self->queue_.front());
          self->queue_.pop();
          if (--self->idle_threads_ == 0 && !self->queue_.empty()) {
            self->HandleQueueBlocked();
          }

          // Execute task with mutex unlocked.
          {
            ScopedWriterUnlock unlock(self->mutex_);
            task.Run();
            task.managed_queue->TaskDone();
          }

          ++self->idle_threads_;
        }
      });
}

void SharedThreadPool::StartOverseerThread() {
  has_overseer_thread_ = true;
  tensorstore::internal::Thread::StartDetached(
      {"pool_overseer"}, [self = IntrusivePtr<SharedThreadPool>(this)] {
        absl::Time idle_start_time = absl::Now();
        absl::MutexLock lock(&self->mutex_);
        while (true) {
          auto now = absl::Now();
          absl::Time deadline;
          if (self->queue_blocked()) {
            deadline = self->queue_blocked_time_ + kThreadStartDelay;
            if (deadline <= now) {
              self->queue_blocked_time_ += kThreadStartDelay;
              self->StartThread();
              idle_start_time = absl::Now();
              continue;
            }
          } else {
            deadline = idle_start_time + kOverseerIdleBeforeExit;
            if (deadline <= now) {
              self->has_overseer_thread_ = false;
              return;
            }
          }
          self->overseer_condvar_.WaitWithDeadline(&self->mutex_, deadline);
        }
      });
}

ManagedTaskQueue::ManagedTaskQueue(IntrusivePtr<SharedThreadPool> pool,
                                   std::size_t thread_limit)
    : pool_(std::move(pool)),
      thread_limit_(thread_limit),
      num_threads_in_use_(0) {}

void ManagedTaskQueue::AddTask(InFlightTask task) {
  {
    absl::MutexLock lock(&mutex_);
    if (num_threads_in_use_ < thread_limit_) {
      ++num_threads_in_use_;
    } else {
      queue_.push(std::move(task));
      return;
    }
  }
  pool_->AddTask(std::move(task));
}

void ManagedTaskQueue::TaskDone() {
  InFlightTask task;
  {
    absl::MutexLock lock(&mutex_);
    if (queue_.empty()) {
      --num_threads_in_use_;
      return;
    }
    assert(num_threads_in_use_ == thread_limit_);
    task = std::move(queue_.front());
    queue_.pop();
  }
  pool_->AddTask(std::move(task));
}

}  // namespace

Executor DetachedThreadPool(std::size_t num_threads) {
  ABSL_CHECK_GT(num_threads, 0);
  static internal::NoDestructor<SharedThreadPool> pool_;
  intrusive_ptr_increment(pool_.get());
  IntrusivePtr<ManagedTaskQueue> managed_task_queue(new ManagedTaskQueue(
      IntrusivePtr<SharedThreadPool>(pool_.get()), num_threads));
  return
      [managed_task_queue = std::move(managed_task_queue)](ExecutorTask task) {
        thread_pool_queued_ops.Increment();
        managed_task_queue->AddTask(InFlightTask{
            std::move(task), absl::GetCurrentTimeNanos(), managed_task_queue});
      };
}

}  // namespace internal
}  // namespace tensorstore
