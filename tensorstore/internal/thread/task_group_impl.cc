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

#include "tensorstore/internal/thread/task_group_impl.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <memory>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/thread/pool_impl.h"
#include "tensorstore/internal/thread/task.h"
#include "tensorstore/internal/thread/task_provider.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_thread_impl {
namespace {

auto& thread_pool_total_queue_time_ns = internal_metrics::Counter<double>::New(
    "/tensorstore/thread_pool/total_queue_time_ns",
    "DetachedThreadPool total queue time for all TaskGroup instances");

auto& thread_pool_max_delay_ns = internal_metrics::MaxGauge<int64_t>::New(
    "/tensorstore/thread_pool/max_delay_ns",
    "DetachedThreadPool max delay for all TaskGroup instances");

auto& thread_pool_total_work_time_ns = internal_metrics::Counter<double>::New(
    "/tensorstore/thread_pool/work_time_ns",
    "DetachedThreadPool total queue time for all TaskGroup instances");

auto& thread_pool_steal_count = internal_metrics::Counter<double>::New(
    "/tensorstore/thread_pool/steal_count",
    "DetachedThreadPool total queue time for all TaskGroup instances");

constexpr absl::Duration kThreadAssignmentLifetime = absl::Milliseconds(20);

thread_local TaskGroup::PerThreadData* per_thread_data = nullptr;

// Tunable parameter: Steal up to 1/2 the pending items (max 16) and move
// them to the global queue_.
inline size_t ItemsToMigrateToGlobalQueue(size_t available) {
  return (std::min)(size_t{16}, available >> 1);
}

// Tunable parameter: Self-assign up to 2 additional tasks (max 1/8 available).
inline size_t ItemsToSelfAssign(size_t default_assign, size_t available) {
  return (std::min)(default_assign, available >> 3);
}

// ThreadMetrics is used to batch-update the tensorstore metrics.
struct ThreadMetrics {
  constexpr static int64_t kUpdateAfterNS = 100000000;  // 100ms
  int64_t total_queue_time_ns = 0;
  int64_t max_delay_ns = 0;
  int64_t work_time_ns = 0;

  int64_t task_queue_ns_ = 0;
  int64_t start_time_ns_ = 0;

  void Update() {
    if (total_queue_time_ns > 0) {
      thread_pool_total_queue_time_ns.IncrementBy(total_queue_time_ns);
      total_queue_time_ns = 0;
    }
    if (max_delay_ns > 0) {
      thread_pool_max_delay_ns.Set(max_delay_ns);
      max_delay_ns = 0;
    }
    if (work_time_ns > 0) {
      thread_pool_total_work_time_ns.IncrementBy(work_time_ns);
      work_time_ns = 0;
    }
  }

  void OnStart(int64_t task_queue_ns) {
    start_time_ns_ = absl::GetCurrentTimeNanos();
    task_queue_ns_ = task_queue_ns;
  }

  int64_t OnStop() {
    int64_t stop_time_ns = absl::GetCurrentTimeNanos();
    int64_t delay_ns = start_time_ns_ - task_queue_ns_;
    total_queue_time_ns += delay_ns;
    max_delay_ns = std::max(max_delay_ns, delay_ns);
    work_time_ns = stop_time_ns - start_time_ns_;
    if (total_queue_time_ns > kUpdateAfterNS) {
      Update();
    }
    return stop_time_ns;
  }
};

}  // namespace

struct TaskGroup::PerThreadData {
  std::atomic<void*> owner = nullptr;
  size_t default_assign = 1;
  InFlightTaskQueue queue{128};
  size_t slot = 0;
};

TaskGroup::TaskGroup(private_t, internal::IntrusivePtr<SharedThreadPool> pool,
                     size_t thread_limit)
    : pool_(std::move(pool)),
      thread_limit_(thread_limit),
      threads_blocked_(0),
      threads_in_use_(0),
      steal_index_(0),
      steal_count_(0) {}

TaskGroup::~TaskGroup() {
  assert(threads_in_use_.load(std::memory_order_relaxed) == 0);
  assert(queue_.empty());
}

int64_t TaskGroup::EstimateThreadsRequired() {
  size_t n = thread_limit_ - threads_in_use_.load(std::memory_order_relaxed);
  if (n == 0 || threads_blocked_.load(std::memory_order_relaxed) != 0) {
    return 0;
  }

  // Otherwise check the available tasks.
  absl::MutexLock lock(&mutex_);
  if (!queue_.empty()) {
    return std::min(n, queue_.size());
  }
  for (auto* p : thread_queues_) {
    if (!p->queue.empty()) return std::min(n, p->queue.size());
  }
  return 0;
}

void TaskGroup::DoWorkOnThread() {
  assert(per_thread_data == nullptr);

  auto data = std::make_shared<PerThreadData>();
  data->owner = this;

  {
    absl::MutexLock lock(&mutex_);
    if (threads_in_use_.load(std::memory_order_relaxed) == thread_limit_) {
      return;
    }
    threads_in_use_.fetch_add(1, std::memory_order_relaxed);
    thread_queues_.push_back(data.get());
    data->slot = thread_queues_.size() - 1;
    per_thread_data = data.get();
  }

  int64_t last_run_ns = absl::GetCurrentTimeNanos();
  ThreadMetrics metrics;

  // As long as there is work available, do it on this thread.
  while (true) {
    // Acquire a task to work on.
    auto task = AcquireTask(data.get(), kThreadAssignmentLifetime);
    if (task) {
      metrics.OnStart(task->start_nanos);
      task->Run();
      last_run_ns = metrics.OnStop();
      continue;
    }

    // Thread has already waited for the deadline, maybe reassign the thread
    // or determine if the thread should exit.
    auto idle = absl::Nanoseconds(absl::GetCurrentTimeNanos() - last_run_ns);
    if (idle > kThreadAssignmentLifetime) {
      // If a thread has been idle for 20ms, migrate it to the global pool.
      break;
    }
  }

  // Update stats.
  metrics.Update();

  {
    absl::MutexLock lock(&mutex_);
    threads_in_use_.fetch_sub(1, std::memory_order_relaxed);
    if (data->slot != thread_queues_.size() - 1) {
      thread_queues_[data->slot] = thread_queues_.back();
      thread_queues_[data->slot]->slot = data->slot;
    }
    thread_queues_.pop_back();
  }

  per_thread_data = nullptr;
}

/// Acquire a task.
std::unique_ptr<InFlightTask> TaskGroup::AcquireTask(PerThreadData* thread_data,
                                                     absl::Duration timeout) {
  struct ScopedIncDec {
    std::atomic<int64_t>& x_;
    ScopedIncDec(std::atomic<int64_t>& x) : x_(x) {
      x_.fetch_add(1, std::memory_order_relaxed);
    }
    ~ScopedIncDec() { x_.fetch_sub(1, std::memory_order_relaxed); }
  };

  // First, attempt to acquire a task from the local queue.
  if (auto* t = thread_data->queue.try_pop(); t != nullptr) {
    return std::unique_ptr<InFlightTask>(t);
  }

  absl::MutexLock lock(&mutex_);
  while (true) {
    // Second, attempt to acquire a task from the global queue.
    if (!queue_.empty()) {
      std::unique_ptr<InFlightTask> task = std::move(queue_.front());
      queue_.pop_front();

      // Tunable parameter: Preemptively assign additional items to self.
      size_t x = ItemsToSelfAssign(thread_data->default_assign, queue_.size());
      while (x--) {
        thread_data->queue.push(queue_.front().release());
        queue_.pop_front();
      }

      if (thread_data->default_assign < 16) {
        thread_data->default_assign *= 2;
      }

      return task;
    }

    thread_data->default_assign = 1;

    // Third, migrate tasks from per-thread queues.
    for (size_t i = 0; i < thread_queues_.size(); ++i, ++steal_index_) {
      if (steal_index_ >= thread_queues_.size()) steal_index_ = 0;
      auto* other_data = thread_queues_[steal_index_];
      if (!other_data || other_data == thread_data) continue;
      std::unique_ptr<InFlightTask> task(other_data->queue.try_steal());
      if (!task) continue;
      // Tunable parameter: Items to steal and move to the global queue.
      size_t x = ItemsToMigrateToGlobalQueue(other_data->queue.size());
      while (x--) {
        std::unique_ptr<InFlightTask> t(other_data->queue.try_steal());
        if (!t) break;
        queue_.push_back(std::move(t));
      }

      thread_pool_steal_count.IncrementBy(1);
      return task;
    }

    // No tasks acquired; wait until more work appears on the global queue.
    ScopedIncDec blocked(threads_blocked_);
    if (!mutex_.AwaitWithTimeout(
            absl::Condition(
                +[](decltype(queue_)* q) { return !q->empty(); }, &queue_),
            timeout)) {
      return nullptr;
    }
  }
  ABSL_UNREACHABLE();
}

/////////////////////////////////////////////////////////////////////////////

void TaskGroup::AddTask(std::unique_ptr<InFlightTask> task) {
  int state = 2;
  if (per_thread_data != nullptr &&
      per_thread_data->owner.load(std::memory_order_relaxed) == this) {
    // Add on the current-thread's queue.
    if (per_thread_data->queue.push(task.get())) {
      task.release();
      state = 0;
    } else {
      state = 1;
    }
  }

  if (state != 0) {
    absl::MutexLock lock(&mutex_);

    if (state == 1) {
      // The local queue is full; migrate up to 1/2 the tasks.
      int n_to_migrate = per_thread_data->queue.size() >> 1;
      for (int i = 0; i < n_to_migrate; i++) {
        InFlightTask* t = per_thread_data->queue.try_pop();
        if (t != nullptr) {
          queue_.push_back(std::unique_ptr<InFlightTask>(t));
        } else {
          break;
        }
      }
    }

    queue_.push_back(std::move(task));
  }

  if (threads_in_use_.load(std::memory_order_relaxed) < thread_limit_) {
    pool_->NotifyWorkAvailable(internal::IntrusivePtr<TaskProvider>(this));
  }
}

void TaskGroup::BulkAddTask(
    tensorstore::span<std::unique_ptr<InFlightTask>> tasks) {
  {
    absl::MutexLock lock(&mutex_);
    for (auto& t : tasks) {
      queue_.push_back(std::move(t));
    }
  }
  if (threads_in_use_.load(std::memory_order_relaxed) < thread_limit_) {
    pool_->NotifyWorkAvailable(internal::IntrusivePtr<TaskProvider>(this));
  }
}

}  // namespace internal_thread_impl
}  // namespace tensorstore
