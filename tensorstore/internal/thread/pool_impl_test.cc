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

#include "tensorstore/internal/thread/pool_impl.h"

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <cassert>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/thread/task.h"
#include "tensorstore/internal/thread/task_group_impl.h"
#include "tensorstore/internal/thread/task_provider.h"
#include "tensorstore/internal/thread/thread.h"
#include "tensorstore/internal/tracing/trace_context.h"

namespace {

using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::MakeIntrusivePtr;
using ::tensorstore::internal_thread_impl::InFlightTask;
using ::tensorstore::internal_thread_impl::SharedThreadPool;
using ::tensorstore::internal_thread_impl::TaskGroup;
using ::tensorstore::internal_thread_impl::TaskProvider;
using TC = ::tensorstore::internal_tracing::TraceContext;

struct SingleTaskProvider : public TaskProvider {
  struct private_t {};

 public:
  static IntrusivePtr<SingleTaskProvider> Make(
      IntrusivePtr<SharedThreadPool> pool, std::unique_ptr<InFlightTask> task) {
    return MakeIntrusivePtr<SingleTaskProvider>(private_t{}, std::move(pool),
                                                std::move(task));
  }

  SingleTaskProvider(private_t, IntrusivePtr<SharedThreadPool> pool,
                     std::unique_ptr<InFlightTask> task)
      : pool_(std::move(pool)), task_(std::move(task)) {}

  ~SingleTaskProvider() override = default;

  int64_t EstimateThreadsRequired() override {
    absl::MutexLock lock(mutex_);
    flags_ += 2;
    return task_ ? 1 : 0;
  }

  void Trigger() {
    pool_->NotifyWorkAvailable(IntrusivePtr<TaskProvider>(this));
  }

  /// Worker Method: Assign a thread to this task provider.
  /// If an assignment cannot be made, returns false.
  void DoWorkOnThread() override {
    std::unique_ptr<InFlightTask> task;

    // Acquire task
    {
      absl::MutexLock lock(mutex_);
      flags_ |= 1;
      if (task_) {
        task = std::move(task_);
      }
    }

    // Run task
    if (task) {
      task->Run();
    }
  }

  IntrusivePtr<SharedThreadPool> pool_;

  absl::Mutex mutex_;
  std::unique_ptr<InFlightTask> task_ ABSL_GUARDED_BY(mutex_);
  int64_t flags_ = 0;
};

// Tests that the thread pool runs a task.
TEST(SharedThreadPoolTest, Basic) {
  auto pool = MakeIntrusivePtr<SharedThreadPool>();

  {
    absl::Notification notification;
    auto provider = SingleTaskProvider::Make(
        pool, std::make_unique<InFlightTask>([&] { notification.Notify(); },
                                             TC(TC::kThread)));

    provider->Trigger();
    provider->Trigger();

    notification.WaitForNotification();
  }
}

// Tests that the thread pool runs a task.
TEST(SharedThreadPoolTest, LotsOfProviders) {
  auto pool = MakeIntrusivePtr<SharedThreadPool>();

  std::vector<IntrusivePtr<SingleTaskProvider>> providers;
  providers.reserve(1000);

  for (int i = 2; i < 1000; i = i * 2) {
    absl::BlockingCounter a(i);
    for (int j = 0; j < i; j++) {
      providers.push_back(SingleTaskProvider::Make(
          pool, std::make_unique<InFlightTask>([&] { a.DecrementCount(); },
                                               TC(TC::kThread))));
    }
    for (auto& p : providers) p->Trigger();
    a.Wait();
    for (auto& p : providers) p->Trigger();
    providers.clear();
  }
}

// Regression test: when an active worker W1 on a TaskGroup
// spawns a task via AddTask() while another worker W2 is blocked in
// AcquireTask() waiting on the global queue_, W2 must be woken (or another
// worker dispatched) so W1 waiting on the child task does not deadlock.
TEST(TaskGroupTest, ActiveWorkerAddTaskWakesBlockedWorker) {
  auto pool = MakeIntrusivePtr<SharedThreadPool>();
  auto group = TaskGroup::Make(pool, /*thread_limit=*/2);

  absl::Notification w2_started;
  absl::Notification w2_release;
  absl::Notification child_done;
  absl::Notification w1_done;

  // Task 1 (runs on W1): waits for W2 to start, releases W2 so W2 blocks in
  // AcquireTask(), then enqueues a child task from within W1 and waits for it.
  group->AddTask(std::make_unique<InFlightTask>(
      [&] {
        w2_started.WaitForNotification();
        w2_release.Notify();
        // Give W2 time to finish Task 2 and enter AcquireTask()'s
        // AwaitWithTimeout on the global queue_.
        absl::SleepFor(absl::Milliseconds(5));

        group->AddTask(std::make_unique<InFlightTask>(
            [&] { child_done.Notify(); }, TC(TC::kThread)));

        EXPECT_TRUE(
            child_done.WaitForNotificationWithTimeout(absl::Milliseconds(500)));
        w1_done.Notify();
      },
      TC(TC::kThread)));

  // Task 2 (runs on W2): signals W1 and immediately returns so W2 enters
  // AcquireTask() and blocks with threads_blocked_ == 1.
  group->AddTask(std::make_unique<InFlightTask>(
      [&] {
        w2_started.Notify();
        w2_release.WaitForNotification();
      },
      TC(TC::kThread)));

  w1_done.WaitForNotification();
  child_done.WaitForNotification();
}

// Regression test: when a worker times out
// in AcquireTask() after kThreadAssignmentLifetime (20ms) and exits
// DoWorkOnThread() while tasks are queued, DoWorkOnThread() must notify
// SharedThreadPool so queued tasks do not hang indefinitely.
TEST(TaskGroupTest, WorkerExitWithPendingWorkNotifiesPool) {
  auto pool = MakeIntrusivePtr<SharedThreadPool>();
  constexpr size_t kNumGroups = 16;
  std::vector<IntrusivePtr<TaskGroup>> groups;
  groups.reserve(kNumGroups);
  for (size_t i = 0; i < kNumGroups; ++i) {
    groups.push_back(TaskGroup::Make(pool, /*thread_limit=*/1));
  }

  // Start one task on each single-thread TaskGroup with a backdated
  // start_nanos so ThreadMetrics::Update() executes when DoWorkOnThread()
  // exits after kThreadAssignmentLifetime (20ms).
  absl::BlockingCounter initial_done(kNumGroups);
  for (size_t i = 0; i < kNumGroups; ++i) {
    auto task = std::make_unique<InFlightTask>(
        [&initial_done] { initial_done.DecrementCount(); }, TC(TC::kThread));
    task->start_nanos = absl::GetCurrentTimeNanos() - 200000000;
    groups[i]->AddTask(std::move(task));
  }
  initial_done.Wait();

  // Wait until right around the 20ms kThreadAssignmentLifetime expiry window,
  // contend on mutex_ via empty BulkAddTask, and enqueue a follow-up task on
  // each group.
  absl::SleepFor(absl::Milliseconds(19));
  std::atomic<bool> stop_contend{false};
  tensorstore::internal::Thread contender({"contender"}, [&] {
    while (!stop_contend.load(std::memory_order_relaxed)) {
      for (auto& g : groups) {
        g->BulkAddTask({});
      }
    }
  });

  absl::SleepFor(absl::Milliseconds(1));
  std::vector<std::unique_ptr<absl::Notification>> follow_up_done;
  follow_up_done.reserve(kNumGroups);
  for (size_t i = 0; i < kNumGroups; ++i) {
    follow_up_done.push_back(std::make_unique<absl::Notification>());
    auto* done = follow_up_done.back().get();
    groups[i]->AddTask(std::make_unique<InFlightTask>(
        [done] { done->Notify(); }, TC(TC::kThread)));
    absl::SleepFor(absl::Microseconds(100));
  }
  stop_contend.store(true, std::memory_order_relaxed);
  contender.Join();

  for (size_t i = 0; i < kNumGroups; ++i) {
    bool ok =
        follow_up_done[i]->WaitForNotificationWithTimeout(absl::Seconds(1));
    EXPECT_TRUE(ok) << "Task hung on group " << i;
    if (!ok) {
      // Trigger pool notification so destructor does not assert on hung queue.
      pool->NotifyWorkAvailable(groups[i]);
      follow_up_done[i]->WaitForNotification();
    }
  }
}

}  // namespace
