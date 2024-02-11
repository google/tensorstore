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

#include <cassert>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/thread/task.h"
#include "tensorstore/internal/thread/task_provider.h"

namespace {

using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::MakeIntrusivePtr;
using ::tensorstore::internal_thread_impl::InFlightTask;
using ::tensorstore::internal_thread_impl::SharedThreadPool;
using ::tensorstore::internal_thread_impl::TaskProvider;

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
    absl::MutexLock lock(&mutex_);
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
      absl::MutexLock lock(&mutex_);
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
        pool, std::make_unique<InFlightTask>([&] { notification.Notify(); }));

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
          pool, std::make_unique<InFlightTask>([&] { a.DecrementCount(); })));
    }
    for (auto& p : providers) p->Trigger();
    a.Wait();
    for (auto& p : providers) p->Trigger();
    providers.clear();
  }
}

}  // namespace
