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

#include <atomic>
#include <chrono>  // NOLINT
#include <cstddef>
#include <memory>
#include <thread>  // NOLINT
#include <vector>

#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "tensorstore/internal/poly.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal {
Executor DetachedThreadPool(std::size_t num_threads);
}  // namespace internal
}  // namespace tensorstore

namespace {
using tensorstore::Executor;
using tensorstore::internal::DetachedThreadPool;

// Tests that the thread pool runs a task.
TEST(DetachedThreadPoolTest, Basic) {
  auto executor = DetachedThreadPool(1);
  absl::Notification notification;
  executor([&] { notification.Notify(); });
  notification.WaitForNotification();
}

// Tests that the thread pool runs two tasks concurrently.
TEST(DetachedThreadPoolTest, Concurrent) {
  auto executor = DetachedThreadPool(2);
  absl::Notification notification1, notification2, notification3;
  executor([&] {
    notification1.Notify();
    notification2.WaitForNotification();
    notification3.Notify();
  });
  executor([&] {
    notification1.WaitForNotification();
    notification2.Notify();
  });
  notification1.WaitForNotification();
  notification2.WaitForNotification();
  notification3.WaitForNotification();
}

// Tests that the thread pool does not run more than the maximum number of tasks
// concurrently.
TEST(DetachedThreadPoolTest, ThreadLimit) {
  constexpr static size_t kThreadLimit = 3;
  auto executor = DetachedThreadPool(kThreadLimit);
  std::atomic<std::size_t> num_running_tasks{0};
  std::vector<absl::Notification> notifications(5);
  for (size_t i = 0; i < notifications.size(); ++i) {
    executor([&num_running_tasks, &notifications, i] {
      auto& notification = notifications[i];
      EXPECT_LE(++num_running_tasks, kThreadLimit);
      std::this_thread::sleep_for(std::chrono::seconds(1));
      --num_running_tasks;
      notification.Notify();
    });
  }
  for (size_t i = 0; i < notifications.size(); ++i) {
    notifications[i].WaitForNotification();
  }
}

// Tests that enqueuing a task from a task's destructor does not deadlock.
TEST(DetachedThreadPoolTest, EnqueueFromTaskDestructor) {
  auto executor = DetachedThreadPool(1);
  absl::Notification notification1;
  absl::Notification notification2;
  struct Task {
    Executor& executor;
    absl::Notification* notification1;
    absl::Notification* notification2;
    void operator()() { notification1->Notify(); }
    Task(const Task&) = delete;
    ~Task() {
      executor(
          [notification2 = this->notification2] { notification2->Notify(); });
    }
  };
  struct TaskWrapper {
    std::unique_ptr<Task> task;
    void operator()() { (*task)(); }
  };
  executor(TaskWrapper{std::unique_ptr<Task>(
      new Task{executor, &notification1, &notification2})});
  notification1.WaitForNotification();
  notification2.WaitForNotification();
}

}  // namespace
