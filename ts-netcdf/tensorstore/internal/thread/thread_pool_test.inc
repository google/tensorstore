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

#ifndef THIRD_PARTY_TENSORSTORE_INTERNAL_THREAD_THREAD_POOL_TEST_INC_
#define THIRD_PARTY_TENSORSTORE_INTERNAL_THREAD_THREAD_POOL_TEST_INC_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/random/random.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/util/executor.h"
#include "absl/synchronization/blocking_counter.h"

namespace {

using ::tensorstore::Executor;
using ::tensorstore::internal::DetachedThreadPool;

// Tests that the thread pool runs a task.
TEST(DetachedThreadPoolTest, Basic) {
  SetupThreadPoolTestEnv();
  auto executor = DetachedThreadPool(1);
  absl::Notification notification;
  executor([&] { notification.Notify(); });
  notification.WaitForNotification();
}

// Tests that the thread pool runs two tasks concurrently.
TEST(DetachedThreadPoolTest, Concurrent) {
  SetupThreadPoolTestEnv();
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
  SetupThreadPoolTestEnv();
  constexpr static size_t kThreadLimit = 3;
  auto executor = DetachedThreadPool(kThreadLimit);
  std::atomic<size_t> num_running_tasks{0};
  std::vector<absl::Notification> notifications(5);
  for (size_t i = 0; i < notifications.size(); ++i) {
    executor([&num_running_tasks, &notifications, i] {
      auto& notification = notifications[i];
      EXPECT_LE(++num_running_tasks, kThreadLimit);
      absl::SleepFor(absl::Seconds(1));
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
  SetupThreadPoolTestEnv();
  struct Task {
    Executor& executor;
    absl::Notification* notification1;
    absl::Notification* notification2;

    void operator()() { notification1->Notify(); }

    Task(Executor& executor, absl::Notification* notification1,
         absl::Notification* notification2)
        : executor(executor),
          notification1(notification1),
          notification2(notification2) {}

    ~Task() {
      executor(
          [notification2 = this->notification2] { notification2->Notify(); });
    }
  };
  struct TaskWrapper {
    std::unique_ptr<Task> task;
    void operator()() { (*task)(); }
  };

  auto executor = DetachedThreadPool(1);
  absl::Notification notification1;
  absl::Notification notification2;

  executor(TaskWrapper{std::make_unique<Task>(std::ref(executor),
                                              &notification1, &notification2)});

  notification1.WaitForNotification();
  notification2.WaitForNotification();
}

// Mimics the benchmark, only on a smaller scale.
TEST(DetachedThreadPoolTest, XorTest) {
  SetupThreadPoolTestEnv();
  const size_t total_size = (128 << 14) / sizeof(uint64_t);
  const size_t buffer_size = 128 / sizeof(uint64_t);
  const size_t n = 128;
  const size_t chunk0 = total_size / n;
  const size_t m = chunk0 / buffer_size;
  const size_t chunk1 = buffer_size;

  std::vector<uint64_t> destination(total_size);
  std::vector<uint64_t> source(buffer_size);

  absl::BitGen rng;
  std::generate(source.begin(), source.end(),
                [&] { return absl::Uniform<uint64_t>(rng); });

  auto executor = DetachedThreadPool(4);
  absl::BlockingCounter done(n * m);
  for (size_t i = 0; i < n; i++) {
    executor([&, i] {
      for (size_t j = 0; j < m; j++) {
        executor([&, i, j] {
          auto* d = destination.data() + (i * chunk0) + (j * chunk1);
          for (size_t k = 0; k < buffer_size; ++k) d[k] ^= source[k];
          done.DecrementCount();
        });
      }
    });
  }
  done.Wait();
}

}  // namespace

#endif  // THIRD_PARTY_TENSORSTORE_INTERNAL_THREAD_THREAD_POOL_TEST_INC_
