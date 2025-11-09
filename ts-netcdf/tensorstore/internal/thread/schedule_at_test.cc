// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/thread/schedule_at.h"

#include <memory>
#include <thread>  // NOLINT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/util/stop_token.h"

namespace {

using ::tensorstore::StopSource;
using ::tensorstore::internal::ScheduleAt;

// Tests that the thread pool runs a task.
TEST(ScheduleAtTest, Basic) {
  absl::Notification a, b;

  auto now = absl::Now();
  ScheduleAt(now + absl::Milliseconds(1), [&] { a.Notify(); });
  ScheduleAt(now + absl::Milliseconds(5), [&] { b.Notify(); });
  EXPECT_FALSE(b.HasBeenNotified());
  b.WaitForNotification();
  EXPECT_TRUE(a.HasBeenNotified());
}

TEST(ScheduleAtTest, RunImmediately) {
  auto notification = std::make_shared<absl::Notification>();
  ScheduleAt(absl::InfinitePast(), [=] { notification->Notify(); });
  notification->WaitForNotification();
}

TEST(ScheduleAtTest, RunMultipleImmediately) {
  auto notification = std::make_shared<absl::Notification>();
  // Block the background thread.
  ScheduleAt(absl::Now(), [=] { notification->WaitForNotification(); });

  auto notification1 = std::make_shared<absl::Notification>();
  auto notification2 = std::make_shared<absl::Notification>();

  ScheduleAt(absl::InfinitePast(), [=] {
    EXPECT_FALSE(notification2->HasBeenNotified());
    notification1->Notify();
  });

  ScheduleAt(absl::InfinitePast(), [=] { notification2->Notify(); });

  // Unblock background thread.
  notification->Notify();

  notification1->WaitForNotification();
  notification2->WaitForNotification();
}

TEST(ScheduleAtTest, Cancel) {
  auto notification = std::make_shared<absl::Notification>();
  EXPECT_EQ(1, notification.use_count());
  StopSource stop_source;
  ScheduleAt(
      absl::InfiniteFuture(), [notification] { notification->Notify(); },
      stop_source.get_token());
  EXPECT_EQ(2, notification.use_count());
  stop_source.request_stop();
  EXPECT_EQ(1, notification.use_count());
  EXPECT_FALSE(notification->HasBeenNotified());
}

TEST(ScheduleAtTest, CancelImmediately) {
  auto notification = std::make_shared<absl::Notification>();
  EXPECT_EQ(1, notification.use_count());
  StopSource stop_source;
  stop_source.request_stop();
  ScheduleAt(
      absl::InfinitePast(), [notification] { notification->Notify(); },
      stop_source.get_token());
  EXPECT_EQ(1, notification.use_count());
  EXPECT_FALSE(notification->HasBeenNotified());
}

TEST(ScheduleAtTest, CancelWhileRunning) {
  auto notification1 = std::make_shared<absl::Notification>();
  StopSource stop_source;
  ScheduleAt(absl::InfinitePast(), [=] {
    notification1->WaitForNotification();
    stop_source.request_stop();
  });
  auto notification2 = std::make_shared<absl::Notification>();
  auto notification3 = std::make_shared<absl::Notification>();
  ScheduleAt(
      absl::InfinitePast(), [=] { notification2->Notify(); },
      stop_source.get_token());
  ScheduleAt(absl::InfinitePast(), [=] { notification3->Notify(); });
  notification1->Notify();
  notification3->WaitForNotification();
  EXPECT_FALSE(notification2->HasBeenNotified());
  EXPECT_EQ(1, notification2.use_count());
}

}  // namespace
