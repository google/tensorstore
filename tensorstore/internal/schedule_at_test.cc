#include "tensorstore/internal/schedule_at.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "tensorstore/util/executor.h"

using tensorstore::internal::ScheduleAt;

namespace {

// Tests that the thread pool runs a task.
TEST(DelayExecutorTest, Basic) {
  absl::Notification a, b;

  auto now = absl::Now();
  ScheduleAt(now + absl::Milliseconds(1), [&] { a.Notify(); });
  ScheduleAt(now + absl::Milliseconds(5), [&] { b.Notify(); });
  EXPECT_FALSE(b.HasBeenNotified());
  b.WaitForNotification();
  EXPECT_TRUE(a.HasBeenNotified());
}

}  // namespace
