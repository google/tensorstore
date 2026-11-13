// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/internal/os/file_lock.h"

#ifdef _WIN32
#include <sys/utime.h>
#include <time.h>
#define utimbuf _utimbuf
#define utime _utime
#else
#include <utime.h>
#endif

#include <fstream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/os/file_descriptor.h"
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/internal/thread/thread.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::IsOk;
using ::tensorstore::IsOkAndHolds;
using ::tensorstore::StatusIs;
using ::tensorstore::internal_os::AcquireExclusiveFile;
using ::tensorstore::internal_os::AcquireFileLock;
using ::tensorstore::internal_os::FileLock;
using ::tensorstore::internal_os::InvalidFileDescriptor;
using ::tensorstore::internal_os::ReadAllToString;
using ::tensorstore::internal_os::RenameOpenFile;
using ::tensorstore::internal_os::TruncateAndOverwrite;
using ::tensorstore::internal_os::WriteToFile;
using ::tensorstore::internal_testing::ScopedTemporaryDirectory;

namespace {

TEST(AcquireFileLockTest, Basic) {
  ScopedTemporaryDirectory tempdir;
  std::string lock_path = tempdir.path() + "/foo.txt.__lock";
  auto lock = AcquireFileLock(lock_path);
  EXPECT_THAT(lock, IsOk());
  EXPECT_NE(lock->fd(), InvalidFileDescriptor());

  FileLock l = std::move(lock).value();
  EXPECT_NE(l.fd(), InvalidFileDescriptor());
  EXPECT_THAT(WriteToFile(l.fd(), "foo", 3), IsOkAndHolds(3));
  EXPECT_THAT(std::move(l).Close(), IsOk());
  EXPECT_THAT(ReadAllToString(lock_path), IsOkAndHolds("foo"));
}

TEST(AcquireFileLockTest, FileExists) {
  ScopedTemporaryDirectory tempdir;
  std::string lock_path = tempdir.path() + "/foo.txt.__lock";
  {
    std::ofstream x(lock_path, std::ios::binary);
    x << "stale_lock_contents";
  }

  auto lock = AcquireFileLock(lock_path);
  ASSERT_THAT(lock, IsOk());
  EXPECT_NE(lock->fd(), InvalidFileDescriptor());
  EXPECT_THAT(WriteToFile(lock->fd(), "foo", 3), IsOkAndHolds(3));
  EXPECT_THAT(std::move(lock).value().Close(), IsOk());
  EXPECT_THAT(ReadAllToString(lock_path), IsOkAndHolds("foo"));
}

TEST(AcquireFileLockTest, TruncatesStaleContentsToEmpty) {
  ScopedTemporaryDirectory tempdir;
  std::string lock_path = tempdir.path() + "/foo.txt.__lock";
  {
    std::ofstream x(lock_path, std::ios::binary);
    x << "stale_lock_contents";
  }

  auto lock = AcquireFileLock(lock_path);
  ASSERT_THAT(lock, IsOk());
  EXPECT_NE(lock->fd(), InvalidFileDescriptor());
  EXPECT_THAT(std::move(lock).value().Close(), IsOk());
  EXPECT_THAT(ReadAllToString(lock_path), IsOkAndHolds(""));
}

TEST(AcquireFileLockTest, ContendedRenameDoesNotTruncateCommittedFile) {
  ScopedTemporaryDirectory tempdir;
  std::string lock_path = tempdir.path() + "/foo.txt.__lock";
  std::string target_path = tempdir.path() + "/foo.txt";

  auto lock_a = AcquireFileLock(lock_path);
  ASSERT_THAT(lock_a, IsOk());
  EXPECT_THAT(WriteToFile(lock_a->fd(), "committed_data", 14),
              IsOkAndHolds(14));

  absl::Notification b_started;
  tensorstore::internal::Thread thread_b({"lock_waiter"}, [&] {
    b_started.Notify();
    auto lock_b = AcquireFileLock(lock_path);
    ASSERT_THAT(lock_b, IsOk());
    EXPECT_THAT(WriteToFile(lock_b->fd(), "new", 3), IsOkAndHolds(3));
    EXPECT_THAT(std::move(lock_b).value().Close(), IsOk());
  });

  b_started.WaitForNotification();
  absl::SleepFor(absl::Milliseconds(50));

  EXPECT_THAT(RenameOpenFile(lock_a->fd(), lock_path, target_path), IsOk());
  EXPECT_THAT(std::move(lock_a).value().Close(), IsOk());

  thread_b.Join();

  EXPECT_THAT(ReadAllToString(target_path), IsOkAndHolds("committed_data"));
  EXPECT_THAT(ReadAllToString(lock_path), IsOkAndHolds("new"));
}

TEST(AcquireExclusiveFileTest, Basic) {
  /// Lock file does not exist.
  ScopedTemporaryDirectory tempdir;
  std::string lock_path = tempdir.path() + "/foo.txt.__lock";
  auto lock = AcquireExclusiveFile(lock_path, absl::Seconds(1));
  EXPECT_THAT(lock, IsOk());
  EXPECT_NE(lock->fd(), InvalidFileDescriptor());

  EXPECT_THAT(std::move(lock).value().Delete(), IsOk());
}

TEST(AcquireExclusiveFileTest, FileExistsNotAcquired) {
  ScopedTemporaryDirectory tempdir;
  std::string lock_path = tempdir.path() + "/foo.txt.__lock";
  {
    std::ofstream x(lock_path);

    // Set the mtime to be stale.
    auto now = absl::Now();
    utimbuf times;
    times.actime = absl::ToTimeT(now - absl::Minutes(60));
    times.modtime = absl::ToTimeT(now - absl::Minutes(60));
    utime(lock_path.c_str(), &times);
  }

  // The lock is stale, so it should be deleted.
  auto lock = AcquireExclusiveFile(lock_path, absl::Milliseconds(100));
  EXPECT_THAT(lock, StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST(TruncateAndOverwrite, Basic) {
  /// Lock file does not exist.
  ScopedTemporaryDirectory tempdir;
  std::string lock_path = tempdir.path() + "/foo.txt";
  auto lock = TruncateAndOverwrite(lock_path);
  EXPECT_THAT(lock, IsOk());
  EXPECT_NE(lock->fd(), InvalidFileDescriptor());

  EXPECT_THAT(std::move(lock).value().Delete(), IsOk());
}

TEST(TruncateAndOverwrite, FileExists) {
  ScopedTemporaryDirectory tempdir;
  std::string lock_path = tempdir.path() + "/foo.txt";
  {
    std::ofstream x(lock_path, std::ios::binary);
    x << "stale_contents";
  }

  auto lock = TruncateAndOverwrite(lock_path);
  ASSERT_THAT(lock, IsOk());
  EXPECT_NE(lock->fd(), InvalidFileDescriptor());
  EXPECT_THAT(WriteToFile(lock->fd(), "foo", 3), IsOkAndHolds(3));
  EXPECT_THAT(std::move(lock).value().Close(), IsOk());
  EXPECT_THAT(ReadAllToString(lock_path), IsOkAndHolds("foo"));
}

}  // namespace
