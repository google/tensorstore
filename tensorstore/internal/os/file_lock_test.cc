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
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::IsOk;
using ::tensorstore::IsOkAndHolds;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_os::AcquireExclusiveFile;
using ::tensorstore::internal_os::AcquireFileLock;
using ::tensorstore::internal_os::FileDescriptorTraits;
using ::tensorstore::internal_os::FileLock;
using ::tensorstore::internal_testing::ScopedTemporaryDirectory;

namespace {

TEST(AcquireFileLockTest, Basic) {
  ScopedTemporaryDirectory tempdir;
  std::string lock_path = tempdir.path() + "/foo.txt.__lock";
  auto lock = AcquireFileLock(lock_path);
  EXPECT_THAT(lock, IsOk());
  EXPECT_NE(lock->fd(), FileDescriptorTraits::Invalid());

  FileLock l = std::move(lock).value();
  EXPECT_NE(l.fd(), FileDescriptorTraits::Invalid());
  EXPECT_THAT(tensorstore::internal_os::WriteToFile(l.fd(), "foo", 3),
              IsOkAndHolds(3));
  std::move(l).Close();
}

TEST(AcquireFileLockTest, FileExists) {
  ScopedTemporaryDirectory tempdir;
  std::string lock_path = tempdir.path() + "/foo.txt.__lock";
  {
    std::ofstream x(lock_path);
  }

  auto lock = AcquireFileLock(lock_path);
  EXPECT_THAT(lock, IsOk());
  EXPECT_NE(lock->fd(), FileDescriptorTraits::Invalid());
  std::move(lock).value().Close();
}

TEST(AcquireExclusiveFileTest, Basic) {
  /// Lock file does not exist.
  ScopedTemporaryDirectory tempdir;
  std::string lock_path = tempdir.path() + "/foo.txt.__lock";
  auto lock = AcquireExclusiveFile(lock_path, absl::Seconds(1));
  EXPECT_THAT(lock, IsOk());
  EXPECT_NE(lock->fd(), FileDescriptorTraits::Invalid());

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
  EXPECT_THAT(lock, MatchesStatus(absl::StatusCode::kDeadlineExceeded));
}

}  // namespace
