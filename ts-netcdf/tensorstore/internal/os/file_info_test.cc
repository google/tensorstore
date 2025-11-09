// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/internal/os/file_info.h"

#include <fstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::IsOk;
using ::tensorstore::internal_os::FileInfo;
using ::tensorstore::internal_os::GetFileInfo;
using ::tensorstore::internal_testing::ScopedTemporaryDirectory;

namespace {

TEST(FileInfoTest, Basics) {
  ScopedTemporaryDirectory tempdir;
  std::string foo_txt = tempdir.path() + "/foo.txt";
  std::string renamed_txt = tempdir.path() + "/renamed.txt";

  const auto now = absl::Now() - absl::Seconds(1);

  // Stat the directory:
  {
    // Check the file info
    FileInfo info;
    EXPECT_THAT(GetFileInfo(tempdir.path(), &info), IsOk());
    EXPECT_FALSE(info.IsRegularFile());
    EXPECT_THAT(info.GetFileId(), ::testing::Ne(0));
    EXPECT_THAT(info.GetDeviceId(), ::testing::Ne(0));
    EXPECT_THAT(info.GetMTime(), ::testing::Ge(now));
    EXPECT_THAT(info.GetCTime(), ::testing::Ge(now));
    EXPECT_THAT(info.GetMode() & 0600, ::testing::Eq(0600));
  }

  {
    std::ofstream foo_stream(foo_txt);
    foo_stream << "foo";
    foo_stream.close();
  }

  // Stat the file by path
  {
    // Check the file info
    FileInfo info;
    EXPECT_THAT(GetFileInfo(foo_txt, &info), IsOk());
    EXPECT_TRUE(info.IsRegularFile());
    EXPECT_THAT(info.GetSize(), 3);
    EXPECT_THAT(info.GetFileId(), ::testing::Ne(0));
    EXPECT_THAT(info.GetDeviceId(), ::testing::Ne(0));
    EXPECT_THAT(info.GetMTime(), ::testing::Ge(now));
    EXPECT_THAT(info.GetCTime(), ::testing::Ge(now));
    EXPECT_THAT(info.GetMode() & 0600, ::testing::Eq(0600));
  }

  // Stat the file:
  {
    // Check the file info
    FileInfo info;
    EXPECT_THAT(GetFileInfo(foo_txt, &info), IsOk());
    EXPECT_TRUE(info.IsRegularFile());
    EXPECT_THAT(info.GetSize(), 3);
    EXPECT_THAT(info.GetFileId(), ::testing::Ne(0));
    EXPECT_THAT(info.GetDeviceId(), ::testing::Ne(0));
    EXPECT_THAT(info.GetMTime(), ::testing::Ge(now));
    EXPECT_THAT(info.GetCTime(), ::testing::Ge(now));
    EXPECT_THAT(info.GetMode() & 0600, ::testing::Eq(0600));
  }
}

}  // namespace
