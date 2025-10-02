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

#include "tensorstore/kvstore/file/util.h"

#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/testing/on_windows.h"
#include "tensorstore/kvstore/key_range.h"

namespace {

using ::tensorstore::KeyRange;
using ::tensorstore::internal_file_util::IsKeyValid;
using ::tensorstore::internal_file_util::LongestDirectoryPrefix;
using ::tensorstore::internal_testing::OnWindows;

TEST(IsKeyValid, InvalidKeys) {
  EXPECT_FALSE(IsKeyValid("", ""));

  EXPECT_FALSE(IsKeyValid("\\", ""));
  EXPECT_FALSE(IsKeyValid("//", ""));
  EXPECT_FALSE(IsKeyValid("\\\\", ""));
  EXPECT_FALSE(IsKeyValid("///", ""));
  EXPECT_FALSE(IsKeyValid("\\\\\\", ""));

  // Paths with / suffixes
  EXPECT_FALSE(IsKeyValid("tmp/root//", ""));
  EXPECT_FALSE(IsKeyValid("tmp/root/", ""));
  EXPECT_FALSE(IsKeyValid("tmp/root\\", ""));
  EXPECT_FALSE(IsKeyValid("/tmp/root//", ""));
  EXPECT_FALSE(IsKeyValid("/tmp/root/", ""));
  EXPECT_FALSE(IsKeyValid("/tmp/root\\", ""));

  // Invalid components.
  EXPECT_FALSE(IsKeyValid("tmp//root", ""));
  EXPECT_FALSE(IsKeyValid("tmp/./root", ""));
  EXPECT_FALSE(IsKeyValid("tmp/../root", ""));
  EXPECT_FALSE(IsKeyValid("tmp/.lock/a", ".lock"));
  EXPECT_FALSE(IsKeyValid("tmp/foo.lock/a", ".lock"));

  EXPECT_FALSE(IsKeyValid("/tmp//root", ""));
  EXPECT_FALSE(IsKeyValid("/tmp/./root", ""));
  EXPECT_FALSE(IsKeyValid("/tmp/../root", ""));
  EXPECT_FALSE(IsKeyValid("/tmp/.lock/a", ".lock"));
  EXPECT_FALSE(IsKeyValid("/tmp/foo.lock/a", ".lock"));
  EXPECT_FALSE(IsKeyValid("tmp\\..\\root", ""));
}

TEST(IsKeyValid, ValidKeys) {
  EXPECT_TRUE(IsKeyValid("tmp", ""));
  EXPECT_TRUE(IsKeyValid("tmp/root", ""));
  EXPECT_TRUE(IsKeyValid("tmp\\root", ""));

  // Absolute paths.
  EXPECT_TRUE(IsKeyValid("/tmp/root", ""));
  EXPECT_TRUE(IsKeyValid("\\tmp\\root", ""));
}

TEST(IsKeyValid, WindowsKeys) {
  // Valid on windows, but with different meanings.
  // Drive-letter paths.
  EXPECT_THAT(IsKeyValid("C:", ""), OnWindows(false, true));
  EXPECT_TRUE(IsKeyValid("c:tmp", ""));
  EXPECT_TRUE(IsKeyValid("C:\\tmp/root", ""));

  // Invalid in both POSIX and Windows.
  EXPECT_FALSE(IsKeyValid("C:/tmp/", ""));
  EXPECT_FALSE(IsKeyValid("//share/path/", ""));
  EXPECT_FALSE(IsKeyValid("\\\\share\\path\\", ""));

  // Network share paths.
  EXPECT_THAT(IsKeyValid("//share/path", ""), OnWindows(true, false));
  EXPECT_THAT(IsKeyValid("\\\\share\\path", ""), OnWindows(true, false));
}

TEST(IsKeyValid, HasEmbeddedNull) {
  EXPECT_FALSE(IsKeyValid(std::string_view("/tmp/\0bar", 9), ""));
}

TEST(LongestDirectoryPrefix, Basic) {
  EXPECT_EQ("", LongestDirectoryPrefix(KeyRange{"a", "b"}));

  EXPECT_EQ("", LongestDirectoryPrefix(KeyRange{"/a", "/b"}));
  EXPECT_EQ("/a", LongestDirectoryPrefix(KeyRange{"/a/a", "/a/b"}));
}

}  // namespace
