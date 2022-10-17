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

#include <gtest/gtest.h>
#include "tensorstore/kvstore/key_range.h"

namespace {

using ::tensorstore::KeyRange;
using ::tensorstore::internal_file_util::IsKeyValid;
using ::tensorstore::internal_file_util::LongestDirectoryPrefix;

TEST(IsKeyValid, Basic) {
  EXPECT_TRUE(IsKeyValid("tmp/root", ""));
  EXPECT_TRUE(IsKeyValid("a", ""));
  EXPECT_TRUE(IsKeyValid("a/b", ""));

  EXPECT_FALSE(IsKeyValid("", ""));
  EXPECT_FALSE(IsKeyValid("/", ""));
  EXPECT_TRUE(IsKeyValid("/tmp/root", ""));
  EXPECT_FALSE(IsKeyValid("/tmp/root/", ""));
  EXPECT_TRUE(IsKeyValid("tmp//root", ""));
  EXPECT_FALSE(IsKeyValid("tmp/./root", ""));
  EXPECT_FALSE(IsKeyValid("tmp/../root", ""));
  EXPECT_FALSE(IsKeyValid("tmp/root/", ""));
  EXPECT_FALSE(IsKeyValid("tmp/.lock/a", ".lock"));
  EXPECT_FALSE(IsKeyValid("tmp/foo.lock/a", ".lock"));

  EXPECT_FALSE(IsKeyValid(std::string_view("tmp/\0bar", 8), ""));
}

TEST(LongestDirectoryPrefix, Basic) {
  EXPECT_EQ("", LongestDirectoryPrefix(KeyRange{"a", "b"}));

  EXPECT_EQ("", LongestDirectoryPrefix(KeyRange{"/a", "/b"}));
  EXPECT_EQ("/a", LongestDirectoryPrefix(KeyRange{"/a/a", "/a/b"}));
}

}  // namespace
