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

#include "tensorstore/internal/string_like.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace {

using tensorstore::internal::StringLikeSpan;

TEST(StringLikeSpan, Default) {
  StringLikeSpan x;
  EXPECT_EQ(0, x.size());
}

TEST(StringLikeSpan, CStrings) {
  std::vector<const char*> c_strings{"a", "b", "c"};
  StringLikeSpan x(c_strings);
  EXPECT_EQ(3, x.size());
  EXPECT_EQ("a", x[0]);
  EXPECT_EQ("b", x[1]);
  EXPECT_EQ("c", x[2]);
}

TEST(StringLikeSpan, StdStrings) {
  std::vector<std::string> std_strings{"a", "b", "c"};
  StringLikeSpan x(std_strings);
  EXPECT_EQ(3, x.size());
  EXPECT_EQ("a", x[0]);
  EXPECT_EQ("b", x[1]);
  EXPECT_EQ("c", x[2]);
}

TEST(StringLikeSpan, StringViews) {
  std::vector<absl::string_view> string_views{"a", "b", "c"};
  StringLikeSpan x(string_views);
  EXPECT_EQ(3, x.size());
  EXPECT_EQ("a", x[0]);
  EXPECT_EQ("b", x[1]);
  EXPECT_EQ("c", x[2]);
}

}  // namespace
