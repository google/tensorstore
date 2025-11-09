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

#include "tensorstore/tscli/lib/glob_to_regex.h"

#include <gtest/gtest.h>

namespace {

using ::tensorstore::cli::GlobToRegex;

TEST(GlobToRegex, Basic) {
  EXPECT_EQ(GlobToRegex("a*b"), "^a[^/]*b$");
  EXPECT_EQ(GlobToRegex("a**b"), "^a.*b$");
  EXPECT_EQ(GlobToRegex("a?b"), "^a[^/]b$");

  EXPECT_EQ(GlobToRegex("a[b"), "^a\\[b$");
  EXPECT_EQ(GlobToRegex("a[A-Z]b"), "^a[A-Z]b$");
  EXPECT_EQ(GlobToRegex("a[!A-Z]b"), "^a[^/A-Z]b$");
  EXPECT_EQ(GlobToRegex("a[A-]b"), "^a[A-]b$");

  EXPECT_EQ(GlobToRegex("a.+{}()|^$b"), "^a\\.\\+\\{\\}\\(\\)\\|\\^\\$b$");
  EXPECT_EQ(GlobToRegex("a\\-b\\"), "^a\\-b\\\\$");

  // See, for example: https://code.visualstudio.com/docs/editor/glob-patterns

  //  [] to declare a range of characters to match (example.[0-9] to match on
  //  example.0, example.1, …)
  EXPECT_EQ(GlobToRegex("example.[0-9]"), "^example\\.[0-9]$");

  //  [!...] to negate a range of characters to match (example.[!0-9] to match
  //  on example.a, example.b, but not example.0)
  EXPECT_EQ(GlobToRegex("example.[!0-9]"), "^example\\.[^/0-9]$");

  // NOTE: {} to group conditions is not supported.
  EXPECT_EQ(GlobToRegex("**/{*.html,*.txt}"),
            "^.*/\\{[^/]*\\.html,[^/]*\\.txt\\}$");
}

}  // namespace
