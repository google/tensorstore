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

#include "tensorstore/kvstore/key_range.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/str_cat.h"

namespace {
using ::tensorstore::KeyRange;

TEST(KeyRangeTest, Comparison) {
  KeyRange r1("a", "b");
  EXPECT_EQ("a", r1.inclusive_min);
  EXPECT_EQ("b", r1.exclusive_max);
  KeyRange r2("a", "c");
  KeyRange r3("", "b");
  KeyRange r4("", "c");
  EXPECT_EQ(r1, r1);
  EXPECT_EQ(r2, r2);
  EXPECT_EQ(r3, r3);
  EXPECT_EQ(r4, r4);
  EXPECT_NE(r1, r2);
  EXPECT_NE(r1, r3);
  EXPECT_NE(r1, r4);
  EXPECT_NE(r2, r3);
  EXPECT_NE(r2, r4);
  EXPECT_NE(r3, r4);
}

TEST(KeyRangeTest, Full) {
  KeyRange full;
  EXPECT_TRUE(full.full());
  EXPECT_EQ(std::string(), full.inclusive_min);
  EXPECT_EQ(std::string(), full.exclusive_max);
  EXPECT_EQ(full, KeyRange({}, {}));
  EXPECT_NE(full, KeyRange("a", "b"));
  EXPECT_NE(full, KeyRange("", "b"));
  EXPECT_NE(full, KeyRange("a", ""));
  EXPECT_FALSE(full.empty());
  EXPECT_EQ("", tensorstore::LongestPrefix(full));
  EXPECT_TRUE(tensorstore::Contains(full, "abc"));
  EXPECT_EQ(KeyRange::Prefix(""), full);
}

TEST(KeyRangeTest, Empty) {
  EXPECT_FALSE(KeyRange("a", "b").empty());
  EXPECT_FALSE(KeyRange("a", "").empty());
  EXPECT_TRUE(KeyRange("b", "a").empty());
  EXPECT_TRUE(KeyRange("b", "b").empty());
}

TEST(KeyRangeTest, Prefix) {
  EXPECT_EQ(KeyRange(), KeyRange::Prefix(""));
  EXPECT_EQ(KeyRange("abc", "abd"), KeyRange::Prefix("abc"));
  EXPECT_EQ(KeyRange("ab\xff", "ac"), KeyRange::Prefix("ab\xff"));
  EXPECT_EQ(KeyRange("ab\xff\xff\xff", "ac"),
            KeyRange::Prefix("ab\xff\xff\xff"));
  EXPECT_EQ(KeyRange("\xff", ""), KeyRange::Prefix("\xff"));
  EXPECT_EQ(KeyRange("\xff\xff\xff", ""), KeyRange::Prefix("\xff\xff\xff"));
}

TEST(KeyRangeTest, Successor) {
  EXPECT_EQ(std::string({'a', 'b', 'c', '\x00'}), KeyRange::Successor("abc"));
  EXPECT_EQ(std::string({'\x00'}), KeyRange::Successor(""));
}

TEST(KeyRangeTest, ContainsKey) {
  EXPECT_TRUE(tensorstore::Contains(KeyRange("a", "c"), "a"));
  EXPECT_TRUE(tensorstore::Contains(KeyRange("a", "c"), "ab"));
  EXPECT_TRUE(tensorstore::Contains(KeyRange("a", "c"), "abc"));
  EXPECT_TRUE(tensorstore::Contains(KeyRange("a", "c"), "b"));
  EXPECT_TRUE(tensorstore::Contains(KeyRange("a", "c"), "ba"));
  EXPECT_FALSE(tensorstore::Contains(KeyRange("a", "c"), "c"));
  EXPECT_FALSE(tensorstore::Contains(KeyRange("a", "c"), "ca"));
  EXPECT_FALSE(tensorstore::Contains(KeyRange("a", "c"), "d"));
}

TEST(KeyRangeTest, ContainsRange) {
  EXPECT_TRUE(tensorstore::Contains(KeyRange(), KeyRange("ab", "cd")));
  EXPECT_TRUE(tensorstore::Contains(KeyRange("a", "c"), KeyRange("a", "c")));
  EXPECT_TRUE(tensorstore::Contains(KeyRange("a", "c"), KeyRange("ab", "c")));
  EXPECT_TRUE(tensorstore::Contains(KeyRange("a", "c"), KeyRange("ab", "ba")));
  EXPECT_TRUE(tensorstore::Contains(KeyRange("a", "c"), KeyRange("b", "ba")));
  EXPECT_TRUE(tensorstore::Contains(KeyRange("a", "c"), KeyRange::Prefix("a")));
  EXPECT_TRUE(
      tensorstore::Contains(KeyRange("a", "c"), KeyRange::Prefix("ab")));
  EXPECT_TRUE(tensorstore::Contains(KeyRange("a", "c"), KeyRange::Prefix("b")));
  EXPECT_FALSE(tensorstore::Contains(KeyRange("a", "c"), KeyRange("a", "ca")));
  EXPECT_FALSE(tensorstore::Contains(KeyRange("a", "c"), KeyRange("0", "a")));
  EXPECT_FALSE(tensorstore::Contains(KeyRange("a", "c"), KeyRange()));
}

TEST(KeyRangeTest, Intersect) {
  EXPECT_EQ(KeyRange("b", "b"),
            tensorstore::Intersect(KeyRange("a", "b"), KeyRange("b", "c")));
  EXPECT_EQ(KeyRange("c", "c"),
            tensorstore::Intersect(KeyRange("a", "b"), KeyRange("c", "d")));
  EXPECT_EQ(KeyRange("b", "b"),
            tensorstore::Intersect(KeyRange("", "b"), KeyRange("b", "")));
  EXPECT_EQ(KeyRange("a", "b"),
            tensorstore::Intersect(KeyRange(), KeyRange("a", "b")));
  EXPECT_EQ(KeyRange("a", "b"),
            tensorstore::Intersect(KeyRange("a", "b"), KeyRange()));
  EXPECT_EQ(KeyRange("a", "b"),
            tensorstore::Intersect(KeyRange("a", "b"), KeyRange("a", "c")));
  EXPECT_EQ(KeyRange("a", "b"),
            tensorstore::Intersect(KeyRange("a", "c"), KeyRange("a", "b")));
  EXPECT_EQ(KeyRange("b", "c"),
            tensorstore::Intersect(KeyRange("a", "c"), KeyRange("b", "c")));
  EXPECT_EQ(KeyRange("aa", "b"),
            tensorstore::Intersect(KeyRange("aa", "c"), KeyRange("a", "b")));
  EXPECT_EQ(KeyRange("aa", "b"),
            tensorstore::Intersect(KeyRange("aa", ""), KeyRange("a", "b")));
}

TEST(KeyRangeTest, LongestPrefix) {
  EXPECT_EQ("", tensorstore::LongestPrefix(KeyRange("a", "c")));
  EXPECT_EQ("a", tensorstore::LongestPrefix(KeyRange("a", "b")));
  EXPECT_EQ("a", tensorstore::LongestPrefix(KeyRange("aa", "b")));
  EXPECT_EQ("abc", tensorstore::LongestPrefix(KeyRange("abc", "abcd")));
  EXPECT_EQ("abc", tensorstore::LongestPrefix(KeyRange("abc", "abd")));
  EXPECT_EQ("ab", tensorstore::LongestPrefix(KeyRange("abc", "abe")));
  EXPECT_EQ("ab\xff", tensorstore::LongestPrefix(KeyRange("ab\xff", "ac")));
  EXPECT_EQ("ab\xff\xff",
            tensorstore::LongestPrefix(KeyRange("ab\xff\xff", "ac")));
  EXPECT_EQ("\xff", tensorstore::LongestPrefix(KeyRange("\xff", "")));
  EXPECT_EQ("\xff\xff", tensorstore::LongestPrefix(KeyRange("\xff\xff", "")));
}

TEST(KeyRangeTest, Ostream) {
  EXPECT_EQ("[\"a\", \"b\")", tensorstore::StrCat(KeyRange("a", "b")));
  EXPECT_EQ("[\"a\", \"ba\")", tensorstore::StrCat(KeyRange("a", "ba")));
}

TEST(KeyRangeTest, CompareKeyAndExclusiveMax) {
  EXPECT_THAT(KeyRange::CompareKeyAndExclusiveMax("a", "a"), ::testing::Eq(0));
  EXPECT_THAT(KeyRange::CompareKeyAndExclusiveMax("a", "b"), ::testing::Lt(0));
  EXPECT_THAT(KeyRange::CompareKeyAndExclusiveMax("b", "a"), ::testing::Gt(0));
  EXPECT_THAT(KeyRange::CompareKeyAndExclusiveMax("", ""), ::testing::Lt(0));
  EXPECT_THAT(KeyRange::CompareKeyAndExclusiveMax("a", ""), ::testing::Lt(0));

  EXPECT_THAT(KeyRange::CompareExclusiveMaxAndKey("a", "a"), ::testing::Eq(0));
  EXPECT_THAT(KeyRange::CompareExclusiveMaxAndKey("a", "b"), ::testing::Lt(0));
  EXPECT_THAT(KeyRange::CompareExclusiveMaxAndKey("b", "a"), ::testing::Gt(0));
  EXPECT_THAT(KeyRange::CompareExclusiveMaxAndKey("", ""), ::testing::Gt(0));
  EXPECT_THAT(KeyRange::CompareExclusiveMaxAndKey("", "a"), ::testing::Gt(0));
}

TEST(KeyRangeTest, CompareExclusiveMax) {
  EXPECT_THAT(KeyRange::CompareExclusiveMax("", ""), ::testing::Eq(0));
  EXPECT_THAT(KeyRange::CompareExclusiveMax("a", "a"), ::testing::Eq(0));
  EXPECT_THAT(KeyRange::CompareExclusiveMax("a", "b"), ::testing::Lt(0));
  EXPECT_THAT(KeyRange::CompareExclusiveMax("b", "a"), ::testing::Gt(0));
  EXPECT_THAT(KeyRange::CompareExclusiveMax("a", ""), ::testing::Lt(0));
  EXPECT_THAT(KeyRange::CompareExclusiveMax("", "a"), ::testing::Gt(0));
}

TEST(KeyRangeTest, AddPrefix) {
  EXPECT_THAT(KeyRange::AddPrefix("", KeyRange("a", "b")),
              ::testing::Eq(KeyRange("a", "b")));
  EXPECT_THAT(KeyRange::AddPrefix("x", KeyRange("a", "b")),
              ::testing::Eq(KeyRange("xa", "xb")));
  EXPECT_THAT(KeyRange::AddPrefix("x", KeyRange("a", "")),
              ::testing::Eq(KeyRange("xa", "y")));
}

TEST(KeyRangeTest, EmptyRange) {
  auto range = KeyRange::EmptyRange();
  EXPECT_TRUE(range.empty());
  EXPECT_EQ(range.inclusive_min, range.exclusive_max);
}

TEST(KeyRangeTest, RemovePrefix) {
  EXPECT_THAT(KeyRange::RemovePrefix("", KeyRange("a", "b")),
              ::testing::Eq(KeyRange("a", "b")));
  EXPECT_THAT(KeyRange::RemovePrefix("a/", KeyRange("a/b", "a/d")),
              ::testing::Eq(KeyRange("b", "d")));
  EXPECT_THAT(KeyRange::RemovePrefix("a/b", KeyRange("a/b", "a/d")),
              ::testing::Eq(KeyRange()));
  EXPECT_THAT(KeyRange::RemovePrefix("a/d", KeyRange("a/b", "a/d")),
              ::testing::Eq(KeyRange::EmptyRange()));
  EXPECT_THAT(KeyRange::RemovePrefix("a/bc", KeyRange("a/b", "a/bcb")),
              ::testing::Eq(KeyRange("", "b")));
  EXPECT_THAT(KeyRange::RemovePrefix("x", KeyRange("xa", "y")),
              ::testing::Eq(KeyRange("a", "")));
}

}  // namespace
