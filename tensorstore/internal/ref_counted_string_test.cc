// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/ref_counted_string.h"

#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal::RefCountedString;
using ::tensorstore::internal::RefCountedStringWriter;

TEST(RefCountedStringTest, DefaultConstruct) {
  RefCountedString s;
  EXPECT_EQ("", std::string_view(s));
  EXPECT_EQ("", std::string(s));
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(nullptr, s.data());
  EXPECT_EQ(0, s.size());
  EXPECT_EQ(nullptr, s.begin());
  EXPECT_EQ(nullptr, s.end());
  EXPECT_EQ(s, s);

  auto other = s;
  EXPECT_EQ(nullptr, other.data());
}

TEST(RefCountedStringTest, EmptyStringConstruct) {
  RefCountedString s("");
  EXPECT_EQ("", std::string_view(s));
  EXPECT_EQ("", std::string(s));
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(nullptr, s.data());
  EXPECT_EQ(0, s.size());
  EXPECT_EQ(nullptr, s.begin());
  EXPECT_EQ(nullptr, s.end());
  EXPECT_EQ(s, s);
}

TEST(RefCountedStringTest, NonEmptyStringConstruct) {
  RefCountedString s("abc");
  EXPECT_EQ("abc", std::string_view(s));
  EXPECT_EQ("abc", std::string(s));
  EXPECT_FALSE(s.empty());
  EXPECT_EQ(3, s.size());
  EXPECT_EQ("abc", s);
  EXPECT_NE("abd", s);
  EXPECT_EQ(s, "abc");
  EXPECT_LT("ab", s);
  EXPECT_LE("abc", s);
  EXPECT_GT("abd", s);
}

TEST(RefCountedStringTest, Copy) {
  RefCountedString x("abc");

  RefCountedString y = x;
  EXPECT_EQ(x.data(), y.data());
}

TEST(RefCountedStringTest, Move) {
  RefCountedString x("abc");
  const char* ptr = x.data();

  RefCountedString y = std::move(x);
  EXPECT_EQ(y, "abc");
  EXPECT_EQ(ptr, y.data());
  EXPECT_TRUE(x.empty());  // NOLINT
}

TEST(RefCountedStringTest, EmptyMoveAssignNonEmpty) {
  RefCountedString x("abc");
  const char* ptr = x.data();

  RefCountedString y;
  y = std::move(x);
  EXPECT_EQ(y, "abc");
  EXPECT_EQ(ptr, y.data());
  EXPECT_TRUE(x.empty());  // NOLINT
}

TEST(RefCountedStringTest, EmptyMoveAssignEmpty) {
  RefCountedString x;
  RefCountedString y;
  y = std::move(x);
  EXPECT_TRUE(y.empty());
  EXPECT_TRUE(x.empty());  // NOLINT
}

TEST(RefCountedStringTest, NonEmptyMoveAssignNonEmpty) {
  RefCountedString x("abc");
  const char* ptr = x.data();

  RefCountedString y("def");
  y = std::move(x);
  EXPECT_EQ(y, "abc");
  EXPECT_EQ(ptr, y.data());
}

TEST(RefCountedStringTest, NonEmptyMoveAssignEmpty) {
  RefCountedString x;

  RefCountedString y("def");
  y = std::move(x);
  EXPECT_TRUE(y.empty());
}

TEST(RefCountedStringTest, NonEmptyCopyAssignNonEmpty) {
  RefCountedString x("abc");
  RefCountedString y("def");

  y = x;
  EXPECT_EQ("abc", y);
}

TEST(RefCountedStringTest, EmptyCopyAssignNonEmpty) {
  RefCountedString x("abc");
  RefCountedString y;

  y = x;
  EXPECT_EQ("abc", y);
}

TEST(RefCountedStringTest, NonEmptyCopyAssignEmpty) {
  RefCountedString x;
  RefCountedString y("def");

  y = x;
  EXPECT_EQ("", y);
}

TEST(RefCountedStringTest, EmptyCopyAssignEmpty) {
  RefCountedString x;
  RefCountedString y;

  y = x;
  EXPECT_EQ("", y);
}

TEST(RefCountedStringTest, NonEmptyAssignFromStringView) {
  RefCountedString x("def");

  x = std::string_view("abc");
  EXPECT_EQ("abc", x);
}

TEST(RefCountedStringTest, EmptyAssignFromStringView) {
  RefCountedString x;

  x = std::string_view("abc");
  EXPECT_EQ("abc", x);
}

TEST(RefCountedStringTest, NonEmptyAssignFromCStr) {
  RefCountedString x("def");

  x = "abc";
  EXPECT_EQ("abc", x);
}

TEST(RefCountedStringTest, EmptyAssignFromCStr) {
  RefCountedString x;

  x = "abc";
  EXPECT_EQ("abc", x);
}

TEST(RefCountedStringTest, SelfAssign) {
  RefCountedString x("abc");
  x = x;
  EXPECT_EQ("abc", x);
}

TEST(RefCountedStringTest, SelfAssignStringView) {
  RefCountedString x("abc");
  x = std::string_view(x);
  EXPECT_EQ("abc", x);
}

TEST(RefCountedStringTest, Comparison) {
  RefCountedString a("abc");
  RefCountedString a1("abc");
  std::string_view a_sv = "abc";
  const char* a_cstr = "abc";

  RefCountedString b("def");
  std::string_view b_sv = "def";
  const char* b_cstr = "def";

  EXPECT_TRUE(a == a);
  EXPECT_TRUE(a == a1);
  EXPECT_TRUE(a == a_sv);
  EXPECT_TRUE(a == a_cstr);
  EXPECT_TRUE(a_sv == a);
  EXPECT_TRUE(a_cstr == a);
  EXPECT_FALSE(a != a);
  EXPECT_FALSE(a != a1);
  EXPECT_FALSE(a != a_sv);
  EXPECT_FALSE(a != a_cstr);
  EXPECT_FALSE(a_sv != a);
  EXPECT_FALSE(a_cstr != a);
  EXPECT_TRUE(a <= a);
  EXPECT_TRUE(a <= a_sv);
  EXPECT_TRUE(a <= a_cstr);
  EXPECT_TRUE(a_sv <= a);
  EXPECT_TRUE(a_cstr <= a);
  EXPECT_TRUE(a <= a1);
  EXPECT_TRUE(a >= a);
  EXPECT_TRUE(a >= a_sv);
  EXPECT_TRUE(a >= a_cstr);
  EXPECT_TRUE(a_sv >= a);
  EXPECT_TRUE(a_cstr >= a);
  EXPECT_TRUE(a >= a1);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(a <= b_sv);
  EXPECT_TRUE(a <= b_cstr);
  EXPECT_TRUE(a_sv <= b);
  EXPECT_TRUE(a_cstr <= b);
  EXPECT_TRUE(a <= b_sv);
  EXPECT_TRUE(a <= b_cstr);
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(a < b_sv);
  EXPECT_TRUE(a < b_cstr);
  EXPECT_TRUE(a_sv < b);
  EXPECT_TRUE(a_cstr < b);
  EXPECT_FALSE(a > b);
  EXPECT_FALSE(a_sv > b);
  EXPECT_FALSE(a_cstr > b);
  EXPECT_FALSE(a > b_sv);
  EXPECT_FALSE(a > b_cstr);
  EXPECT_FALSE(a >= b);
  EXPECT_FALSE(a >= b_sv);
  EXPECT_FALSE(a >= b_cstr);
  EXPECT_FALSE(a_sv >= b);
  EXPECT_FALSE(a_cstr >= b);
}

TEST(RefCountedStringTest, StdStringConversion) {
  std::string s = static_cast<std::string>(RefCountedString("abc"));
  EXPECT_EQ("abc", s);
}

TEST(RefCountedStringTest, Indexing) {
  RefCountedString x = "abc";
  EXPECT_EQ('a', x[0]);
  EXPECT_EQ('c', x[2]);
}

TEST(RefCountedStringTest, Writer) {
  RefCountedStringWriter writer(3);
  memcpy(writer.data(), "abc", 3);
  RefCountedString s = std::move(writer);
  EXPECT_EQ("abc", s);
}

}  // namespace
