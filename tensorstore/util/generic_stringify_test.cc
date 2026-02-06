// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/util/generic_stringify.h"

#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::GenericStringify;
using ::tensorstore::Result;
using ::testing::StrEq;

TEST(GenericStringifyTest, Null) {
  EXPECT_THAT(absl::StrCat(GenericStringify(nullptr)), StrEq("null"));
  EXPECT_THAT(absl::StrCat(GenericStringify(std::nullopt)), StrEq("null"));
}

TEST(GenericStringifyTest, String) {
  EXPECT_THAT(absl::StrCat(GenericStringify("abc")), StrEq("abc"));
  EXPECT_THAT(absl::StrCat(GenericStringify(std::string("abc"))), StrEq("abc"));
  EXPECT_THAT(absl::StrCat(GenericStringify(std::string_view("abc"))),
              StrEq("abc"));
}

TEST(GenericStringifyTest, Numeric) {
  EXPECT_THAT(absl::StrCat(GenericStringify(1)), StrEq("1"));
  EXPECT_THAT(absl::StrCat(GenericStringify(1U)), StrEq("1"));
  EXPECT_THAT(absl::StrCat(GenericStringify(1.5f)), StrEq("1.500000"));
  EXPECT_THAT(absl::StrCat(GenericStringify(1.5)), StrEq("1.500000"));
  EXPECT_THAT(absl::StrCat(GenericStringify('a')), StrEq("a"));
}

TEST(GenericStringifyTest, Bool) {
  EXPECT_THAT(absl::StrCat(GenericStringify(true)), StrEq("true"));
  EXPECT_THAT(absl::StrCat(GenericStringify(false)), StrEq("false"));
}

enum class MyEnum { kA = 1, kB = 2 };

TEST(GenericStringifyTest, Enum) {
  EXPECT_THAT(absl::StrCat(GenericStringify(MyEnum::kA)), StrEq("1"));
}

TEST(GenericStringifyTest, Optional) {
  EXPECT_THAT(absl::StrCat(GenericStringify(std::optional<int>(1))),
              StrEq("<1>"));
  EXPECT_THAT(absl::StrCat(GenericStringify(std::optional<std::string>("abc"))),
              StrEq("<abc>"));
  const std::optional<int> opt_null = std::nullopt;
  EXPECT_THAT(absl::StrCat(GenericStringify(opt_null)), StrEq("null"));
}

TEST(GenericStringifyTest, Result) {
  EXPECT_THAT(absl::StrCat(GenericStringify(Result<int>(1))), StrEq("<OK: 1>"));
  EXPECT_THAT(absl::StrCat(GenericStringify(
                  Result<int>(absl::InvalidArgumentError("Test")))),
              testing::StartsWith("<INVALID_ARGUMENT: Test"));
}

TEST(GenericStringifyTest, Pair) {
  EXPECT_THAT(
      absl::StrCat(GenericStringify(std::pair<int, std::string>(1, "a"))),
      StrEq("{1, a}"));
}

TEST(GenericStringifyTest, Tuple) {
  EXPECT_THAT(absl::StrCat(GenericStringify(std::tuple<>{})), StrEq("{}"));
  EXPECT_THAT(absl::StrCat(GenericStringify(std::make_tuple(1, "a", 3.5))),
              StrEq("{1, a, 3.500000}"));
}

struct MyStringifiable {
  int value;
  template <typename Sink>
  friend void AbslStringify(Sink& sink, MyStringifiable v) {
    absl::Format(&sink, "MyStringifiable{%d}", v.value);
  }
};

TEST(GenericStringifyTest, AbslStringify) {
  std::optional<MyStringifiable> optional_stringifiable(MyStringifiable{10});
  EXPECT_THAT(absl::StrCat(GenericStringify(optional_stringifiable)),
              StrEq("<MyStringifiable{10}>"));
}

struct OstreamablePoint {
  int x;
  int y;

  [[maybe_unused]] friend std::ostream& operator<<(std::ostream& os,
                                                   const OstreamablePoint& p) {
    return os << "P(" << p.x << "," << p.y << ")";
  }
};

TEST(GenericStringifyTest, Ostreamable) {
  EXPECT_THAT(absl::StrCat(GenericStringify(OstreamablePoint{1, 2})),
              StrEq("P(1,2)"));
}

TEST(GenericStringifyTest, Vector) {
  std::vector<int> vec{1, 2, 3};
  EXPECT_THAT(absl::StrCat(GenericStringify(vec)), StrEq("{1, 2, 3}"));
}

TEST(GenericStringifyTest, Span) {
  std::vector<int> vec{1, 2, 3};
  EXPECT_THAT(absl::StrCat(GenericStringify(tensorstore::span<int>(vec))),
              StrEq("{1, 2, 3}"));
}

}  // namespace
