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

#include "tensorstore/kvstore/byte_range.h"

#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using tensorstore::ByteRange;
using tensorstore::MatchesStatus;
using tensorstore::OptionalByteRangeRequest;
using tensorstore::StrCat;
using tensorstore::internal::GetSubCord;

TEST(ByteRangeTest, SatisfiesInvariants) {
  EXPECT_TRUE((ByteRange{0, 1}).SatisfiesInvariants());
  EXPECT_TRUE((ByteRange{0, 0}).SatisfiesInvariants());
  EXPECT_TRUE((ByteRange{0, 100}).SatisfiesInvariants());
  EXPECT_TRUE((ByteRange{10, 100}).SatisfiesInvariants());
  EXPECT_TRUE((ByteRange{100, 100}).SatisfiesInvariants());
  EXPECT_FALSE((ByteRange{100, 99}).SatisfiesInvariants());
  EXPECT_FALSE((ByteRange{100, 0}).SatisfiesInvariants());
}

TEST(ByteRangeTest, Size) {
  EXPECT_EQ(5, (ByteRange{2, 7}.size()));
  EXPECT_EQ(0, (ByteRange{2, 2}.size()));
}

TEST(ByteRangeTest, Comparison) {
  ByteRange a{1, 2};
  ByteRange b{1, 3};
  ByteRange c{2, 3};
  EXPECT_TRUE(a == a);
  EXPECT_TRUE(b == b);
  EXPECT_TRUE(c == c);
  EXPECT_FALSE(a != a);
  EXPECT_FALSE(b != b);
  EXPECT_FALSE(c != c);
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a != b);
  EXPECT_NE(a, c);
  EXPECT_NE(b, c);
}

TEST(ByteRangeTest, Ostream) {
  EXPECT_EQ("[1, 10)", tensorstore::StrCat(ByteRange{1, 10}));
}

TEST(OptionalByteRangeRequestTest, DefaultConstruct) {
  OptionalByteRangeRequest r;
  EXPECT_EQ(0, r.inclusive_min);
  EXPECT_EQ(std::nullopt, r.exclusive_max);
}

TEST(OptionalByteRangeRequestTest, ConstructInclusiveMin) {
  OptionalByteRangeRequest r(5);
  EXPECT_EQ(5, r.inclusive_min);
  EXPECT_EQ(std::nullopt, r.exclusive_max);
}

TEST(OptionalByteRangeRequestTest, ConstructInclusiveMinExclusiveMax) {
  OptionalByteRangeRequest r(5, 10);
  EXPECT_EQ(5, r.inclusive_min);
  EXPECT_EQ(10, r.exclusive_max);
}

TEST(OptionalByteRangeRequestTest, ConstructByteRange) {
  OptionalByteRangeRequest r(ByteRange{5, 10});
  EXPECT_EQ(5, r.inclusive_min);
  EXPECT_EQ(10, r.exclusive_max);
}

TEST(OptionalByteRangeRequestTest, Comparison) {
  OptionalByteRangeRequest a{1, 2};
  OptionalByteRangeRequest b{1, 3};
  OptionalByteRangeRequest c{2, 3};
  OptionalByteRangeRequest d{1, std::nullopt};
  EXPECT_TRUE(a == a);
  EXPECT_TRUE(b == b);
  EXPECT_TRUE(c == c);
  EXPECT_TRUE(d == d);
  EXPECT_FALSE(a != a);
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(a != c);
  EXPECT_TRUE(a != d);
  EXPECT_TRUE(b != d);
  EXPECT_TRUE(c != d);
}

TEST(OptionalByteRangeRequestTest, SatisfiesInvariants) {
  EXPECT_TRUE(OptionalByteRangeRequest().SatisfiesInvariants());
  EXPECT_TRUE(OptionalByteRangeRequest(10).SatisfiesInvariants());
  EXPECT_TRUE(OptionalByteRangeRequest(0, 1).SatisfiesInvariants());
  EXPECT_TRUE(OptionalByteRangeRequest(0, 0).SatisfiesInvariants());
  EXPECT_TRUE(OptionalByteRangeRequest(0, 100).SatisfiesInvariants());
  EXPECT_TRUE(OptionalByteRangeRequest(10, 100).SatisfiesInvariants());
  EXPECT_TRUE(OptionalByteRangeRequest(100, 100).SatisfiesInvariants());
  EXPECT_FALSE(OptionalByteRangeRequest(100, 99).SatisfiesInvariants());
  EXPECT_FALSE(OptionalByteRangeRequest(100, 0).SatisfiesInvariants());
}

TEST(OptionalByteRangeRequestTest, Ostream) {
  EXPECT_EQ("[5, 10)", StrCat(OptionalByteRangeRequest(5, 10)));
  EXPECT_EQ("[5, ?)", StrCat(OptionalByteRangeRequest(5)));
}

TEST(OptionalByteRangeRequestTest, Validate) {
  EXPECT_THAT(OptionalByteRangeRequest(5, 10).Validate(20),
              ::testing::Optional(ByteRange{5, 10}));
  EXPECT_THAT(OptionalByteRangeRequest(5, 10).Validate(10),
              ::testing::Optional(ByteRange{5, 10}));
  EXPECT_THAT(OptionalByteRangeRequest(5).Validate(10),
              ::testing::Optional(ByteRange{5, 10}));
  EXPECT_THAT(OptionalByteRangeRequest(5, 10).Validate(9),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Requested byte range \\[5, 10\\) is not valid for "
                            "value of size 9"));
  EXPECT_THAT(
      OptionalByteRangeRequest(15, 15).Validate(9),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    "Requested byte range \\[15, 15\\) is not valid for "
                    "value of size 9"));
}

TEST(GetSubStringTest, Basic) {
  EXPECT_EQ("bcd", GetSubCord(absl::Cord("abcde"), {1, 4}));
  EXPECT_EQ("bcd", GetSubCord(absl::Cord("abcde"), {1, 4}));
  EXPECT_EQ("abcde", GetSubCord(absl::Cord("abcde"), {0, 5}));
}

}  // namespace
