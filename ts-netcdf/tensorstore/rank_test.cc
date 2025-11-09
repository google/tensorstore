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

#include "tensorstore/rank.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::dynamic_rank;
using ::tensorstore::InlineRankLimit;
using ::tensorstore::MatchesStatus;
using ::tensorstore::RankConstraint;
using ::tensorstore::StaticRankCast;
using ::tensorstore::unchecked;

// Static rank conversion tests
static_assert(RankConstraint::Implies(3, 3));
static_assert(RankConstraint::Implies(3, dynamic_rank));
static_assert(RankConstraint::Implies(dynamic_rank, dynamic_rank));
static_assert(!RankConstraint::Implies(3, 2));
static_assert(!RankConstraint::Implies(dynamic_rank, 3));

static_assert(RankConstraint::EqualOrUnspecified(3, 3));
static_assert(RankConstraint::EqualOrUnspecified(dynamic_rank, dynamic_rank));
static_assert(RankConstraint::EqualOrUnspecified(dynamic_rank, 3));
static_assert(RankConstraint::EqualOrUnspecified(3, dynamic_rank));
static_assert(!RankConstraint::EqualOrUnspecified(3, 2));

static_assert(RankConstraint::Add(2, 3) == 5);
static_assert(RankConstraint::Add({2, 3, 4}) == 9);
static_assert(RankConstraint::Add({2}) == 2);
static_assert(RankConstraint::Add({}) == 0);
static_assert(RankConstraint::Add(dynamic_rank, 3) == dynamic_rank);
static_assert(RankConstraint::Add(3, dynamic_rank) == dynamic_rank);
static_assert(RankConstraint::Add(dynamic_rank, dynamic_rank) == dynamic_rank);

static_assert(RankConstraint::Subtract(5, 2) == 3);
static_assert(RankConstraint::Subtract(dynamic_rank, 3) == dynamic_rank);
static_assert(RankConstraint::Subtract(3, dynamic_rank) == dynamic_rank);
static_assert(RankConstraint::Subtract(dynamic_rank, dynamic_rank) ==
              dynamic_rank);

static_assert(RankConstraint::And(dynamic_rank, 5) == 5);
static_assert(RankConstraint::And(5, dynamic_rank) == 5);
static_assert(RankConstraint::And(dynamic_rank, dynamic_rank) == dynamic_rank);
static_assert(RankConstraint::And({5, 5, dynamic_rank}) == 5);
static_assert(RankConstraint::And({3}) == 3);
static_assert(RankConstraint::And({}) == dynamic_rank);

static_assert(RankConstraint::LessOrUnspecified(1, 2) == true);
static_assert(RankConstraint::LessOrUnspecified(1, 1) == false);
static_assert(RankConstraint::LessOrUnspecified(dynamic_rank, 2) == true);
static_assert(RankConstraint::LessOrUnspecified(1, dynamic_rank) == true);
static_assert(RankConstraint::LessOrUnspecified(dynamic_rank, dynamic_rank) ==
              true);

static_assert(RankConstraint::LessEqualOrUnspecified(1, 2) == true);
static_assert(RankConstraint::LessEqualOrUnspecified(1, 1) == true);
static_assert(RankConstraint::LessEqualOrUnspecified(1, 0) == false);
static_assert(RankConstraint::LessEqualOrUnspecified(dynamic_rank, 2) == true);
static_assert(RankConstraint::LessEqualOrUnspecified(1, dynamic_rank) == true);
static_assert(RankConstraint::LessEqualOrUnspecified(dynamic_rank,
                                                     dynamic_rank) == true);

static_assert(RankConstraint::GreaterOrUnspecified(2, 1) == true);
static_assert(RankConstraint::GreaterOrUnspecified(1, 1) == false);
static_assert(RankConstraint::GreaterOrUnspecified(dynamic_rank, 2) == true);
static_assert(RankConstraint::GreaterOrUnspecified(1, dynamic_rank) == true);
static_assert(RankConstraint::GreaterOrUnspecified(dynamic_rank,
                                                   dynamic_rank) == true);

static_assert(RankConstraint::GreaterEqualOrUnspecified(2, 1) == true);
static_assert(RankConstraint::GreaterEqualOrUnspecified(1, 1) == true);
static_assert(RankConstraint::GreaterEqualOrUnspecified(0, 1) == false);
static_assert(RankConstraint::GreaterEqualOrUnspecified(dynamic_rank, 2) ==
              true);
static_assert(RankConstraint::GreaterEqualOrUnspecified(1, dynamic_rank) ==
              true);
static_assert(RankConstraint::GreaterEqualOrUnspecified(dynamic_rank,
                                                        dynamic_rank) == true);

TEST(RankCastTest, Basic) {
  auto x =
      StaticRankCast<3>(std::integral_constant<DimensionIndex, 3>()).value();
  static_assert(
      std::is_same_v<decltype(x), std::integral_constant<DimensionIndex, 3>>);
  auto y = StaticRankCast<dynamic_rank>(x).value();
  EXPECT_EQ(3, y);
  static_assert(std::is_same_v<decltype(y), DimensionIndex>);
  auto a = StaticRankCast<3>(DimensionIndex(3)).value();
  auto b = StaticRankCast<dynamic_rank>(DimensionIndex(3)).value();

  static_assert(
      std::is_same_v<decltype(a), std::integral_constant<DimensionIndex, 3>>);
  static_assert(std::is_same_v<decltype(b), DimensionIndex>);

  EXPECT_THAT((StaticRankCast<3>(DimensionIndex(2))),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot cast rank of 2 to rank of 3"));
  EXPECT_THAT((StaticRankCast<3>(DimensionIndex(3))),
              ::testing::Optional(tensorstore::StaticRank<3>()));
  EXPECT_THAT((StaticRankCast<3>(DimensionIndex(dynamic_rank))),
              ::testing::Optional(tensorstore::StaticRank<3>()));
}

TEST(RankCastDeathTest, DynamicToStatic) {
  // Casting from the dynamic rank of 1 to the static rank of 3 should result in
  // an assertion failure at run time.
  EXPECT_DEBUG_DEATH((StaticRankCast<3, unchecked>(DimensionIndex(1))),
                     "StaticCast is not valid");
}

static_assert(InlineRankLimit(dynamic_rank(0)) == 0);
static_assert(InlineRankLimit(dynamic_rank(1)) == 1);
static_assert(InlineRankLimit(dynamic_rank(2)) == 2);
static_assert(RankConstraint::FromInlineRank(dynamic_rank(0)) == -1);
static_assert(RankConstraint::FromInlineRank(dynamic_rank(1)) == -1);
static_assert(RankConstraint::FromInlineRank(dynamic_rank(2)) == -1);
static_assert(RankConstraint::FromInlineRank(0) == 0);
static_assert(RankConstraint::FromInlineRank(1) == 1);
static_assert(RankConstraint::FromInlineRank(2) == 2);

}  // namespace
