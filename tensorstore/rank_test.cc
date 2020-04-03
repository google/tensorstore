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

using tensorstore::AddStaticRanks;
using tensorstore::DimensionIndex;
using tensorstore::dynamic_rank;
using tensorstore::InlineRankLimit;
using tensorstore::IsRankExplicitlyConvertible;
using tensorstore::IsRankImplicitlyConvertible;
using tensorstore::IsStaticRankGreater;
using tensorstore::IsStaticRankGreaterEqual;
using tensorstore::IsStaticRankLess;
using tensorstore::IsStaticRankLessEqual;
using tensorstore::MatchesStatus;
using tensorstore::MaxStaticRank;
using tensorstore::MinStaticRank;
using tensorstore::NormalizeRankSpec;
using tensorstore::StaticRankCast;
using tensorstore::SubtractStaticRanks;
using tensorstore::unchecked;

// Static rank conversion tests
static_assert(IsRankImplicitlyConvertible(3, 3), "");
static_assert(IsRankImplicitlyConvertible(3, dynamic_rank), "");
static_assert(IsRankImplicitlyConvertible(dynamic_rank, dynamic_rank), "");
static_assert(!IsRankImplicitlyConvertible(3, 2), "");
static_assert(!IsRankImplicitlyConvertible(dynamic_rank, 3), "");

static_assert(IsRankExplicitlyConvertible(3, 3), "");
static_assert(IsRankExplicitlyConvertible(dynamic_rank, dynamic_rank), "");
static_assert(IsRankExplicitlyConvertible(dynamic_rank, 3), "");
static_assert(IsRankExplicitlyConvertible(3, dynamic_rank), "");
static_assert(!IsRankExplicitlyConvertible(3, 2), "");

static_assert(AddStaticRanks(2, 3) == 5, "");
static_assert(AddStaticRanks(2, 3, 4) == 9, "");
static_assert(AddStaticRanks(2) == 2, "");
static_assert(AddStaticRanks() == 0, "");
static_assert(AddStaticRanks(dynamic_rank, 3) == dynamic_rank, "");
static_assert(AddStaticRanks(3, dynamic_rank) == dynamic_rank, "");
static_assert(AddStaticRanks(dynamic_rank, dynamic_rank) == dynamic_rank, "");

static_assert(SubtractStaticRanks(5, 2) == 3, "");
static_assert(SubtractStaticRanks(dynamic_rank, 3) == dynamic_rank, "");
static_assert(SubtractStaticRanks(3, dynamic_rank) == dynamic_rank, "");
static_assert(SubtractStaticRanks(dynamic_rank, dynamic_rank) == dynamic_rank,
              "");

static_assert(MinStaticRank(3, 5) == 3, "");
static_assert(MinStaticRank(dynamic_rank, 5) == 5, "");
static_assert(MinStaticRank(5, dynamic_rank) == 5, "");
static_assert(MinStaticRank(dynamic_rank, dynamic_rank) == dynamic_rank, "");
static_assert(MinStaticRank(3, 5, dynamic_rank) == 3, "");
static_assert(MinStaticRank(3) == 3, "");
static_assert(MinStaticRank() == dynamic_rank, "");

static_assert(MaxStaticRank(3, 5) == 5, "");
static_assert(MaxStaticRank(dynamic_rank, 5) == 5, "");
static_assert(MaxStaticRank(5, dynamic_rank) == 5, "");
static_assert(MaxStaticRank(dynamic_rank, dynamic_rank) == dynamic_rank, "");
static_assert(MaxStaticRank(3, 5, dynamic_rank) == 5, "");
static_assert(MaxStaticRank(3) == 3, "");
static_assert(MaxStaticRank() == dynamic_rank, "");

static_assert(IsStaticRankLess(1, 2) == true, "");
static_assert(IsStaticRankLess(1, 1) == false, "");
static_assert(IsStaticRankLess(dynamic_rank, 2) == true, "");
static_assert(IsStaticRankLess(1, dynamic_rank) == true, "");
static_assert(IsStaticRankLess(dynamic_rank, dynamic_rank) == true, "");

static_assert(IsStaticRankLessEqual(1, 2) == true, "");
static_assert(IsStaticRankLessEqual(1, 1) == true, "");
static_assert(IsStaticRankLessEqual(1, 0) == false, "");
static_assert(IsStaticRankLessEqual(dynamic_rank, 2) == true, "");
static_assert(IsStaticRankLessEqual(1, dynamic_rank) == true, "");
static_assert(IsStaticRankLessEqual(dynamic_rank, dynamic_rank) == true, "");

static_assert(IsStaticRankGreater(2, 1) == true, "");
static_assert(IsStaticRankGreater(1, 1) == false, "");
static_assert(IsStaticRankGreater(dynamic_rank, 2) == true, "");
static_assert(IsStaticRankGreater(1, dynamic_rank) == true, "");
static_assert(IsStaticRankGreater(dynamic_rank, dynamic_rank) == true, "");

static_assert(IsStaticRankGreaterEqual(2, 1) == true, "");
static_assert(IsStaticRankGreaterEqual(1, 1) == true, "");
static_assert(IsStaticRankGreaterEqual(0, 1) == false, "");
static_assert(IsStaticRankGreaterEqual(dynamic_rank, 2) == true, "");
static_assert(IsStaticRankGreaterEqual(1, dynamic_rank) == true, "");
static_assert(IsStaticRankGreaterEqual(dynamic_rank, dynamic_rank) == true, "");

TEST(RankCastTest, Basic) {
  auto x =
      StaticRankCast<3>(std::integral_constant<DimensionIndex, 3>()).value();
  static_assert(std::is_same<decltype(x),
                             std::integral_constant<DimensionIndex, 3>>::value,
                "");
  auto y = StaticRankCast<dynamic_rank>(x).value();
  EXPECT_EQ(3, y);
  static_assert(std::is_same<decltype(y), DimensionIndex>::value, "");
  auto a = StaticRankCast<3>(DimensionIndex(3)).value();
  auto b = StaticRankCast<dynamic_rank>(DimensionIndex(3)).value();

  static_assert(std::is_same<decltype(a),
                             std::integral_constant<DimensionIndex, 3>>::value,
                "");
  static_assert(std::is_same<decltype(b), DimensionIndex>::value, "");

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
static_assert(NormalizeRankSpec(dynamic_rank(0)) == -1);
static_assert(NormalizeRankSpec(dynamic_rank(1)) == -1);
static_assert(NormalizeRankSpec(dynamic_rank(2)) == -1);
static_assert(NormalizeRankSpec(0) == 0);
static_assert(NormalizeRankSpec(1) == 1);
static_assert(NormalizeRankSpec(2) == 2);

}  // namespace
