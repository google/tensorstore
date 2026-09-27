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

#include "tensorstore/util/division.h"

#include <stdint.h>

#include <limits>

#include <gtest/gtest.h>

namespace {

static_assert(3 == tensorstore::FloorOfRatio(10, 3));
static_assert(-4 == tensorstore::FloorOfRatio(-10, 3));
static_assert(-4 == tensorstore::FloorOfRatio(10, -3));
static_assert(3 == tensorstore::FloorOfRatio(-10, -3));

// Exact multiples for FloorOfRatio.
static_assert(3 == tensorstore::FloorOfRatio(9, 3));
static_assert(-3 == tensorstore::FloorOfRatio(-9, 3));
static_assert(-3 == tensorstore::FloorOfRatio(9, -3));
static_assert(3 == tensorstore::FloorOfRatio(-9, -3));

// Zero-quotient (|numerator| < |denominator|) for FloorOfRatio.
static_assert(0 == tensorstore::FloorOfRatio(1, 3));
static_assert(-1 == tensorstore::FloorOfRatio(-1, 3));
static_assert(-1 == tensorstore::FloorOfRatio(1, -3));
static_assert(0 == tensorstore::FloorOfRatio(-1, -3));

// Zero numerator for FloorOfRatio.
static_assert(0 == tensorstore::FloorOfRatio(0, 3));
static_assert(0 == tensorstore::FloorOfRatio(0, -3));

static_assert(4 == tensorstore::CeilOfRatio(10, 3));
static_assert(-3 == tensorstore::CeilOfRatio(-10, 3));
static_assert(-3 == tensorstore::CeilOfRatio(10, -3));
static_assert(4 == tensorstore::CeilOfRatio(-10, -3));

// Exact multiples for CeilOfRatio.
static_assert(3 == tensorstore::CeilOfRatio(9, 3));
static_assert(-3 == tensorstore::CeilOfRatio(-9, 3));
static_assert(-3 == tensorstore::CeilOfRatio(9, -3));
static_assert(3 == tensorstore::CeilOfRatio(-9, -3));

// Zero-quotient (|numerator| < |denominator|) for CeilOfRatio.
static_assert(1 == tensorstore::CeilOfRatio(1, 3));
static_assert(0 == tensorstore::CeilOfRatio(-1, 3));
static_assert(0 == tensorstore::CeilOfRatio(1, -3));
static_assert(1 == tensorstore::CeilOfRatio(-1, -3));

// Zero numerator for CeilOfRatio.
static_assert(0 == tensorstore::CeilOfRatio(0, 3));
static_assert(0 == tensorstore::CeilOfRatio(0, -3));

TEST(DivisionTest, CeilAndFloorOfRatio) {
  // int64_t boundary values where casting to double loses precision.
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  EXPECT_EQ(kMax, tensorstore::FloorOfRatio<int64_t>(kMax, 1));
  EXPECT_EQ(kMax, tensorstore::CeilOfRatio<int64_t>(kMax, 1));
  EXPECT_EQ(kMax - 1, tensorstore::FloorOfRatio<int64_t>(kMax - 1, 1));
  EXPECT_EQ(kMax / 2, tensorstore::FloorOfRatio<int64_t>(kMax, 2));
  EXPECT_EQ(kMax / 2 + 1, tensorstore::CeilOfRatio<int64_t>(kMax, 2));
  EXPECT_EQ(0, tensorstore::FloorOfRatio<int64_t>(kMax - 1, kMax));
  EXPECT_EQ(1, tensorstore::CeilOfRatio<int64_t>(kMax - 1, kMax));
  EXPECT_EQ(-1, tensorstore::FloorOfRatio<int64_t>(-(kMax - 1), kMax));
  EXPECT_EQ(0, tensorstore::CeilOfRatio<int64_t>(-(kMax - 1), kMax));
  EXPECT_EQ(kMin / 2, tensorstore::FloorOfRatio<int64_t>(kMin, 2));
  EXPECT_EQ(kMin / 2, tensorstore::CeilOfRatio<int64_t>(kMin, 2));
  EXPECT_EQ(kMin / 3 - 1, tensorstore::FloorOfRatio<int64_t>(kMin, 3));
  EXPECT_EQ(kMin / 3, tensorstore::CeilOfRatio<int64_t>(kMin, 3));

  // Exhaustive property checks across all sign combinations on a grid.
  for (int n = -25; n <= 25; ++n) {
    for (int d = -10; d <= 10; ++d) {
      if (d == 0) continue;
      const int floor_val = tensorstore::FloorOfRatio(n, d);
      const int ceil_val = tensorstore::CeilOfRatio(n, d);
      EXPECT_EQ(ceil_val, -tensorstore::FloorOfRatio(-n, d));
      if (n % d == 0) {
        EXPECT_EQ(n / d, floor_val);
        EXPECT_EQ(n / d, ceil_val);
      } else {
        EXPECT_EQ(floor_val + 1, ceil_val);
        if (d > 0) {
          EXPECT_EQ(n, floor_val * d + tensorstore::NonnegativeMod(n, d));
        }
      }
    }
  }
}

static_assert(0 == tensorstore::RoundUpTo(0, 1));
static_assert(10 == tensorstore::RoundUpTo(7, 5));
static_assert(10 == tensorstore::RoundUpTo(10, 5));
static_assert(2147483645 == tensorstore::RoundUpTo(2147483641, 5));
static_assert(0xffffffffu == tensorstore::RoundUpTo(0xfffffffeu, 3u));
static_assert(0xffffffffu == tensorstore::RoundUpTo<uint32_t>(2, 0xffffffffu));
// Overflow scenario.
static_assert(0u == tensorstore::RoundUpTo<uint32_t>(0xffffffffu, 2u));

static_assert(3 == tensorstore::NonnegativeMod(10, 7));
static_assert(4 == tensorstore::NonnegativeMod(-10, 7));

static_assert(5 == tensorstore::GreatestCommonDivisor(5, 10));
static_assert(5 == tensorstore::GreatestCommonDivisor(10, 15));
static_assert(5 == tensorstore::GreatestCommonDivisor(10, -15));
static_assert(5 == tensorstore::GreatestCommonDivisor(-10, 15));
static_assert(5 == tensorstore::GreatestCommonDivisor(-10, -15));
static_assert(5 == tensorstore::GreatestCommonDivisor(15, 10));
static_assert(5u == tensorstore::GreatestCommonDivisor(15u, 10u));
static_assert(15 == tensorstore::GreatestCommonDivisor(15, 0));
static_assert(15 == tensorstore::GreatestCommonDivisor(-15, 0));
static_assert(15 == tensorstore::GreatestCommonDivisor(0, 15));
static_assert(8 == tensorstore::GreatestCommonDivisor<int32_t>(-0x80000000, 8));
static_assert(8 ==
              tensorstore::GreatestCommonDivisor<int32_t>(-0x80000000, -8));
static_assert(8 == tensorstore::GreatestCommonDivisor<int32_t>(8, -0x80000000));
static_assert(8 ==
              tensorstore::GreatestCommonDivisor<int32_t>(-8, -0x80000000));

static_assert(1 ==
              tensorstore::GreatestCommonDivisor<int32_t>(-0x80000000, -1));
}  // namespace
