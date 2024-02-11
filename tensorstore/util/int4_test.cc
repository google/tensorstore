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

#include "tensorstore/util/int4.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_gtest.h"

namespace {

using Int4 = tensorstore::Int4Padded;

Int4 Bitcast(int8_t x) { return absl::bit_cast<Int4>(x); }

// Int4 has so few valid values it's practical to list them all.
constexpr std::pair<int8_t, Int4> kInt8ToInt4[] = {
    {-10, Int4(6)}, {-9, Int4(7)},  {-8, Int4(-8)}, {-7, Int4(-7)},
    {-6, Int4(-6)}, {-5, Int4(-5)}, {-4, Int4(-4)}, {-3, Int4(-3)},
    {-2, Int4(-2)}, {-1, Int4(-1)}, {0, Int4(0)},   {1, Int4(1)},
    {2, Int4(2)},   {3, Int4(3)},   {4, Int4(4)},   {5, Int4(5)},
    {6, Int4(6)},   {7, Int4(7)},   {8, Int4(-8)},  {9, Int4(-7)},
    {10, Int4(-6)},
};
constexpr std::pair<Int4, int8_t> kInt4ToInt8[] = {
    {Int4(-8), -8}, {Int4(-7), -7}, {Int4(-6), -6}, {Int4(-5), -5},
    {Int4(-4), -4}, {Int4(-3), -3}, {Int4(-2), -2}, {Int4(-1), -1},
    {Int4(0), 0},   {Int4(1), 1},   {Int4(2), 2},   {Int4(3), 3},
    {Int4(4), 4},   {Int4(5), 5},   {Int4(6), 6},   {Int4(7), 7},
};

TEST(Int4Test, Int8ToInt4) {
  for (const auto& [i8, i4] : kInt8ToInt4) {
    EXPECT_EQ(static_cast<Int4>(i8), i4);
  }
}

TEST(Int4Test, Int4ToInt8) {
  for (const auto& [i4, i8] : kInt4ToInt8) {
    EXPECT_EQ(static_cast<int8_t>(i4), i8);
  }
}

template <typename X>
void TestInt4ToXToInt4() {
  for (const auto& [i4, i8] : kInt4ToInt8) {
    EXPECT_EQ(static_cast<Int4>(static_cast<X>(i4)), i4);
  }
}

TEST(Int4Test, Int4ToInt32ToInt4) { TestInt4ToXToInt4<int32_t>(); }
TEST(Int4Test, Int4ToFloatToInt4) { TestInt4ToXToInt4<float>(); }
TEST(Int4Test, Int4ToDoubleToInt4) { TestInt4ToXToInt4<double>(); }

TEST(Int4Test, Arithmetic) {
  EXPECT_EQ(Int4(1) + Int4(2), Int4(3));
  EXPECT_EQ(Int4(7) + Int4(2), Int4(-7));
  EXPECT_EQ(Int4(3) - Int4(5), Int4(-2));
  EXPECT_EQ(Int4(5) * Int4(-7), Int4(-3));
  EXPECT_EQ(Int4(-8) / Int4(3), Int4(-2));
  EXPECT_EQ(Int4(-7) % Int4(3), Int4(-1));
}

TEST(Int4Test, BitwiseBinary) {
  EXPECT_EQ(Int4(0b0110) & Int4(0b1011), Int4(0b0010));
  EXPECT_EQ(Int4(0b0110) | Int4(0b1011), Int4(0b1111));
  EXPECT_EQ(Int4(0b0110) ^ Int4(0b1011), Int4(0b1101));
}

TEST(Int4Test, BitwiseUnaryInverse) {
  EXPECT_EQ(~Int4(0b1011), Int4(0b0100));
  EXPECT_EQ(~Int4(0b0110), Int4(0b1001));
}

TEST(Int4Test, BitwiseShift) {
  EXPECT_EQ(Int4(0b0011) << Int4(0), Int4(0b0011));
  EXPECT_EQ(Int4(0b0011) << Int4(1), Int4(0b0110));
  EXPECT_EQ(Int4(0b0011) << Int4(2), Int4(0b1100));
  EXPECT_EQ(Int4(0b0011) << Int4(3), Int4(0b1000));
  EXPECT_EQ(Int4(0b0011) << Int4(4), Int4(0b0000));
  EXPECT_EQ(Int4(0b0011) << Int4(5), Int4(0b0000));

  EXPECT_EQ(Int4(0b0011) << int8_t{0}, Int4(0b0011));
  EXPECT_EQ(Int4(0b0011) << int8_t{1}, Int4(0b0110));
  EXPECT_EQ(Int4(0b0011) << int8_t{2}, Int4(0b1100));
  EXPECT_EQ(Int4(0b0011) << int8_t{3}, Int4(0b1000));
  EXPECT_EQ(Int4(0b0011) << int8_t{4}, Int4(0b0000));
  EXPECT_EQ(Int4(0b0011) << int8_t{5}, Int4(0b0000));

  EXPECT_EQ(Int4(0b0100) >> Int4(0), Int4(0b0100));
  EXPECT_EQ(Int4(0b0100) >> Int4(1), Int4(0b0010));
  EXPECT_EQ(Int4(0b0100) >> Int4(2), Int4(0b0001));
  EXPECT_EQ(Int4(0b0100) >> Int4(3), Int4(0b0000));
  EXPECT_EQ(Int4(0b0100) >> Int4(4), Int4(0b0000));
  EXPECT_EQ(Int4(0b0100) >> Int4(5), Int4(0b0000));

  EXPECT_EQ(Int4(0b0100) >> int8_t{0}, Int4(0b0100));
  EXPECT_EQ(Int4(0b0100) >> int8_t{1}, Int4(0b0010));
  EXPECT_EQ(Int4(0b0100) >> int8_t{2}, Int4(0b0001));
  EXPECT_EQ(Int4(0b0100) >> int8_t{3}, Int4(0b0000));
  EXPECT_EQ(Int4(0b0100) >> int8_t{4}, Int4(0b0000));
  EXPECT_EQ(Int4(0b0100) >> int8_t{5}, Int4(0b0000));

  // signed int => arithmetic right shift

  EXPECT_EQ(Int4(0b1010) >> Int4(0), Int4(0b1010));
  EXPECT_EQ(Int4(0b1010) >> Int4(1), Int4(0b1101));
  EXPECT_EQ(Int4(0b1010) >> Int4(2), Int4(0b1110));
  EXPECT_EQ(Int4(0b1010) >> Int4(3), Int4(0b1111));
  EXPECT_EQ(Int4(0b1010) >> Int4(4), Int4(0b1111));
  EXPECT_EQ(Int4(0b1010) >> Int4(5), Int4(0b1111));

  EXPECT_EQ(Int4(0b1010) >> int8_t{0}, Int4(0b1010));
  EXPECT_EQ(Int4(0b1010) >> int8_t{1}, Int4(0b1101));
  EXPECT_EQ(Int4(0b1010) >> int8_t{2}, Int4(0b1110));
  EXPECT_EQ(Int4(0b1010) >> int8_t{3}, Int4(0b1111));
  EXPECT_EQ(Int4(0b1010) >> int8_t{4}, Int4(0b1111));
  EXPECT_EQ(Int4(0b1010) >> int8_t{5}, Int4(0b1111));
}

TEST(Int4Test, Abs) {
  EXPECT_EQ(abs(Int4(7)), Int4(7));
  EXPECT_EQ(abs(Int4(0)), Int4(0));
  EXPECT_EQ(abs(Int4(-7)), Int4(7));
  EXPECT_EQ(abs(Int4(-8)), Int4(-8));
}

TEST(Int4Test, Pow) {
  EXPECT_EQ(pow(Int4(2), Int4(0)), Int4(1));
  EXPECT_EQ(pow(Int4(2), Int4(1)), Int4(2));
  EXPECT_EQ(pow(Int4(2), Int4(2)), Int4(4));
}

TEST(Int4Test, Comparison) {
  for (int i = 0; i <= 15; i++) {
    const Int4 a = kInt4ToInt8[i].first;
    EXPECT_EQ(a, a);
    EXPECT_LE(a, a);
    EXPECT_GE(a, a);
    for (int j = i + 1; j <= 15; j++) {
      const Int4 b = kInt4ToInt8[j].first;
      EXPECT_NE(a, b);
      EXPECT_LT(a, b);
      EXPECT_LE(a, b);
      EXPECT_GT(b, a);
      EXPECT_GE(b, a);
    }
  }
}

// The upper 4 bits of the representation is canonically sign-extended so that
// the int4 is encoded as the numerically identical int8. This property cannot
// be guaranteed in the context of TensorStore (due to memcpy, reinterpret_cast,
// persistence, ...) and in general one int4 has 16 alternative encodings.
// All of these are considered equivalent (`operator ==` returns true).
TEST(Int4Test, EquivalentRepresentationsCompareEqual) {
  for (int low_nibble = 0; low_nibble <= 15; low_nibble++) {
    const Int4 answer = Int4(low_nibble);
    for (int high_nibble_a = 0; high_nibble_a <= 15; high_nibble_a++) {
      for (int high_nibble_b = 0; high_nibble_b <= 15; high_nibble_b++) {
        const int8_t a = low_nibble | (high_nibble_a << 4);
        const int8_t b = low_nibble | (high_nibble_b << 4);
        const Int4 a4 = Bitcast(a);
        const Int4 b4 = Bitcast(b);
        EXPECT_EQ(a4, answer);
        EXPECT_EQ(b4, answer);
        EXPECT_EQ(a4, b4);
      }
    }
  }
}

// Ditto but testing inequality.
TEST(Int4Test, NonCanonicalRepresentationsCompareCorrectly) {
  EXPECT_LT(Bitcast(0xD3), Bitcast(0xE5));
  EXPECT_LE(Bitcast(0xD3), Bitcast(0xE5));
  EXPECT_GT(Bitcast(0x33), Bitcast(0x4A));
  EXPECT_GE(Bitcast(0x33), Bitcast(0x4A));
}

TEST(Int4Test, JsonConversion) {
  for (const auto& [i4, i8] : kInt4ToInt8) {
    EXPECT_THAT(::nlohmann::json(i4), tensorstore::MatchesJson(i8));
  }
}

}  // namespace
