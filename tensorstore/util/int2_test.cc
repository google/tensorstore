// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/util/int2.h"

#include <stdint.h>

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include <nlohmann/json_fwd.hpp>
#include "tensorstore/internal/testing/json_gtest.h"

namespace {

using Int2 = tensorstore::Int2Padded;

Int2 Bitcast(int8_t x) { return absl::bit_cast<Int2>(x); }

// Int2 has so few valid values it's practical to list them all.
constexpr std::pair<int8_t, Int2> kInt8ToInt2[] = {
    {-4, Int2(0)}, {-3, Int2(1)}, {-2, Int2(-2)}, {-1, Int2(-1)},
    {0, Int2(0)},  {1, Int2(1)},  {2, Int2(-2)},  {3, Int2(-1)},
};
constexpr std::pair<Int2, int8_t> kInt2ToInt8[] = {
    {Int2(-2), -2},
    {Int2(-1), -1},
    {Int2(0), 0},
    {Int2(1), 1},
};

TEST(Int2Test, Int8ToInt2) {
  for (const auto& [i8, i2] : kInt8ToInt2) {
    EXPECT_EQ(static_cast<Int2>(i8), i2);
  }
}

TEST(Int2Test, Int2ToInt8) {
  for (const auto& [i2, i8] : kInt2ToInt8) {
    EXPECT_EQ(static_cast<int8_t>(i2), i8);
  }
}

template <typename X>
void TestInt2ToXToInt2() {
  for (const auto& [i4, i8] : kInt2ToInt8) {
    EXPECT_EQ(static_cast<Int2>(static_cast<X>(i4)), i4);
  }
}

TEST(Int2Test, Int2ToInt32ToInt2) { TestInt2ToXToInt2<int32_t>(); }
TEST(Int2Test, Int2ToFloatToInt2) { TestInt2ToXToInt2<float>(); }
TEST(Int2Test, Int2ToDoubleToInt2) { TestInt2ToXToInt2<double>(); }

TEST(Int2Test, Arithmetic) {
  EXPECT_EQ(Int2(1) + Int2(1), Int2(-2));
  EXPECT_EQ(Int2(0) + Int2(1), Int2(1));
  EXPECT_EQ(Int2(0) - Int2(1), Int2(-1));
  EXPECT_EQ(Int2(-1) - Int2(1), Int2(-2));
  EXPECT_EQ(Int2(-2) * Int2(-1), Int2(-2));
}

TEST(Int2Test, BitwiseBinary) {
  EXPECT_EQ(Int2(0b10) & Int2(0b11), Int2(0b10));
  EXPECT_EQ(Int2(0b10) | Int2(0b11), Int2(0b11));
  EXPECT_EQ(Int2(0b10) ^ Int2(0b11), Int2(0b01));
}

TEST(Int2Test, BitwiseUnaryInverse) {
  EXPECT_EQ(~Int2(0b11), Int2(0b00));
  EXPECT_EQ(~Int2(0b10), Int2(0b01));
}

TEST(Int2Test, BitwiseShift) {
  EXPECT_EQ(Int2(0b11) << Int2(0), Int2(0b11));
  EXPECT_EQ(Int2(0b11) << Int2(1), Int2(0b10));

  EXPECT_EQ(Int2(0b11) << int8_t{0}, Int2(0b11));
  EXPECT_EQ(Int2(0b11) << int8_t{1}, Int2(0b10));
  EXPECT_EQ(Int2(0b11) << int8_t{2}, Int2(0b00));
  EXPECT_EQ(Int2(0b11) << int8_t{3}, Int2(0b00));

  EXPECT_EQ(Int2(0b10) >> Int2(0), Int2(0b10));
  EXPECT_EQ(Int2(0b10) >> Int2(1), Int2(0b11));
  EXPECT_EQ(Int2(0b11) >> Int2(1), Int2(0b11));

  EXPECT_EQ(Int2(0b11) >> int8_t{0}, Int2(0b11));
  EXPECT_EQ(Int2(0b10) >> int8_t{1}, Int2(0b11));
  EXPECT_EQ(Int2(0b11) >> int8_t{1}, Int2(0b11));

  EXPECT_EQ(Int2(0b01) >> Int2(0), Int2(0b01));
  EXPECT_EQ(Int2(0b01) >> Int2(1), Int2(0b00));

  EXPECT_EQ(Int2(0b01) >> int8_t{0}, Int2(0b01));
  EXPECT_EQ(Int2(0b01) >> int8_t{1}, Int2(0b00));
}

TEST(Int2Test, Abs) {
  EXPECT_EQ(abs(Int2(1)), Int2(1));
  EXPECT_EQ(abs(Int2(0)), Int2(0));
  EXPECT_EQ(abs(Int2(-1)), Int2(1));
  EXPECT_EQ(abs(Int2(-2)), Int2(-2));
}

TEST(Int2Test, Pow) {
  EXPECT_EQ(pow(Int2(1), Int2(0)), Int2(1));
  EXPECT_EQ(pow(Int2(1), Int2(1)), Int2(1));
}

TEST(Int2Test, Comparison) {
  for (int i = 0; i <= 3; i++) {
    const Int2 a = kInt2ToInt8[i].first;
    EXPECT_EQ(a, a);
    EXPECT_LE(a, a);
    EXPECT_GE(a, a);
    for (int j = i + 1; j <= 3; j++) {
      const Int2 b = kInt2ToInt8[j].first;
      EXPECT_NE(a, b);
      EXPECT_LT(a, b);
      EXPECT_LE(a, b);
      EXPECT_GT(b, a);
      EXPECT_GE(b, a);
    }
  }
}

// The upper 6 bits of the representation is canonically sign-extended so that
// the Int2 is encoded as the numerically identical int8. This property cannot
// be guaranteed in the context of TensorStore (due to memcpy, reinterpret_cast,
// persistence, ...) and in general one Int2 has 16 alternative encodings.
// All of these are considered equivalent (`operator ==` returns true).
TEST(Int2Test, EquivalentRepresentationsCompareEqual) {
  for (int low_nibble = 0; low_nibble <= 15; low_nibble++) {
    const Int2 answer = Int2(low_nibble);
    for (int high_nibble_a = 0; high_nibble_a <= 15; high_nibble_a++) {
      for (int high_nibble_b = 0; high_nibble_b <= 15; high_nibble_b++) {
        const int8_t a = low_nibble | (high_nibble_a << 4);
        const int8_t b = low_nibble | (high_nibble_b << 4);
        const Int2 a4 = Bitcast(a);
        const Int2 b4 = Bitcast(b);
        EXPECT_EQ(a4, answer);
        EXPECT_EQ(b4, answer);
        EXPECT_EQ(a4, b4);
      }
    }
  }
}

// Ditto but testing inequality.
TEST(Int2Test, NonCanonicalRepresentationsCompareCorrectly) {
  EXPECT_LT(Bitcast(0xD3), Bitcast(0xE5));
  EXPECT_LE(Bitcast(0xD3), Bitcast(0xE5));
  EXPECT_GT(Bitcast(0x33), Bitcast(0x4A));
  EXPECT_GE(Bitcast(0x33), Bitcast(0x4A));
}

TEST(Int2Test, JsonConversion) {
  for (const auto& [i2, i8] : kInt2ToInt8) {
    EXPECT_THAT(::nlohmann::json(i2), tensorstore::MatchesJson(i8));
  }
}

}  // namespace
