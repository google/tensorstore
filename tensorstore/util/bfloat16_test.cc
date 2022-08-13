// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/util/bfloat16.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/data_type.h"
#include "tensorstore/internal/bit_operations.h"
#include "tensorstore/internal/json_gtest.h"

// The implementation below is derived from Tensorflow and Eigen:
//
//  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace {
using ::tensorstore::bfloat16_t;
using ::tensorstore::internal::bit_cast;
using ::tensorstore::internal::Float32ToBfloat16RoundNearestEven;
using ::tensorstore::internal::Float32ToBfloat16Truncate;

::testing::Matcher<bfloat16_t> MatchesBits(uint16_t bits) {
  return ::testing::ResultOf([](bfloat16_t y) { return bit_cast<uint16_t>(y); },
                             ::testing::Eq(bits));
}

::testing::Matcher<float> NearFloat(float x, float relative_error = 1e-3) {
  return ::testing::FloatNear(x, std::abs(x) * relative_error);
}

float BinaryToFloat(uint32_t sign, uint32_t exponent, uint32_t high_mantissa,
                    uint32_t low_mantissa) {
  float dest;
  uint32_t src =
      (sign << 31) + (exponent << 23) + (high_mantissa << 16) + low_mantissa;
  memcpy(static_cast<void*>(&dest), static_cast<const void*>(&src),
         sizeof(dest));
  return dest;
}

void TestTruncate(float input, float expected_truncation,
                  float expected_rounding) {
  bfloat16_t truncated = Float32ToBfloat16Truncate(input);
  bfloat16_t rounded = Float32ToBfloat16RoundNearestEven(input);
  if (std::isnan(input)) {
    EXPECT_TRUE(std::isnan(truncated));
    EXPECT_TRUE(std::isnan(rounded));
    return;
  }
  EXPECT_EQ(expected_truncation, static_cast<float>(truncated));
  EXPECT_EQ(expected_rounding, static_cast<float>(rounded));
}

template <typename T>
void TestRoundtrips() {
  for (T value : {
           -std::numeric_limits<T>::infinity(),
           std::numeric_limits<T>::infinity(),
           T(-1.0),
           T(-0.5),
           T(-0.0),
           T(1.0),
           T(0.5),
           T(0.0),
       }) {
    EXPECT_EQ(value, static_cast<T>(static_cast<bfloat16_t>(value)));
  }
}

TEST(Bfloat16Test, FloatRoundtrips) { TestRoundtrips<float>(); }

TEST(Bfloat16Test, DoubleRoundtrips) { TestRoundtrips<double>(); }

TEST(Bfloat16Test, Float16Roundtrips) {
  TestRoundtrips<tensorstore::float16_t>();
}

TEST(Bfloat16Test, ConversionFromFloat) {
  EXPECT_THAT(bfloat16_t(1.0f), MatchesBits(0x3f80));
  EXPECT_THAT(bfloat16_t(0.5f), MatchesBits(0x3f00));
  EXPECT_THAT(bfloat16_t(0.33333f), MatchesBits(0x3eab));
  EXPECT_THAT(bfloat16_t(3.38e38f), MatchesBits(0x7f7e));
  EXPECT_THAT(bfloat16_t(3.40e38f), MatchesBits(0x7f80));  // Becomes infinity.
}

TEST(Bfloat16Test, RoundToNearestEven) {
  float val1 = static_cast<float>(bit_cast<bfloat16_t>(uint16_t(0x3c00)));
  float val2 = static_cast<float>(bit_cast<bfloat16_t>(uint16_t(0x3c01)));
  float val3 = static_cast<float>(bit_cast<bfloat16_t>(uint16_t(0x3c02)));
  EXPECT_THAT(bfloat16_t(0.5f * (val1 + val2)), MatchesBits(0x3c00));
  EXPECT_THAT(bfloat16_t(0.5f * (val2 + val3)), MatchesBits(0x3c02));
}

TEST(Bfloat16Test, ConversionFromInt) {
  EXPECT_THAT(bfloat16_t(-1), MatchesBits(0xbf80));
  EXPECT_THAT(bfloat16_t(0), MatchesBits(0x0000));
  EXPECT_THAT(bfloat16_t(1), MatchesBits(0x3f80));
  EXPECT_THAT(bfloat16_t(2), MatchesBits(0x4000));
  EXPECT_THAT(bfloat16_t(3), MatchesBits(0x4040));
  EXPECT_THAT(bfloat16_t(12), MatchesBits(0x4140));
}

TEST(Bfloat16Test, ConversionFromBool) {
  EXPECT_THAT(bfloat16_t(false), MatchesBits(0x0000));
  EXPECT_THAT(bfloat16_t(true), MatchesBits(0x3f80));
}

TEST(Bfloat16Test, ConversionToBool) {
  EXPECT_EQ(static_cast<bool>(bfloat16_t(3)), true);
  EXPECT_EQ(static_cast<bool>(bfloat16_t(0.33333f)), true);
  EXPECT_EQ(bfloat16_t(-0.0), false);
  EXPECT_EQ(static_cast<bool>(bfloat16_t(0.0)), false);
}

TEST(Bfloat16Test, ExplicitConversionToFloat) {
  EXPECT_EQ(static_cast<float>(bit_cast<bfloat16_t, uint16_t>(0x0000)), 0.0f);
  EXPECT_EQ(static_cast<float>(bit_cast<bfloat16_t, uint16_t>(0x3f80)), 1.0f);
}

TEST(Bfloat16Test, ImplicitConversionToFloat) {
  EXPECT_EQ((bit_cast<bfloat16_t, uint16_t>(0x0000)), 0.0f);
  EXPECT_EQ((bit_cast<bfloat16_t, uint16_t>(0x3f80)), 1.0f);
}

TEST(Bfloat16Test, Zero) {
  EXPECT_EQ(bfloat16_t(0.0f), bfloat16_t(0.0f));
  EXPECT_EQ(bfloat16_t(-0.0f), bfloat16_t(0.0f));
  EXPECT_EQ(bfloat16_t(-0.0f), bfloat16_t(-0.0f));
  EXPECT_THAT(bfloat16_t(0.0f), MatchesBits(0x0000));
  EXPECT_THAT(bfloat16_t(-0.0f), MatchesBits(0x8000));
}

TEST(Bfloat16Test, DefaultConstruct) {
  EXPECT_EQ(static_cast<float>(bfloat16_t()), 0.0f);
}

TEST(Bfloat16Test, Truncate0) {
  TestTruncate(BinaryToFloat(0, 0x80, 0x48, 0xf5c3),   // +0x1.91eb86p+1
               BinaryToFloat(0, 0x80, 0x48, 0x0000),   // +0x1.900000p+1
               BinaryToFloat(0, 0x80, 0x49, 0x0000));  // +0x1.920000p+1
}
TEST(Bfloat16Test, Truncate1) {
  TestTruncate(BinaryToFloat(1, 0x80, 0x48, 0xf5c3),   // -0x1.91eb86p+1
               BinaryToFloat(1, 0x80, 0x48, 0x0000),   // -0x1.900000p+1
               BinaryToFloat(1, 0x80, 0x49, 0x0000));  // -0x1.920000p+1
}

TEST(Bfloat16Test, Truncate2) {
  TestTruncate(BinaryToFloat(0, 0x80, 0x48, 0x8000),   // +0x1.910000p+1
               BinaryToFloat(0, 0x80, 0x48, 0x0000),   // +0x1.900000p+1
               BinaryToFloat(0, 0x80, 0x48, 0x0000));  // +0x1.900000p+1
}

TEST(Bfloat16Test, Truncate3) {
  TestTruncate(BinaryToFloat(0, 0xff, 0x00, 0x0001),   // nan
               BinaryToFloat(0, 0xff, 0x40, 0x0000),   // nan
               BinaryToFloat(0, 0xff, 0x40, 0x0000));  // nan
}

TEST(Bfloat16Test, Truncate4) {
  TestTruncate(BinaryToFloat(0, 0xff, 0x7f, 0xffff),   // nan
               BinaryToFloat(0, 0xff, 0x40, 0x0000),   // nan
               BinaryToFloat(0, 0xff, 0x40, 0x0000));  // nan
}

TEST(Bfloat16Test, Truncate5) {
  TestTruncate(BinaryToFloat(1, 0x80, 0x48, 0xc000),   // -0x1.918000p+1
               BinaryToFloat(1, 0x80, 0x48, 0x0000),   // -0x1.900000p+1
               BinaryToFloat(1, 0x80, 0x49, 0x0000));  // -0x1.920000p+1
}

TEST(Bfloat16Test, Truncate6) {
  TestTruncate(BinaryToFloat(0, 0x80, 0x48, 0x0000),   // 0x1.900000p+1
               BinaryToFloat(0, 0x80, 0x48, 0x0000),   // 0x1.900000p+1
               BinaryToFloat(0, 0x80, 0x48, 0x0000));  // 0x1.900000p+1
}

TEST(Bfloat16Test, Truncate7) {
  TestTruncate(BinaryToFloat(0, 0x80, 0x48, 0x4000),   // 0x1.908000p+1
               BinaryToFloat(0, 0x80, 0x48, 0x0000),   // 0x1.900000p+1
               BinaryToFloat(0, 0x80, 0x48, 0x0000));  // 0x1.900000p+1
}

TEST(Bfloat16Test, Truncate8) {
  TestTruncate(BinaryToFloat(0, 0x80, 0x48, 0x8000),   // 0x1.910000p+1
               BinaryToFloat(0, 0x80, 0x48, 0x0000),   // 0x1.900000p+1
               BinaryToFloat(0, 0x80, 0x48, 0x0000));  // 0x1.900000p+1
}

TEST(Bfloat16Test, Truncate9) {
  TestTruncate(BinaryToFloat(0, 0x00, 0x48, 0x8000),   // 0x1.220000p-127
               BinaryToFloat(0, 0x00, 0x48, 0x0000),   // 0x1.200000p-127
               BinaryToFloat(0, 0x00, 0x48, 0x0000));  // 0x1.200000p-127
}

TEST(Bfloat16Test, Truncate10) {
  TestTruncate(BinaryToFloat(0, 0x00, 0x7f, 0xc000),   // 0x1.ff0000p-127
               BinaryToFloat(0, 0x00, 0x7f, 0x0000),   // 0x1.fc0000p-127
               BinaryToFloat(0, 0x00, 0x80, 0x0000));  // 0x1.000000p-126
}

TEST(Bfloat16Test, Conversion) {
  for (int i = 0; i < 100; ++i) {
    float a = i + 1.25;
    bfloat16_t b = static_cast<bfloat16_t>(a);
    float c = static_cast<float>(b);
    EXPECT_LE(std::abs(c - a), a / 128);
  }
}

TEST(Bfloat16Test, Epsilon) {
  EXPECT_LE(1.0f,
            static_cast<float>(std::numeric_limits<bfloat16_t>::epsilon() +
                               bfloat16_t(1.0f)));
  EXPECT_EQ(1.0f,
            static_cast<float>(std::numeric_limits<bfloat16_t>::epsilon() /
                                   bfloat16_t(2.0f) +
                               bfloat16_t(1.0f)));
}

TEST(Bfloat16Test, NextAfter) {
  const bfloat16_t one(1), two(2), zero(0),
      nan = std::numeric_limits<bfloat16_t>::quiet_NaN(),
      epsilon = std::numeric_limits<bfloat16_t>::epsilon(),
      denorm_min = std::numeric_limits<bfloat16_t>::denorm_min();
  EXPECT_EQ(epsilon, nextafter(one, two) - one);
  EXPECT_EQ(-epsilon / 2, nextafter(one, zero) - one);
  EXPECT_EQ(one, nextafter(one, one));
  EXPECT_EQ(denorm_min, nextafter(zero, one));
  EXPECT_EQ(-denorm_min, nextafter(zero, -one));
  const bfloat16_t values[] = {zero, -zero, nan};
  for (int i = 0; i < 3; ++i) {
    auto a = values[i];
    for (int j = 0; j < 3; ++j) {
      if (i == j) continue;
      auto b = values[j];
      auto next_float =
          std::nextafter(static_cast<float>(a), static_cast<float>(b));
      auto next_bfloat16 = nextafter(a, b);
      EXPECT_EQ(std::isnan(next_float), isnan(next_bfloat16));
      if (!std::isnan(next_float)) {
        EXPECT_EQ(next_float, next_bfloat16);
      }
    }
  }
  EXPECT_EQ(std::numeric_limits<bfloat16_t>::infinity(),
            nextafter(std::numeric_limits<bfloat16_t>::max(),
                      std::numeric_limits<bfloat16_t>::infinity()));
}

TEST(Bfloat16Test, Negate) {
  EXPECT_EQ(static_cast<float>(-bfloat16_t(3.0f)), -3.0f);
  EXPECT_EQ(static_cast<float>(-bfloat16_t(-4.5f)), 4.5f);
}

// Visual Studio errors out on divisions by 0
#ifndef _MSC_VER
TEST(Bfloat16Test, DivisionByZero) {
  EXPECT_TRUE(std::isnan(static_cast<float>(bfloat16_t(0.0 / 0.0))));
  EXPECT_TRUE(std::isinf(static_cast<float>(bfloat16_t(1.0 / 0.0))));
  EXPECT_TRUE(std::isinf(static_cast<float>(bfloat16_t(-1.0 / 0.0))));

  // Visual Studio errors out on divisions by 0
  EXPECT_TRUE(std::isnan(bfloat16_t(0.0 / 0.0)));
  EXPECT_TRUE(std::isinf(bfloat16_t(1.0 / 0.0)));
  EXPECT_TRUE(std::isinf(bfloat16_t(-1.0 / 0.0)));
}
#endif

TEST(Bfloat16Test, NonFinite) {
  EXPECT_FALSE(std::isinf(
      static_cast<float>(bfloat16_t(3.38e38f))));  // Largest finite number.
  EXPECT_FALSE(std::isnan(static_cast<float>(bfloat16_t(0.0f))));
  EXPECT_TRUE(
      std::isinf(static_cast<float>(bit_cast<bfloat16_t, uint16_t>(0xff80))));
  EXPECT_TRUE(
      std::isnan(static_cast<float>(bit_cast<bfloat16_t, uint16_t>(0xffc0))));
  EXPECT_TRUE(
      std::isinf(static_cast<float>(bit_cast<bfloat16_t, uint16_t>(0x7f80))));
  EXPECT_TRUE(
      std::isnan(static_cast<float>(bit_cast<bfloat16_t, uint16_t>(0x7fc0))));

  // Exactly same checks as above, just directly on the bfloat16 representation.
  EXPECT_FALSE(isinf(bit_cast<bfloat16_t, uint16_t>(0x7bff)));
  EXPECT_FALSE(isnan(bit_cast<bfloat16_t, uint16_t>(0x0000)));
  EXPECT_TRUE(isinf(bit_cast<bfloat16_t, uint16_t>(0xff80)));
  EXPECT_TRUE(isnan(bit_cast<bfloat16_t, uint16_t>(0xffc0)));
  EXPECT_TRUE(isinf(bit_cast<bfloat16_t, uint16_t>(0x7f80)));
  EXPECT_TRUE(isnan(bit_cast<bfloat16_t, uint16_t>(0x7fc0)));

  EXPECT_THAT(bfloat16_t(BinaryToFloat(0x0, 0xff, 0x40, 0x0)),  // +nan
              MatchesBits(0x7fe0));
  EXPECT_THAT(bfloat16_t(BinaryToFloat(0x1, 0xff, 0x40, 0x0)),  // -nan
              MatchesBits(0xffe0));
  EXPECT_THAT(
      Float32ToBfloat16Truncate(BinaryToFloat(0x0, 0xff, 0x40, 0x0)),  // +nan
      MatchesBits(0x7fe0));
  EXPECT_THAT(
      Float32ToBfloat16Truncate(BinaryToFloat(0x1, 0xff, 0x40, 0x0)),  // -nan
      MatchesBits(0xffe0));
}

TEST(Bfloat16Test, NumericLimits) {
  static_assert(std::numeric_limits<bfloat16_t>::is_signed);

  EXPECT_EQ(
      bit_cast<uint16_t>(std::numeric_limits<bfloat16_t>::infinity()),
      bit_cast<uint16_t>(bfloat16_t(std::numeric_limits<float>::infinity())));
  // There is no guarantee that casting a 32-bit NaN to bfloat16 has a precise
  // bit pattern.  We test that it is in fact a NaN, then test the signaling
  // bit (msb of significand is 1 for quiet, 0 for signaling).
  constexpr uint16_t BFLOAT16_QUIET_BIT = 0x0040;
  EXPECT_TRUE(isnan(std::numeric_limits<bfloat16_t>::quiet_NaN()));
  EXPECT_TRUE(isnan(bfloat16_t(std::numeric_limits<float>::quiet_NaN())));
  EXPECT_GT((bit_cast<uint16_t>(std::numeric_limits<bfloat16_t>::quiet_NaN()) &
             BFLOAT16_QUIET_BIT),
            0);
  EXPECT_GT(
      (bit_cast<uint16_t>(bfloat16_t(std::numeric_limits<float>::quiet_NaN())) &
       BFLOAT16_QUIET_BIT),
      0);

  EXPECT_TRUE(isnan(std::numeric_limits<bfloat16_t>::signaling_NaN()));
  EXPECT_TRUE(isnan(bfloat16_t(std::numeric_limits<float>::signaling_NaN())));
  EXPECT_EQ(
      0, (bit_cast<uint16_t>(std::numeric_limits<bfloat16_t>::signaling_NaN()) &
          BFLOAT16_QUIET_BIT));
#ifndef _MSC_VER
  // MSVC seems to not preserve signaling bit.
  EXPECT_EQ(0, (bit_cast<uint16_t>(
                    bfloat16_t(std::numeric_limits<float>::signaling_NaN())) &
                BFLOAT16_QUIET_BIT));
#endif

  EXPECT_GT(std::numeric_limits<bfloat16_t>::min(), bfloat16_t(0.f));
  EXPECT_GT(std::numeric_limits<bfloat16_t>::denorm_min(), bfloat16_t(0.f));
  EXPECT_EQ(std::numeric_limits<bfloat16_t>::denorm_min() / bfloat16_t(2),
            bfloat16_t(0.f));
}

TEST(Bfloat16Test, Arithmetic) {
  EXPECT_EQ(static_cast<float>(bfloat16_t(2) + bfloat16_t(2)), 4);
  EXPECT_EQ(static_cast<float>(bfloat16_t(2) + bfloat16_t(-2)), 0);
  EXPECT_THAT(static_cast<float>(bfloat16_t(0.33333f) + bfloat16_t(0.66667f)),
              NearFloat(1.0f));
  EXPECT_EQ(static_cast<float>(bfloat16_t(2.0f) * bfloat16_t(-5.5f)), -11.0f);
  EXPECT_THAT(static_cast<float>(bfloat16_t(1.0f) / bfloat16_t(3.0f)),
              NearFloat(0.3339f));
  EXPECT_EQ(static_cast<float>(-bfloat16_t(4096.0f)), -4096.0f);
  EXPECT_EQ(static_cast<float>(-bfloat16_t(-4096.0f)), 4096.0f);
}

TEST(Bfloat16Test, Comparison) {
  EXPECT_TRUE(bfloat16_t(1.0f) > bfloat16_t(0.5f));
  EXPECT_TRUE(bfloat16_t(0.5f) < bfloat16_t(1.0f));
  EXPECT_FALSE((bfloat16_t(1.0f) < bfloat16_t(0.5f)));
  EXPECT_FALSE((bfloat16_t(0.5f) > bfloat16_t(1.0f)));

  EXPECT_FALSE((bfloat16_t(4.0f) > bfloat16_t(4.0f)));
  EXPECT_FALSE((bfloat16_t(4.0f) < bfloat16_t(4.0f)));

  EXPECT_FALSE((bfloat16_t(0.0f) < bfloat16_t(-0.0f)));
  EXPECT_FALSE((bfloat16_t(-0.0f) < bfloat16_t(0.0f)));
  EXPECT_FALSE((bfloat16_t(0.0f) > bfloat16_t(-0.0f)));
  EXPECT_FALSE((bfloat16_t(-0.0f) > bfloat16_t(0.0f)));

  EXPECT_TRUE(bfloat16_t(0.2f) > bfloat16_t(-1.0f));
  EXPECT_TRUE(bfloat16_t(-1.0f) < bfloat16_t(0.2f));
  EXPECT_TRUE(bfloat16_t(-16.0f) < bfloat16_t(-15.0f));

  EXPECT_TRUE(bfloat16_t(1.0f) == bfloat16_t(1.0f));
  EXPECT_TRUE(bfloat16_t(1.0f) != bfloat16_t(2.0f));

  // Comparisons with NaNs and infinities.
#ifndef _MSC_VER
  // Visual Studio errors out on divisions by 0
  EXPECT_FALSE((bfloat16_t(0.0 / 0.0) == bfloat16_t(0.0 / 0.0)));
  EXPECT_TRUE(bfloat16_t(0.0 / 0.0) != bfloat16_t(0.0 / 0.0));

  EXPECT_FALSE((bfloat16_t(1.0) == bfloat16_t(0.0 / 0.0)));
  EXPECT_FALSE((bfloat16_t(1.0) < bfloat16_t(0.0 / 0.0)));
  EXPECT_FALSE((bfloat16_t(1.0) > bfloat16_t(0.0 / 0.0)));
  EXPECT_TRUE(bfloat16_t(1.0) != bfloat16_t(0.0 / 0.0));

  EXPECT_TRUE(bfloat16_t(1.0) < bfloat16_t(1.0 / 0.0));
  EXPECT_TRUE(bfloat16_t(1.0) > bfloat16_t(-1.0 / 0.0));
#endif
}

constexpr float PI = 3.14159265358979323846f;

TEST(Bfloat16Test, BasicFunctions) {
  EXPECT_EQ(static_cast<float>(tensorstore::abs(bfloat16_t(3.5f))), 3.5f);
  EXPECT_EQ(static_cast<float>(tensorstore::abs(bfloat16_t(3.5f))), 3.5f);
  EXPECT_EQ(static_cast<float>(tensorstore::abs(bfloat16_t(-3.5f))), 3.5f);
  EXPECT_EQ(static_cast<float>(tensorstore::abs(bfloat16_t(-3.5f))), 3.5f);

  EXPECT_EQ(static_cast<float>(tensorstore::floor(bfloat16_t(3.5f))), 3.0f);
  EXPECT_EQ(static_cast<float>(tensorstore::floor(bfloat16_t(3.5f))), 3.0f);
  EXPECT_EQ(static_cast<float>(tensorstore::floor(bfloat16_t(-3.5f))), -4.0f);
  EXPECT_EQ(static_cast<float>(tensorstore::floor(bfloat16_t(-3.5f))), -4.0f);

  EXPECT_EQ(static_cast<float>(tensorstore::ceil(bfloat16_t(3.5f))), 4.0f);
  EXPECT_EQ(static_cast<float>(tensorstore::ceil(bfloat16_t(3.5f))), 4.0f);
  EXPECT_EQ(static_cast<float>(tensorstore::ceil(bfloat16_t(-3.5f))), -3.0f);
  EXPECT_EQ(static_cast<float>(tensorstore::ceil(bfloat16_t(-3.5f))), -3.0f);

  EXPECT_FLOAT_EQ(static_cast<float>(tensorstore::sqrt(bfloat16_t(0.0f))),
                  0.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(tensorstore::sqrt(bfloat16_t(0.0f))),
                  0.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(tensorstore::sqrt(bfloat16_t(4.0f))),
                  2.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(tensorstore::sqrt(bfloat16_t(4.0f))),
                  2.0f);

  EXPECT_FLOAT_EQ(
      static_cast<float>(tensorstore::pow(bfloat16_t(0.0f), bfloat16_t(1.0f))),
      0.0f);
  EXPECT_FLOAT_EQ(
      static_cast<float>(tensorstore::pow(bfloat16_t(0.0f), bfloat16_t(1.0f))),
      0.0f);
  EXPECT_FLOAT_EQ(
      static_cast<float>(tensorstore::pow(bfloat16_t(2.0f), bfloat16_t(2.0f))),
      4.0f);
  EXPECT_FLOAT_EQ(
      static_cast<float>(tensorstore::pow(bfloat16_t(2.0f), bfloat16_t(2.0f))),
      4.0f);

  EXPECT_EQ(static_cast<float>(tensorstore::exp(bfloat16_t(0.0f))), 1.0f);
  EXPECT_EQ(static_cast<float>(tensorstore::exp(bfloat16_t(0.0f))), 1.0f);
  EXPECT_THAT(static_cast<float>(tensorstore::exp(bfloat16_t(PI))),
              NearFloat(20.f + static_cast<float>(PI)));
  EXPECT_THAT(static_cast<float>(tensorstore::exp(bfloat16_t(PI))),
              NearFloat(20.f + static_cast<float>(PI)));

  EXPECT_EQ(static_cast<float>(tensorstore::expm1(bfloat16_t(0.0f))), 0.0f);
  EXPECT_EQ(static_cast<float>(tensorstore::expm1(bfloat16_t(0.0f))), 0.0f);
  EXPECT_THAT(static_cast<float>(tensorstore::expm1(bfloat16_t(2.0f))),
              NearFloat(6.375f));
  EXPECT_THAT(static_cast<float>(tensorstore::expm1(bfloat16_t(2.0f))),
              NearFloat(6.375f));

  EXPECT_EQ(static_cast<float>(tensorstore::log(bfloat16_t(1.0f))), 0.0f);
  EXPECT_EQ(static_cast<float>(tensorstore::log(bfloat16_t(1.0f))), 0.0f);
  EXPECT_THAT(static_cast<float>(tensorstore::log(bfloat16_t(10.0f))),
              NearFloat(2.296875f));
  EXPECT_THAT(static_cast<float>(tensorstore::log(bfloat16_t(10.0f))),
              NearFloat(2.296875f));

  EXPECT_EQ(static_cast<float>(tensorstore::log1p(bfloat16_t(0.0f))), 0.0f);
  EXPECT_EQ(static_cast<float>(tensorstore::log1p(bfloat16_t(0.0f))), 0.0f);
  EXPECT_THAT(static_cast<float>(tensorstore::log1p(bfloat16_t(10.0f))),
              NearFloat(2.390625f));
  EXPECT_THAT(static_cast<float>(tensorstore::log1p(bfloat16_t(10.0f))),
              NearFloat(2.390625f));
}

TEST(Bfloat16Test, TrigonometricFunctions) {
  EXPECT_THAT(tensorstore::cos(bfloat16_t(0.0f)),
              NearFloat(bfloat16_t(std::cos(0.0f))));
  EXPECT_THAT(tensorstore::cos(bfloat16_t(0.0f)),
              NearFloat(bfloat16_t(std::cos(0.0f))));
  EXPECT_FLOAT_EQ(tensorstore::cos(bfloat16_t(PI)), bfloat16_t(std::cos(PI)));
  EXPECT_NEAR(tensorstore::cos(bfloat16_t(PI / 2)),
              bfloat16_t(std::cos(PI / 2)), 1e-3);
  EXPECT_NEAR(tensorstore::cos(bfloat16_t(3 * PI / 2)),
              bfloat16_t(std::cos(3 * PI / 2)), 1e-2);
  EXPECT_THAT(tensorstore::cos(bfloat16_t(3.5f)),
              NearFloat(bfloat16_t(std::cos(3.5f))));

  EXPECT_FLOAT_EQ(tensorstore::sin(bfloat16_t(0.0f)),
                  bfloat16_t(std::sin(0.0f)));
  EXPECT_FLOAT_EQ(tensorstore::sin(bfloat16_t(0.0f)),
                  bfloat16_t(std::sin(0.0f)));
  EXPECT_NEAR(tensorstore::sin(bfloat16_t(PI)), bfloat16_t(std::sin(PI)), 1e-3);
  EXPECT_THAT(tensorstore::sin(bfloat16_t(PI / 2)),
              NearFloat(bfloat16_t(std::sin(PI / 2))));
  EXPECT_THAT(tensorstore::sin(bfloat16_t(3 * PI / 2)),
              NearFloat(bfloat16_t(std::sin(3 * PI / 2))));
  EXPECT_THAT(tensorstore::sin(bfloat16_t(3.5f)),
              NearFloat(bfloat16_t(std::sin(3.5f))));

  EXPECT_FLOAT_EQ(tensorstore::tan(bfloat16_t(0.0f)),
                  bfloat16_t(std::tan(0.0f)));
  EXPECT_FLOAT_EQ(tensorstore::tan(bfloat16_t(0.0f)),
                  bfloat16_t(std::tan(0.0f)));
  EXPECT_NEAR(tensorstore::tan(bfloat16_t(PI)), bfloat16_t(std::tan(PI)), 1e-3);
  EXPECT_THAT(tensorstore::tan(bfloat16_t(3.5f)),
              NearFloat(bfloat16_t(std::tan(3.5f))));
}

TEST(Bfloat16Test, JsonConversion) {
  EXPECT_THAT(::nlohmann::json(bfloat16_t(1.5)), tensorstore::MatchesJson(1.5));
}

}  // namespace
