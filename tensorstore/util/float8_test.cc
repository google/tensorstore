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

#include "tensorstore/util/float8.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/strings/str_cat.h"
#include <half.hpp>
#include "tensorstore/util/bfloat16.h"

// The implementation below is derived from jax-ml/ml_dtypes:
// https://github.com/jax-ml/ml_dtypes/blob/main/ml_dtypes/tests/float8_test.cc

/* Copyright 2022 The ml_dtypes Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

namespace tensorstore {
namespace {

using std::isfinite;
using std::isinf;  // NOLINT
using std::isnan;  // NOLINT

template <typename Float8_>
class Float8Test : public ::testing::Test {};

// Helper utility for prettier test names.
struct Float8TestParamNames {
  template <typename TypeParam>
  static std::string GetName(int idx) {
    if constexpr (std::is_same_v<TypeParam, Float8e4m3fn>) {
      return "Float8e4m3fn";
    } else if constexpr (std::is_same_v<TypeParam, Float8e4m3b11fnuz>) {
      return "Float8e4m3b11fnuz";
    } else if constexpr (std::is_same_v<TypeParam, Float8e5m2>) {
      return "Float8e5m2";
    } else if constexpr (std::is_same_v<TypeParam, Float8e4m3fnuz>) {
      return "Float8e4m3fnuz";
    } else if constexpr (std::is_same_v<TypeParam, Float8e5m2fnuz>) {
      return "Float8e5m2fnuz";
    }
    return absl::StrCat(idx);
  }
};

using Float8Types =
    ::testing::Types<Float8e4m3fn, Float8e5m2, Float8e4m3b11fnuz,
                     Float8e4m3fnuz, Float8e5m2fnuz>;
TYPED_TEST_SUITE(Float8Test, Float8Types, Float8TestParamNames);

TEST(Float8E4m3fnTest, NumericLimits) {
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e4m3fn>::quiet_NaN()));
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e4m3fn>::signaling_NaN()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e4m3fn>::min()),
            std::exp2(-6));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e4m3fn>::max()), 448);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e4m3fn>::lowest()),
            -448);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e4m3fn>::epsilon()),
            0.125);
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<Float8e4m3fn>::round_error()),
      0.5);
  // No infinity, represent as NaN.
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e4m3fn>::infinity()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e4m3fn>::denorm_min()),
            std::exp2(-9));
  EXPECT_EQ(std::numeric_limits<Float8e4m3fn>::digits, 4);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fn>::digits10, 0);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fn>::max_digits10, 3);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fn>::min_exponent, -5);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fn>::min_exponent10, -1);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fn>::max_exponent, 9);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fn>::max_exponent10, 2);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fn>::is_iec559, false);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fn>::has_infinity, false);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fn>::has_signaling_NaN, false);
}

TEST(Float8E4m3b11fnuzTest, NumericLimits) {
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e4m3b11fnuz>::quiet_NaN()));
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e4m3b11fnuz>::signaling_NaN()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e4m3b11fnuz>::min()),
            std::exp2(-10));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e4m3b11fnuz>::max()),
            30);
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<Float8e4m3b11fnuz>::lowest()),
      -30);
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<Float8e4m3b11fnuz>::epsilon()),
      0.125);
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<Float8e4m3b11fnuz>::round_error()),
      0.5);
  // No infinity, represent as NaN.
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e4m3b11fnuz>::infinity()));
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<Float8e4m3b11fnuz>::denorm_min()),
      std::exp2(-13));
  EXPECT_EQ(std::numeric_limits<Float8e4m3b11fnuz>::digits, 4);
  EXPECT_EQ(std::numeric_limits<Float8e4m3b11fnuz>::digits10, 0);
  EXPECT_EQ(std::numeric_limits<Float8e4m3b11fnuz>::max_digits10, 3);
  EXPECT_EQ(std::numeric_limits<Float8e4m3b11fnuz>::min_exponent, -9);
  EXPECT_EQ(std::numeric_limits<Float8e4m3b11fnuz>::min_exponent10, -3);
  EXPECT_EQ(std::numeric_limits<Float8e4m3b11fnuz>::max_exponent, 5);
  EXPECT_EQ(std::numeric_limits<Float8e4m3b11fnuz>::max_exponent10, 1);
  EXPECT_EQ(std::numeric_limits<Float8e4m3b11fnuz>::is_iec559, false);
  EXPECT_EQ(std::numeric_limits<Float8e4m3b11fnuz>::has_infinity, false);
  EXPECT_EQ(std::numeric_limits<Float8e4m3b11fnuz>::has_signaling_NaN, false);
}

TEST(Float8E4m3fnuzTest, NumericLimits) {
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e4m3fnuz>::quiet_NaN()));
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e4m3fnuz>::signaling_NaN()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e4m3fnuz>::min()),
            std::exp2(-7));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e4m3fnuz>::max()),
            240);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e4m3fnuz>::lowest()),
            -240);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e4m3fnuz>::epsilon()),
            0.125);
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<Float8e4m3fnuz>::round_error()),
      0.5);
  // No infinity, represent as NaN.
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e4m3fnuz>::infinity()));
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<Float8e4m3fnuz>::denorm_min()),
      std::exp2(-10));
  EXPECT_EQ(std::numeric_limits<Float8e4m3fnuz>::digits, 4);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fnuz>::digits10, 0);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fnuz>::max_digits10, 3);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fnuz>::min_exponent, -6);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fnuz>::min_exponent10, -2);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fnuz>::max_exponent, 8);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fnuz>::max_exponent10, 2);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fnuz>::is_iec559, false);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fnuz>::has_infinity, false);
  EXPECT_EQ(std::numeric_limits<Float8e4m3fnuz>::has_signaling_NaN, false);
}

TEST(Float8E5m2Test, NumericLimits) {
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e5m2>::quiet_NaN()));
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e5m2>::signaling_NaN()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e5m2>::min()),
            std::exp2(-14));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e5m2>::max()), 57344);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e5m2>::lowest()),
            -57344);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e5m2>::epsilon()),
            0.25);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e5m2>::round_error()),
            0.5);
  EXPECT_TRUE(isinf(std::numeric_limits<Float8e5m2>::infinity()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e5m2>::denorm_min()),
            std::exp2(-16));
  EXPECT_EQ(std::numeric_limits<Float8e5m2>::digits, 3);
  EXPECT_EQ(std::numeric_limits<Float8e5m2>::digits10, 0);
  EXPECT_EQ(std::numeric_limits<Float8e5m2>::max_digits10, 2);
  EXPECT_EQ(std::numeric_limits<Float8e5m2>::min_exponent, -13);
  EXPECT_EQ(std::numeric_limits<Float8e5m2>::min_exponent10, -4);
  EXPECT_EQ(std::numeric_limits<Float8e5m2>::max_exponent, 16);
  EXPECT_EQ(std::numeric_limits<Float8e5m2>::max_exponent10, 4);
  EXPECT_EQ(std::numeric_limits<Float8e5m2>::is_iec559, true);
  EXPECT_EQ(std::numeric_limits<Float8e5m2>::has_infinity, true);
  EXPECT_EQ(std::numeric_limits<Float8e5m2>::has_signaling_NaN, true);
}

TEST(Float8E5m2fnuzTest, NumericLimits) {
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e5m2fnuz>::quiet_NaN()));
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e5m2fnuz>::signaling_NaN()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e5m2fnuz>::min()),
            std::exp2(-15));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e5m2fnuz>::max()),
            57344);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e5m2fnuz>::lowest()),
            -57344);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e5m2fnuz>::epsilon()),
            0.25);
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<Float8e5m2fnuz>::round_error()),
      0.5);
  // No infinity, represented as NaN.
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e5m2fnuz>::infinity()));
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<Float8e5m2fnuz>::denorm_min()),
      std::exp2(-17));
  EXPECT_EQ(std::numeric_limits<Float8e5m2fnuz>::digits, 3);
  EXPECT_EQ(std::numeric_limits<Float8e5m2fnuz>::digits10, 0);
  EXPECT_EQ(std::numeric_limits<Float8e5m2fnuz>::max_digits10, 2);
  EXPECT_EQ(std::numeric_limits<Float8e5m2fnuz>::min_exponent, -14);
  EXPECT_EQ(std::numeric_limits<Float8e5m2fnuz>::min_exponent10, -4);
  EXPECT_EQ(std::numeric_limits<Float8e5m2fnuz>::max_exponent, 16);
  EXPECT_EQ(std::numeric_limits<Float8e5m2fnuz>::max_exponent10, 4);
  EXPECT_EQ(std::numeric_limits<Float8e5m2fnuz>::is_iec559, false);
  EXPECT_EQ(std::numeric_limits<Float8e5m2fnuz>::has_infinity, false);
  EXPECT_EQ(std::numeric_limits<Float8e5m2fnuz>::has_signaling_NaN, false);
}

TYPED_TEST(Float8Test, FromRep) {
  using Float8 = TypeParam;
  Float8 x = Float8::FromRep(0x4F);
  EXPECT_EQ(x.rep(), 0x4F);
}

TYPED_TEST(Float8Test, Negate) {
  using Float8 = TypeParam;
  Float8 x = -Float8::FromRep(0x4F);
  EXPECT_EQ(x.rep(), 0x80 | 0x4F);

  Float8 nan = -std::numeric_limits<Float8>::quiet_NaN();
  EXPECT_TRUE(isnan(nan));
}

TYPED_TEST(Float8Test, BitCasts) {
  using Float8 = TypeParam;
  Float8 x = Float8::FromRep(0x47);
  EXPECT_EQ(absl::bit_cast<uint8_t>(x), 0x47);
  EXPECT_EQ(absl::bit_cast<Float8>(x.rep()).rep(), 0x47);
}

TYPED_TEST(Float8Test, UpCasts) {
  using Float8 = TypeParam;

  // Loop through each float8 value.
  for (int i = 0x00; i <= 0xFF; ++i) {
    // Cast up to each other floating-point type, and verify they are the same.
    Float8 f8 = Float8::FromRep(i);
    double f64 = static_cast<double>(f8);
    float f32 = static_cast<float>(f8);
    tensorstore::BFloat16 bf16 = static_cast<tensorstore::BFloat16>(f8);
    ::half_float::half f16 = static_cast<::half_float::half>(f8);

    if (isnan(f8)) {
      EXPECT_TRUE(std::isnan(f64));
      EXPECT_TRUE(std::isnan(f32));
      EXPECT_TRUE(tensorstore::isnan(bf16));
      EXPECT_TRUE(::half_float::isnan(f16));
    } else {
      EXPECT_EQ(f64, f32);
      EXPECT_EQ(f32, bf16);
      EXPECT_EQ(bf16, f16);
    }
  }
}

TYPED_TEST(Float8Test, DownCasts) {
  using Float8 = TypeParam;
  for (int i = 0x00; i <= 0xFF; ++i) {
    float x = static_cast<float>(Float8::FromRep(i));

    Float8 f64 = static_cast<Float8>(static_cast<double>(x));
    Float8 f32 = static_cast<Float8>(static_cast<float>(x));
    Float8 bf16 = static_cast<Float8>(static_cast<tensorstore::BFloat16>(x));
    Float8 f16 = static_cast<Float8>(static_cast<::half_float::half>(x));

    if (std::isnan(x)) {
      EXPECT_TRUE(isnan(f64));
      EXPECT_TRUE(isnan(f32));
      EXPECT_TRUE(isnan(bf16));
      EXPECT_TRUE(isnan(f16));
    } else {
      EXPECT_EQ(f64.rep(), i) << i;
      EXPECT_EQ(f32.rep(), i) << i;
      EXPECT_EQ(bf16.rep(), i) << i;
      EXPECT_EQ(f16.rep(), i) << i;
    }
  }
}

TYPED_TEST(Float8Test, ConvertFromWithSaturation) {
  using Float8 = TypeParam;

  // Saturation above max value.
  Float8 upper =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
          static_cast<float>(std::numeric_limits<Float8>::max()) * 2);
  EXPECT_EQ(upper, std::numeric_limits<Float8>::max());

  Float8 lower =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
          static_cast<float>(std::numeric_limits<Float8>::lowest()) * 2);
  EXPECT_EQ(lower, std::numeric_limits<Float8>::lowest());

  // Special values remain with saturation.
  Float8 nan =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/true>(
          std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(isnan(nan));
  Float8 inf =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/true>(
          std::numeric_limits<float>::infinity());
  // E4M3 doesn't have inf, so check inf -> NaN conversion.
  EXPECT_TRUE(std::numeric_limits<Float8>::has_infinity ? isinf(inf)
                                                        : isnan(inf));
  Float8 ninf =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/true>(
          -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::numeric_limits<Float8>::has_infinity ? isinf(ninf)
                                                        : isnan(ninf));
}

TYPED_TEST(Float8Test, ConvertFromWithTruncation) {
  using Float8 = TypeParam;

  // Truncation and rounding of a number ever-so-slightly less than 2.
  float less_than_two = absl::bit_cast<float>(0x3FFFFFFF);
  Float8 truncated =
      Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
          less_than_two);
  EXPECT_LT(static_cast<float>(truncated), 2);

  Float8 rounded =
      Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
          less_than_two);
  EXPECT_EQ(static_cast<float>(rounded), 2);

  double kLarge = 0x1.c001p+16;
  EXPECT_EQ(
      (Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
           kLarge)
           .rep()),
      std::numeric_limits<Float8>::infinity().rep());
  EXPECT_EQ(
      (Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
           kLarge)
           .rep()),
      std::numeric_limits<Float8>::infinity().rep());

  // Truncation and rounding of a subnormal.
  for (int i = 0x01; i < 0x04; ++i) {
    float less_than_subnorm =
        std::nexttoward(static_cast<float>(Float8::FromRep(i)), 0);

    Float8 truncated_subnorm =
        Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
            less_than_subnorm);
    EXPECT_EQ(truncated_subnorm.rep(), i - 1);

    Float8 rounded_subnorm =
        Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
            less_than_subnorm);
    EXPECT_EQ(rounded_subnorm.rep(), i);
  }
}

TYPED_TEST(Float8Test, ConvertTo) {
  using Float8 = TypeParam;

  // Converting to higher precision types doesn't result in either
  // truncation or saturation, so let's just ensure they all provide the
  // same results.
  for (int i = 0x00; i <= 0xFF; ++i) {
    // Cast up to each other floating-point type, and verify they are the same.
    Float8 f8 = Float8::FromRep(i);
    float f32 = static_cast<float>(f8);
    if (isnan(f8)) {
      EXPECT_TRUE(isnan(Float8::template ConvertTo<float, /*kSaturate=*/false,
                                                   /*kTruncate=*/false>(f8)));
      EXPECT_TRUE(isnan(Float8::template ConvertTo<float, /*kSaturate=*/false,
                                                   /*kTruncate=*/true>(f8)));
      EXPECT_TRUE(isnan(Float8::template ConvertTo<float, /*kSaturate=*/true,
                                                   /*kTruncate=*/false>(f8)));
      EXPECT_TRUE(isnan(Float8::template ConvertTo<float, /*kSaturate=*/true,
                                                   /*kTruncate=*/true>(f8)));
    } else {
      EXPECT_EQ(f32, (Float8::template ConvertTo<float, /*kSaturate=*/false,
                                                 /*kTruncate=*/false>(f8)));
      EXPECT_EQ(f32, (Float8::template ConvertTo<float, /*kSaturate=*/false,
                                                 /*kTruncate=*/true>(f8)));
      EXPECT_EQ(f32, (Float8::template ConvertTo<float, /*kSaturate=*/true,
                                                 /*kTruncate=*/false>(f8)));
      EXPECT_EQ(f32, (Float8::template ConvertTo<float, /*kSaturate=*/true,
                                                 /*kTruncate=*/true>(f8)));
    }
  }
}

TEST(Float8Test, Float8E5m2_To_Float8E4m3) {
  // Saturation.
  Float8e5m2 max = std::numeric_limits<Float8e5m2>::max();
  Float8e4m3fn saturated = Float8e4m3fn::ConvertFrom</*kSaturate=*/true>(max);
  EXPECT_EQ(saturated, std::numeric_limits<Float8e4m3fn>::max());
  saturated = Float8e5m2::ConvertTo<Float8e4m3fn, /*kSaturate=*/true>(max);
  EXPECT_EQ(saturated, std::numeric_limits<Float8e4m3fn>::max());

  // Truncation - only occurs for e4m3 subnormals.
  Float8e5m2 less_than_subnorm = Float8e5m2::FromRep(0x1F);  // 2^-7 - 2^-10.
  Float8e4m3fn rounded_subnorm =
      Float8e4m3fn::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
          less_than_subnorm);
  EXPECT_EQ(rounded_subnorm.rep(), 0x04);
  Float8e4m3fn truncated_subnorm =
      Float8e4m3fn::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
          less_than_subnorm);
  EXPECT_EQ(truncated_subnorm.rep(), 0x03);
}

TEST(Float8Test, Half_To_Float8E4m3) {
  ::half_float::half big_half(0x1.dfcp+8f);
  Float8e4m3fn big_e4m3 =
      Float8e4m3fn::ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
          big_half);
  EXPECT_EQ(big_e4m3.rep(), std::numeric_limits<Float8e4m3fn>::max().rep());
}

TEST(Float8Test, Float8E5m2_To_Float8E4m3b11fnuz) {
  // Saturation.
  Float8e5m2 max = std::numeric_limits<Float8e5m2>::max();
  Float8e4m3b11fnuz saturated =
      Float8e4m3b11fnuz::ConvertFrom</*kSaturate=*/true>(max);
  EXPECT_EQ(saturated, std::numeric_limits<Float8e4m3b11fnuz>::max());
  saturated = Float8e5m2::ConvertTo<Float8e4m3b11fnuz, /*kSaturate=*/true>(max);
  EXPECT_EQ(saturated, std::numeric_limits<Float8e4m3b11fnuz>::max());

  // Truncation - only occurs for e4m3 subnormals.
  Float8e5m2 less_than_subnorm = Float8e5m2::FromRep(0x0F);  // 2^-11 - 2^-14.
  Float8e4m3b11fnuz rounded_subnorm =
      Float8e4m3b11fnuz::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
          less_than_subnorm);
  EXPECT_EQ(rounded_subnorm.rep(), 0x04);
  Float8e4m3b11fnuz truncated_subnorm =
      Float8e4m3b11fnuz::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
          less_than_subnorm);
  EXPECT_EQ(truncated_subnorm.rep(), 0x03);

  // Saturation.
  for (uint8_t i = 0; i < std::numeric_limits<Float8e5m2>::infinity().rep();
       ++i) {
    Float8e5m2 big_e5m2 = absl::bit_cast<Float8e5m2>(i);
    EXPECT_TRUE(isfinite(big_e5m2)) << uint16_t{i};
    float big_float = static_cast<float>(big_e5m2);
    auto big_e4m3 =
        Float8e4m3b11fnuz::ConvertFrom</*kSaturate=*/true,
                                       /*kTruncate=*/false>(big_float);
    if (i > 0x4f) {
      EXPECT_EQ(big_e4m3.rep(),
                std::numeric_limits<Float8e4m3b11fnuz>::max().rep())
          << uint16_t{i};
    }
    EXPECT_EQ((Float8e4m3b11fnuz::ConvertFrom</*kSaturate=*/true,
                                              /*kTruncate=*/false>(big_e5m2)
                   .rep()),
              big_e4m3.rep())
        << i;
    EXPECT_EQ((Float8e4m3b11fnuz::ConvertFrom</*kSaturate=*/true,
                                              /*kTruncate=*/false>(-big_e5m2)
                   .rep()),
              (-big_e4m3).rep())
        << i;
  }
}

TEST(Float8Test, Float8E4m3b11fnuz_To_Float8E4m3) {
  // Saturation.
  Float8e4m3b11fnuz max = std::numeric_limits<Float8e4m3b11fnuz>::max();
  Float8e4m3fn saturated = Float8e4m3fn::ConvertFrom</*kSaturate=*/true>(max);
  EXPECT_EQ(static_cast<float>(saturated),
            static_cast<float>(std::numeric_limits<Float8e4m3b11fnuz>::max()));
  saturated =
      Float8e4m3b11fnuz::ConvertTo<Float8e4m3fn, /*kSaturate=*/true>(max);
  EXPECT_EQ(static_cast<float>(saturated),
            static_cast<float>(std::numeric_limits<Float8e4m3b11fnuz>::max()));

  // Truncation - only occurs for e4m3 subnormals.
  Float8e4m3b11fnuz less_than_subnorm =
      Float8e4m3b11fnuz::FromRep(0b0011'110);  // 2^-7 - 2^-10.
  Float8e4m3fn rounded_subnorm =
      Float8e4m3fn::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
          less_than_subnorm);
  EXPECT_EQ(rounded_subnorm.rep(), 0x04);
  Float8e4m3fn truncated_subnorm =
      Float8e4m3fn::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
          less_than_subnorm);
  EXPECT_EQ(truncated_subnorm.rep(), 0x03);

  // Saturation.
  for (uint8_t i = 0;
       i < std::numeric_limits<Float8e4m3b11fnuz>::infinity().rep(); ++i) {
    Float8e4m3b11fnuz big_e4m3b11fnuz = absl::bit_cast<Float8e4m3b11fnuz>(i);
    EXPECT_TRUE(isfinite(big_e4m3b11fnuz)) << uint16_t{i};
    float big_float = static_cast<float>(big_e4m3b11fnuz);
    auto big_e4m3 =
        Float8e4m3fn::ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
            big_float);
    EXPECT_EQ(
        (Float8e4m3fn::ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
             big_e4m3b11fnuz)
             .rep()),
        big_e4m3.rep())
        << i;
    EXPECT_EQ(
        (Float8e4m3fn::ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
             -big_e4m3b11fnuz)
             .rep()),
        (big_float > 0.0f ? -big_e4m3 : big_e4m3).rep())
        << i;
  }
}

TEST(Float8Test, Float8E4m3_To_Float8E5m2) {
  // Truncation and rounding of a number ever-so-slightly less than 2.
  Float8e4m3fn less_than_two = Float8e4m3fn::FromRep(0x3F);
  Float8e5m2 truncated =
      Float8e5m2::template ConvertFrom</*kSaturate=*/false,
                                       /*kTruncate=*/true>(less_than_two);
  EXPECT_LT(static_cast<float>(truncated), 2);

  Float8e5m2 rounded =
      Float8e5m2::template ConvertFrom</*kSaturate=*/false,
                                       /*kTruncate=*/false>(less_than_two);
  EXPECT_EQ(static_cast<float>(rounded), 2);
}

TEST(Float8Test, Half_To_Float8E5m2) {
  // Special values, NaN.
  ::half_float::half inf =
      absl::bit_cast<::half_float::half>(static_cast<uint16_t>(0x7C00));
  EXPECT_EQ(static_cast<Float8e5m2>(inf).rep(), 0x7C);
  ::half_float::half ninf =
      absl::bit_cast<::half_float::half>(static_cast<uint16_t>(0xFC00));
  EXPECT_EQ(static_cast<Float8e5m2>(ninf).rep(), 0xFC);

  ::half_float::half nan =
      absl::bit_cast<::half_float::half>(static_cast<uint16_t>(0x7C01));
  EXPECT_EQ(static_cast<Float8e5m2>(nan).rep(), 0x7E);
  ::half_float::half nnan =
      absl::bit_cast<::half_float::half>(static_cast<uint16_t>(0xFC01));
  EXPECT_EQ(static_cast<Float8e5m2>(nnan).rep(), 0xFE);

  // Rounding vs truncation.
  ::half_float::half less_than_two =
      absl::bit_cast<::half_float::half>(static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ((Float8e5m2::ConvertFrom</*kSaturate=*/false,
                                     /*kTruncate=*/false>(less_than_two)
                 .rep()),
            0x40);
  EXPECT_EQ((Float8e5m2::ConvertFrom</*kSaturate=*/false,
                                     /*kTruncate=*/true>(less_than_two)
                 .rep()),
            0x3F);
  EXPECT_EQ((Float8e5m2::ConvertFrom</*kSaturate=*/false,
                                     /*kTruncate=*/false>(-less_than_two)
                 .rep()),
            0xC0);
  EXPECT_EQ((Float8e5m2::ConvertFrom</*kSaturate=*/false,
                                     /*kTruncate=*/true>(-less_than_two)
                 .rep()),
            0xBF);

  // Saturation.
  for (uint16_t i = static_cast<uint16_t>(absl::bit_cast<uint8_t>(
                        std::numeric_limits<Float8e5m2>::max()))
                    << 8;
       i < absl::bit_cast<uint16_t>(
               std::numeric_limits<::half_float::half>::infinity());
       ++i) {
    ::half_float::half big_half = absl::bit_cast<::half_float::half>(i);
    float big_float = static_cast<float>(big_half);
    EXPECT_EQ((Float8e5m2::ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
                   big_half)
                   .rep()),
              (Float8e5m2::ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
                   big_float)
                   .rep()))
        << i;
    EXPECT_EQ((Float8e5m2::ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
                   -big_half)
                   .rep()),
              (Float8e5m2::ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
                   -big_float)
                   .rep()))
        << i;
  }
}

using ::testing::Eq;
using ::testing::IsTrue;
MATCHER_P(EqOrIsNan, other, "") {
  if (isnan(other)) {
    return ExplainMatchResult(IsTrue(), isnan(arg), result_listener);
  }
  return ExplainMatchResult(Eq(other), arg, result_listener);
}

TYPED_TEST(Float8Test, CallTheOperator) {
  using Float8 = TypeParam;

  for (int i = 0x00; i <= 0xFF; ++i) {
    Float8 a = Float8::FromRep(i);
    for (int j = 0x00; j <= 0xFF; ++j) {
      Float8 b = Float8::FromRep(j);

      EXPECT_THAT(a + b, EqOrIsNan<Float8>(Float8{float{a} + float{b}}));
      EXPECT_THAT(a - b, EqOrIsNan<Float8>(Float8{float{a} - float{b}}));
      EXPECT_THAT(a * b, EqOrIsNan<Float8>(Float8{float{a} * float{b}}));
      EXPECT_THAT(a / b, EqOrIsNan<Float8>(Float8{float{a} / float{b}}));

      Float8 c;
      EXPECT_THAT((c = a, c += b),
                  EqOrIsNan<Float8>(Float8{float{a} + float{b}}));
      EXPECT_THAT((c = a, c -= b),
                  EqOrIsNan<Float8>(Float8{float{a} - float{b}}));
      EXPECT_THAT((c = a, c *= b),
                  EqOrIsNan<Float8>(Float8{float{a} * float{b}}));
      EXPECT_THAT((c = a, c /= b),
                  EqOrIsNan<Float8>(Float8{float{a} / float{b}}));

      EXPECT_EQ(a == b, float{a} == float{b}) << float{a} << " vs " << float{b};
      EXPECT_EQ(a != b, float{a} != float{b});
      EXPECT_EQ(a < b, float{a} < float{b});
      EXPECT_EQ(a <= b, float{a} <= float{b});
      EXPECT_EQ(a > b, float{a} > float{b});
      EXPECT_EQ(a >= b, float{a} >= float{b});
    }
  }
}

TYPED_TEST(Float8Test, CallTheConstOperator) {
  using Float8 = TypeParam;

  for (int i = 0x00; i <= 0xFF; ++i) {
    const Float8 a = Float8::FromRep(i);
    for (int j = 0x00; j <= 0xFF; ++j) {
      const Float8 b = Float8::FromRep(j);

      EXPECT_THAT(a + b, EqOrIsNan<Float8>(Float8{float{a} + float{b}}));
      EXPECT_THAT(a - b, EqOrIsNan<Float8>(Float8{float{a} - float{b}}));
      EXPECT_THAT(a * b, EqOrIsNan<Float8>(Float8{float{a} * float{b}}));
      EXPECT_THAT(a / b, EqOrIsNan<Float8>(Float8{float{a} / float{b}}));

      Float8 c;
      EXPECT_THAT((c = a, c += b),
                  EqOrIsNan<Float8>(Float8{float{a} + float{b}}));
      EXPECT_THAT((c = a, c -= b),
                  EqOrIsNan<Float8>(Float8{float{a} - float{b}}));
      EXPECT_THAT((c = a, c *= b),
                  EqOrIsNan<Float8>(Float8{float{a} * float{b}}));
      EXPECT_THAT((c = a, c /= b),
                  EqOrIsNan<Float8>(Float8{float{a} / float{b}}));

      EXPECT_EQ(a == b, float{a} == float{b}) << float{a} << " vs " << float{b};
      EXPECT_EQ(a != b, float{a} != float{b});
      EXPECT_EQ(a < b, float{a} < float{b}) << float{a} << " vs " << float{b};
      EXPECT_EQ(a <= b, float{a} <= float{b});
      EXPECT_EQ(a > b, float{a} > float{b}) << float{a} << " vs " << float{b};
      EXPECT_EQ(a >= b, float{a} >= float{b});
    }
  }
}

TEST(Float855m2Test, SmallCastToDenormal) {
  // Special edge-case where rounding to a normalized value would
  // normally round down, but rounding to a subnormal rounds up.
  float x = std::ldexp(1.3125, -15);
  Float8e5m2 y = static_cast<Float8e5m2>(x);
  float z = static_cast<float>(y);
  EXPECT_EQ(z, std::ldexp(1.5, -15));
}

// Helper utility for prettier test names.
struct Float8CastTestParamNames {
  template <typename TypeParam>
  static std::string GetName(int idx) {
    using first_type = typename TypeParam::first_type;
    using second_type = typename TypeParam::second_type;
    return absl::StrCat(::testing::internal::GetTypeName<first_type>(), "_",
                        ::testing::internal::GetTypeName<second_type>());
  }
};

#define GEN_LONG_DOUBLE_PAIR(Type) std::pair<Type, long double>,

#define GEN_DEST_TYPES(Type)                                               \
  GEN_LONG_DOUBLE_PAIR(Type)                                               \
  std::pair<Type, double>, std::pair<Type, float>,                         \
      std::pair<Type, tensorstore::BFloat16>,                              \
      std::pair<Type, ::half_float::half>, std::pair<Type, Float8e4m3fn>,  \
      std::pair<Type, Float8e4m3b11fnuz>, std::pair<Type, Float8e4m3fnuz>, \
      std::pair<Type, Float8e5m2fnuz>, std::pair<Type, Float8e5m2>,        \
      std::pair<Type, bool>, std::pair<Type, int32_t>,                     \
      std::pair<Type, int64_t>

#define GEN_TYPE_PAIRS()                                           \
  GEN_DEST_TYPES(Float8e4m3fn), GEN_DEST_TYPES(Float8e4m3b11fnuz), \
      GEN_DEST_TYPES(Float8e5m2), GEN_DEST_TYPES(Float8e4m3fnuz),  \
      GEN_DEST_TYPES(Float8e5m2fnuz)

using Float8CastTypePairs = ::testing::Types<GEN_TYPE_PAIRS()>;

template <typename CastPair>
class Float8CastTest : public ::testing::Test {};
TYPED_TEST_SUITE(Float8CastTest, Float8CastTypePairs, Float8CastTestParamNames);

TYPED_TEST(Float8CastTest, CastThroughFloat) {
  using Float8 = typename TypeParam::first_type;
  using DestType = typename TypeParam::second_type;

  for (int i = 0x00; i <= 0xFF; ++i) {
    Float8 f8 = Float8::FromRep(i);

    if constexpr (std::numeric_limits<DestType>::is_integer &&
                  !std::is_same_v<DestType, bool>) {
      if (!isfinite(f8)) {
        continue;
      }
    }

    DestType dest = static_cast<DestType>(f8);
    DestType expected = static_cast<DestType>(static_cast<float>(f8));

    if constexpr (std::numeric_limits<DestType>::is_integer) {
      EXPECT_EQ(dest, expected);
    } else {
      EXPECT_THAT(dest, EqOrIsNan<DestType>(expected));
    }
  }
}

}  // namespace
}  // namespace tensorstore
