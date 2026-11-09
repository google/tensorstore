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

#include <stdint.h>

#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

// Helper utility for prettier test names.
template <typename T>
std::string TypeToName() {
  if constexpr (std::is_same_v<T, Float8e3m4>) return "Float8e3m4";
  if constexpr (std::is_same_v<T, Float8e4m3fn>) return "Float8e4m3fn";
  if constexpr (std::is_same_v<T, Float8e4m3b11fnuz>)
    return "Float8e4m3b11fnuz";
  if constexpr (std::is_same_v<T, Float8e5m2>) return "Float8e5m2";
  if constexpr (std::is_same_v<T, Float8e4m3fnuz>) return "Float8e4m3fnuz";
  if constexpr (std::is_same_v<T, Float8e5m2fnuz>) return "Float8e5m2fnuz";
  if constexpr (std::is_same_v<T, Float8e8m0fnu>) return "Float8e8m0fnu";
  if constexpr (std::is_same_v<T, long double>) return "long_double";
  if constexpr (std::is_same_v<T, double>) return "double";
  if constexpr (std::is_same_v<T, float>) return "float";
  if constexpr (std::is_same_v<T, tensorstore::BFloat16>) return "BFloat16";
  if constexpr (std::is_same_v<T, ::half_float::half>) return "float16";
  if constexpr (std::is_same_v<T, bool>) return "bool";
  if constexpr (std::is_same_v<T, int32_t>) return "int32_t";
  if constexpr (std::is_same_v<T, int64_t>) return "int64_t";
  return "unknown";
}

struct Float8TestParamNames {
  template <typename TypeParam>
  static std::string GetName(int idx) {
    return TypeToName<TypeParam>();
  }
};

TEST(Float8E3m4Test, NumericLimits) {
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e3m4>::quiet_NaN()));
  EXPECT_TRUE(isnan(std::numeric_limits<Float8e3m4>::signaling_NaN()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e3m4>::min()), 0.25);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e3m4>::max()), 15.5);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e3m4>::lowest()),
            -15.5);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e3m4>::epsilon()),
            0.0625);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e3m4>::round_error()),
            0.5);
  EXPECT_TRUE(isinf(std::numeric_limits<Float8e3m4>::infinity()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float8e3m4>::denorm_min()),
            std::exp2(-6));
  EXPECT_EQ(std::numeric_limits<Float8e3m4>::digits, 5);
  EXPECT_EQ(std::numeric_limits<Float8e3m4>::digits10, 1);
  EXPECT_EQ(std::numeric_limits<Float8e3m4>::max_digits10, 3);
  EXPECT_EQ(std::numeric_limits<Float8e3m4>::min_exponent, -1);
  EXPECT_EQ(std::numeric_limits<Float8e3m4>::min_exponent10, 0);
  EXPECT_EQ(std::numeric_limits<Float8e3m4>::max_exponent, 4);
  EXPECT_EQ(std::numeric_limits<Float8e3m4>::max_exponent10, 1);
  EXPECT_EQ(std::numeric_limits<Float8e3m4>::is_iec559, true);
  EXPECT_EQ(std::numeric_limits<Float8e3m4>::has_infinity, true);
  EXPECT_EQ(std::numeric_limits<Float8e3m4>::has_quiet_NaN, true);
  EXPECT_EQ(std::numeric_limits<Float8e3m4>::has_signaling_NaN, true);
}

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

TEST(Float8E8m0fnuTest, NumericLimits) {
  using limits = std::numeric_limits<Float8e8m0fnu>;
  EXPECT_FALSE(limits::is_signed);
  EXPECT_TRUE(isnan(limits::quiet_NaN()));
  EXPECT_TRUE(isnan(limits::signaling_NaN()));
  EXPECT_TRUE(isnan(limits::infinity()));  // No infinity.
  EXPECT_EQ(static_cast<float>(limits::min()), 0x1p-127);
  EXPECT_EQ(static_cast<float>(limits::max()), 0x1p+127);
  EXPECT_EQ(static_cast<float>(limits::lowest()), 0x1p-127);
  EXPECT_EQ(static_cast<float>(limits::epsilon()), 1.0);
  EXPECT_EQ(static_cast<float>(limits::round_error()), 0.5);
  EXPECT_EQ(limits::digits, 1);
  EXPECT_EQ(limits::digits10, 0);
  EXPECT_EQ(limits::max_digits10, 2);
  EXPECT_EQ(limits::min_exponent, -126);
  EXPECT_EQ(limits::min_exponent10, -38);
  EXPECT_EQ(limits::max_exponent, 128);
  EXPECT_EQ(limits::max_exponent10, 38);
  EXPECT_EQ(limits::is_iec559, false);
  EXPECT_EQ(limits::has_infinity, false);
  EXPECT_EQ(limits::has_quiet_NaN, true);
  EXPECT_EQ(limits::has_signaling_NaN, false);
}

TEST(Float8E8m0fnuTest, NegativeInput) {
  using Float8 = Float8e8m0fnu;
  std::vector<float> inputs = {
      -std::fabs(std::numeric_limits<float>::quiet_NaN()),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::lowest(),
      -1.0f,
      -0.5f,
      -std::numeric_limits<float>::min(),
      -std::numeric_limits<float>::denorm_min(),
      -0.0f,
  };
  for (float input : inputs) {
    Float8 f8 = Float8::template ConvertFrom</*kSaturate=*/false,
                                             /*kTruncate=*/false>(input);
    EXPECT_TRUE(isnan(f8));
  }
  for (float input : inputs) {
    Float8 f8 = Float8::template ConvertFrom</*kSaturate=*/true,
                                             /*kTruncate=*/false>(input);
    if (std::isfinite(input)) {
      EXPECT_EQ(f8, std::numeric_limits<Float8>::lowest())
          << "input: " << input << " f8: " << f8;
    } else {
      EXPECT_TRUE(isnan(f8));
    }
  }
}

TEST(Float8E8m0fnuTest, ConvertFromWideIntegers) {
  // 2^24 + 2^23 - 1 rounds to 2^24 + 2^23 as a float, which is halfway between
  // 2^24 and 2^25.
  EXPECT_EQ(static_cast<float>(Float8e8m0fnu(int32_t{0x17F'FFFF})), 0x1p24);
  EXPECT_EQ(static_cast<float>(Float8e8m0fnu(int32_t{0x180'0001})), 0x1p25);
  EXPECT_EQ(static_cast<float>(Float8e8m0fnu(uint64_t{0xBFFF'FFFF'FFFF'FFFF})),
            0x1p63);
  EXPECT_EQ(
      static_cast<float>(Float8e8m0fnu(std::numeric_limits<uint64_t>::max())),
      0x1p64);
  // Truncation must not be affected by rounding up when narrowing to float.
  EXPECT_EQ(static_cast<float>(Float8e8m0fnu::ConvertFrom</*kSaturate=*/false,
                                                          /*kTruncate=*/true>(
                std::numeric_limits<int64_t>::max())),
            0x1p62);
  EXPECT_EQ(static_cast<float>(Float8e8m0fnu::ConvertFrom</*kSaturate=*/false,
                                                          /*kTruncate=*/true>(
                std::numeric_limits<uint64_t>::max())),
            0x1p63);
}

TEST(Float8E8m0fnuTest, ConvertFromLowestBinade) {
  EXPECT_EQ(static_cast<BFloat16>(Float8e8m0fnu::FromRep(0x00)), 0x1p-127f);
  EXPECT_EQ(static_cast<float>(Float8e8m0fnu::FromRep(0x00)), 0x1p-127f);
  EXPECT_EQ(static_cast<double>(Float8e8m0fnu::FromRep(0x00)), 0x1p-127);

  // 2^-127 is the smallest value, not a subnormal: inputs below the midpoint of
  // [2^-127, 2^-126] round down, and positive inputs below 2^-127 saturate to
  // 2^-127.
  const std::vector<std::pair<float, uint8_t>> cases = {
      {0x1p-149f, 0x00},        {0x1.fffff8p-128f, 0x00}, {0x1p-127f, 0x00},
      {0x1.000004p-127f, 0x00}, {0x1.1p-127f, 0x00},      {0x1.4p-127f, 0x00},
      {0x1.7ffffcp-127f, 0x00}, {0x1.8p-127f, 0x01},      {0x1.4p-126f, 0x01},
      {0x1.8p-126f, 0x02},
  };
  for (const auto& [input, rep] : cases) {
    EXPECT_EQ(Float8e8m0fnu(input).rep(), rep) << "input: " << input;
    EXPECT_EQ((Float8e8m0fnu::ConvertFrom</*kSaturate=*/true>(input)).rep(),
              rep)
        << "input: " << input;
    EXPECT_EQ(Float8e8m0fnu(static_cast<double>(input)).rep(), rep)
        << "input: " << input;
    const tensorstore::BFloat16 bf16_input(input);
    if (static_cast<float>(bf16_input) == input) {
      EXPECT_EQ(Float8e8m0fnu(bf16_input).rep(), rep) << "input: " << input;
    }
  }
  EXPECT_EQ(Float8e8m0fnu(0x1p-1022).rep(), 0x00);
  EXPECT_EQ(Float8e8m0fnu(0x1.7ffffffffffffp-127).rep(), 0x00);
  EXPECT_EQ(Float8e8m0fnu(0x1.8p-127).rep(), 0x01);
  EXPECT_EQ((Float8e8m0fnu::ConvertFrom</*kSaturate=*/false,
                                        /*kTruncate=*/true>(0x1.cp-127f))
                .rep(),
            0x00);
  EXPECT_EQ((Float8e8m0fnu::ConvertFrom</*kSaturate=*/false,
                                        /*kTruncate=*/true>(0x1.cp-126f))
                .rep(),
            0x01);
}

template <typename Float8_>
class Float8Test : public ::testing::Test {};

using Float8Types = ::testing::Types<Float8e3m4, Float8e4m3fn, Float8e4m3fnuz,
                                     Float8e4m3b11fnuz, Float8e5m2,
                                     Float8e5m2fnuz, Float8e8m0fnu>;

TYPED_TEST_SUITE(Float8Test, Float8Types, Float8TestParamNames);

TYPED_TEST(Float8Test, DefaultConstruction) {
  using Float = TypeParam;
  const Float zero = Float::FromRep(0);

  // sd is static, so it must be zero initialized.
  static Float sd;
  EXPECT_EQ(sd, zero);

  // z is zero initialized.
  Float z{};
  EXPECT_EQ(z, zero);

  // v is value initialized to zero.
  Float v = Float();
  EXPECT_EQ(v, zero);
}

TYPED_TEST(Float8Test, FromRep) {
  using Float8 = TypeParam;

  for (int i = 0x00; i <= 0xFF; ++i) {
    // Cast up to each other floating-point type, and verify they are the same.
    Float8 f8 = Float8::FromRep(static_cast<uint8_t>(i));
    EXPECT_EQ(i, f8.rep());

    if (isnan(f8) || isinf(f8)) continue;
    EXPECT_EQ(static_cast<double>(static_cast<float>(f8)),
              static_cast<double>(f8))
        << i;
    EXPECT_EQ(static_cast<float>(static_cast<double>(f8)),
              static_cast<float>(f8))
        << i;
  }
}

TYPED_TEST(Float8Test, Negate) {
  using Float8 = TypeParam;
  if (!std::numeric_limits<Float8>::is_signed) {
    GTEST_SKIP() << "Type doesn't support negative numbers";
  }

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
    Float8 f8 = Float8::FromRep(static_cast<uint8_t>(i));
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
      if (!std::is_same_v<Float8, Float8e8m0fnu>) {
        EXPECT_EQ(bf16, f16);
      }
    }
  }
}

TYPED_TEST(Float8Test, DownCasts) {
  using Float8 = TypeParam;
  for (int i = 0x00; i <= 0xFF; ++i) {
    float x = static_cast<float>(Float8::FromRep(static_cast<uint8_t>(i)));

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
      if (!std::is_same_v<Float8, Float8e8m0fnu>) {
        EXPECT_EQ(f16.rep(), i) << i;
      }
    }
  }
}

TYPED_TEST(Float8Test, ConvertFromWithSaturation) {
  using Float8 = TypeParam;

  // Saturation above max value.
  Float8 upper =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
          static_cast<double>(std::numeric_limits<Float8>::max()) * 2);
  EXPECT_EQ(upper, std::numeric_limits<Float8>::max());

  if (std::numeric_limits<Float8>::is_signed) {
    Float8 lower =
        Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
            static_cast<double>(std::numeric_limits<Float8>::lowest()) * 2);
    EXPECT_EQ(lower, std::numeric_limits<Float8>::lowest());
  }

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

  double kLarge = 0x1p+128;
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

TYPED_TEST(Float8Test, FloatCompoundAddAssign) {
  using Float8 = TypeParam;
  float acc = 1.25f;
  float& ref = (acc += Float8(2.0f));
  EXPECT_EQ(&ref, &acc);
  EXPECT_FLOAT_EQ(acc, 3.25f);
}

TYPED_TEST(Float8Test, FloatCompoundSubAssign) {
  using Float8 = TypeParam;
  float acc = 3.25f;
  float& ref = (acc -= Float8(2.0f));
  EXPECT_EQ(&ref, &acc);
  EXPECT_FLOAT_EQ(acc, 1.25f);
}

TYPED_TEST(Float8Test, FloatCompoundMulAssign) {
  using Float8 = TypeParam;
  float acc = 1.25f;
  float& ref = (acc *= Float8(2.0f));
  EXPECT_EQ(&ref, &acc);
  EXPECT_FLOAT_EQ(acc, 2.5f);
}

TYPED_TEST(Float8Test, FloatCompoundDivAssign) {
  using Float8 = TypeParam;
  float acc = 2.5f;
  float& ref = (acc /= Float8(2.0f));
  EXPECT_EQ(&ref, &acc);
  EXPECT_FLOAT_EQ(acc, 1.25f);
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

TEST(Float8Test, Float8E3m4_To_Float8E5m2) {
  // Truncation and rounding of a number ever-so-slightly less than 2.
  Float8e3m4 less_than_two = Float8e3m4::FromRep(0x3F);
  Float8e5m2 truncated =
      Float8e5m2::template ConvertFrom</*kSaturate=*/false,
                                       /*kTruncate=*/true>(less_than_two);
  EXPECT_LT(static_cast<float>(truncated), 2);

  Float8e5m2 rounded =
      Float8e5m2::template ConvertFrom</*kSaturate=*/false,
                                       /*kTruncate=*/false>(less_than_two);
  EXPECT_EQ(static_cast<float>(rounded), 2);
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

TEST(Float8Test, Half_To_Float8E3m4) {
  // Special values, NaN.
  ::half_float::half inf =
      absl::bit_cast<::half_float::half>(static_cast<uint16_t>(0x7C00));
  EXPECT_EQ(static_cast<Float8e3m4>(inf).rep(), 0x70);
  ::half_float::half ninf =
      absl::bit_cast<::half_float::half>(static_cast<uint16_t>(0xFC00));
  EXPECT_EQ(static_cast<Float8e3m4>(ninf).rep(), 0xF0);

  ::half_float::half nan =
      absl::bit_cast<::half_float::half>(static_cast<uint16_t>(0x7C01));
  EXPECT_EQ(static_cast<Float8e3m4>(nan).rep(), 0x78);
  ::half_float::half nnan =
      absl::bit_cast<half_float::half>(static_cast<uint16_t>(0xFC01));
  EXPECT_EQ(static_cast<Float8e3m4>(nnan).rep(), 0xF8);

  // Rounding vs truncation.
  ::half_float::half less_than_two =
      absl::bit_cast<::half_float::half>(static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ((Float8e3m4::ConvertFrom</*kSaturate=*/false,
                                     /*kTruncate=*/false>(less_than_two)
                 .rep()),
            0x40);
  EXPECT_EQ((Float8e3m4::ConvertFrom</*kSaturate=*/false,
                                     /*kTruncate=*/true>(less_than_two)
                 .rep()),
            0x3F);
  EXPECT_EQ((Float8e3m4::ConvertFrom</*kSaturate=*/false,
                                     /*kTruncate=*/false>(-less_than_two)
                 .rep()),
            0xC0);
  EXPECT_EQ((Float8e3m4::ConvertFrom</*kSaturate=*/false,
                                     /*kTruncate=*/true>(-less_than_two)
                 .rep()),
            0xBF);

  // Saturation.
  // f8e3m4<max>=0.110.1111 0x1.Fp+3 f16=0.10010.1111000000 uint16=0x4BC0
  // f8e3m4<inf>=0.111.0000 0x1.0p+4 f16=0.10011.0000000000 uint16=0x4C00
  for (uint16_t i = 0x4BC0; i < 0x4C00; ++i) {
    ::half_float::half big_half = absl::bit_cast<::half_float::half>(i);
    float big_float = static_cast<float>(big_half);
    EXPECT_EQ((Float8e3m4::ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
                   big_half)
                   .rep()),
              (Float8e3m4::ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
                   big_float)
                   .rep()))
        << i;
    EXPECT_EQ((Float8e3m4::ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
                   -big_half)
                   .rep()),
              (Float8e3m4::ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
                   -big_float)
                   .rep()))
        << i;
  }
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

#define COMP(x)                         \
  EXPECT_EQ(a x b, float{a} x float{b}) \
      << a << #x << b << " vs " << float{a} << #x << float{b}

      COMP(==);
      COMP(!=);
      COMP(<);  // NOLINT
      COMP(<=);
      COMP(>);  // NOLINT
      COMP(>=);
#undef COMP
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

#define COMP(x)                         \
  EXPECT_EQ(a x b, float{a} x float{b}) \
      << a << #x << b << " vs " << float{a} << #x << float{b}

      COMP(==);
      COMP(!=);
      COMP(<);
      COMP(<=);
      COMP(>);
      COMP(>=);
#undef COMP
    }
  }
}

TEST(Float8E3m4Test, SmallCastToDenormal) {
  // Special edge-case where rounding to a normalized value would
  // normally round down, but rounding to a subnormal rounds up.
  float x = 0x0.8Ap-2;  // btw denormals
  Float8e3m4 y = static_cast<Float8e3m4>(x);
  float z = static_cast<float>(y);
  EXPECT_EQ(z, 0x0.9p-2);  // rounded up to the next denormal
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
    return absl::StrCat(TypeToName<first_type>(), "_",
                        TypeToName<second_type>());
  }
};

// clang-format off
#define GEN_TYPE_PAIRS(T) \
    std::pair<T, long double>, \
    std::pair<T, double>, \
    std::pair<T, float>, \
    std::pair<T, tensorstore::BFloat16>, \
    std::pair<T, ::half_float::half>, \
    std::pair<T, Float8e3m4>, \
    std::pair<T, Float8e4m3fn>, \
    std::pair<T, Float8e4m3b11fnuz>, \
    std::pair<T, Float8e4m3fnuz>, \
    std::pair<T, Float8e5m2fnuz>, \
    std::pair<T, Float8e5m2>, \
    std::pair<T, Float8e8m0fnu>, \
    std::pair<T, bool>, \
    std::pair<T, int32_t>, \
    std::pair<T, int64_t>

using Float8CastTypePairs = ::testing::Types<
    GEN_TYPE_PAIRS(Float8e3m4),
    GEN_TYPE_PAIRS(Float8e4m3fn),
    GEN_TYPE_PAIRS(Float8e4m3b11fnuz),
    GEN_TYPE_PAIRS(Float8e5m2),
    GEN_TYPE_PAIRS(Float8e4m3fnuz),
    GEN_TYPE_PAIRS(Float8e5m2fnuz),
    GEN_TYPE_PAIRS(Float8e8m0fnu)>;
// clang-format on

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
      if (!isfinite(f8) ||
          static_cast<float>(std::numeric_limits<DestType>::max()) <=
              static_cast<float>(f8)) {
        continue;
      }
    }

    DestType dest;
    if constexpr (!std::is_integral_v<DestType> &&
                  !std::is_same_v<DestType, long double>) {
      dest = Float8::template ConvertTo<DestType>(f8);
    } else {
      dest = static_cast<DestType>(f8);
    }
    DestType expected = static_cast<DestType>(static_cast<float>(f8));

    // Keep for MSVC build so that isnan() is only called on floating points
    if constexpr (std::numeric_limits<DestType>::is_integer) {
      EXPECT_EQ(dest, expected);
    } else {
      EXPECT_THAT(dest, EqOrIsNan<DestType>(expected));
    }
  }
}

TEST(Float8ConstexprConversionTest, ConstexprCasts) {
  constexpr Float8e4m3fn kOneE4m3 = Float8e4m3fn::FromRep(0x38);
  constexpr Float8e5m2 kOneE5m2(kOneE4m3);
  constexpr Float8e4m3fn kFromFloat(1.0f);
  constexpr Float8e4m3fn kFromDouble(1.0);
  constexpr Float8e4m3fn kFromInt(1);
  static_assert(kOneE5m2.rep() == 0x3C);
  static_assert(kFromFloat.rep() == 0x38);
  static_assert(kFromDouble.rep() == 0x38);
  static_assert(kFromInt.rep() == 0x38);
  static_assert(static_cast<float>(kOneE4m3) == 1.0f);
  static_assert(static_cast<double>(kOneE4m3) == 1.0);
  static_assert(static_cast<int8_t>(kOneE4m3) == 1);
  static_assert(static_cast<float>(static_cast<BFloat16>(kOneE4m3)) == 1.0f);
  static_assert(absl::bit_cast<uint16_t>(
                    static_cast<::half_float::half>(kOneE4m3)) == 0x3C00);
  static_assert(absl::bit_cast<uint16_t>(
                    static_cast<::half_float::half>(kOneE5m2)) == 0x3C00);
  static_assert(static_cast<std::complex<float>>(Float8e4m3fn(3.5f)) ==
                std::complex<float>(3.5f, 0.0f));
  static_assert(BFloat16(1.5f) == 1.5f);
  static_assert(BFloat16(2) == 2.0f);
  static_assert(BFloat16(true) == 1.0f);
  static_assert(kOneE4m3 == Float8e4m3fn(1.0f));
  static_assert(kOneE4m3 != Float8e4m3fn(2.0f));
  static_assert(kOneE4m3 < Float8e4m3fn(2.0f));
  static_assert(kOneE4m3 <= Float8e4m3fn(1.0f));
  static_assert(Float8e4m3fn(2.0f) > kOneE4m3);
  static_assert(Float8e4m3fn(2.0f) >= kOneE4m3);
  static_assert((kOneE4m3 + kOneE4m3) == Float8e4m3fn(2.0f));
  static_assert((Float8e4m3fn(3.0f) - kOneE4m3) == Float8e4m3fn(2.0f));
  static_assert((Float8e4m3fn(2.0f) * Float8e4m3fn(3.0f)) ==
                Float8e4m3fn(6.0f));
  static_assert((Float8e4m3fn(6.0f) / Float8e4m3fn(2.0f)) ==
                Float8e4m3fn(3.0f));
  static_assert(Float8e8m0fnu(1.0f) == Float8e8m0fnu::FromRep(0x7F));
  static_assert(Float8e8m0fnu(1.0f) < Float8e8m0fnu(2.0f));
  static_assert(static_cast<float>(Float8e3m4(1.0f)) == 1.0f);
  static_assert(static_cast<float>(Float8e4m3b11fnuz(1.0f)) == 1.0f);
  static_assert(static_cast<float>(Float8e4m3fnuz(1.0f)) == 1.0f);
  static_assert(static_cast<float>(Float8e5m2fnuz(1.0f)) == 1.0f);
  static_assert(static_cast<float>(Float8e8m0fnu(1.0f)) == 1.0f);
  static_assert((Float8e4m3b11fnuz(3.0f) - Float8e4m3b11fnuz(1.0f)) ==
                Float8e4m3b11fnuz(2.0f));
  static_assert((Float8e4m3fnuz(3.0f) - Float8e4m3fnuz(1.0f)) ==
                Float8e4m3fnuz(2.0f));
  static_assert((Float8e5m2fnuz(3.0f) - Float8e5m2fnuz(1.0f)) ==
                Float8e5m2fnuz(2.0f));
  static_assert((Float8e8m0fnu(4.0f) - Float8e8m0fnu(2.0f)) ==
                Float8e8m0fnu(2.0f));
}

TEST(Float8StringifyTest, AbslStringify) {
  EXPECT_EQ("1", absl::StrCat(Float8e4m3fn(1.0f)));
  EXPECT_EQ("0.5", absl::StrCat(Float8e4m3fn(0.5f)));
  EXPECT_EQ("1", absl::StrCat(Float8e5m2(1.0f)));
  EXPECT_EQ("0.5", absl::StrCat(Float8e5m2(0.5f)));
}

}  // namespace
}  // namespace tensorstore
