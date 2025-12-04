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

#include "tensorstore/util/mxfloat.h"

#include <stdint.h>

#include <cmath>
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

// Helper utility for prettier test names.
template <typename T>
std::string TypeToName() {
  if constexpr (std::is_same_v<T, Float6e2m3fn>) return "Float6e2m3fn";
  if constexpr (std::is_same_v<T, Float6e3m2fn>) return "Float6e3m2fn";
  if constexpr (std::is_same_v<T, Float4e2m1fn>) return "Float4e2m1fn";
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

template <typename MXFloat_>
class MXFloatTest : public ::testing::Test {};

// Helper utility for prettier test names.
struct MXFloatTestParamNames {
  template <typename TypeParam>
  static std::string GetName(int idx) {
    return TypeToName<TypeParam>();
  }
};

TEST(Float6e2m3fnTest, NumericLimits) {
  EXPECT_FALSE(isnan(std::numeric_limits<Float6e2m3fn>::quiet_NaN()));
  EXPECT_FALSE(isnan(std::numeric_limits<Float6e2m3fn>::signaling_NaN()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float6e2m3fn>::min()), 1.0f);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float6e2m3fn>::max()), 7.5f);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float6e2m3fn>::lowest()),
            -7.5f);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float6e2m3fn>::epsilon()),
            0.125f);
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<Float6e2m3fn>::round_error()),
      0.25f);
  EXPECT_FALSE(isinf(std::numeric_limits<Float6e2m3fn>::infinity()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float6e2m3fn>::denorm_min()),
            0.125f);
  EXPECT_EQ(std::numeric_limits<Float6e2m3fn>::digits, 4);
  EXPECT_EQ(std::numeric_limits<Float6e2m3fn>::digits10, 0);
  EXPECT_EQ(std::numeric_limits<Float6e2m3fn>::max_digits10, 3);
  EXPECT_EQ(std::numeric_limits<Float6e2m3fn>::is_iec559, false);
  EXPECT_EQ(std::numeric_limits<Float6e2m3fn>::has_infinity, false);
  EXPECT_EQ(std::numeric_limits<Float6e2m3fn>::has_quiet_NaN, false);
  EXPECT_EQ(std::numeric_limits<Float6e2m3fn>::has_signaling_NaN, false);
}

TEST(Float6e3m2fnTest, NumericLimits) {
  EXPECT_FALSE(isnan(std::numeric_limits<Float6e3m2fn>::quiet_NaN()));
  EXPECT_FALSE(isnan(std::numeric_limits<Float6e3m2fn>::signaling_NaN()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float6e3m2fn>::min()),
            0.25f);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float6e3m2fn>::max()),
            28.0f);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float6e3m2fn>::lowest()),
            -28.0f);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float6e3m2fn>::epsilon()),
            0.25f);
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<Float6e3m2fn>::round_error()),
      1.0f);
  EXPECT_FALSE(isinf(std::numeric_limits<Float6e3m2fn>::infinity()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float6e3m2fn>::denorm_min()),
            0.0625f);
  EXPECT_EQ(std::numeric_limits<Float6e3m2fn>::digits, 3);
  EXPECT_EQ(std::numeric_limits<Float6e3m2fn>::digits10, 0);
  EXPECT_EQ(std::numeric_limits<Float6e3m2fn>::max_digits10, 2);
  EXPECT_EQ(std::numeric_limits<Float6e3m2fn>::is_iec559, false);
  EXPECT_EQ(std::numeric_limits<Float6e3m2fn>::has_infinity, false);
  EXPECT_EQ(std::numeric_limits<Float6e3m2fn>::has_quiet_NaN, false);
  EXPECT_EQ(std::numeric_limits<Float6e3m2fn>::has_signaling_NaN, false);
}

TEST(Float4e2m1fnTest, NumericLimits) {
  EXPECT_FALSE(isnan(std::numeric_limits<Float4e2m1fn>::quiet_NaN()));
  EXPECT_FALSE(isnan(std::numeric_limits<Float4e2m1fn>::signaling_NaN()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float4e2m1fn>::min()), 1.0f);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float4e2m1fn>::max()), 6.0f);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float4e2m1fn>::lowest()),
            -6.0f);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float4e2m1fn>::epsilon()),
            0.5f);
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<Float4e2m1fn>::round_error()),
      1.0f);
  EXPECT_FALSE(isinf(std::numeric_limits<Float4e2m1fn>::infinity()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Float4e2m1fn>::denorm_min()),
            0.5f);
  EXPECT_EQ(std::numeric_limits<Float4e2m1fn>::digits, 2);
  EXPECT_EQ(std::numeric_limits<Float4e2m1fn>::digits10, 0);
  EXPECT_EQ(std::numeric_limits<Float4e2m1fn>::max_digits10, 2);
  EXPECT_EQ(std::numeric_limits<Float4e2m1fn>::is_iec559, false);
  EXPECT_EQ(std::numeric_limits<Float4e2m1fn>::has_infinity, false);
  EXPECT_EQ(std::numeric_limits<Float4e2m1fn>::has_quiet_NaN, false);
  EXPECT_EQ(std::numeric_limits<Float4e2m1fn>::has_signaling_NaN, false);
}

using MXFloatTypes = ::testing::Types<Float6e2m3fn, Float6e3m2fn, Float4e2m1fn>;
TYPED_TEST_SUITE(MXFloatTest, MXFloatTypes, MXFloatTestParamNames);

TYPED_TEST(MXFloatTest, DefaultConstruction) {
  using Float = TypeParam;
  const Float zero(0);

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

TYPED_TEST(MXFloatTest, FromRep) {
  using Float = TypeParam;
  for (int i = 0; i < (1 << Float::kBits); ++i) {
    Float x = Float::FromRep(i);
    EXPECT_EQ(x.rep(), i);
  }
}

TYPED_TEST(MXFloatTest, Negate) {
  using Float = TypeParam;
  constexpr uint8_t sign_bit = 1 << (Float::kBits - 1);
  for (int i = 0; i < (1 << Float::kBits); ++i) {
    Float x = Float::FromRep(i);
    EXPECT_EQ((-x).rep(), i ^ sign_bit);
  }
}

TYPED_TEST(MXFloatTest, BitCasts) {
  using Float = TypeParam;
  for (int i = 0; i < (1 << Float::kBits); ++i) {
    Float x = Float::FromRep(i);
    EXPECT_EQ(absl::bit_cast<uint8_t>(x), i);
    EXPECT_EQ(absl::bit_cast<Float>(x.rep()).rep(), i);
  }
}

TYPED_TEST(MXFloatTest, UpCasts) {
  using Float = TypeParam;
  for (int i = 0x00; i < (1 << Float::kBits); ++i) {
    Float f = Float::FromRep(i);
    double f64 = static_cast<double>(f);
    float f32 = static_cast<float>(f);
    tensorstore::BFloat16 bf16 = static_cast<tensorstore::BFloat16>(f);
    ::half_float::half f16 = static_cast<::half_float::half>(f);
    EXPECT_EQ(f64, f32);
    EXPECT_EQ(f32, static_cast<float>(bf16));
    EXPECT_EQ(static_cast<float>(bf16), static_cast<float>(f16));
  }
}

TYPED_TEST(MXFloatTest, DownCasts) {
  using Float = TypeParam;
  for (int i = 0x00; i < (1 << Float::kBits); ++i) {
    float x = static_cast<float>(Float::FromRep(i));
    Float f64 = static_cast<Float>(static_cast<double>(x));
    Float f32 = static_cast<Float>(static_cast<float>(x));
    Float bf16 = static_cast<Float>(static_cast<tensorstore::BFloat16>(x));
    Float f16 = static_cast<Float>(static_cast<::half_float::half>(x));
    EXPECT_EQ(f64.rep(), i) << i;
    EXPECT_EQ(f32.rep(), i) << i;
    EXPECT_EQ(bf16.rep(), i) << i;
    EXPECT_EQ(f16.rep(), i) << i;
  }
}

TYPED_TEST(MXFloatTest, ConvertFromWithSaturation) {
  using Float = TypeParam;
  // Saturation above max value.
  Float upper =
      Float::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
          std::numeric_limits<float>::infinity());
  EXPECT_EQ(upper, std::numeric_limits<Float>::max());
  Float lower =
      Float::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
          -std::numeric_limits<float>::infinity());
  EXPECT_EQ(lower, std::numeric_limits<Float>::lowest());
  // NaN conversion.
  Float nan =
      Float::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/true>(
          std::numeric_limits<float>::quiet_NaN());
  EXPECT_EQ(nan, std::numeric_limits<Float>::quiet_NaN());
}

using ::testing::Eq;
using ::testing::IsTrue;
MATCHER_P(EqOrIsNan, other, "") {
  if (isnan(other)) {
    return ExplainMatchResult(IsTrue(), isnan(arg), result_listener);
  }
  return ExplainMatchResult(Eq(other), arg, result_listener);
}

TYPED_TEST(MXFloatTest, CallTheOperator) {
  using Float = TypeParam;

  for (int i = 0x00; i < (1 << Float::kBits); ++i) {
    Float a = Float::FromRep(i);
    for (int j = 0x00; j < (1 << Float::kBits); ++j) {
      Float b = Float::FromRep(j);

      EXPECT_THAT(a + b, EqOrIsNan<Float>(Float{float{a} + float{b}}));
      EXPECT_THAT(a - b, EqOrIsNan<Float>(Float{float{a} - float{b}}));
      EXPECT_THAT(a * b, EqOrIsNan<Float>(Float{float{a} * float{b}}));
      EXPECT_THAT(a / b, EqOrIsNan<Float>(Float{float{a} / float{b}}));

      Float c;
      EXPECT_THAT((c = a, c += b), Float{float{a} + float{b}});
      EXPECT_THAT((c = a, c -= b), Float{float{a} - float{b}});
      EXPECT_THAT((c = a, c *= b), Float{float{a} * float{b}});
      EXPECT_THAT((c = a, c /= b), Float{float{a} / float{b}});

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

TYPED_TEST(MXFloatTest, CallTheConstOperator) {
  using Float = TypeParam;

  for (int i = 0x00; i < (1 << Float::kBits); ++i) {
    const Float a = Float::FromRep(i);
    for (int j = 0x00; j < (1 << Float::kBits); ++j) {
      const Float b = Float::FromRep(j);

      EXPECT_THAT(a + b, EqOrIsNan<Float>(Float{float{a} + float{b}}));
      EXPECT_THAT(a - b, EqOrIsNan<Float>(Float{float{a} - float{b}}));
      EXPECT_THAT(a * b, EqOrIsNan<Float>(Float{float{a} * float{b}}));
      EXPECT_THAT(a / b, EqOrIsNan<Float>(Float{float{a} / float{b}}));

      Float c;
      EXPECT_THAT((c = a, c += b),
                  EqOrIsNan<Float>(Float{float{a} + float{b}}));
      EXPECT_THAT((c = a, c -= b),
                  EqOrIsNan<Float>(Float{float{a} - float{b}}));
      EXPECT_THAT((c = a, c *= b),
                  EqOrIsNan<Float>(Float{float{a} * float{b}}));
      EXPECT_THAT((c = a, c /= b),
                  EqOrIsNan<Float>(Float{float{a} / float{b}}));

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

struct MXFloatCastTestParamNames {
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
    std::pair<T, Float6e2m3fn>, \
    std::pair<T, Float6e3m2fn>, \
    std::pair<T, Float4e2m1fn>, \
    std::pair<T, bool>, \
    std::pair<T, int32_t>, \
    std::pair<T, int64_t>

using MXFloatCastTypePairs = ::testing::Types<
    GEN_TYPE_PAIRS(Float6e2m3fn),
    GEN_TYPE_PAIRS(Float6e3m2fn),
    GEN_TYPE_PAIRS(Float4e2m1fn)>;
// clang-format on

template <typename CastPair>
class MXFloatCastTest : public ::testing::Test {};
TYPED_TEST_SUITE(MXFloatCastTest, MXFloatCastTypePairs,
                 MXFloatCastTestParamNames);

TYPED_TEST(MXFloatCastTest, CastThroughFloat) {
  using Float = typename TypeParam::first_type;
  using DestType = typename TypeParam::second_type;

  for (int i = 0x00; i < (1 << Float::kBits); ++i) {
    Float fmx = Float::FromRep(i);

    if constexpr (std::numeric_limits<DestType>::is_integer &&
                  !std::is_same_v<DestType, bool>) {
      if (!isfinite(static_cast<float>(fmx)) ||
          static_cast<float>(std::numeric_limits<DestType>::max()) <=
              static_cast<float>(fmx)) {
        continue;
      }
    }

    DestType dest;
    if constexpr (!std::is_integral_v<DestType> &&
                  !std::is_same_v<DestType, long double>) {
      dest = Float::template ConvertTo<DestType>(fmx);
    } else {
      dest = static_cast<DestType>(fmx);
    }
    DestType expected = static_cast<DestType>(static_cast<float>(fmx));

    // Keep for MSVC build so that isnan() is only called on floating points
    if constexpr (std::numeric_limits<DestType>::is_integer) {
      EXPECT_EQ(dest, expected);
    } else {
      EXPECT_THAT(dest, EqOrIsNan<DestType>(expected));
    }
  }
}

}  // namespace
}  // namespace tensorstore
