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

#ifndef TENSORSTORE_UTIL_MXFLOAT_H_
#define TENSORSTORE_UTIL_MXFLOAT_H_

// The implementation below is derived from jax-ml/ml_dtypes:
// https://github.com/jax-ml/ml_dtypes/blob/main/ml_dtypes/include/mxfloat.h

/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

// Microscaling (MX) floating point formats, as described in
//   https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
//
// Note: this implements the underlying raw data types (e.g. E2M1FN), not the
// composite types (e.g. MXFP4).

#include <stddef.h>
#include <stdint.h>

#include <limits>

#include "tensorstore/util/float8.h"

namespace tensorstore {
namespace mxfloat_internal {

// Use 8-bit storage for 6-bit and 4-bit types.
template <typename Derived>
class MXFloat6Base : public float8_internal::Float8Base<Derived> {
  using Base = float8_internal::Float8Base<Derived>;
  friend class float8_internal::Float8Base<Derived>;
  using Base::Base;

 public:
  static constexpr int kBits = 6;

  explicit operator bool() const { return (Base::rep() & 0x1F) != 0; }
  constexpr Derived operator-() const {
    return Derived::FromRep(Base::rep() ^ 0x20);
  }
  Derived operator-(const Derived& other) const {
    return Base::operator-(other);
  }
};

template <typename Derived>
class MXFloat4Base : public float8_internal::Float8Base<Derived> {
  using Base = float8_internal::Float8Base<Derived>;
  friend class float8_internal::Float8Base<Derived>;
  using Base::Base;

 public:
  static constexpr int kBits = 4;

  explicit operator bool() const { return (Base::rep() & 0x07) != 0; }
  constexpr Derived operator-() const {
    return Derived::FromRep(Base::rep() ^ 0x08);
  }
  Derived operator-(const Derived& other) const {
    return Base::operator-(other);
  }
};

class Float6e2m3fn : public MXFloat6Base<Float6e2m3fn> {
  // Exponent: 2, Mantissa: 3, bias: 1.
  // Extended range: no inf, no NaN.
  using Base = MXFloat6Base<Float6e2m3fn>;
  friend class float8_internal::Float8Base<Float6e2m3fn>;
  using Base::Base;

 public:
  template <typename T, float8_internal::RequiresIsDerivedFromFloat8Base<T> = 0>
  explicit Float6e2m3fn(T f8) : Float6e2m3fn(ConvertFrom(f8)) {}
};

class Float6e3m2fn : public MXFloat6Base<Float6e3m2fn> {
  // Exponent: 3, Mantissa: 2, bias: 3.
  // Extended range: no inf, no NaN.
  using Base = MXFloat6Base<Float6e3m2fn>;
  friend class float8_internal::Float8Base<Float6e3m2fn>;
  using Base::Base;

 public:
  template <typename T, float8_internal::RequiresIsDerivedFromFloat8Base<T> = 0>
  explicit Float6e3m2fn(T f8) : Float6e3m2fn(ConvertFrom(f8)) {}
};

class Float4e2m1fn : public MXFloat4Base<Float4e2m1fn> {
  // Exponent: 2, Mantissa: 1, bias: 1.
  // Extended range: no inf, no NaN.
  using Base = MXFloat4Base<Float4e2m1fn>;
  friend class float8_internal::Float8Base<Float4e2m1fn>;
  using Base::Base;

 public:
  template <typename T, float8_internal::RequiresIsDerivedFromFloat8Base<T> = 0>
  explicit Float4e2m1fn(T f8) : Float4e2m1fn(ConvertFrom(f8)) {}
};

// Common properties for specializing std::numeric_limits.
template <int E, int M>
struct numeric_limits_mxfloat_tpl {
 protected:
  static constexpr int kExponentBias = (1 << (E - 1)) - 1;
  static constexpr int kMantissaBits = M;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = false;
  static constexpr bool has_signaling_NaN = false;
#if !defined(__cplusplus) || __cplusplus < 202302L
  static constexpr std::float_denorm_style has_denorm = std::denorm_present;
  static constexpr bool has_denorm_loss = false;
#endif
  static constexpr std::float_round_style round_style = std::round_to_nearest;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = kMantissaBits + 1;
  static constexpr int digits10 = float8_internal::Digits10FromDigits(digits);
  static constexpr int max_digits10 =
      float8_internal::MaxDigits10FromDigits(digits);
  static constexpr int radix = std::numeric_limits<float>::radix;
  static constexpr int min_exponent = (1 - kExponentBias) + 1;
  static constexpr int min_exponent10 =
      float8_internal::MinExponent10FromMinExponent(min_exponent);
  static constexpr int max_exponent = kExponentBias + 2;
  static constexpr int max_exponent10 =
      float8_internal::MaxExponent10FromMaxExponentAndDigits(max_exponent,
                                                             digits);
  static constexpr bool traps = std::numeric_limits<float>::traps;
  static constexpr bool tinyness_before =
      std::numeric_limits<float>::tinyness_before;
  // NOLINTEND
};

struct numeric_limits_float6_e2m3fn : public numeric_limits_mxfloat_tpl<2, 3> {
  // 1.0 * 2^(0) = 1
  static constexpr Float6e2m3fn min() {
    return Float6e2m3fn::FromRep(0b0'01'000);
  }
  // -1.875 * 2^(2) = -7.5
  static constexpr Float6e2m3fn lowest() {
    return Float6e2m3fn::FromRep(0b1'11'111);
  }
  // 1.875 * 2^(2) = 7.5
  static constexpr Float6e2m3fn max() {
    return Float6e2m3fn::FromRep(0b0'11'111);
  }
  // 0.125 * 2^(0) = 0.125
  static constexpr Float6e2m3fn epsilon() {
    return Float6e2m3fn::FromRep(0b0'00'001);
  }
  // 0.25 * 2^(0) = 0.25
  static constexpr Float6e2m3fn round_error() {
    return Float6e2m3fn::FromRep(0b0'00'010);
  }
  // 0.25 * 2^(0) = 0.125
  static constexpr Float6e2m3fn denorm_min() {
    return Float6e2m3fn::FromRep(0b0'00'001);
  }

  // Conversion from NaNs is implementation-defined (by MX specification).
  static constexpr Float6e2m3fn quiet_NaN() {
    return Float6e2m3fn::FromRep(0b1'00'000);
  }
  static constexpr Float6e2m3fn signaling_NaN() {
    return Float6e2m3fn::FromRep(0b1'00'000);
  }
  static constexpr Float6e2m3fn infinity() {
    return Float6e2m3fn::FromRep(0b0'11'111);
  }
};

struct numeric_limits_float6_e3m2fn : public numeric_limits_mxfloat_tpl<3, 2> {
  // 1.0 * 2^(-2) = 0.25
  static constexpr Float6e3m2fn min() {
    return Float6e3m2fn::FromRep(0b0'001'00);
  }
  // -1.75 * 2^(4) = -28
  static constexpr Float6e3m2fn lowest() {
    return Float6e3m2fn::FromRep(0b1'111'11);
  }
  // 1.75 * 2^(4) = 28
  static constexpr Float6e3m2fn max() {
    return Float6e3m2fn::FromRep(0b0'111'11);
  }
  // 1.0 * 2^(-2) = 0.25
  static constexpr Float6e3m2fn epsilon() {
    return Float6e3m2fn::FromRep(0b0'001'00);
  }
  // 1.0 * 2^(0) = 1
  static constexpr Float6e3m2fn round_error() {
    return Float6e3m2fn::FromRep(0b0'011'00);
  }
  // 0.25 * 2^(-2) = 0.0625
  static constexpr Float6e3m2fn denorm_min() {
    return Float6e3m2fn::FromRep(0b0'000'01);
  }

  // Conversion from NaNs is implementation-defined (by MX specification).
  static constexpr Float6e3m2fn quiet_NaN() {
    return Float6e3m2fn::FromRep(0b1'000'00);
  }
  static constexpr Float6e3m2fn signaling_NaN() {
    return Float6e3m2fn::FromRep(0b1'000'00);
  }
  static constexpr Float6e3m2fn infinity() {
    return Float6e3m2fn::FromRep(0b0'111'11);
  }
};

struct numeric_limits_float4_e2m1fn : public numeric_limits_mxfloat_tpl<2, 1> {
  // 1.0 * 2^(0) = 1
  static constexpr Float4e2m1fn min() {
    return Float4e2m1fn::FromRep(0b0'01'0);
  }
  // -1.5 * 2^(2) = -6
  static constexpr Float4e2m1fn lowest() {
    return Float4e2m1fn::FromRep(0b1'11'1);
  }
  // 1.5 * 2^(2) = 6
  static constexpr Float4e2m1fn max() {
    return Float4e2m1fn::FromRep(0b0'11'1);
  }
  // 0.5 * 2^(0) = 0.5
  static constexpr Float4e2m1fn epsilon() {
    return Float4e2m1fn::FromRep(0b0'00'1);
  }
  // 1.0 * 2^(0) = 1
  static constexpr Float4e2m1fn round_error() {
    return Float4e2m1fn::FromRep(0b0'01'0);
  }
  // 0.5 * 2^(0) = 0.5
  static constexpr Float4e2m1fn denorm_min() {
    return Float4e2m1fn::FromRep(0b0'00'1);
  }

  // Conversion from NaNs is implementation-defined (by MX specification).
  static constexpr Float4e2m1fn quiet_NaN() {
    return Float4e2m1fn::FromRep(0b1'00'0);
  }
  static constexpr Float4e2m1fn signaling_NaN() {
    return Float4e2m1fn::FromRep(0b1'00'0);
  }
  static constexpr Float4e2m1fn infinity() {
    return Float4e2m1fn::FromRep(0b0'11'1);
  }
};

constexpr inline Float6e2m3fn abs(const Float6e2m3fn& a) {
  return Float6e2m3fn::FromRep(a.rep() & 0b0'11'111);
}

constexpr inline bool(isnan)(const Float6e2m3fn& a) { return false; }

constexpr inline Float6e3m2fn abs(const Float6e3m2fn& a) {
  return Float6e3m2fn::FromRep(a.rep() & 0b0'111'11);
}
constexpr inline bool(isinf)(const Float6e2m3fn& a) { return false; }

constexpr inline bool(isnan)(const Float6e3m2fn& a) { return false; }

constexpr inline bool(isinf)(const Float6e3m2fn& a) { return false; }

constexpr inline Float4e2m1fn abs(const Float4e2m1fn& a) {
  return Float4e2m1fn::FromRep(a.rep() & 0b0'11'1);
}

constexpr inline bool(isnan)(const Float4e2m1fn& a) { return false; }

constexpr inline bool(isinf)(const Float4e2m1fn& a) { return false; }

// Define traits required for floating point conversion.
template <typename T, int E, int M>
struct TraitsBase : public float8_internal::TraitsBase<T> {
  static constexpr int kBits = E + M + 1;
  static constexpr int kMantissaBits = M;
  static constexpr int kExponentBits = E;
  static constexpr int kExponentBias = (1 << (E - 1)) - 1;
  static constexpr uint8_t kExponentMask = ((1 << E) - 1) << M;
};

}  // namespace mxfloat_internal

// Exported types.

/// Storage-only MX floating-point data types.
///
/// See
/// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
///
/// \ingroup Data types
using Float6e2m3fn = mxfloat_internal::Float6e2m3fn;
using Float6e3m2fn = mxfloat_internal::Float6e3m2fn;
using Float4e2m1fn = mxfloat_internal::Float4e2m1fn;

}  // namespace tensorstore

// Standard library overrides.
namespace std {

template <>
struct numeric_limits<tensorstore::mxfloat_internal::Float6e2m3fn>
    : public tensorstore::mxfloat_internal::numeric_limits_float6_e2m3fn {};

template <>
struct numeric_limits<tensorstore::mxfloat_internal::Float6e3m2fn>
    : public tensorstore::mxfloat_internal::numeric_limits_float6_e3m2fn {};

template <>
struct numeric_limits<tensorstore::mxfloat_internal::Float4e2m1fn>
    : public tensorstore::mxfloat_internal::numeric_limits_float4_e2m1fn {};

}  // namespace std

// Conversion traits.
namespace tensorstore {
namespace float8_internal {

template <>
struct Traits<Float6e2m3fn>
    : public mxfloat_internal::TraitsBase<Float6e2m3fn, 2, 3> {};

template <>
struct Traits<Float6e3m2fn>
    : public mxfloat_internal::TraitsBase<Float6e3m2fn, 3, 2> {};

template <>
struct Traits<Float4e2m1fn>
    : public mxfloat_internal::TraitsBase<Float4e2m1fn, 2, 1> {};

#ifdef _MSC_VER
// MSVC requires overloads for fpclassify to work with custom types.
// We use the internal implementations of isnan/isinf/abs.
#define TENSORSTORE_INTERNAL_MXFLOAT_FPCLASSIFY(MXFloat)                     \
  inline int fpclassify(MXFloat a) noexcept {                                \
    if (tensorstore::mxfloat_internal::isnan(a)) return FP_NAN;              \
    if (tensorstore::mxfloat_internal::isinf(a)) return FP_INFINITE;         \
    MXFloat abs_value = tensorstore::mxfloat_internal::abs(a);               \
    /* Cast to bool checks if rep != 0 */                                    \
    if (!static_cast<bool>(abs_value)) return FP_ZERO;                       \
    using Traits = tensorstore::float8_internal::Traits<MXFloat>;            \
    if ((abs_value.rep() & Traits::kExponentMask) == 0) return FP_SUBNORMAL; \
    return FP_NORMAL;                                                        \
  }

TENSORSTORE_INTERNAL_MXFLOAT_FPCLASSIFY(Float6e2m3fn);
TENSORSTORE_INTERNAL_MXFLOAT_FPCLASSIFY(Float6e3m2fn);
TENSORSTORE_INTERNAL_MXFLOAT_FPCLASSIFY(Float4e2m1fn);
#undef TENSORSTORE_INTERNAL_MXFLOAT_FPCLASSIFY
#endif

}  // namespace float8_internal
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_MXFLOAT_H_
