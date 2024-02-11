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

#ifndef TENSORSTORE_UTIL_FLOAT8_H_
#define TENSORSTORE_UTIL_FLOAT8_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>
#include <utility>

#include "absl/base/casts.h"
#include <half.hpp>
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/util/bfloat16.h"

// The implementation below is derived from jax-ml/ml_dtypes:
// https://github.com/jax-ml/ml_dtypes/blob/main/ml_dtypes/include/float8.h

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// 8-bit Floating Point Interchange Format, as described by
//   https://arxiv.org/abs/2209.05433

#if (defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L)
#include <bit>
#endif

namespace tensorstore {
namespace float8_internal {

// Forward-declarations of classes.
class Float8e4m3fn;
class Float8e4m3fnuz;
class Float8e4m3b11fnuz;
class Float8e5m2;
class Float8e5m2fnuz;

template <typename Derived>
class Float8Base {
 protected:
  // Constructor tag to allow constexpr construction from bit representation.
  struct ConstructFromRepTag {};
  constexpr Float8Base(uint8_t rep, ConstructFromRepTag) : rep_{rep} {}

 public:
  constexpr Float8Base() : rep_(0) {}

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit Float8Base(T f)
      : Float8Base(ConvertFrom(static_cast<float>(f)).rep(),
                   ConstructFromRepTag{}) {}
  explicit Float8Base(double f64)
      : Float8Base(ConvertFrom(f64).rep(), ConstructFromRepTag{}) {}
  explicit Float8Base(float f32)
      : Float8Base(ConvertFrom(f32).rep(), ConstructFromRepTag{}) {}
  explicit Float8Base(BFloat16 bf16)
      : Float8Base(ConvertFrom(bf16).rep(), ConstructFromRepTag{}) {}
  explicit Float8Base(::half_float::half f16)
      : Float8Base(ConvertFrom(f16).rep(), ConstructFromRepTag{}) {}

  constexpr uint8_t rep() const { return rep_; }

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit operator T() const {
    return static_cast<T>(static_cast<float>(derived()));
  }
  explicit operator double() const { return ConvertTo<double>(derived()); }
  explicit operator float() const { return ConvertTo<float>(derived()); }
  explicit operator BFloat16() const { return ConvertTo<BFloat16>(derived()); }
  explicit operator ::half_float::half() const {
    return ConvertTo<::half_float::half>(derived());
  }
  explicit operator bool() const { return (rep() & 0x7F) != 0; }

  constexpr Derived operator-() const {
    return Derived(static_cast<uint8_t>(rep() ^ 0x80), ConstructFromRepTag{});
  }

  constexpr const Derived& derived() const {
    return *static_cast<const Derived*>(this);
  }

  constexpr Derived& derived() { return *static_cast<Derived*>(this); }

  static constexpr Derived FromRep(uint8_t rep) {
    return Derived(rep, ConstructFromRepTag{});
  }

  // Conversions allowing saturation and truncation.
  template <bool kSaturate = false, bool kTruncate = false, typename From>
  static Derived ConvertFrom(const From& from);

  template <typename To, bool kSaturate = false, bool kTruncate = false>
  static To ConvertTo(const Derived& from);

  // Operators via float32.
  Derived operator+(const Derived& other) const {
    return Derived{float{derived()} + float{other}};
  }

  Derived operator-(const Derived& other) const {
    return Derived{float{derived()} - float{other}};
  }

  Derived operator*(const Derived& other) const {
    return Derived{float{derived()} * float{other}};
  }

  Derived operator/(const Derived& other) const {
    return Derived{float{derived()} / float{other}};
  }

  constexpr bool operator==(const Derived& other) const {
    return Compare(derived(), other) == Ordering::kEquivalent;
  }

  constexpr bool operator!=(const Derived& other) const {
    return Compare(derived(), other) != Ordering::kEquivalent;
  }

  bool operator<(const Derived& other) const {
    return Compare(derived(), other) == Ordering::kLess;
  }

  bool operator<=(const Derived& other) const {
    return Compare(derived(), other) <= Ordering::kEquivalent;
  }

  bool operator>(const Derived& other) const {
    return Compare(derived(), other) == Ordering::kGreater;
  }

  bool operator>=(const Derived& other) const {
    Ordering ordering = Compare(derived(), other);
    return ordering == Ordering::kGreater || ordering == Ordering::kEquivalent;
  }

  // Compound assignment.
  Derived& operator+=(const Derived& other) {
    derived() = derived() + other;
    return derived();
  }

  // for downsample_nditerable
  friend float operator+=(const float& a, Derived b) {
    return a + static_cast<float>(b);
  }

  Derived& operator-=(const Derived& other) {
    derived() = derived() - other;
    return derived();
  }

  Derived& operator*=(const Derived& other) {
    derived() = derived() * other;
    return derived();
  }

  Derived& operator/=(const Derived& other) {
    derived() = derived() / other;
    return derived();
  }

  // Note: Comparison operators do not need to be defined since they are
  // provided automatically by the implicit conversion to `float`.

  // Conversion to `::nlohmann::json`.
  template <template <typename U, typename V, typename... Args>
            class ObjectType /* = std::map*/,
            template <typename U, typename... Args>
            class ArrayType /* = std::vector*/,
            class StringType /*= std::string*/, class BooleanType /* = bool*/,
            class NumberIntegerType /* = std::int64_t*/,
            class NumberUnsignedType /* = std::uint64_t*/,
            class NumberFloatType /* = double*/,
            template <typename U> class AllocatorType /* = std::allocator*/,
            template <typename T, typename SFINAE = void>
            class JSONSerializer /* = adl_serializer*/,
            class BinaryType /* = std::vector<std::uint8_t>*/>
  friend void to_json(
      ::nlohmann::basic_json<ObjectType, ArrayType, StringType, BooleanType,
                             NumberIntegerType, NumberUnsignedType,
                             NumberFloatType, AllocatorType, JSONSerializer,
                             BinaryType>& j,
      Derived v) {
    j = static_cast<NumberFloatType>(v);
  }

 private:
  static std::pair<uint8_t, uint8_t> SignAndMagnitude(Derived x) {
    const uint8_t x_abs_bits = absl::bit_cast<uint8_t>(abs(x));
    const uint8_t x_bits = absl::bit_cast<uint8_t>(x);
    const uint8_t x_sign = x_bits ^ x_abs_bits;
    return {x_sign, x_abs_bits};
  }
  static int8_t SignAndMagnitudeToTwosComplement(uint8_t sign,
                                                 uint8_t magnitude) {
    return magnitude ^ (static_cast<int8_t>(sign) < 0 ? -1 : 0);
  }

  enum Ordering : int8_t {
    kLess = -1,
    kEquivalent = 0,
    kGreater = 1,
    kUnordered = 2,
  };

  friend Ordering Compare(const Derived& lhs, const Derived& rhs) {
    if (isnan(lhs) || isnan(rhs)) {
      return Ordering::kUnordered;
    }
    auto [lhs_sign, lhs_mag] = SignAndMagnitude(lhs);
    auto [rhs_sign, rhs_mag] = SignAndMagnitude(rhs);
    if (lhs_mag == 0 && rhs_mag == 0) {
      return Ordering::kEquivalent;
    }
    int8_t lhs_twos_complement =
        SignAndMagnitudeToTwosComplement(lhs_sign, lhs_mag);
    int8_t rhs_twos_complement =
        SignAndMagnitudeToTwosComplement(rhs_sign, rhs_mag);
    if (lhs_twos_complement < rhs_twos_complement) {
      return Ordering::kLess;
    }
    if (lhs_twos_complement > rhs_twos_complement) {
      return Ordering::kGreater;
    }
    return Ordering::kEquivalent;
  }

  uint8_t rep_;
};

class Float8e4m3fn : public Float8Base<Float8e4m3fn> {
  // Exponent: 4, Mantissa: 3, bias: 7.
  // Extended range: no inf, NaN represented by 0bS111'1111.
  // The "fn" suffix is for consistency with the corresponding LLVM/MLIR type,
  // signaling this type is not consistent with IEEE-754.  The "f" indicates
  // it is finite values only. The "n" indicates it includes NaNs, but only
  // at the outer range.
 private:
  using Base = Float8Base<Float8e4m3fn>;
  friend class Float8Base<Float8e4m3fn>;
  using Base::Float8Base;

 public:
  explicit Float8e4m3fn(const Float8e5m2& f8) : Float8e4m3fn(ConvertFrom(f8)) {}
  explicit Float8e4m3fn(const Float8e4m3b11fnuz& f8)
      : Float8e4m3fn(ConvertFrom(f8)) {}
};

class Float8e4m3b11fnuz : public Float8Base<Float8e4m3b11fnuz> {
  // Exponent: 4, Mantissa: 3, bias: 11.
  // Extended range: no inf, NaN represented by 0b1000'0000.
 private:
  using Base = Float8Base<Float8e4m3b11fnuz>;
  friend class Float8Base<Float8e4m3b11fnuz>;
  using Base::Float8Base;

 public:
  explicit Float8e4m3b11fnuz(const Float8e5m2& f8)
      : Float8e4m3b11fnuz(ConvertFrom(f8)) {}
  explicit Float8e4m3b11fnuz(const Float8e5m2fnuz& f8)
      : Float8e4m3b11fnuz(ConvertFrom(f8)) {}
  explicit Float8e4m3b11fnuz(const Float8e4m3fn& f8)
      : Float8e4m3b11fnuz(ConvertFrom(f8)) {}
  explicit Float8e4m3b11fnuz(const Float8e4m3fnuz& f8)
      : Float8e4m3b11fnuz(ConvertFrom(f8)) {}

  constexpr Float8e4m3b11fnuz operator-() const {
    if ((rep() & 0x7f) == 0x00) {
      return *this;
    }
    return Base::operator-();
  }

  Float8e4m3b11fnuz operator-(const Float8e4m3b11fnuz& other) const {
    return Base::operator-(other);
  }

  explicit operator bool() const { return rep() != 0; }
};

class Float8e4m3fnuz : public Float8Base<Float8e4m3fnuz> {
  // 8-bit floating point with 3 bit mantissa.
  //
  // An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits
  // mantissa. The suffix "fnuz" is consistent with LLVM/MLIR naming and is
  // derived from the differences to IEEE floating point conventions. `F` is
  // for "finite" (no infinities), `N` for with special NaN encoding, `UZ` for
  // unsigned zero.
  //
  // This type has the following characteristics:
  // * bit encoding: S1E4M3 - `0bSEEEEMMM`
  // * exponent bias: 8
  // * infinities: Not supported
  // * NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits
  // set to all 0s - `0b10000000`
  // * denormals when exponent is 0
 private:
  using Base = Float8Base<Float8e4m3fnuz>;
  friend class Float8Base<Float8e4m3fnuz>;
  using Base::Float8Base;

 public:
  explicit Float8e4m3fnuz(const Float8e5m2& f8)
      : Float8e4m3fnuz(ConvertFrom(f8)) {}
  explicit Float8e4m3fnuz(const Float8e5m2fnuz& f8)
      : Float8e4m3fnuz(ConvertFrom(f8)) {}
  explicit Float8e4m3fnuz(const Float8e4m3b11fnuz& f8)
      : Float8e4m3fnuz(ConvertFrom(f8)) {}
  explicit Float8e4m3fnuz(const Float8e4m3fn& f8)
      : Float8e4m3fnuz(ConvertFrom(f8)) {}

  constexpr Float8e4m3fnuz operator-() const {
    if ((rep() & 0x7f) == 0x00) {
      return *this;
    }
    return Base::operator-();
  }

  Float8e4m3fnuz operator-(const Float8e4m3fnuz& other) const {
    return Base::operator-(other);
  }

  explicit operator bool() const { return rep() != 0; }
};

class Float8e5m2 : public Float8Base<Float8e5m2> {
  // Exponent: 5, Mantissa: 2, bias: 15.
  // IEEE 754.
 private:
  using Base = Float8Base<Float8e5m2>;
  friend class Float8Base<Float8e5m2>;
  using Base::Float8Base;

 public:
  explicit Float8e5m2(Float8e4m3fn f8) : Float8e5m2(ConvertFrom(f8)) {}
  explicit Float8e5m2(Float8e4m3fnuz f8) : Float8e5m2(ConvertFrom(f8)) {}
  explicit Float8e5m2(Float8e4m3b11fnuz f8) : Float8e5m2(ConvertFrom(f8)) {}
  explicit Float8e5m2(Float8e5m2fnuz& f8) : Float8e5m2(ConvertFrom(f8)) {}
};

class Float8e5m2fnuz : public Float8Base<Float8e5m2fnuz> {
  // 8-bit floating point with 2 bit mantissa.
  //
  // An 8-bit floating point type with 1 sign bit, 5 bits exponent and 2 bits
  // mantissa. The suffix "fnuz" is consistent with LLVM/MLIR naming and is
  // derived from the differences to IEEE floating point conventions. `F` is
  // for "finite" (no infinities), `N` for with special NaN encoding, `UZ` for
  // unsigned zero.
  //
  // This type has the following characteristics:
  // * bit encoding: S1E5M2 - `0bSEEEEEMM`
  // * exponent bias: 16
  // * infinities: Not supported
  // * NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits
  // set to all 0s - `0b10000000`
  // * denormals when exponent is 0
 private:
  using Base = Float8Base<Float8e5m2fnuz>;
  friend class Float8Base<Float8e5m2fnuz>;
  using Base::Float8Base;

 public:
  explicit Float8e5m2fnuz(const Float8e5m2& f8)
      : Float8e5m2fnuz(ConvertFrom(f8)) {}
  explicit Float8e5m2fnuz(const Float8e4m3b11fnuz& f8)
      : Float8e5m2fnuz(ConvertFrom(f8)) {}
  explicit Float8e5m2fnuz(const Float8e4m3fn& f8)
      : Float8e5m2fnuz(ConvertFrom(f8)) {}
  explicit Float8e5m2fnuz(const Float8e4m3fnuz& f8)
      : Float8e5m2fnuz(ConvertFrom(f8)) {}

  constexpr Float8e5m2fnuz operator-() const {
    if ((rep() & 0x7f) == 0x00) {
      return *this;
    }
    return Base::operator-();
  }

  Float8e5m2fnuz operator-(const Float8e5m2fnuz& other) const {
    return Base::operator-(other);
  }

  explicit operator bool() const { return rep() != 0; }
};

constexpr double ConstexprAbs(double x) { return x < 0.0 ? -x : x; }

constexpr double ConstexprCeil(double x) {
  constexpr double kIntegerThreshold =
      uint64_t{1} << (std::numeric_limits<double>::digits - 1);
  // Too big or NaN inputs get returned unchanged.
  if (!(ConstexprAbs(x) < kIntegerThreshold)) {
    return x;
  }
  const double x_trunc = static_cast<double>(static_cast<int64_t>(x));
  return x_trunc < x ? x_trunc + 1.0 : x_trunc;
}

constexpr double ConstexprFloor(double x) { return -ConstexprCeil(-x); }

constexpr double kLog10Of2 = 0.3010299956639812;
// C17 5.2.4.2.2p11:
// "number of decimal digits, q, such that any floating-point number with q
// decimal digits can be rounded into a floating-point number with p radix b
// digits and back again without change to the q decimal digits"
// floor((p - 1) * log10(2));
constexpr int Digits10FromDigits(int digits) {
  return static_cast<int>(ConstexprFloor((digits - 1) * kLog10Of2));
}

// C17 5.2.4.2.2p11:
// "number of decimal digits, n, such that any floating-point number with p
// radix b digits can be rounded to a floating-point number with n decimal
// digits and back again without change to the value"
// ceil(1 + p * log10(2));
constexpr int MaxDigits10FromDigits(int digits) {
  return static_cast<int>(ConstexprCeil(1.0 + (digits * kLog10Of2)));
}

// C17 5.2.4.2.2p11:
// "minimum negative integer such that 10 raised to that power is in the range
// of normalized floating-point numbers"
// ceil(log10(2**(emin - 1))) == ceil((emin - 1) * log10(2));
constexpr int MinExponent10FromMinExponent(int min_exponent) {
  return static_cast<int>(ConstexprCeil((min_exponent - 1) * kLog10Of2));
}

// C17 5.2.4.2.2p11:
// "maximum integer such that 10 raised to that power is in the range of
// representable finite floating-point numbers"
// floor(log10((1 - 2**-p) * 2**emax)) == floor(log10(1 - 2**-p) +
// emax * log10(2))
constexpr int MaxExponent10FromMaxExponentAndDigits(int max_exponent,
                                                    int digits) {
  // We only support digits in {3,4}. This table would grow if we wanted to
  // handle more values.
  constexpr double kLog10OfOnePredecessor[] = {
      // log10(1 - 2**-3)
      -0.057991946977686754,
      // log10(1 - 2**-4)
      -0.028028723600243537,
  };
  return static_cast<int>(ConstexprFloor(kLog10OfOnePredecessor[digits - 3] +
                                         max_exponent * kLog10Of2));
}

// Structures for use in specializing std::numeric_limits.
struct numeric_limits_float8_base {
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const bool is_specialized = true;
  static inline constexpr const bool is_signed = true;
  static inline constexpr const bool is_integer = false;
  static inline constexpr const bool is_exact = false;
  static inline constexpr const bool has_quiet_NaN = true;
  static inline constexpr const std::float_denorm_style has_denorm =
      std::denorm_present;
  static inline constexpr const bool has_denorm_loss = false;
  static inline constexpr const std::float_round_style round_style =
      std::round_to_nearest;
  static inline constexpr const bool is_bounded = true;
  static inline constexpr const bool is_modulo = false;
  static inline constexpr const int radix = std::numeric_limits<float>::radix;
  static inline constexpr const bool traps = std::numeric_limits<float>::traps;
  static inline constexpr const bool tinyness_before =
      std::numeric_limits<float>::tinyness_before;
  // NOLINTEND
};

struct numeric_limits_float8_e4m3fn : public numeric_limits_float8_base {
 private:
  static inline constexpr const int kExponentBias = 7;
  static inline constexpr const int kMantissaBits = 3;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = kMantissaBits + 1;
  static inline constexpr const int digits10 = Digits10FromDigits(digits);
  static inline constexpr const int max_digits10 =
      MaxDigits10FromDigits(digits);
  static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent =
      (0b1111 - 7) + 1;  // Extended format.
  static inline constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_signaling_NaN = false;
  // NOLINTEND

  // 1.0 * 2^(0b0001 - 7) = 1.0 * 2^-6 = 0.015625
  static constexpr Float8e4m3fn min() {
    return Float8e4m3fn::FromRep(0b0'0001 << kMantissaBits);
  }
  // -(1 + 0b110 * 2^-3) * 2^(0b1111 - 7) = -1.75 * 2^8 = 448
  static constexpr Float8e4m3fn lowest() {
    return Float8e4m3fn::FromRep(0b1'1111'110);
  }
  // (1 + 0b110 * 2^-3) * 2**(0b1111 - 7) = 1.75 * 2^8 = 448
  static constexpr Float8e4m3fn max() {
    return Float8e4m3fn::FromRep(0b0'1111'110);
  }
  // 1.0 * 2^-3 = 0.125
  static constexpr Float8e4m3fn epsilon() {
    return Float8e4m3fn::FromRep((-kMantissaBits + kExponentBias)
                                 << kMantissaBits);
  }
  // 1.0 * 2^-1 = 0.5
  static constexpr Float8e4m3fn round_error() {
    return Float8e4m3fn::FromRep((-1 + kExponentBias) << kMantissaBits);
  }
  static constexpr Float8e4m3fn infinity() {
    return Float8e4m3fn::FromRep(0b0'1111'111);
  }
  // NaN.
  static constexpr Float8e4m3fn quiet_NaN() {
    return Float8e4m3fn::FromRep(0b0'1111'111);
  }
  static constexpr Float8e4m3fn signaling_NaN() {
    return Float8e4m3fn::FromRep(0b0'1111'111);
  }
  // 1.0 * 2^(-7 - 3 + 1) = 1.0 * 2^-9 = 0.001953125
  static constexpr Float8e4m3fn denorm_min() {
    return Float8e4m3fn::FromRep(0b0'0000'001);
  }
};

struct numeric_limits_float8_e4m3b11fnuz : public numeric_limits_float8_base {
 private:
  static inline constexpr const int kExponentBias = 11;
  static inline constexpr const int kMantissaBits = 3;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = kMantissaBits + 1;
  static inline constexpr const int digits10 = Digits10FromDigits(digits);
  static inline constexpr const int max_digits10 =
      MaxDigits10FromDigits(digits);
  static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent =
      (0b1111 - kExponentBias) + 1;  // Extended format.
  static inline constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_signaling_NaN = false;
  // NOLINTEND

  // 1.0 * 2^(0b0001 - 11) = 1.0 * 2^-10 = 0.0009765625
  static constexpr Float8e4m3b11fnuz min() {
    return Float8e4m3b11fnuz::FromRep(1 << kMantissaBits);
  }
  // -(1 + 0b111 * 2^-3) * 2^(0b1111 - 11) = -1.875 * 2^4 = -30
  static constexpr Float8e4m3b11fnuz lowest() {
    return Float8e4m3b11fnuz::FromRep(0b1'1111'111);
  }
  // (1 + 0b111 * 2^-3) * 2^(0b1111 - 11) = 1.875 * 2^4 = 30
  static constexpr Float8e4m3b11fnuz max() {
    return Float8e4m3b11fnuz::FromRep(0b0'1111'111);
  }
  // 1.0 * 2^-3 = 0.125
  static constexpr Float8e4m3b11fnuz epsilon() {
    return Float8e4m3b11fnuz::FromRep((-kMantissaBits + kExponentBias)
                                      << kMantissaBits);
  }
  // 1.0 * 2^-1 = 0.5
  static constexpr Float8e4m3b11fnuz round_error() {
    return Float8e4m3b11fnuz::FromRep((-1 + kExponentBias) << kMantissaBits);
  }
  static constexpr Float8e4m3b11fnuz infinity() {
    return Float8e4m3b11fnuz::FromRep(0b1'0000'000);
  }
  // NaN.
  static constexpr Float8e4m3b11fnuz quiet_NaN() {
    return Float8e4m3b11fnuz::FromRep(0b1'0000'000);
  }
  static constexpr Float8e4m3b11fnuz signaling_NaN() {
    return Float8e4m3b11fnuz::FromRep(0b1'0000'000);
  }
  // 1.0 * 2^(-11 - 3 + 1) = 1.0 * 2^-13 = 0.0001220703125
  static constexpr Float8e4m3b11fnuz denorm_min() {
    return Float8e4m3b11fnuz::FromRep(0b0'0000'001);
  }
};

struct numeric_limits_float8_e4m3fnuz : public numeric_limits_float8_base {
 private:
  static inline constexpr const int kExponentBias = 8;
  static inline constexpr const int kMantissaBits = 3;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = kMantissaBits + 1;
  static inline constexpr const int digits10 = Digits10FromDigits(digits);
  static inline constexpr const int max_digits10 =
      MaxDigits10FromDigits(digits);
  static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent =
      (0b1111 - kExponentBias) + 1;  // Extended format.
  static inline constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_signaling_NaN = false;
  // NOLINTEND

  static constexpr Float8e4m3fnuz min() {
    return Float8e4m3fnuz::FromRep(0x08);
  }
  static constexpr Float8e4m3fnuz lowest() {
    return Float8e4m3fnuz::FromRep(0xFF);
  }
  static constexpr Float8e4m3fnuz max() {
    return Float8e4m3fnuz::FromRep(0x7F);
  }
  static constexpr Float8e4m3fnuz epsilon() {
    return Float8e4m3fnuz::FromRep((-kMantissaBits + kExponentBias)
                                   << kMantissaBits);
  }
  static constexpr Float8e4m3fnuz round_error() {
    return Float8e4m3fnuz::FromRep((-1 + kExponentBias) << kMantissaBits);
  }
  static constexpr Float8e4m3fnuz infinity() {
    return Float8e4m3fnuz::FromRep(0x80);
  }  // NaN.
  static constexpr Float8e4m3fnuz quiet_NaN() {
    return Float8e4m3fnuz::FromRep(0x80);
  }
  static constexpr Float8e4m3fnuz signaling_NaN() {
    return Float8e4m3fnuz::FromRep(0x80);
  }
  static constexpr Float8e4m3fnuz denorm_min() {
    return Float8e4m3fnuz::FromRep(0x01);
  }
};

struct numeric_limits_float8_e5m2 : public numeric_limits_float8_base {
 private:
  static inline constexpr const int kExponentBias = 15;
  static inline constexpr const int kMantissaBits = 2;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = kMantissaBits + 1;
  static inline constexpr const int digits10 = Digits10FromDigits(digits);
  static inline constexpr const int max_digits10 =
      MaxDigits10FromDigits(digits);
  static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent = 0b11111 - kExponentBias;
  static inline constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static inline constexpr const bool is_iec559 = true;
  static inline constexpr const bool has_infinity = true;
  static inline constexpr const bool has_signaling_NaN = true;
  // NOLINTEND

  // 1.0 * 2^(0b00001 - 15) = 1.0 * 2^-14 = 0.00006103515625
  static constexpr Float8e5m2 min() {
    return Float8e5m2::FromRep(1 << kMantissaBits);
  }
  // -(1 + 0b11 * 2^-2) * 2^(0b11110 - 15) = -1.75 * 2^15 = -57344
  static constexpr Float8e5m2 lowest() {
    return Float8e5m2::FromRep(0b1'11110'11);
  }
  // (1 + 0b11 * 2^-2) * 2^(0b11110 - 15) = 1.75 * 2^15 = 57344
  static constexpr Float8e5m2 max() {
    return Float8e5m2::FromRep(0b0'11110'11);
  }
  // 1.0 * 2^-2 = 0.25
  static constexpr Float8e5m2 epsilon() {
    return Float8e5m2::FromRep((-kMantissaBits + kExponentBias)
                               << kMantissaBits);
  }
  // 1.0 * 2^-1 = 0.5
  static constexpr Float8e5m2 round_error() {
    return Float8e5m2::FromRep((-1 + kExponentBias) << kMantissaBits);
  }
  static constexpr Float8e5m2 infinity() {
    return Float8e5m2::FromRep(0b0'11111'00);
  }
  static constexpr Float8e5m2 quiet_NaN() {
    // IEEE 754-2019 6.2.1: "All binary NaN bit strings have the sign bit S set
    // to 0 or 1 and all the bits of the biased exponent field E set to 1
    // (see 3.4). A quiet NaN bit string should be encoded with the first bit
    // (d1) of the trailing significand field T being 1."
    return Float8e5m2::FromRep(0b0'11111'10);
  }
  static constexpr Float8e5m2 signaling_NaN() {
    // IEEE 754-2019 6.2.1: "A signaling NaN bit string should be encoded with
    // the first bit of the trailing significand field being 0."
    return Float8e5m2::FromRep(0b0'11111'01);
  }
  // 1.0 * 2^(-15 - 2 + 1) = 1.0 * 2^-16 = 0.0000152587890625
  static constexpr Float8e5m2 denorm_min() {
    return Float8e5m2::FromRep(0b0'00000'01);
  }
};

struct numeric_limits_float8_e5m2fnuz : public numeric_limits_float8_base {
 private:
  static inline constexpr const int kExponentBias = 16;
  static inline constexpr const int kMantissaBits = 2;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = kMantissaBits + 1;
  static inline constexpr const int digits10 = Digits10FromDigits(digits);
  static inline constexpr const int max_digits10 =
      MaxDigits10FromDigits(digits);
  static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent =
      (0b11111 - kExponentBias) + 1;
  static inline constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_signaling_NaN = false;
  // NOLINTEND

  static constexpr Float8e5m2fnuz min() {
    return Float8e5m2fnuz::FromRep(0x04);
  }
  static constexpr Float8e5m2fnuz lowest() {
    return Float8e5m2fnuz::FromRep(0xFF);
  }
  static constexpr Float8e5m2fnuz max() {
    return Float8e5m2fnuz::FromRep(0x7F);
  }
  static constexpr Float8e5m2fnuz epsilon() {
    return Float8e5m2fnuz::FromRep((-kMantissaBits + kExponentBias)
                                   << kMantissaBits);
  }
  static constexpr Float8e5m2fnuz round_error() {
    return Float8e5m2fnuz::FromRep((-1 + kExponentBias) << kMantissaBits);
  }
  static constexpr Float8e5m2fnuz infinity() {
    return Float8e5m2fnuz::FromRep(0x80);
  }  // NaN.
  static constexpr Float8e5m2fnuz quiet_NaN() {
    return Float8e5m2fnuz::FromRep(0x80);
  }
  static constexpr Float8e5m2fnuz signaling_NaN() {
    return Float8e5m2fnuz::FromRep(0x80);
  }
  static constexpr Float8e5m2fnuz denorm_min() {
    return Float8e5m2fnuz::FromRep(0x01);
  }
};

}  // namespace float8_internal
}  // namespace tensorstore

namespace std {
// Standard-library overrides.
template <>
struct numeric_limits<tensorstore::float8_internal::Float8e4m3fn>
    : public tensorstore::float8_internal::numeric_limits_float8_e4m3fn {};

template <>
struct numeric_limits<tensorstore::float8_internal::Float8e4m3b11fnuz>
    : public tensorstore::float8_internal::numeric_limits_float8_e4m3b11fnuz {};

template <>
struct numeric_limits<tensorstore::float8_internal::Float8e4m3fnuz>
    : public tensorstore::float8_internal::numeric_limits_float8_e4m3fnuz {};

template <>
struct numeric_limits<tensorstore::float8_internal::Float8e5m2>
    : public tensorstore::float8_internal::numeric_limits_float8_e5m2 {};

template <>
struct numeric_limits<tensorstore::float8_internal::Float8e5m2fnuz>
    : public tensorstore::float8_internal::numeric_limits_float8_e5m2fnuz {};

}  // namespace std

namespace tensorstore {
namespace float8_internal {

constexpr inline Float8e4m3fn abs(const Float8e4m3fn& a) {
  return Float8e4m3fn::FromRep(a.rep() & 0b0'1111'111);
}

constexpr inline bool(isnan)(const Float8e4m3fn& a) {
  return abs(a).rep() == std::numeric_limits<Float8e4m3fn>::quiet_NaN().rep();
}

constexpr inline Float8e4m3b11fnuz abs(const Float8e4m3b11fnuz& a) {
  return (a.rep() & 0b0'1111'111) == 0
             ? Float8e4m3b11fnuz::FromRep(a.rep())
             : Float8e4m3b11fnuz::FromRep(a.rep() & 0b0'1111'111);
}

constexpr inline bool(isnan)(const Float8e4m3b11fnuz& a) {
  return a.rep() == std::numeric_limits<Float8e4m3b11fnuz>::quiet_NaN().rep();
}

constexpr inline Float8e4m3fnuz abs(const Float8e4m3fnuz& a) {
  return (a.rep() & 0x7F) == 0 ? Float8e4m3fnuz::FromRep(a.rep())
                               : Float8e4m3fnuz::FromRep(a.rep() & 0x7F);
}

constexpr inline bool(isnan)(const Float8e4m3fnuz& a) {
  return abs(a).rep() == std::numeric_limits<Float8e4m3fnuz>::quiet_NaN().rep();
}

constexpr inline Float8e5m2 abs(const Float8e5m2& a) {
  return Float8e5m2::FromRep(a.rep() & 0b0'11111'11);
}

constexpr inline bool(isnan)(const Float8e5m2& a) {
  return abs(a).rep() > std::numeric_limits<Float8e5m2>::infinity().rep();
}

constexpr inline Float8e5m2fnuz abs(const Float8e5m2fnuz& a) {
  return (a.rep() & 0x7F) == 0 ? Float8e5m2fnuz::FromRep(a.rep())
                               : Float8e5m2fnuz::FromRep(a.rep() & 0x7F);
}

constexpr inline bool isnan(const Float8e5m2fnuz& a) { return a.rep() == 0x80; }

template <typename Float8>
constexpr inline bool(isinf)(const Float8Base<Float8>& a) {
  return std::numeric_limits<Float8>::has_infinity
             ? abs(a.derived()).rep() ==
                   std::numeric_limits<Float8>::infinity().rep()
             : false;  // No inf representation.
}

template <typename Float8>
constexpr inline bool(isfinite)(const Float8Base<Float8>& a) {
  return !isnan(a.derived()) && !isinf(a.derived());
}

template <typename Float8>
std::ostream& operator<<(std::ostream& os, const Float8Base<Float8>& f8) {
  os << static_cast<float>(f8.derived());
  return os;
}

//==============================================================================
// Inline conversion routines between float8 and other types.
//==============================================================================

template <size_t Size>
struct get_integer_by_size {
  typedef void signed_type;
  typedef void unsigned_type;
};
template <>
struct get_integer_by_size<1> {
  typedef int8_t signed_type;
  typedef uint8_t unsigned_type;
};
template <>
struct get_integer_by_size<2> {
  typedef int16_t signed_type;
  typedef uint16_t unsigned_type;
};
template <>
struct get_integer_by_size<4> {
  typedef int32_t signed_type;
  typedef uint32_t unsigned_type;
};
template <>
struct get_integer_by_size<8> {
  typedef int64_t signed_type;
  typedef uint64_t unsigned_type;
};

// Helper for getting a bit representation provided a byte size.
template <int kNumBytes>
using GetUnsignedInteger =
    typename get_integer_by_size<kNumBytes>::unsigned_type;

// Converts between two floating-point types.
template <typename From, typename To, bool kSaturate, bool kTruncate,
          typename EnableIf = void>
struct ConvertImpl;

// Convert to same type.  We need explicit specializations for all combinations
// of template parameters to avoid ambiguities.
template <typename Scalar>
struct IdentityConversion {
  static inline Scalar run(const Scalar& from) { return from; }
};

template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/false, /*kTruncate=*/false,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};
template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/false, /*kTruncate=*/true,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};
template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/true, /*kTruncate=*/false,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};
template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/true, /*kTruncate=*/true,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};

template <typename Float>
struct TraitsBase {
  using BitsType = GetUnsignedInteger<sizeof(Float)>;
  static constexpr int kBits = sizeof(Float) * CHAR_BIT;
  static constexpr int kMantissaBits = std::numeric_limits<Float>::digits - 1;
  static constexpr int kExponentBits = kBits - kMantissaBits - 1;
  static constexpr BitsType kExponentMask = ((BitsType{1} << kExponentBits) - 1)
                                            << kMantissaBits;
  static constexpr BitsType kMantissaMask = (BitsType{1} << kMantissaBits) - 1;
  static constexpr int kExponentBias = (1 << (kExponentBits - 1)) - 1;
};

template <typename Float>
struct Traits : public TraitsBase<Float> {};

template <>
struct Traits<Float8e4m3b11fnuz> : public TraitsBase<Float8e4m3b11fnuz> {
  static constexpr int kExponentBias = 11;
};

template <>
struct Traits<Float8e4m3fnuz> : public TraitsBase<Float8e4m3fnuz> {
  using Base = TraitsBase<Float8e4m3fnuz>;
  static constexpr int kExponentBias = Base::kExponentBias + 1;
};

template <>
struct Traits<Float8e5m2fnuz> : public TraitsBase<Float8e5m2fnuz> {
  using Base = TraitsBase<Float8e5m2fnuz>;
  static constexpr int kExponentBias = Base::kExponentBias + 1;
};

template <typename Bits>
constexpr inline Bits RoundBitsToNearestEven(Bits bits, int roundoff) {
  // Round to nearest even by adding a bias term.
  // Consider a bit pattern
  //   FFF...FLRTT...T,
  // where bits RTT...T need to be rounded-off.  We add a bias term to the
  // bit pattern s.t. a carry is introduced to round up only if
  // - L is 1, R is 1, OR
  // - L is 0, R is 1, any T is one.
  // We do this by adding L to a bit pattern consisting of all T = 1.
  Bits bias = roundoff == 0
                  ? 0
                  : ((bits >> roundoff) & 1) + (Bits{1} << (roundoff - 1)) - 1;
  return bits + bias;
}

#if (defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L)
using std::countl_zero;
#else
static constexpr inline int countl_zero(uint64_t x) {
  int zeroes = 60;
  if (x >> 32) {
    zeroes -= 32;
    x >>= 32;
  }
  if (x >> 16) {
    zeroes -= 16;
    x >>= 16;
  }
  if (x >> 8) {
    zeroes -= 8;
    x >>= 8;
  }
  if (x >> 4) {
    zeroes -= 4;
    x >>= 4;
  }
  return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[x] + zeroes;
}
static constexpr inline int countl_zero(uint32_t x) {
  int zeroes = 28;
  if (x >> 16) {
    zeroes -= 16;
    x >>= 16;
  }
  if (x >> 8) {
    zeroes -= 8;
    x >>= 8;
  }
  if (x >> 4) {
    zeroes -= 4;
    x >>= 4;
  }
  return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[x] + zeroes;
}
static constexpr inline int countl_zero(uint16_t x) {
  int zeroes = 12;
  if (x >> 8) {
    zeroes -= 8;
    x >>= 8;
  }
  if (x >> 4) {
    zeroes -= 4;
    x >>= 4;
  }
  return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[x] + zeroes;
}
static constexpr inline int countl_zero(uint8_t x) {
  int zeroes = 4;
  if (x >> 4) {
    zeroes -= 4;
    x >>= 4;
  }
  return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[x] + zeroes;
}
#endif

template <typename From, typename To, bool kSaturate, bool kTruncate>
struct ConvertImpl<From, To, kSaturate, kTruncate,
                   std::enable_if_t<!std::is_same_v<From, To>>> {
  using FromTraits = Traits<From>;
  using FromBits = typename FromTraits::BitsType;
  static constexpr int kFromBits = FromTraits::kBits;
  static constexpr int kFromMantissaBits = FromTraits::kMantissaBits;
  static constexpr int kFromExponentBits = FromTraits::kExponentBits;
  static constexpr int kFromExponentBias = FromTraits::kExponentBias;
  static constexpr FromBits kFromExponentMask = FromTraits::kExponentMask;

  using ToTraits = Traits<To>;
  using ToBits = typename ToTraits::BitsType;
  static constexpr int kToBits = ToTraits::kBits;
  static constexpr int kToMantissaBits = ToTraits::kMantissaBits;
  static constexpr int kToExponentBits = ToTraits::kExponentBits;
  static constexpr int kToExponentBias = ToTraits::kExponentBias;
  static constexpr ToBits kToExponentMask = ToTraits::kExponentMask;

  // `WideBits` is wide enough to accommodate the largest exponent and mantissa
  // in either `From` or `To`.
  static constexpr int kWideBits =
      (std::max(kToMantissaBits, kFromMantissaBits)) +  // Max significand.
      (std::max(kToExponentBits, kFromExponentBits));   // Max exponent.
  static constexpr int kWideBytes = (kWideBits + (CHAR_BIT - 1)) / CHAR_BIT;
  using WideBits = GetUnsignedInteger<kWideBytes>;
  static constexpr int kExponentOffset = kToExponentBias - kFromExponentBias;
  static constexpr int kDigitShift = kToMantissaBits - kFromMantissaBits;

  static inline To run(const From& from) {
    using std::abs;
    using std::isinf;  // NOLINT
    using std::isnan;  // NOLINT

    // Shift bits to destination type, without sign bit.
    const bool from_sign_bit =
        absl::bit_cast<FromBits>(from) >> (kFromBits - 1);
    const FromBits from_bits = absl::bit_cast<FromBits>(abs(from));

    // Special values, preserving sign.
    if (isinf(from)) {
      return from_sign_bit ? -std::numeric_limits<To>::infinity()
                           : std::numeric_limits<To>::infinity();
    }
    if (isnan(from)) {
      return from_sign_bit ? -std::numeric_limits<To>::quiet_NaN()
                           : std::numeric_limits<To>::quiet_NaN();
    }
    if (from_bits == 0) {
      return from_sign_bit ? -To{} : To{};
    }

    const int biased_from_exponent = from_bits >> kFromMantissaBits;

    // `To` supports more exponents near zero which means that some subnormal
    // values in `From` may become normal.
    if constexpr (std::numeric_limits<To>::min_exponent <
                  std::numeric_limits<From>::min_exponent) {
      if (biased_from_exponent == 0) {
        // Subnormals.
        WideBits bits = from_bits;

        // Determine exponent in target type.
        const int normalization_factor =
            countl_zero(from_bits) - (kFromBits - kFromMantissaBits) + 1;
        const int biased_exponent = kExponentOffset - normalization_factor + 1;
        if (biased_exponent <= 0) {
          // Result is subnormal.  Adjust the subnormal bits to account for
          // the difference in exponent bias.
          if constexpr (kExponentOffset < sizeof(WideBits) * CHAR_BIT) {
            bits <<= kExponentOffset;
          }
        } else {
          // Result is normal. Shift the mantissa to account for the number of
          // leading zero digits, and clear the hidden bit.
          bits <<= normalization_factor;
          bits &= ~(WideBits{1} << kFromMantissaBits);
          // Insert the exponent bits.
          bits |= static_cast<WideBits>(biased_exponent) << kFromMantissaBits;
        }

        // Truncate/round mantissa if necessary.
        if constexpr (kDigitShift > 0) {
          bits <<= kDigitShift;
        } else {
          if constexpr (!kTruncate) {
            bits = RoundBitsToNearestEven(bits, -kDigitShift);
          }
          bits >>= -kDigitShift;
        }
        To to = absl::bit_cast<To>(static_cast<ToBits>(bits));
        return from_sign_bit ? -to : to;
      }
    }
    // `To` supports fewer exponents near zero which means that some values in
    // `From` may become subnormal.
    if constexpr (std::numeric_limits<To>::min_exponent >
                  std::numeric_limits<From>::min_exponent) {
      const int unbiased_exponent = biased_from_exponent - kFromExponentBias;
      const int biased_to_exponent = unbiased_exponent + kToExponentBias;
      // Subnormals and zero.
      if (biased_to_exponent <= 0) {
        // Round and shift mantissa down.
        FromBits from_has_leading_one = (biased_from_exponent > 0 ? 1 : 0);
        int exponent_shift =
            -kDigitShift - biased_to_exponent + from_has_leading_one;
        // Insert the implicit leading 1 bit on the mantissa for normalized
        // inputs.
        FromBits rounded_from_bits =
            (from_bits & FromTraits::kMantissaMask) |
            (from_has_leading_one << kFromMantissaBits);
        ToBits bits = 0;
        // To avoid UB, limit rounding and shifting to the full mantissa plus
        // leading 1.
        if (exponent_shift <= kFromMantissaBits + 1) {
          if constexpr (!kTruncate) {
            // NOTE: we need to round again from the original from_bits,
            // otherwise the lower precision bits may already be lost.  There is
            // an edge-case where rounding to a normalized value would normally
            // round down, but for a subnormal, we need to round up.
            rounded_from_bits =
                RoundBitsToNearestEven(rounded_from_bits, exponent_shift);
          }
          bits = (rounded_from_bits >> exponent_shift);
        }
        // Insert sign and return.
        To to = absl::bit_cast<To>(bits);
        return from_sign_bit ? -to : to;
      }
    }

    // Round the mantissa if it is shrinking.
    WideBits rounded_from_bits = from_bits;
    if constexpr (kDigitShift < 0) {
      if constexpr (!kTruncate) {
        rounded_from_bits = RoundBitsToNearestEven(from_bits, -kDigitShift);
      }
      // Zero-out tail bits.
      rounded_from_bits &= ~((WideBits{1} << (-kDigitShift)) - 1);
    }

    // Re-bias the exponent.
    rounded_from_bits += static_cast<WideBits>(kExponentOffset)
                         << kFromMantissaBits;

    ToBits bits;
    // Check for overflows by aligning the significands. We always align the
    // narrower significand to the wider significand.
    const WideBits kToHighestRep =
        absl::bit_cast<ToBits>(std::numeric_limits<To>::max());
    WideBits aligned_highest{kToHighestRep};
    if constexpr (kDigitShift < 0) {
      aligned_highest <<= -kDigitShift;
      // Shift down, all dropped bits should already be zero.
      bits = static_cast<ToBits>(rounded_from_bits >> -kDigitShift);
    } else if constexpr (kDigitShift >= 0) {
      // Shift up, inserting zeros in the newly created digits.
      rounded_from_bits <<= kDigitShift;
      bits = ToBits{rounded_from_bits};
    }

    To to = absl::bit_cast<To>(bits);
    // `From` supports larger values than `To`, we may overflow.
    if constexpr (std::make_pair(std::numeric_limits<To>::max_exponent,
                                 std::numeric_limits<To>::digits) <
                  std::make_pair(std::numeric_limits<From>::max_exponent,
                                 std::numeric_limits<From>::digits)) {
      if (rounded_from_bits > aligned_highest) {
        // Overflowed values map to highest or infinity depending on kSaturate.
        to = kSaturate ? std::numeric_limits<To>::max()
                       : std::numeric_limits<To>::infinity();
      }
    }
    // Insert sign bit.
    return from_sign_bit ? -to : to;
  }
};

// Saturation has no impact when casting e4m3 to e5m2.
template <bool kTruncate>
struct ConvertImpl<Float8e4m3fn, Float8e5m2, true, kTruncate> {
  static inline Float8e5m2 run(const Float8e4m3fn& from) {
    return ConvertImpl<Float8e4m3fn, Float8e5m2, false, kTruncate>::run(from);
  }
};

template <bool kSaturate, bool kTruncate>
struct ConvertImpl<::half_float::half, Float8e5m2, kSaturate, kTruncate> {
  static inline Float8e5m2 run(const ::half_float::half& from) {
    uint16_t from_bits = absl::bit_cast<uint16_t>(from);

    // Special values (Inf or NaN).
    uint16_t abs_bits = from_bits & 0x7FFF;
    if (abs_bits == 0x7C00) {
      return Float8e5m2::FromRep(from_bits >> 8);
    } else if (abs_bits > 0x7C00) {
      // IEEE 754-2019 6.2.1: "A quiet NaN bit string should be encoded with the
      // first bit (d1) of the trailing significand field T being 1."
      // IEEE 754-2019 6.2.3: "Conversion of a quiet NaN to a floating-point
      // format of the same or a different radix that does not allow the payload
      // to be preserved, shall return a quiet NaN [...]"
      return Float8e5m2::FromRep((from_bits >> 8) | 0b0'00000'10);
    }

    if constexpr (!kTruncate) {
      from_bits = RoundBitsToNearestEven(from_bits, 8);
      // Rounding can cause an overflow to infinity. Clamp to the largest finite
      // value if saturation is requested.
      if constexpr (kSaturate) {
        const Float8e5m2 kHighest = std::numeric_limits<Float8e5m2>::max();
        if ((from_bits & 0x7F00) > static_cast<uint16_t>(kHighest.rep()) << 8) {
          const bool from_sign_bit = from_bits >> 15;
          return from_sign_bit ? -kHighest : kHighest;
        }
      }
    }
    return Float8e5m2::FromRep(from_bits >> 8);
  }
};

template <>
struct ConvertImpl<Float8e5m2, ::half_float::half, /*kSaturate=*/false,
                   /*kTruncate=*/false> {
  static inline ::half_float::half run(const Float8e5m2& from) {
    return absl::bit_cast<::half_float::half>(
        static_cast<uint16_t>(static_cast<uint16_t>(from.rep()) << 8));
  }
};

// Direct casts of e5m2 to ::half_float::half simply shifts bits over.
template <bool kSaturate, bool kTruncate>
struct ConvertImpl<Float8e5m2, ::half_float::half, kSaturate, kTruncate> {
  static inline ::half_float::half run(const Float8e5m2& from) {
    return absl::bit_cast<::half_float::half>(
        static_cast<uint16_t>(static_cast<uint16_t>(from.rep()) << 8));
  }
};

template <bool kSaturate, bool kTruncate>
struct ConvertImpl<Float8e5m2fnuz, ::half_float::half, kSaturate, kTruncate> {
  static inline ::half_float::half run(const Float8e5m2fnuz& from) {
    return static_cast<::half_float::half>(static_cast<float>(from));
  }
};

template <typename Derived>
template <bool kSaturate, bool kTruncate, typename From>
Derived Float8Base<Derived>::ConvertFrom(const From& from) {
  return ConvertImpl<From, Derived, kSaturate, kTruncate>::run(from);
}

template <typename Derived>
template <typename To, bool kSaturate, bool kTruncate>
To Float8Base<Derived>::ConvertTo(const Derived& from) {
  return ConvertImpl<Derived, To, kSaturate, kTruncate>::run(from);
}

#ifdef _MSC_VER
#define TENSORSTORE_INTERNAL_FPCLASSIFY(Float8)                     \
  inline int fpclassify(Float8 a) noexcept {                        \
    if (tensorstore::float8_internal::isnan(a)) return FP_NAN;      \
    if (tensorstore::float8_internal::isinf(a)) return FP_INFINITE; \
    Float8 abs_value = tensorstore::float8_internal::abs(a);        \
    if (abs_value.rep() == 0x00) return FP_ZERO;                    \
    if ((abs_value.rep() & Traits<Float8>::kExponentMask) == 0)     \
      return FP_SUBNORMAL;                                          \
    return FP_NORMAL;                                               \
  }

TENSORSTORE_INTERNAL_FPCLASSIFY(Float8e4m3fn);
TENSORSTORE_INTERNAL_FPCLASSIFY(Float8e4m3fnuz);
TENSORSTORE_INTERNAL_FPCLASSIFY(Float8e4m3b11fnuz);
TENSORSTORE_INTERNAL_FPCLASSIFY(Float8e5m2);
TENSORSTORE_INTERNAL_FPCLASSIFY(Float8e5m2fnuz);
#undef TENSORSTORE_INTERNAL_FPCLASSIFY
#endif

}  // namespace float8_internal

// Exported types.
using Float8e4m3fn = float8_internal::Float8e4m3fn;
using Float8e4m3fnuz = float8_internal::Float8e4m3fnuz;
using Float8e4m3b11fnuz = float8_internal::Float8e4m3b11fnuz;
using Float8e5m2 = float8_internal::Float8e5m2;
using Float8e5m2fnuz = float8_internal::Float8e5m2fnuz;

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_FLOAT8_H_
