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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <ostream>
#include <type_traits>
#include <utility>

#include "absl/base/casts.h"
#include "absl/numeric/bits.h"
#include "absl/strings/str_format.h"
#include <half.hpp>
#include <nlohmann/json_fwd.hpp>
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

// Strongly suggest to the compiler that these functions are inlined.
#ifndef TENSORSTORE_FLOATINLINE
#if defined(_MSC_VER)
#define TENSORSTORE_FLOATINLINE __forceinline
#else
#define TENSORSTORE_FLOATINLINE inline
#endif
#endif  // TENSORSTORE_FLOATINLINE

namespace tensorstore {
namespace float8_internal {

// Forward-declarations of classes.
class Float8e3m4;
class Float8e4m3fn;
class Float8e4m3fnuz;
class Float8e4m3b11fnuz;
class Float8e5m2;
class Float8e5m2fnuz;
class Float8e8m0fnu;

template <typename Derived>
class Float8Base {
 protected:
  // Constructor tag to allow constexpr construction from bit representation.
  struct ConstructFromRepTag {};
  constexpr Float8Base(uint8_t rep, ConstructFromRepTag) : rep_{rep} {}

 public:
  static constexpr int kBits = 8;
  Float8Base() = default;

  template <typename T>
  constexpr explicit Float8Base(
      T i, std::enable_if_t<std::is_integral_v<T>, int> = 0)
      : Float8Base(ConvertFrom(i).rep(), ConstructFromRepTag{}) {}

  template <typename T>
  constexpr explicit Float8Base(
      T i,
      std::enable_if_t<!std::is_integral_v<T> && !std::is_floating_point_v<T>,
                       int> = 0)
      : Float8Base(ConvertFrom(static_cast<float>(i)).rep(),
                   ConstructFromRepTag{}) {}

  template <typename T>
  constexpr explicit Float8Base(
      T f, std::enable_if_t<std::is_floating_point_v<T>, int> = 0)
      : Float8Base(ConvertFrom(f).rep(), ConstructFromRepTag{}) {}

  constexpr explicit Float8Base(BFloat16 bf16)
      : Float8Base(ConvertFrom(bf16).rep(), ConstructFromRepTag{}) {}
  constexpr explicit Float8Base(::half_float::half f16)
      : Float8Base(ConvertFrom(f16).rep(), ConstructFromRepTag{}) {}

  constexpr uint8_t rep() const { return rep_; }

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  constexpr explicit operator T() const {
    return static_cast<T>(static_cast<float>(derived()));
  }
  constexpr explicit operator double() const {
    return ConvertTo<double>(derived());
  }
  constexpr explicit operator float() const {
    return ConvertTo<float>(derived());
  }
  constexpr explicit operator BFloat16() const {
    return ConvertTo<BFloat16>(derived());
  }
  constexpr explicit operator ::half_float::half() const {
    return ConvertTo<::half_float::half>(derived());
  }
  constexpr explicit operator bool() const { return (rep() & 0x7F) != 0; }

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
  static constexpr Derived ConvertFrom(From from);

  template <typename To, bool kSaturate = false, bool kTruncate = false>
  static constexpr To ConvertTo(const Derived& from);

  // Operators via float32.
  constexpr Derived operator+(const Derived& other) const {
    return Derived{float{derived()} + float{other}};
  }

  constexpr Derived operator-(const Derived& other) const {
    return Derived{float{derived()} - float{other}};
  }

  constexpr Derived operator*(const Derived& other) const {
    return Derived{float{derived()} * float{other}};
  }

  constexpr Derived operator/(const Derived& other) const {
    return Derived{float{derived()} / float{other}};
  }

  constexpr bool operator==(const Derived& other) const {
    return Compare(derived(), other) == Ordering::kEquivalent;
  }

  constexpr bool operator!=(const Derived& other) const {
    return Compare(derived(), other) != Ordering::kEquivalent;
  }

  TENSORSTORE_FLOATINLINE constexpr bool operator<(const Derived& other) const {
    return Compare(derived(), other) == Ordering::kLess;
  }

  TENSORSTORE_FLOATINLINE constexpr bool operator<=(
      const Derived& other) const {
    return Compare(derived(), other) <= Ordering::kEquivalent;
  }

  TENSORSTORE_FLOATINLINE constexpr bool operator>(const Derived& other) const {
    return Compare(derived(), other) == Ordering::kGreater;
  }

  TENSORSTORE_FLOATINLINE constexpr bool operator>=(
      const Derived& other) const {
    Ordering ordering = Compare(derived(), other);
    return ordering == Ordering::kGreater || ordering == Ordering::kEquivalent;
  }

  // Compound assignment.
  TENSORSTORE_FLOATINLINE constexpr Derived& operator+=(const Derived& other) {
    derived() = derived() + other;
    return derived();
  }

  TENSORSTORE_FLOATINLINE constexpr Derived& operator-=(const Derived& other) {
    derived() = derived() - other;
    return derived();
  }

  TENSORSTORE_FLOATINLINE constexpr Derived& operator*=(const Derived& other) {
    derived() = derived() * other;
    return derived();
  }

  TENSORSTORE_FLOATINLINE constexpr Derived& operator/=(const Derived& other) {
    derived() = derived() / other;
    return derived();
  }

  // for downsample_nditerable
  friend TENSORSTORE_FLOATINLINE constexpr float& operator+=(float& a,
                                                             Derived b) {
    return a += static_cast<float>(b);
  }
  friend TENSORSTORE_FLOATINLINE constexpr float& operator-=(float& a,
                                                             Derived b) {
    return a -= static_cast<float>(b);
  }
  friend TENSORSTORE_FLOATINLINE constexpr float& operator*=(float& a,
                                                             Derived b) {
    return a *= static_cast<float>(b);
  }
  friend TENSORSTORE_FLOATINLINE constexpr float& operator/=(float& a,
                                                             Derived b) {
    return a /= static_cast<float>(b);
  }

  // Note: Comparison operators do not need to be defined since they are
  // provided automatically by the implicit conversion to `float`.

  // Conversion to `::nlohmann::json`.
  template <template <typename U, typename V,
                      typename... Args> class ObjectType /* = std::map*/,
            template <typename U,
                      typename... Args> class ArrayType /* = std::vector*/,
            class StringType /*= std::string*/, class BooleanType /* = bool*/,
            class NumberIntegerType /* = int64_t*/,
            class NumberUnsignedType /* = uint64_t*/,
            class NumberFloatType /* = double*/,
            template <typename U> class AllocatorType /* = std::allocator*/,
            template <typename T,
                      typename SFINAE =
                          void> class JSONSerializer /* = adl_serializer*/,
            class BinaryType /* = std::vector<uint8_t>*/>
  friend void to_json(
      ::nlohmann::basic_json<ObjectType, ArrayType, StringType, BooleanType,
                             NumberIntegerType, NumberUnsignedType,
                             NumberFloatType, AllocatorType, JSONSerializer,
                             BinaryType>& j,
      Derived v) {
    j = static_cast<NumberFloatType>(v);
  }

 private:
  static TENSORSTORE_FLOATINLINE constexpr std::pair<uint8_t, uint8_t>
  SignAndMagnitude(Derived x) {
    const uint8_t x_abs_bits = absl::bit_cast<uint8_t>(abs(x));
    const uint8_t x_bits = absl::bit_cast<uint8_t>(x);
    const uint8_t x_sign = (x_bits ^ x_abs_bits) << (CHAR_BIT - Derived::kBits);
    return {x_sign, x_abs_bits};
  }
  static TENSORSTORE_FLOATINLINE constexpr int8_t
  SignAndMagnitudeToTwosComplement(uint8_t sign, uint8_t magnitude) {
    return magnitude ^ (static_cast<int8_t>(sign) < 0 ? -1 : 0);
  }

  enum Ordering : int8_t {
    kLess = -1,
    kEquivalent = 0,
    kGreater = 1,
    kUnordered = 2,
  };

  friend constexpr Ordering Compare(const Derived& lhs, const Derived& rhs) {
    if (isnan(lhs) || isnan(rhs)) {
      return Ordering::kUnordered;
    }
    if constexpr (!std::numeric_limits<Derived>::is_signed) {
      return lhs.rep() < rhs.rep()   ? Ordering::kLess
             : lhs.rep() > rhs.rep() ? Ordering::kGreater
                                     : Ordering::kEquivalent;
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

template <typename T>
using RequiresIsDerivedFromFloat8Base =
    std::enable_if_t<std::is_base_of_v<Float8Base<T>, T>, int>;

class Float8e3m4 : public Float8Base<Float8e3m4> {
  // Exponent: 3, Mantissa: 4, bias: 3.
  // IEEE 754.
 private:
  using Base = Float8Base<Float8e3m4>;
  friend class Float8Base<Float8e3m4>;
  using Base::Float8Base;

 public:
  template <typename T, RequiresIsDerivedFromFloat8Base<T> = 0>
  constexpr explicit Float8e3m4(T f8) : Float8e3m4(ConvertFrom(f8)) {}
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
  template <typename T, RequiresIsDerivedFromFloat8Base<T> = 0>
  constexpr explicit Float8e4m3fn(T f8) : Float8e4m3fn(ConvertFrom(f8)) {}
};

class Float8e4m3b11fnuz : public Float8Base<Float8e4m3b11fnuz> {
  // Exponent: 4, Mantissa: 3, bias: 11.
  // Extended range: no inf, NaN represented by 0b1000'0000.
 private:
  using Base = Float8Base<Float8e4m3b11fnuz>;
  friend class Float8Base<Float8e4m3b11fnuz>;
  using Base::Float8Base;

 public:
  template <typename T, RequiresIsDerivedFromFloat8Base<T> = 0>
  constexpr explicit Float8e4m3b11fnuz(T f8)
      : Float8e4m3b11fnuz(ConvertFrom(f8)) {}

  using Base::operator-;

  constexpr Float8e4m3b11fnuz operator-() const {
    if ((rep() & 0x7f) == 0x00) {
      return *this;
    }
    return Base::operator-();
  }

  constexpr explicit operator bool() const { return rep() != 0; }
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
  template <typename T, RequiresIsDerivedFromFloat8Base<T> = 0>
  constexpr explicit Float8e4m3fnuz(T f8) : Float8e4m3fnuz(ConvertFrom(f8)) {}

  using Base::operator-;

  constexpr Float8e4m3fnuz operator-() const {
    if ((rep() & 0x7f) == 0x00) {
      return *this;
    }
    return Base::operator-();
  }

  constexpr explicit operator bool() const { return rep() != 0; }
};

class Float8e5m2 : public Float8Base<Float8e5m2> {
  // Exponent: 5, Mantissa: 2, bias: 15.
  // IEEE 754.
 private:
  using Base = Float8Base<Float8e5m2>;
  friend class Float8Base<Float8e5m2>;
  using Base::Float8Base;

 public:
  template <typename T, RequiresIsDerivedFromFloat8Base<T> = 0>
  constexpr explicit Float8e5m2(T f8) : Float8e5m2(ConvertFrom(f8)) {}
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
  template <typename T, RequiresIsDerivedFromFloat8Base<T> = 0>
  constexpr explicit Float8e5m2fnuz(T f8) : Float8e5m2fnuz(ConvertFrom(f8)) {}

  using Base::operator-;

  constexpr Float8e5m2fnuz operator-() const {
    if ((rep() & 0x7f) == 0x00) {
      return *this;
    }
    return Base::operator-();
  }

  constexpr explicit operator bool() const { return rep() != 0; }
};

class Float8e8m0fnu : public Float8Base<Float8e8m0fnu> {
  // Exponent: 8, Mantissa: 0, bias: 127.
  // No inf, no zero, NaN represented by 0xFF.
 private:
  using Base = Float8Base<Float8e8m0fnu>;
  friend class Float8Base<Float8e8m0fnu>;
  using Base::Float8Base;

 public:
  template <typename T, RequiresIsDerivedFromFloat8Base<T> = 0>
  constexpr explicit Float8e8m0fnu(T f8) : Float8e8m0fnu(ConvertFrom(f8)) {}

  using Base::operator-;

  constexpr Float8e8m0fnu operator-() const {
    // No negative numbers supported in E8M0 => NaN
    return Float8e8m0fnu::FromRep(0xFF);
  }

  constexpr explicit operator bool() const {
    // No zero supported in E8M0 format.
    return true;
  }
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
  // We only support digits in {1,2,3,4,5}. This table would grow if we wanted
  // to handle more values.
  constexpr double kLog10OfOnePredecessor[] = {
      // log10(1 - 2**-1)
      -0.3010299956639812,
      // log10(1 - 2**-2)
      -0.12493873660829993,
      // log10(1 - 2**-3)
      -0.057991946977686754,
      // log10(1 - 2**-4)
      -0.028028723600243537,
      // log10(1 - 2**-5)
      -0.013788284485633295,
  };
  return static_cast<int>(ConstexprFloor(kLog10OfOnePredecessor[digits - 1] +
                                         max_exponent * kLog10Of2));
}

// Structures for use in specializing std::numeric_limits.
struct numeric_limits_float8_base {
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static constexpr const bool is_specialized = true;
  static constexpr const bool is_signed = true;
  static constexpr const bool is_integer = false;
  static constexpr const bool is_exact = false;
  static constexpr const bool has_quiet_NaN = true;
// has_denorm and has_denorm_loss are deprecated in C++23.
#if !defined(__cplusplus) || __cplusplus < 202302L
  static constexpr const std::float_denorm_style has_denorm =
      std::denorm_present;
  static constexpr const bool has_denorm_loss = false;
#endif
  static constexpr const std::float_round_style round_style =
      std::round_to_nearest;
  static constexpr const bool is_bounded = true;
  static constexpr const bool is_modulo = false;
  static constexpr const int radix = std::numeric_limits<float>::radix;
  static constexpr const bool traps = std::numeric_limits<float>::traps;
  static constexpr const bool tinyness_before =
      std::numeric_limits<float>::tinyness_before;
  // NOLINTEND
};

struct numeric_limits_float8_e3m4 : public numeric_limits_float8_base {
 private:
  static constexpr const int kExponentBias = 3;
  static constexpr const int kMantissaBits = 4;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static constexpr const int digits = kMantissaBits + 1;
  static constexpr const int digits10 = Digits10FromDigits(digits);
  static constexpr const int max_digits10 = MaxDigits10FromDigits(digits);
  static constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static constexpr const int max_exponent = 0b111 - kExponentBias;
  static constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static constexpr const bool is_iec559 = true;
  static constexpr const bool has_infinity = true;
  static constexpr const bool has_signaling_NaN = true;
  // NOLINTEND

  // 1.0 * 2^(0b001 - 3) = 1.0 * 2^-2 = 1/4 (min normal)
  static constexpr Float8e3m4 min() {
    return Float8e3m4::FromRep(1 << kMantissaBits);
  }
  // -(1 + 0b1111 * 2^-2) * 2^(0b110 - 3) = -(1 + 15/16) * 2^3 = -15.5
  static constexpr Float8e3m4 lowest() {
    return Float8e3m4::FromRep(0b1'110'1111);
  }
  // (1 + 0b1111 * 2^-2) * 2^(0b110 - 3) = (1 + 15/16) * 2^3 = 15.5
  static constexpr Float8e3m4 max() {
    return Float8e3m4::FromRep(0b0'110'1111);
  }
  // (1 + 1/16) * 2^0 - 1.0 = 1.0 + 1/16 - 1.0 = 1/16
  // Encoded as denormal number 2^-2 * 1/4
  static constexpr Float8e3m4 epsilon() {
    return Float8e3m4::FromRep(0b0'000'0100);
  }
  // 1.0 * 2^-1 = 0.5
  static constexpr Float8e3m4 round_error() {
    return Float8e3m4::FromRep((-1 + kExponentBias) << kMantissaBits);
  }
  static constexpr Float8e3m4 infinity() {
    return Float8e3m4::FromRep(0b0'111'0000);
  }
  static constexpr Float8e3m4 quiet_NaN() {
    // IEEE 754-2019 6.2.1: "All binary NaN bit strings have the sign bit S set
    // to 0 or 1 and all the bits of the biased exponent field E set to 1
    // (see 3.4). A quiet NaN bit string should be encoded with the first bit
    // (d1) of the trailing significand field T being 1."
    return Float8e3m4::FromRep(0b0'111'1000);
  }
  static constexpr Float8e3m4 signaling_NaN() {
    // IEEE 754-2019 6.2.1: "A signaling NaN bit string should be encoded with
    // the first bit of the trailing significand field being 0."
    return Float8e3m4::FromRep(0b0'111'0100);
  }
  // 2^(-2) * 2^(-4) = 2^-6 = 1/64 (min denormal)
  static constexpr Float8e3m4 denorm_min() {
    return Float8e3m4::FromRep(0b0'000'0001);
  }
};

struct numeric_limits_float8_e4m3fn : public numeric_limits_float8_base {
 private:
  static constexpr const int kExponentBias = 7;
  static constexpr const int kMantissaBits = 3;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static constexpr const int digits = kMantissaBits + 1;
  static constexpr const int digits10 = Digits10FromDigits(digits);
  static constexpr const int max_digits10 = MaxDigits10FromDigits(digits);
  static constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static constexpr const int max_exponent =
      (0b1111 - 7) + 1;  // Extended format.
  static constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static constexpr const bool is_iec559 = false;
  static constexpr const bool has_infinity = false;
  static constexpr const bool has_signaling_NaN = false;
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
  static constexpr const int kExponentBias = 11;
  static constexpr const int kMantissaBits = 3;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static constexpr const int digits = kMantissaBits + 1;
  static constexpr const int digits10 = Digits10FromDigits(digits);
  static constexpr const int max_digits10 = MaxDigits10FromDigits(digits);
  static constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static constexpr const int max_exponent =
      (0b1111 - kExponentBias) + 1;  // Extended format.
  static constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static constexpr const bool is_iec559 = false;
  static constexpr const bool has_infinity = false;
  static constexpr const bool has_signaling_NaN = false;
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
  static constexpr const int kExponentBias = 8;
  static constexpr const int kMantissaBits = 3;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static constexpr const int digits = kMantissaBits + 1;
  static constexpr const int digits10 = Digits10FromDigits(digits);
  static constexpr const int max_digits10 = MaxDigits10FromDigits(digits);
  static constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static constexpr const int max_exponent =
      (0b1111 - kExponentBias) + 1;  // Extended format.
  static constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static constexpr const bool is_iec559 = false;
  static constexpr const bool has_infinity = false;
  static constexpr const bool has_signaling_NaN = false;
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
  }
  // NaN.
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
  static constexpr const int kExponentBias = 15;
  static constexpr const int kMantissaBits = 2;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static constexpr const int digits = kMantissaBits + 1;
  static constexpr const int digits10 = Digits10FromDigits(digits);
  static constexpr const int max_digits10 = MaxDigits10FromDigits(digits);
  static constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static constexpr const int max_exponent = 0b11111 - kExponentBias;
  static constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static constexpr const bool is_iec559 = true;
  static constexpr const bool has_infinity = true;
  static constexpr const bool has_signaling_NaN = true;
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
  static constexpr const int kExponentBias = 16;
  static constexpr const int kMantissaBits = 2;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static constexpr const int digits = kMantissaBits + 1;
  static constexpr const int digits10 = Digits10FromDigits(digits);
  static constexpr const int max_digits10 = MaxDigits10FromDigits(digits);
  static constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static constexpr const int max_exponent = (0b11111 - kExponentBias) + 1;
  static constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static constexpr const bool is_iec559 = false;
  static constexpr const bool has_infinity = false;
  static constexpr const bool has_signaling_NaN = false;
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

struct numeric_limits_float8_e8m0fnu : public numeric_limits_float8_base {
 private:
  static constexpr const int kExponentBias = 127;
  static constexpr const int kMantissaBits = 0;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static constexpr const bool is_signed = false;
// has_denorm and has_denorm_loss are deprecated in C++23.
#if !defined(__cplusplus) || __cplusplus < 202302L
  static constexpr const std::float_denorm_style has_denorm =
      std::denorm_absent;
#endif
  static constexpr const int digits = kMantissaBits + 1;
  static constexpr const int digits10 = Digits10FromDigits(digits);
  static constexpr const int max_digits10 = MaxDigits10FromDigits(digits);
  // 2**-127 smallest valid normalized value.
  static constexpr const int min_exponent = -kExponentBias + 1;
  static constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  // 128 encoding used for NaN.
  static constexpr const int max_exponent = kExponentBias + 1;
  static constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static constexpr const bool is_iec559 = false;
  static constexpr const bool has_infinity = false;
  static constexpr const bool has_signaling_NaN = false;
  // NOLINTEND

  static constexpr Float8e8m0fnu min() { return Float8e8m0fnu::FromRep(0x00); }
  static constexpr Float8e8m0fnu lowest() {
    return Float8e8m0fnu::FromRep(0x00);
  }
  static constexpr Float8e8m0fnu max() { return Float8e8m0fnu::FromRep(0xFE); }
  static constexpr Float8e8m0fnu epsilon() {
    return Float8e8m0fnu::FromRep((-kMantissaBits + kExponentBias)
                                  << kMantissaBits);
  }
  static constexpr Float8e8m0fnu round_error() {
    return Float8e8m0fnu::FromRep((-1 + kExponentBias) << kMantissaBits);
  }
  static constexpr Float8e8m0fnu quiet_NaN() {
    return Float8e8m0fnu::FromRep(0xFF);
  }
  static constexpr Float8e8m0fnu signaling_NaN() {
    return Float8e8m0fnu::FromRep(0xFF);
  }
  static constexpr Float8e8m0fnu infinity() {
    return Float8e8m0fnu::FromRep(0xFF);
  }
  static constexpr Float8e8m0fnu denorm_min() {
    // No denorm => smallest value.
    return Float8e8m0fnu::FromRep(0x00);
  }
};

}  // namespace float8_internal
}  // namespace tensorstore

namespace std {
// Standard-library overrides.
template <>
struct numeric_limits<tensorstore::float8_internal::Float8e3m4>
    : public tensorstore::float8_internal::numeric_limits_float8_e3m4 {};

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

template <>
struct numeric_limits<tensorstore::float8_internal::Float8e8m0fnu>
    : public tensorstore::float8_internal::numeric_limits_float8_e8m0fnu {};

}  // namespace std

namespace tensorstore {
namespace float8_internal {

constexpr Float8e3m4 abs(const Float8e3m4& a) {
  return Float8e3m4::FromRep(a.rep() & 0b0'111'1111);
}

constexpr bool(isnan)(const Float8e3m4& a) {
  return abs(a).rep() > std::numeric_limits<Float8e3m4>::infinity().rep();
}

constexpr Float8e4m3fn abs(const Float8e4m3fn& a) {
  return Float8e4m3fn::FromRep(a.rep() & 0b0'1111'111);
}

constexpr bool(isnan)(const Float8e4m3fn& a) {
  return abs(a).rep() == std::numeric_limits<Float8e4m3fn>::quiet_NaN().rep();
}

constexpr Float8e4m3b11fnuz abs(const Float8e4m3b11fnuz& a) {
  return (a.rep() & 0b0'1111'111) == 0
             ? Float8e4m3b11fnuz::FromRep(a.rep())
             : Float8e4m3b11fnuz::FromRep(a.rep() & 0b0'1111'111);
}

constexpr bool(isnan)(const Float8e4m3b11fnuz& a) {
  return a.rep() == std::numeric_limits<Float8e4m3b11fnuz>::quiet_NaN().rep();
}

constexpr Float8e4m3fnuz abs(const Float8e4m3fnuz& a) {
  return (a.rep() & 0x7F) == 0 ? Float8e4m3fnuz::FromRep(a.rep())
                               : Float8e4m3fnuz::FromRep(a.rep() & 0x7F);
}

constexpr bool(isnan)(const Float8e4m3fnuz& a) {
  return abs(a).rep() == std::numeric_limits<Float8e4m3fnuz>::quiet_NaN().rep();
}

constexpr Float8e5m2 abs(const Float8e5m2& a) {
  return Float8e5m2::FromRep(a.rep() & 0b0'11111'11);
}

constexpr bool(isnan)(const Float8e5m2& a) {
  return abs(a).rep() > std::numeric_limits<Float8e5m2>::infinity().rep();
}

constexpr Float8e5m2fnuz abs(const Float8e5m2fnuz& a) {
  return (a.rep() & 0x7F) == 0 ? Float8e5m2fnuz::FromRep(a.rep())
                               : Float8e5m2fnuz::FromRep(a.rep() & 0x7F);
}

constexpr bool isnan(const Float8e5m2fnuz& a) { return a.rep() == 0x80; }

constexpr Float8e8m0fnu abs(const Float8e8m0fnu& a) { return a; }

constexpr bool(isnan)(const Float8e8m0fnu& a) { return a.rep() == 0xFF; }

template <typename Float8>
constexpr bool(isinf)(const Float8Base<Float8>& a) {
  if constexpr (std::numeric_limits<Float8>::has_infinity) {
    return abs(a.derived()).rep() ==
           std::numeric_limits<Float8>::infinity().rep();
  } else {
    // No inf representation.
    return false;
  }
}

template <typename Float8>
constexpr bool(isfinite)(const Float8Base<Float8>& a) {
  return !isnan(a.derived()) && !isinf(a.derived());
}

template <typename Sink, typename Float8>
void AbslStringify(Sink& sink, const Float8Base<Float8>& f8) {
  absl::Format(&sink, "%v", static_cast<float>(f8.derived()));
}

template <typename Float8>
std::ostream& operator<<(std::ostream& os, const Float8Base<Float8>& f8) {
  return os << absl::StreamFormat("%v", f8);
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

// Helper for getting a bytes size which is a power of two.
template <int Size>
struct NextPowerOfTwo {
  static constexpr int value = Size;
};
template <>
struct NextPowerOfTwo<3> {
  static constexpr int value = 4;
};
template <>
struct NextPowerOfTwo<5> {
  static constexpr int value = 8;
};
template <>
struct NextPowerOfTwo<6> {
  static constexpr int value = 8;
};
template <>
struct NextPowerOfTwo<7> {
  static constexpr int value = 8;
};

// Helper for getting a bit representation provided a byte size.
template <int kNumBytes>
using GetUnsignedInteger =
    typename get_integer_by_size<kNumBytes>::unsigned_type;

template <int kNumBits>
using GetUnsignedIntegerBits =
    GetUnsignedInteger<NextPowerOfTwo<(kNumBits + 7) / 8>::value>;

// Converts between two floating-point types.
template <typename From, typename To, bool kSaturate, bool kTruncate,
          typename EnableIf = void>
struct ConvertImpl;

// Convert to same type.  We need explicit specializations for all combinations
// of template parameters to avoid ambiguities.
template <typename Scalar>
struct IdentityConversion {
  static constexpr Scalar run(const Scalar& from) { return from; }
};

template <typename Scalar, bool kSaturate, bool kTruncate>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/kSaturate,
                   /*kTruncate=*/kTruncate>
    : public IdentityConversion<Scalar> {};

template <typename Float>
struct TraitsBase {
  using BitsType = GetUnsignedInteger<sizeof(Float)>;
  static constexpr bool kIsSigned = std::numeric_limits<Float>::is_signed;
  static constexpr bool kHasZero = true;

  static constexpr int kBits = sizeof(Float) * CHAR_BIT;
  static constexpr int kMantissaBits = std::numeric_limits<Float>::digits - 1;
  // Extra bit used in exponent for unsigned float.
  static constexpr int kExponentBits =
      kBits - kMantissaBits - static_cast<int>(kIsSigned);
  static constexpr BitsType kExponentMask = ((BitsType{1} << kExponentBits) - 1)
                                            << kMantissaBits;
  static constexpr BitsType kMantissaMask = (BitsType{1} << kMantissaBits) - 1;
  static constexpr int kExponentBias = (1 << (kExponentBits - 1)) - 1;

  // Type that can hold mantissa, exponent, and one extra (overflow) bit.
  // This may be larger than BitsType for unsigned types.
  using WideBitsType =
      GetUnsignedIntegerBits<kMantissaBits + kExponentBits + 1>;
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

template <>
struct Traits<Float8e8m0fnu> : public TraitsBase<Float8e8m0fnu> {
  using Base = TraitsBase<Float8e8m0fnu>;
  // No zero in E8M0 OCP MX format description.
  static constexpr bool kHasZero = false;
};

template <typename Bits>
constexpr Bits RoundBitsToNearestEven(Bits bits, int roundoff) {
  // Round to nearest even by adding a bias term.
  // Consider a bit pattern
  //   FFF...FLRTT...T,
  // where bits RTT...T need to be rounded-off.  We add a bias term to the
  // bit pattern s.t. a carry is introduced to round up only if
  // - L is 1, R is 1, OR
  // - L is 0, R is 1, any T is one.
  // We do this by adding L to a bit pattern consisting of all T = 1.
  Bits bias = ((bits >> roundoff) & 1) + (Bits{1} << (roundoff - 1)) - 1;
  return bits + bias;
}

#if (defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L)
using std::countl_zero;
#else
using absl::countl_zero;
#endif

#if (defined(__cpp_lib_int_pow2) && __cpp_lib_int_pow2 >= 202002L)
using std::bit_width;
#else
using absl::bit_width;
#endif

template <typename From, typename To, bool kSaturate, bool kTruncate>
struct ConvertImpl<From, To, kSaturate, kTruncate,
                   std::enable_if_t<!std::is_same_v<From, To>>> {
  using FromTraits = Traits<From>;
  using FromBits = typename FromTraits::BitsType;
  static constexpr bool kFromIsSigned = FromTraits::kIsSigned;
  static constexpr bool kFromHasZero = FromTraits::kHasZero;
  static constexpr int kFromBits = FromTraits::kBits;
  static constexpr int kFromMantissaBits = FromTraits::kMantissaBits;
  static constexpr int kFromExponentBits = FromTraits::kExponentBits;
  static constexpr int kFromExponentBias = FromTraits::kExponentBias;
  // FromBits has to be able to hold the implicit bit at `kFromMantissaBits +
  // 1` and handle shifts wide enough to clear the implicit bit.
  static_assert(std::numeric_limits<FromBits>::digits > kFromMantissaBits + 1);

  using ToTraits = Traits<To>;
  using ToBits = typename ToTraits::BitsType;
  using WideToBits = typename ToTraits::WideBitsType;
  static constexpr bool kToIsSigned = ToTraits::kIsSigned;
  static constexpr bool kToHasZero = ToTraits::kHasZero;
  static constexpr int kToMantissaBits = ToTraits::kMantissaBits;
  static constexpr int kToExponentBits = ToTraits::kExponentBits;
  static constexpr int kToExponentBias = ToTraits::kExponentBias;
  // See Step 4 for why a target without zero gets one more binade.
  static constexpr int kToEncodingBias = kToExponentBias + !kToHasZero;

  static constexpr To run(From from) {
    const FromBits from_bits = absl::bit_cast<FromBits>(from);
    const bool from_sign = (from_bits >> (kFromBits - 1)) && kFromIsSigned;
    FromBits from_abs_bits;
    bool from_isinf;
    bool from_isnan;
    // Check exponent and mantissa bits as integers instead of calling
    // isinf/isnan on `from`. When auto-vectorizing on ARM64, Clang can compile
    // `isinf || isnan` into a floating-point `>` comparison against +inf,
    // which sets the hardware FE_INVALID exception flag on quiet NaNs.
    // Integer comparisons never raise floating-point exceptions.
    if constexpr (std::numeric_limits<From>::is_iec559 ||
                  !std::is_base_of_v<Float8Base<From>, From>) {
      from_abs_bits = from_bits & ~(FromBits{1} << (kFromBits - 1));
      from_isinf = from_abs_bits == FromTraits::kExponentMask;
      from_isnan = from_abs_bits > FromTraits::kExponentMask;
    } else {
      From from_abs = abs(from);
      from_abs_bits = absl::bit_cast<FromBits>(from_abs);
      from_isinf = isinf(from_abs);
      from_isnan = isnan(from_abs);
    }

    // 1. Handle non-finite values (Infinity and NaN).
    if (from_isinf) {
      return from_sign ? -std::numeric_limits<To>::infinity()
                       : std::numeric_limits<To>::infinity();
    }
    if (from_isnan) {
      return from_sign ? -std::numeric_limits<To>::quiet_NaN()
                       : std::numeric_limits<To>::quiet_NaN();
    }

    // 2. Handle Zero.
    // Note: Types without zero (e.g., E8M0 where kFromHasZero=false) encode
    // values in 0x00 and pass through as normal numbers.
    if (from_abs_bits == 0 && kFromHasZero) {
      if constexpr (kToHasZero) {
        To zero = absl::bit_cast<To>(ToBits{0});
        return from_sign && kToIsSigned ? -zero : zero;
      } else {
        // If the target has no zero (e.g., E8M0), we saturate to the minimum
        // denormalized value.
        return kSaturate ? std::numeric_limits<To>::denorm_min()
                         : std::numeric_limits<To>::quiet_NaN();
      }
    }

    // `To` unsigned floating format: NaN or saturate.
    if constexpr (kFromIsSigned && !kToIsSigned) {
      if (from_sign) {
        return kSaturate ? std::numeric_limits<To>::lowest()
                         : std::numeric_limits<To>::quiet_NaN();
      }
    }

    // 3. Unpack and Normalize Input.
    // We convert the input into a normalized (unbiased_exponent, mantissa)
    // pair. The mantissa is shifted such that the implicit leading 1 is at
    // bit `kFromMantissaBits`.
    int unbiased_exponent;
    FromBits normalized_mantissa;

    const FromBits from_biased_exp = from_abs_bits >> kFromMantissaBits;
    const FromBits from_fraction = from_abs_bits & FromTraits::kMantissaMask;

    constexpr FromBits kImplicitBit = FromBits{1} << kFromMantissaBits;
    if (from_biased_exp != 0 || !kFromHasZero) {
      // Normal number (or E8M0).
      unbiased_exponent = from_biased_exp - kFromExponentBias;
      normalized_mantissa = kImplicitBit | from_fraction;
    } else {
      // Input is subnormal. We normalize it by shifting the fraction until the
      // MSB is 1, adjusting the exponent accordingly.
      // `countl_zero` counts from the MSB of the storage type, so we adjust
      // based on the actual bit width of the type.
      int type_leading_zeros = countl_zero(from_fraction);
      int storage_bits = sizeof(FromBits) * CHAR_BIT;
      int fraction_leading_zeros =
          type_leading_zeros - (storage_bits - kFromMantissaBits);

      int normalization_shift = fraction_leading_zeros + 1;
      normalized_mantissa = from_fraction << normalization_shift;
      unbiased_exponent = (1 - kFromExponentBias) - normalization_shift;
    }

    // 4. Calculate Target Parameters.
    // We calculate `target_biased_exponent_base` representing:
    // `BiasedExp - 1`.
    //
    // The "-1" is critical to the packing algorithm (Step 6). We maintain the
    // mantissa with its implicit leading 1. When we eventually add the
    // mantissa to the exponent bits, that leading 1 acts as `1 <<
    // kToMantissaBits`, effectively adding 1 to the exponent.
    //
    // If the result is negative, it indicates the value is too small for the
    // normal range of the target (underflow), requiring denormalization.
    //
    // A target without zero (e.g., E8M0) has no subnormals: biased exponent 0
    // is its smallest normal binade. We encode it as if it had zero and
    // subnormals one binade below that, and Step 6 shifts the encoding down.
    int target_biased_exponent_base = unbiased_exponent + kToEncodingBias - 1;

    // 5. Apply Shift and Rounding, and 6. Pack and Handle Overflow.
    // We calculate the shift needed to align the source mantissa to the target
    // mantissa.
    // Positive shift: Right shift (truncation/rounding needed).
    // Negative shift: Left shift (padding needed).
    constexpr int kNormalShift = kFromMantissaBits - kToMantissaBits;
    WideToBits result_bits_wide;
    if (target_biased_exponent_base >= 0) {
      if constexpr (kNormalShift > 0) {
        if constexpr (!kTruncate) {
          // Rounding may cause a carry (e.g., 1.11... -> 10.00...).
          // This carry will naturally flow into the exponent during packing.
          normalized_mantissa =
              RoundBitsToNearestEven(normalized_mantissa, kNormalShift);
        }
        result_bits_wide = normalized_mantissa >> kNormalShift;
      } else {
        result_bits_wide = static_cast<ToBits>(normalized_mantissa)
                           << -kNormalShift;
      }
      // Reassemble the bits using integer addition.
      //
      // As noted in Step 4, `target_biased_exponent_base` holds
      // `BiasedExp - 1`, and `result_bits_wide` still holds the implicit
      // leading 1 at position `kToMantissaBits`.
      //
      // Therefore:
      //   (BiasedExp - 1) << Shift  +  (1 << Shift | Fraction)
      //   = ((BiasedExp - 1) + 1) << Shift  +  Fraction
      //   = BiasedExp << Shift  +  Fraction
      //
      // This addition also handles rounding overflow automatically. If
      // rounding created a carry (mantissa = 2.0), it adds an extra 1 to the
      // exponent (creating `BiasedExp + 1`). `WideToBits` is required to store
      // the highest bit in this case.
      result_bits_wide += static_cast<WideToBits>(target_biased_exponent_base)
                          << kToMantissaBits;
    } else {
      const int denormal_adjustment = -target_biased_exponent_base;
      // A shift of `kFromMantissaBits + 1` removes the implicit bit, but the
      // result may still round to 1. A shift of `kFromMantissaBits + 2`
      // guarantees a result of 0 after rounding. We clamp the shift to this
      // value.
      const int alignment_shift =
          std::min(kNormalShift + denormal_adjustment, kFromMantissaBits + 2);
      if (alignment_shift > 0) {
        if constexpr (!kTruncate) {
          normalized_mantissa =
              RoundBitsToNearestEven(normalized_mantissa, alignment_shift);
        }
        result_bits_wide = normalized_mantissa >> alignment_shift;
      } else {
        result_bits_wide = static_cast<ToBits>(normalized_mantissa)
                           << -alignment_shift;
      }
    }
    if constexpr (!kToHasZero) {
      // Undo the extra binade from Step 4. Values that rounded or truncated to
      // zero or a subnormal clamp to the smallest representable value (which is
      // the nearest value when rounding, and matches llvm::APFloat rmTowardZero
      // when truncating since no representable value is <= x).
      constexpr WideToBits kSmallest = WideToBits{1} << kToMantissaBits;
      result_bits_wide = std::max(result_bits_wide, kSmallest) - kSmallest;
    }

    const ToBits kToMaxFinite =
        absl::bit_cast<ToBits>(std::numeric_limits<To>::max());

    // Check for overflow.
    // Condition 1: Exponent calculated exceeds target max.
    // Condition 2: The final packed bits exceed the max finite representation
    // (handles cases where rounding pushed a max-value into infinity).
    ToBits result_bits;
    if (target_biased_exponent_base >=
            std::numeric_limits<To>::max_exponent + kToEncodingBias ||
        result_bits_wide > kToMaxFinite) {
      if constexpr (kSaturate) {
        result_bits = kToMaxFinite;
      } else {
        result_bits =
            absl::bit_cast<ToBits>(std::numeric_limits<To>::infinity());
      }
    } else {
      result_bits = static_cast<ToBits>(result_bits_wide);
    }

    // 7. Apply Sign
    To result = absl::bit_cast<To>(result_bits);
    return from_sign ? -result : result;
  }
};

// Saturation has no impact when casting e4m3fn to e5m2.
template <bool kTruncate>
struct ConvertImpl<Float8e4m3fn, Float8e5m2, true, kTruncate> {
  static constexpr Float8e5m2 run(Float8e4m3fn from) {
    return ConvertImpl<Float8e4m3fn, Float8e5m2, false, kTruncate>::run(from);
  }
};

template <bool kSaturate, bool kTruncate>
struct ConvertImpl<::half_float::half, Float8e5m2, kSaturate, kTruncate> {
  static constexpr Float8e5m2 run(const ::half_float::half& from) {
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

// Direct casts of e5m2 to ::half_float::half simply shifts bits over.
template <bool kSaturate, bool kTruncate>
struct ConvertImpl<Float8e5m2, ::half_float::half, kSaturate, kTruncate> {
  static constexpr ::half_float::half run(const Float8e5m2& from) {
    return absl::bit_cast<::half_float::half>(
        static_cast<uint16_t>(static_cast<uint16_t>(from.rep()) << 8));
  }
};

// Converts `from` to float such that rounding (or truncating) the result to a
// type with at least two fewer significand bits than float gives the same
// result as rounding (or truncating) `from` to that type directly.
template <typename From>
constexpr float RoundToOddFloat(From from) {
  static_assert(std::numeric_limits<float>::is_iec559);
  static_assert(std::numeric_limits<float>::radix == 2);
  constexpr int kFloatDigits = std::numeric_limits<float>::digits;
  if constexpr (std::is_integral_v<From>) {
    if constexpr (std::numeric_limits<From>::digits <= kFloatDigits) {
      // Every value of `From` is representable without rounding.
      return static_cast<float>(from);
    } else {
      static_assert(std::numeric_limits<From>::digits <=
                        std::numeric_limits<uint64_t>::digits,
                    "Integer types wider than 64 bits are not supported.");
      bool is_negative = false;
      uint64_t magnitude = static_cast<uint64_t>(from);
      if constexpr (std::is_signed_v<From>) {
        if (from < 0) {
          is_negative = true;
          // Unsigned negation is exact, even for the minimum value.
          magnitude = uint64_t{0} - magnitude;
        }
      }
      if ((magnitude >> kFloatDigits) == 0) {
        return static_cast<float>(from);
      }
      const int bit_width = float8_internal::bit_width(magnitude);
      // Keep the `kFloatDigits` leading bits, and set the lowest of them if
      // any of the discarded bits are set.
      const int shift = bit_width - kFloatDigits;
      const uint64_t sticky = (magnitude & ((uint64_t{1} << shift) - 1)) != 0;
      magnitude = ((magnitude >> shift) | sticky) << shift;
      // `magnitude` has at most `kFloatDigits` significant bits now, so this
      // conversion is exact.
      const float abs_narrow = static_cast<float>(magnitude);
      return is_negative ? -abs_narrow : abs_narrow;
    }
  } else {
    static_assert(std::is_floating_point_v<From>);
    if constexpr (std::numeric_limits<From>::digits <= kFloatDigits) {
      return static_cast<float>(from);
    } else if constexpr (std::numeric_limits<From>::is_iec559 &&
                         sizeof(From) == sizeof(uint64_t)) {
      static_assert(std::numeric_limits<From>::digits >= kFloatDigits + 2);
      static_assert(std::numeric_limits<float>::min_exponent >=
                    std::numeric_limits<From>::min_exponent + 2);
      const uint64_t wide_bits = absl::bit_cast<uint64_t>(from);
      const bool is_negative = (wide_bits >> 63) != 0;
      const uint64_t abs_wide_bits = wide_bits & 0x7FFF'FFFF'FFFF'FFFFULL;
      if (abs_wide_bits > 0x7FF0'0000'0000'0000ULL) {
        return is_negative ? -std::numeric_limits<float>::quiet_NaN()
                           : std::numeric_limits<float>::quiet_NaN();
      }
      const From abs_wide = absl::bit_cast<From>(abs_wide_bits);
      float abs_narrow = static_cast<float>(abs_wide);
      const uint64_t abs_narrow_as_wide_bits =
          absl::bit_cast<uint64_t>(static_cast<From>(abs_narrow));
      uint32_t narrow_bits = absl::bit_cast<uint32_t>(abs_narrow);
      const bool keep_narrow =
          (abs_wide_bits == abs_narrow_as_wide_bits) || (narrow_bits & 1);
      const bool narrow_is_rd = abs_wide_bits > abs_narrow_as_wide_bits;
      narrow_bits += keep_narrow ? 0 : narrow_is_rd ? 1 : -1;
      abs_narrow = absl::bit_cast<float>(narrow_bits);
      return is_negative ? -abs_narrow : abs_narrow;
    } else {
      static_assert(std::numeric_limits<From>::digits >= kFloatDigits + 2);
      static_assert(std::numeric_limits<float>::min_exponent >=
                    std::numeric_limits<From>::min_exponent + 2);
      if (from != from) {
        return std::signbit(from) ? -std::numeric_limits<float>::quiet_NaN()
                                  : std::numeric_limits<float>::quiet_NaN();
      }
      const bool is_negative = std::signbit(from);
      const From abs_wide = std::fabs(from);
      float abs_narrow = static_cast<float>(abs_wide);
      const From abs_narrow_as_wide = static_cast<From>(abs_narrow);

      uint32_t narrow_bits = absl::bit_cast<uint32_t>(abs_narrow);
      const bool keep_narrow =
          (abs_wide == abs_narrow_as_wide) || (narrow_bits & 1);
      const bool narrow_is_rd = abs_wide > abs_narrow_as_wide;
      narrow_bits += keep_narrow ? 0 : narrow_is_rd ? 1 : -1;
      abs_narrow = absl::bit_cast<float>(narrow_bits);
      return is_negative ? -abs_narrow : abs_narrow;
    }
  }
}

template <typename Derived>
template <bool kSaturate, bool kTruncate, typename From>
constexpr Derived Float8Base<Derived>::ConvertFrom(const From from) {
  // Integers and floating point types wider than double (float80, binary128,
  // etc.) have no direct conversion. Narrowing them to float first must not
  // round twice, so use round to odd.
  if constexpr (std::is_integral_v<From> || (std::is_floating_point_v<From> &&
                                             sizeof(From) > sizeof(double))) {
    return ConvertImpl<float, Derived, kSaturate, kTruncate>::run(
        RoundToOddFloat(from));
  } else {
    return ConvertImpl<From, Derived, kSaturate, kTruncate>::run(from);
  }
}

template <typename Derived>
template <typename To, bool kSaturate, bool kTruncate>
constexpr To Float8Base<Derived>::ConvertTo(const Derived& from) {
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

TENSORSTORE_INTERNAL_FPCLASSIFY(Float8e3m4);
TENSORSTORE_INTERNAL_FPCLASSIFY(Float8e4m3fn);
TENSORSTORE_INTERNAL_FPCLASSIFY(Float8e4m3fnuz);
TENSORSTORE_INTERNAL_FPCLASSIFY(Float8e4m3b11fnuz);
TENSORSTORE_INTERNAL_FPCLASSIFY(Float8e5m2);
TENSORSTORE_INTERNAL_FPCLASSIFY(Float8e5m2fnuz);
TENSORSTORE_INTERNAL_FPCLASSIFY(Float8e8m0fnu);
#undef TENSORSTORE_INTERNAL_FPCLASSIFY
#endif

}  // namespace float8_internal

// Exported types.

/// Storage-only 8-bit floating-point data types.
///
/// See https://github.com/jax-ml/ml_dtypes for details.
///
/// \ingroup Data types
using Float8e3m4 = float8_internal::Float8e3m4;
using Float8e4m3fn = float8_internal::Float8e4m3fn;
using Float8e4m3fnuz = float8_internal::Float8e4m3fnuz;
using Float8e4m3b11fnuz = float8_internal::Float8e4m3b11fnuz;
using Float8e5m2 = float8_internal::Float8e5m2;
using Float8e5m2fnuz = float8_internal::Float8e5m2fnuz;
using Float8e8m0fnu = float8_internal::Float8e8m0fnu;

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_FLOAT8_H_
