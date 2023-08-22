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

#ifndef TENSORSTORE_UTIL_BFLOAT16_H_
#define TENSORSTORE_UTIL_BFLOAT16_H_

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "absl/base/casts.h"
#include "tensorstore/internal/json_fwd.h"

// The implementation below is derived from Tensorflow and Eigen:
// https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/arch/Default/BFloat16.h
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

namespace tensorstore {
class BFloat16;
}  // namespace tensorstore

namespace std {
template <>
struct numeric_limits<::tensorstore::BFloat16>;
}  // namespace std

namespace tensorstore {
namespace internal {
BFloat16 NumericFloat32ToBfloat16RoundNearestEven(float v);
BFloat16 Float32ToBfloat16RoundNearestEven(float v);
float Bfloat16ToFloat(BFloat16 v);
}  // namespace internal

/// Storage-only bfloat16 type.
///
/// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
///
/// This differs from ``Eigen::bfloat16`` and ``tensorflow::bfloat16`` in
/// that it preserves subnormals rather than flushing them to zero, and also
/// preserves signaling NaN.
///
/// \ingroup Data types
class BFloat16 {
 public:
  /// Zero initialization.
  ///
  /// \id zero
  constexpr BFloat16() : rep_(0) {}

  /// Possibly lossy conversion from any type convertible to `float`.
  ///
  /// \id convert
  template <typename T,
            typename = std::enable_if_t<std::is_convertible_v<T, float>>>
  explicit BFloat16(T x) {
    if constexpr (std::is_same_v<T, bool>) {
      rep_ = static_cast<uint16_t>(x) * 0x3f80;
    } else if constexpr (std::numeric_limits<T>::is_integer) {
      *this = internal::NumericFloat32ToBfloat16RoundNearestEven(
          static_cast<float>(x));
    } else {
      *this =
          internal::Float32ToBfloat16RoundNearestEven(static_cast<float>(x));
    }
  }

  /// Lossless conversion to `float`.
  operator float() const { return internal::Bfloat16ToFloat(*this); }

  /// Possibly lossy conversion from `float`.
  ///
  /// \id float
  /// \membergroup Assignment operators
  BFloat16& operator=(float v) { return *this = static_cast<BFloat16>(v); }

  /// Bool assignment.
  ///
  /// \id bool
  /// \membergroup Assignment operators
  BFloat16& operator=(bool v) { return *this = static_cast<BFloat16>(v); }

  /// Possibly lossy conversion from any integer type.
  ///
  /// \id integer
  /// \membergroup Assignment operators
  template <typename T>
  std::enable_if_t<std::numeric_limits<T>::is_integer, BFloat16&> operator=(
      T v) {  // NOLINT: misc-unconventional-assign-operator
    return *this = static_cast<BFloat16>(v);
  }

  /// Define arithmetic operators.  Mixed operations involving other floating
  /// point are supported automatically by the implicit conversion to float.
  /// However, for operations with only `BFloat16` parameters, or with one
  /// `BFloat16` parameter and one integer parameter, we provide an
  /// implementation that converts the result back to `BFloat16` for
  /// consistency with the builtin floating point operations.
#define TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_OP(OP)                 \
  friend BFloat16 operator OP(BFloat16 a, BFloat16 b) {                 \
    return BFloat16(static_cast<float>(a) OP static_cast<float>(b));    \
  }                                                                     \
  template <typename T>                                                 \
  friend std::enable_if_t<std::numeric_limits<T>::is_integer, BFloat16> \
  operator OP(BFloat16 a, T b) {                                        \
    return BFloat16(static_cast<float>(a) OP b);                        \
  }                                                                     \
  template <typename T>                                                 \
  friend std::enable_if_t<std::numeric_limits<T>::is_integer, BFloat16> \
  operator OP(T a, BFloat16 b) {                                        \
    return BFloat16(a OP static_cast<float>(b));                        \
  }                                                                     \
  /**/

#define TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_ASSIGN_OP(OP)           \
  friend BFloat16& operator OP##=(BFloat16& a, BFloat16 b) {             \
    return a = BFloat16(static_cast<float>(a) OP static_cast<float>(b)); \
  }                                                                      \
  template <typename T>                                                  \
  friend std::enable_if_t<std::numeric_limits<T>::is_integer, BFloat16&> \
  operator OP##=(BFloat16& a, T b) {                                     \
    return a = BFloat16(static_cast<float>(a) OP b);                     \
  }                                                                      \
  /**/

  /// Addition operator.
  ///
  /// \membergroup Arithmetic operators
  /// \id binary
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_OP(+)

  /// Addition assignment operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_ASSIGN_OP(+)

  /// Subtraction operator.
  ///
  /// \membergroup Arithmetic operators
  /// \id binary
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_OP(-)

  /// Subtraction assignment operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_ASSIGN_OP(-)

  /// Multiplication operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_OP(*)

  /// Multiplication assignment operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_ASSIGN_OP(*)

  /// Division operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_OP(/)

  /// Division assignment operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_ASSIGN_OP(/)

#undef TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_OP
#undef TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_ASSIGN_OP

  /// Unary negation.
  ///
  /// \membergroup Arithmetic operators
  /// \id negate
  friend BFloat16 operator-(BFloat16 a) {
    BFloat16 result;
    result.rep_ = a.rep_ ^ 0x8000;
    return result;
  }

  /// Unary plus.
  ///
  /// \membergroup Arithmetic operators
  /// \id unary
  friend BFloat16 operator+(BFloat16 a) { return a; }

  /// Pre-increment.
  ///
  /// \id pre
  /// \membergroup Arithmetic operators
  friend BFloat16 operator++(BFloat16& a) {
    a += BFloat16(1);
    return a;
  }

  /// Pre-decrement.
  ///
  /// \membergroup Arithmetic operators
  /// \id pre
  friend BFloat16 operator--(BFloat16& a) {
    a -= BFloat16(1);
    return a;
  }

  /// Post-increment.
  ///
  /// \membergroup Arithmetic operators
  /// \id post
  friend BFloat16 operator++(BFloat16& a, int) {
    BFloat16 original_value = a;
    ++a;
    return original_value;
  }

  /// Post-decrement.
  ///
  /// \membergroup Arithmetic operators
  /// \id post
  friend BFloat16 operator--(BFloat16& a, int) {
    BFloat16 original_value = a;
    --a;
    return original_value;
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
      BFloat16 v) {
    j = static_cast<NumberFloatType>(v);
  }

  // Treat as private:

  // Workaround for non-constexpr bit_cast implementation.
  struct bitcast_construct_t {};
  explicit constexpr BFloat16(bitcast_construct_t, uint16_t rep) : rep_(rep) {}
  uint16_t rep_;
};

/// Returns true if `x` is +/-infinity.
///
/// \membergroup Classification functions
/// \relates BFloat16
inline bool isinf(BFloat16 x) { return std::isinf(static_cast<float>(x)); }

/// Returns `true` if `x` is negative.
///
/// \relates BFloat16
/// \membergroup Floating-point manipulation functions
inline bool signbit(BFloat16 x) { return std::signbit(static_cast<float>(x)); }

/// Returns `true` if `x` is NaN.
///
/// \membergroup Classification functions
/// \relates BFloat16
inline bool isnan(BFloat16 x) { return std::isnan(static_cast<float>(x)); }

/// Returns `true` if `x` is finite.
///
/// \membergroup Classification functions
/// \relates BFloat16
inline bool isfinite(BFloat16 x) {
  return std::isfinite(static_cast<float>(x));
}

/// Returns the absolute value of `x`.
///
/// \membergroup Basic operations
/// \relates BFloat16
inline BFloat16 abs(BFloat16 x) {
  x.rep_ &= 0x7fff;
  return x;
}

/// Computes :math:`e` raised to the given power (:math:`e^x`).
///
/// \membergroup Exponential functions
/// \relates BFloat16
inline BFloat16 exp(BFloat16 x) {
  return BFloat16(std::exp(static_cast<float>(x)));
}

/// Computes :math:`2` raised to the given power (:math:`2^x`).
///
/// \membergroup Exponential functions
/// \relates BFloat16
inline BFloat16 exp2(BFloat16 x) {
  return BFloat16(std::exp2(static_cast<float>(x)));
}

/// Computes :math:`e` raised to the given power, minus 1 (:math:`e^x-1`).
///
/// \membergroup Exponential functions
/// \relates BFloat16
inline BFloat16 expm1(BFloat16 x) {
  return BFloat16(std::expm1(static_cast<float>(x)));
}

/// Computes the natural (base :math:`e`) logarithm (:math:`\ln x`)
///
/// \membergroup Exponential functions
/// \relates BFloat16
inline BFloat16 log(BFloat16 x) {
  return BFloat16(std::log(static_cast<float>(x)));
}

/// Computes the natural (base :math:`e`) logarithm of 1 plus the given number
/// (:math:`\ln (1 + x)`).
///
/// \membergroup Exponential functions
/// \relates BFloat16
inline BFloat16 log1p(BFloat16 x) {
  return BFloat16(std::log1p(static_cast<float>(x)));
}

/// Computes the base-10 logarithm of the given number (:math:`\log_{10} x`).
///
/// \membergroup Exponential functions
/// \relates BFloat16
inline BFloat16 log10(BFloat16 x) {
  return BFloat16(std::log10(static_cast<float>(x)));
}

/// Computes the base-2 logarithm of the given number (:math:`\log_2 x`).
///
/// \membergroup Exponential functions
/// \relates BFloat16
inline BFloat16 log2(BFloat16 x) {
  return BFloat16(std::log2(static_cast<float>(x)));
}

/// Computes the square root of the given number (:math:`\sqrt{x}`).
///
/// \membergroup Power functions
/// \relates BFloat16
inline BFloat16 sqrt(BFloat16 x) {
  return BFloat16(std::sqrt(static_cast<float>(x)));
}

/// Raises a number to the given power (:math:`x^y`).
///
/// \membergroup Power functions
/// \relates BFloat16
inline BFloat16 pow(BFloat16 x, BFloat16 y) {
  return BFloat16(std::pow(static_cast<float>(x), static_cast<float>(y)));
}

/// Computes the sine of the given number (:math:`\sin x`).
///
/// \membergroup Trigonometric functions
/// \relates BFloat16
inline BFloat16 sin(BFloat16 x) {
  return BFloat16(std::sin(static_cast<float>(x)));
}

/// Computes the cosine of the given number (:math:`cos x`).
///
/// \membergroup Trigonometric functions
/// \relates BFloat16
inline BFloat16 cos(BFloat16 x) {
  return BFloat16(std::cos(static_cast<float>(x)));
}

/// Computes the tangent.
///
/// \membergroup Trigonometric functions
/// \relates BFloat16
inline BFloat16 tan(BFloat16 x) {
  return BFloat16(std::tan(static_cast<float>(x)));
}

/// Computes the inverse sine.
///
/// \membergroup Trigonometric functions
/// \relates BFloat16
inline BFloat16 asin(BFloat16 x) {
  return BFloat16(std::asin(static_cast<float>(x)));
}

/// Computes the inverse cosine.
///
/// \membergroup Trigonometric functions
/// \relates BFloat16
inline BFloat16 acos(BFloat16 x) {
  return BFloat16(std::acos(static_cast<float>(x)));
}

/// Computes the inverse tangent.
///
/// \membergroup Trigonometric functions
/// \relates BFloat16
inline BFloat16 atan(BFloat16 x) {
  return BFloat16(std::atan(static_cast<float>(x)));
}

/// Computes the hyperbolic sine.
///
/// \membergroup Hyperbolic functions
/// \relates BFloat16
inline BFloat16 sinh(BFloat16 x) {
  return BFloat16(std::sinh(static_cast<float>(x)));
}

/// Computes the hyperbolic cosine.
///
/// \membergroup Hyperbolic functions
/// \relates BFloat16
inline BFloat16 cosh(BFloat16 x) {
  return BFloat16(std::cosh(static_cast<float>(x)));
}

/// Computes the hyperbolic tangent.
///
/// \membergroup Hyperbolic functions
/// \relates BFloat16
inline BFloat16 tanh(BFloat16 x) {
  return BFloat16(std::tanh(static_cast<float>(x)));
}

/// Computes the inverse hyperbolic sine.
///
/// \membergroup Hyperbolic functions
/// \relates BFloat16
inline BFloat16 asinh(BFloat16 x) {
  return BFloat16(std::asinh(static_cast<float>(x)));
}

/// Computes the inverse hyperbolic cosine.
///
/// \membergroup Hyperbolic functions
/// \relates BFloat16
inline BFloat16 acosh(BFloat16 x) {
  return BFloat16(std::acosh(static_cast<float>(x)));
}

/// Computes the inverse hyperbolic tangent.
///
/// \membergroup Hyperbolic functions
/// \relates BFloat16
inline BFloat16 atanh(BFloat16 x) {
  return BFloat16(std::atanh(static_cast<float>(x)));
}

/// Computes the nearest integer not less than the given value.
///
/// \membergroup Rounding functions
/// \relates BFloat16
inline BFloat16 floor(BFloat16 x) {
  return BFloat16(std::floor(static_cast<float>(x)));
}

/// Computes the nearest integer not greater in absolute value.
///
/// \membergroup Rounding functions
/// \relates BFloat16
inline BFloat16 trunc(BFloat16 x) {
  return BFloat16(std::trunc(static_cast<float>(x)));
}

/// Computes the nearest integer using the current rounding mode.
///
/// \membergroup Rounding functions
/// \relates BFloat16
inline BFloat16 rint(BFloat16 x) {
  return BFloat16(std::rint(static_cast<float>(x)));
}

/// Computes the nearest integer not less than the given value.
///
/// \membergroup Rounding functions
/// \relates BFloat16
inline BFloat16 ceil(BFloat16 x) {
  return BFloat16(std::ceil(static_cast<float>(x)));
}

/// Computes the floating-point remainder of the division operation `x / y`.
///
/// \membergroup Basic operations
/// \relates BFloat16
inline BFloat16 fmod(BFloat16 x, BFloat16 y) {
  return BFloat16(std::fmod(static_cast<float>(x), static_cast<float>(y)));
}

/// Computes the minimum of two values.
///
/// \membergroup Basic operations
/// \relates BFloat16
inline BFloat16 fmin(BFloat16 a, BFloat16 b) {
  return BFloat16(std::fmin(static_cast<float>(a), static_cast<float>(b)));
}

/// Computes the maximum of two values.
///
/// \membergroup Basic operations
/// \relates BFloat16
inline BFloat16 fmax(BFloat16 a, BFloat16 b) {
  return BFloat16(std::fmax(static_cast<float>(a), static_cast<float>(b)));
}

/// Next representable value towards the given value.
///
/// \membergroup Floating-point manipulation functions
/// \relates BFloat16
inline BFloat16 nextafter(BFloat16 from, BFloat16 to) {
  const uint16_t from_as_int = absl::bit_cast<uint16_t>(from),
                 to_as_int = absl::bit_cast<uint16_t>(to);
  const uint16_t sign_mask = 1 << 15;
  float from_as_float(from), to_as_float(to);
  if (std::isnan(from_as_float) || std::isnan(to_as_float)) {
    return BFloat16(std::numeric_limits<float>::quiet_NaN());
  }
  if (from_as_int == to_as_int) {
    return to;
  }
  if (from_as_float == 0) {
    if (to_as_float == 0) {
      return to;
    } else {
      // Smallest subnormal signed like `to`.
      return absl::bit_cast<BFloat16, uint16_t>((to_as_int & sign_mask) | 1);
    }
  }
  uint16_t from_sign = from_as_int & sign_mask;
  uint16_t to_sign = to_as_int & sign_mask;
  uint16_t from_abs = from_as_int & ~sign_mask;
  uint16_t to_abs = to_as_int & ~sign_mask;
  uint16_t magnitude_adjustment =
      (from_abs > to_abs || from_sign != to_sign) ? 0xFFFF : 0x0001;
  return absl::bit_cast<BFloat16, uint16_t>(from_as_int + magnitude_adjustment);
}

namespace internal {

inline uint16_t GetFloat32High16(float v) {
  return static_cast<uint16_t>(absl::bit_cast<uint32_t>(v) >> 16);
}

/// Converts float32 -> bfloat16, rounding towards zero (truncating).
inline BFloat16 Float32ToBfloat16Truncate(float v) {
  // IEEE 764 binary32 floating point representation:
  //
  // Exponent  | Fraction = 0 | Fraction != 0
  // ----------------------------------------
  // 0         | zero         | subnormal
  // 0x1..0xfe | normal value | normal value
  // 0xff      | +/- infinity | NaN
  //
  // For exponents not equal to `0xff`, we can simply truncate the low 16 bits
  // of the fraction.  This may result in a subnormal value becoming zero, but
  // that is not a problem.
  //
  // However, if the exponent is equal to `0xff`, truncating the low 16 bits of
  // the fraction may convert a NaN to infinity.
  uint32_t bits = absl::bit_cast<uint32_t>(v);
  if (std::isnan(v)) {
    // Set bit 21 (second to highest fraction bit) to 1, to ensure the truncated
    // fraction still indicates a NaN.  This preserves the sign and also
    // preserves the high bit of the fraction (quiet/signalling NaN bit).
    bits |= (static_cast<uint32_t>(1) << 21);
  }
  return absl::bit_cast<BFloat16, uint16_t>(bits >> 16);
}

/// Converts finite float32 -> bfloat16, rounding to the nearest, or to even in
/// the case of a tie.
///
/// The input must not be NaN.  This is more efficient than the general
/// `Float32ToBfloat16RoundNearestEven` conversion in the case that the input is
/// known not to be NaN (e.g. for integer conversion by way of float).
inline BFloat16 NumericFloat32ToBfloat16RoundNearestEven(float v) {
  assert(!std::isnan(v));
  // Fast rounding algorithm that rounds a half value to nearest even. This
  // reduces expected error when we convert a large number of floats. Here
  // is how it works:
  //
  // Definitions:
  // To convert a float 32 to bfloat16, a float 32 can be viewed as 32 bits
  // with the following tags:
  //
  // Sign |  Exp (8 bits) | Frac (23 bits)
  //  S     EEEEEEEE         FFFFFFLRTTTTTTTTTTTTTTT
  //
  //  S: Sign bit.
  //  E: Exponent bits.
  //  F: First 6 bits of fraction.
  //  L: Least significant bit of resulting bfloat16 if we truncate away the
  //  rest of the float32. This is also the 7th bit of fraction
  //  R: Rounding bit, 8th bit of fraction.
  //  T: Sticky bits, rest of fraction, 15 bits.
  //
  // To round half to nearest even, there are 3 cases where we want to round
  // down (simply truncate the result of the bits away, which consists of
  // rounding bit and sticky bits) and two cases where we want to round up
  // (truncate then add one to the result).
  //
  // The fast converting algorithm simply adds lsb (L) to 0x7fff (15 bits of
  // 1s) as the rounding bias, adds the rounding bias to the input, then
  // truncates the last 16 bits away.
  //
  // To understand how it works, we can analyze this algorithm case by case:
  //
  // 1. L = 0, R = 0:
  //   Expect: round down, this is less than half value.
  //
  //   Algorithm:
  //   - Rounding bias: 0x7fff + 0 = 0x7fff
  //   - Adding rounding bias to input may create any carry, depending on
  //   whether there is any value set to 1 in T bits.
  //   - R may be set to 1 if there is a carry.
  //   - L remains 0.
  //   - Note that this case also handles Inf and -Inf, where all fraction
  //   bits, including L, R and Ts are all 0. The output remains Inf after
  //   this algorithm.
  //
  // 2. L = 1, R = 0:
  //   Expect: round down, this is less than half value.
  //
  //   Algorithm:
  //   - Rounding bias: 0x7fff + 1 = 0x8000
  //   - Adding rounding bias to input doesn't change sticky bits but
  //   adds 1 to rounding bit.
  //   - L remains 1.
  //
  // 3. L = 0, R = 1, all of T are 0:
  //   Expect: round down, this is exactly at half, the result is already
  //   even (L=0).
  //
  //   Algorithm:
  //   - Rounding bias: 0x7fff + 0 = 0x7fff
  //   - Adding rounding bias to input sets all sticky bits to 1, but
  //   doesn't create a carry.
  //   - R remains 1.
  //   - L remains 0.
  //
  // 4. L = 1, R = 1:
  //   Expect: round up, this is exactly at half, the result needs to be
  //   round to the next even number.
  //
  //   Algorithm:
  //   - Rounding bias: 0x7fff + 1 = 0x8000
  //   - Adding rounding bias to input doesn't change sticky bits, but
  //   creates a carry from rounding bit.
  //   - The carry sets L to 0, creates another carry bit and propagate
  //   forward to F bits.
  //   - If all the F bits are 1, a carry then propagates to the exponent
  //   bits, which then creates the minimum value with the next exponent
  //   value. Note that we won't have the case where exponents are all 1,
  //   since that's either a NaN (handled in the other if condition) or inf
  //   (handled in case 1).
  //
  // 5. L = 0, R = 1, any of T is 1:
  //   Expect: round up, this is greater than half.
  //
  //   Algorithm:
  //   - Rounding bias: 0x7fff + 0 = 0x7fff
  //   - Adding rounding bias to input creates a carry from sticky bits,
  //   sets rounding bit to 0, then create another carry.
  //   - The second carry sets L to 1.
  //
  // Examples:
  //
  //  Exact half value that is already even:
  //    Input:
  //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
  //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
  //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0     1000000000000000
  //
  //     This falls into case 3. We truncate the rest of 16 bits and no
  //     carry is created into F and L:
  //
  //    Output:
  //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
  //     S     E E E E E E E E      F F F F F F L
  //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0
  //
  //  Exact half value, round to next even number:
  //    Input:
  //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
  //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
  //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 0 1     1000000000000000
  //
  //     This falls into case 4. We create a carry from R and T,
  //     which then propagates into L and F:
  //
  //    Output:
  //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
  //     S     E E E E E E E E      F F F F F F L
  //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0
  //
  //
  //  Max denormal value round to min normal value:
  //    Input:
  //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
  //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
  //     0     0 0 0 0 0 0 0 0      1 1 1 1 1 1 1     1111111111111111
  //
  //     This falls into case 4. We create a carry from R and T,
  //     propagate into L and F, which then propagates into exponent
  //     bits:
  //
  //    Output:
  //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
  //     S     E E E E E E E E      F F F F F F L
  //     0     0 0 0 0 0 0 0 1      0 0 0 0 0 0 0
  //
  //  Max normal value round to Inf:
  //    Input:
  //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
  //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
  //     0     1 1 1 1 1 1 1 0      1 1 1 1 1 1 1     1111111111111111
  //
  //     This falls into case 4. We create a carry from R and T,
  //     propagate into L and F, which then propagates into exponent
  //     bits:
  //
  //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
  //     S     E E E E E E E E      F F F F F F L
  //     0     1 1 1 1 1 1 1 1      0 0 0 0 0 0 0
  uint32_t input = absl::bit_cast<uint32_t>(v);
  const uint32_t lsb = (input >> 16) & 1;
  const uint32_t rounding_bias = 0x7fff + lsb;
  input += rounding_bias;
  return absl::bit_cast<BFloat16, uint16_t>(input >> 16);
}

/// Converts float32 -> bfloat16, rounding to the nearest, or to even in the
/// case of a tie.
inline BFloat16 Float32ToBfloat16RoundNearestEven(float v) {
  if (std::isnan(v)) {
    return tensorstore::BFloat16(
        tensorstore::BFloat16::bitcast_construct_t{},
        static_cast<uint16_t>((absl::bit_cast<uint32_t>(v) | 0x00200000u) >>
                              16));
  }
  return NumericFloat32ToBfloat16RoundNearestEven(v);
}

inline float Bfloat16ToFloat(BFloat16 v) {
  return absl::bit_cast<float>(
      static_cast<uint32_t>(absl::bit_cast<uint16_t>(v)) << 16);
}

}  // namespace internal
}  // namespace tensorstore

namespace std {
template <>
struct numeric_limits<tensorstore::BFloat16> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr float_denorm_style has_denorm = std::denorm_present;
  static constexpr bool has_denorm_loss = false;
  static constexpr std::float_round_style round_style =
      numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = numeric_limits<float>::min_exponent;
  static constexpr int min_exponent10 = numeric_limits<float>::min_exponent10;
  static constexpr int max_exponent = numeric_limits<float>::max_exponent;
  static constexpr int max_exponent10 = numeric_limits<float>::max_exponent10;
  static constexpr bool traps = numeric_limits<float>::traps;
  static constexpr bool tinyness_before =
      numeric_limits<float>::tinyness_before;

  /// Smallest positive normalized value: 0x1p-126 = 1.175494351e-38
  static constexpr tensorstore::BFloat16 min() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x0080));
  }

  /// Lowest finite negative value: -0x1.fep+127 = -3.38953139e+38
  static constexpr tensorstore::BFloat16 lowest() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0xff7f));
  }

  /// Largest finite value: 0x1.fep127 = 3.38953139e+38
  static constexpr tensorstore::BFloat16 max() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x7f7f));
  }

  /// Machine epsilon: 0x1p-7 = 0.0078125
  ///
  /// Equal to `nextafter(BFloat16(1), BFloat16(2)) - BFloat16(1)`.
  static constexpr tensorstore::BFloat16 epsilon() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x3c00));
  }

  /// Largest possible rounding error in ULPs (units in the last place): 0.5
  static constexpr tensorstore::BFloat16 round_error() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x3f00));
  }
  static constexpr tensorstore::BFloat16 infinity() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x7f80));
  }
  static constexpr tensorstore::BFloat16 quiet_NaN() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x7fc0));
  }
  static constexpr tensorstore::BFloat16 signaling_NaN() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x7f81));
  }

  /// Smallest positive subnormal value: 0x1p-133 = 9.18354962e-41
  static constexpr tensorstore::BFloat16 denorm_min() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x0001));
  }
};

}  // namespace std

#endif  // TENSORSTORE_UTIL_BFLOAT16_H_
