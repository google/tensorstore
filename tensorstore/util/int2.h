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

#ifndef TENSORSTORE_UTIL_INT2_H_
#define TENSORSTORE_UTIL_INT2_H_

#include <stdint.h>

#include <cmath>
#include <cstdint>  // IWYU pragma: keep
#include <limits>
#include <type_traits>

#include <nlohmann/json_fwd.hpp>

namespace tensorstore {
class Int2Padded;
}  // namespace tensorstore

namespace std {
template <>
struct numeric_limits<::tensorstore::Int2Padded>;
}  // namespace std

namespace tensorstore {

namespace internal {

constexpr int8_t SignedTrunc2(int8_t x) {
  return static_cast<int8_t>(static_cast<uint8_t>(x) << 6) >> 6;
}

}  // namespace internal

/// Int2 type padded to int8.
///
/// \ingroup Data types
class Int2Padded {
 public:
  /// Zero initialization.
  ///
  /// \id zero
  constexpr Int2Padded() : rep_(0) {}

  /// Possibly lossy conversion from any type convertible to `int8_t`.
  ///
  /// \id convert
  template <typename T,
            typename = std::enable_if_t<std::is_convertible_v<T, int8_t>>>
  constexpr explicit Int2Padded(T x)
      : rep_(internal::SignedTrunc2(static_cast<int8_t>(x))) {}

  /// Lossless conversion to `int8_t`.
  constexpr operator int8_t() const {
    // NOTE: Re-applying truncation and sign-extension to avoid handing out
    // a non-canonical representation.
    return internal::SignedTrunc2(rep_);
  }

  /// Bool assignment.
  ///
  /// \id bool
  /// \membergroup Assignment operators
  Int2Padded& operator=(bool v) { return *this = static_cast<Int2Padded>(v); }

  /// Possibly lossy conversion from any integer type.
  ///
  /// \id integer
  /// \membergroup Assignment operators
  template <typename T>
  std::enable_if_t<std::numeric_limits<T>::is_integer, Int2Padded&> operator=(
      T v) {  // NOLINT: misc-unconventional-assign-operator
    return *this = static_cast<Int2Padded>(v);
  }

  /// Define arithmetic operators.  Mixed operations involving other integers
  /// are supported automatically by the implicit conversion to int8_t.
  /// However, for operations with only `Int2Padded` parameters, or with one
  /// `Int2Padded` parameter and one integer parameter, we provide an
  /// implementation that converts the result back to `Int2Padded` for
  /// consistency with the builtin integer operations.
#define TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_OP(OP)                \
  friend Int2Padded operator OP(Int2Padded a, Int2Padded b) {             \
    return Int2Padded(a.rep_ OP b.rep_);                                  \
  }                                                                       \
  template <typename T>                                                   \
  friend std::enable_if_t<std::numeric_limits<T>::is_integer, Int2Padded> \
  operator OP(Int2Padded a, T b) {                                        \
    return Int2Padded(a.rep_ OP b);                                       \
  }                                                                       \
  template <typename T>                                                   \
  friend std::enable_if_t<std::numeric_limits<T>::is_integer, Int2Padded> \
  operator OP(T a, Int2Padded b) {                                        \
    return Int2Padded(a OP b.rep_);                                       \
  }                                                                       \
  /**/

#define TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_ASSIGN_OP(OP)          \
  friend Int2Padded& operator OP##=(Int2Padded & a, Int2Padded b) {        \
    return a = Int2Padded(a.rep_ OP b.rep_);                               \
  }                                                                        \
  template <typename T>                                                    \
  friend std::enable_if_t<std::numeric_limits<T>::is_integer, Int2Padded&> \
  operator OP##=(Int2Padded & a, T b) {                                    \
    return a = Int2Padded(a.rep_ OP b);                                    \
  }                                                                        \
  /**/

  /// Addition operator.
  ///
  /// \membergroup Arithmetic operators
  /// \id binary
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_OP(+)

  /// Addition assignment operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_ASSIGN_OP(+)

  /// Subtraction operator.
  ///
  /// \membergroup Arithmetic operators
  /// \id binary
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_OP(-)

  /// Subtraction assignment operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_ASSIGN_OP(-)

  /// Multiplication operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_OP(*)

  /// Multiplication assignment operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_ASSIGN_OP(*)

  /// Division operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_OP(/)

  /// Division assignment operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_ASSIGN_OP(/)

  /// Modulo operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_OP(%)

  /// Modulo assignment operator.
  ///
  /// \membergroup Arithmetic operators
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_ASSIGN_OP(%)

  /// Bitwise and operator.
  ///
  /// \membergroup Bitwise operators
  /// \id binary
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_OP(&)

  /// Bitwise and assignment operator.
  ///
  /// \membergroup Bitwise operators
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_ASSIGN_OP(&)

  /// Bitwise or operator.
  ///
  /// \membergroup Bitwise operators
  /// \id binary
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_OP(|)

  /// Bitwise or assignment operator.
  ///
  /// \membergroup Bitwise operators
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_ASSIGN_OP(|)

  /// Bitwise xor operator.
  ///
  /// \membergroup Bitwise operators
  /// \id binary
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_OP(^)

  /// Bitwise xor assignment operator.
  ///
  /// \membergroup Bitwise operators
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_ASSIGN_OP(^)

  /// Bitwise left shift operator.
  ///
  /// \membergroup Bitwise operators
  /// \id binary
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_OP(<<)

  /// Bitwise left shift assignment operator.
  ///
  /// \membergroup Bitwise operators
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_ASSIGN_OP(<<)

  /// Bitwise right shift operator.
  ///
  /// \membergroup Bitwise operators
  /// \id binary
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_OP(>>)

  /// Bitwise right shift assignment operator.
  ///
  /// \membergroup Bitwise operators
  TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_ASSIGN_OP(>>)

#undef TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_OP
#undef TENSORSTORE_INTERNAL_INT2_PADDED_ARITHMETIC_ASSIGN_OP

  /// Unary inverse.
  ///
  /// \membergroup Bitwise operators
  /// \id negate
  friend Int2Padded operator~(Int2Padded a) {
    Int2Padded result;
    result.rep_ = internal::SignedTrunc2(~a.rep_);
    return result;
  }

  /// Unary negation.
  ///
  /// \membergroup Arithmetic operators
  /// \id negate
  friend Int2Padded operator-(Int2Padded a) {
    Int2Padded result;
    result.rep_ = internal::SignedTrunc2(-a.rep_);
    return result;
  }

  /// Unary plus.
  ///
  /// \membergroup Arithmetic operators
  /// \id unary
  friend Int2Padded operator+(Int2Padded a) { return a; }

  /// Pre-increment.
  ///
  /// \id pre
  /// \membergroup Arithmetic operators
  friend Int2Padded operator++(Int2Padded& a) {
    a += Int2Padded(1);
    return a;
  }

  /// Pre-decrement.
  ///
  /// \membergroup Arithmetic operators
  /// \id pre
  friend Int2Padded operator--(Int2Padded& a) {
    a -= Int2Padded(1);
    return a;
  }

  /// Post-increment.
  ///
  /// \membergroup Arithmetic operators
  /// \id post
  friend Int2Padded operator++(Int2Padded& a, int) {
    Int2Padded original_value = a;
    ++a;
    return original_value;
  }

  /// Post-decrement.
  ///
  /// \membergroup Arithmetic operators
  /// \id post
  friend Int2Padded operator--(Int2Padded& a, int) {
    Int2Padded original_value = a;
    --a;
    return original_value;
  }

  // Note: Comparison operators do not need to be defined since they are
  // provided automatically by the implicit conversion to `int8_t`.

  // Conversion to `::nlohmann::json`.
  template <template <typename U, typename V,
                      typename... Args> class ObjectType /* = std::map*/,
            template <typename U,
                      typename... Args> class ArrayType /* = std::vector*/,
            class StringType /*= std::string*/, class BooleanType /* = bool*/,
            class NumberIntegerType /* = std::int64_t*/,
            class NumberUnsignedType /* = std::uint64_t*/,
            class NumberFloatType /* = double*/,
            template <typename U> class AllocatorType /* = std::allocator*/,
            template <typename T,
                      typename SFINAE =
                          void> class JSONSerializer /* = adl_serializer*/,
            class BinaryType /* = std::vector<std::uint8_t>*/>
  friend void to_json(
      ::nlohmann::basic_json<ObjectType, ArrayType, StringType, BooleanType,
                             NumberIntegerType, NumberUnsignedType,
                             NumberFloatType, AllocatorType, JSONSerializer,
                             BinaryType>& j,
      Int2Padded v) {
    j = static_cast<NumberIntegerType>(v);
  }

  constexpr friend bool operator==(const Int2Padded& a, const Int2Padded& b) {
    return internal::SignedTrunc2(a.rep_) == internal::SignedTrunc2(b.rep_);
  }

  constexpr friend bool operator!=(const Int2Padded& a, const Int2Padded& b) {
    return !(a == b);
  }

  // Treat as private:

  // Workaround for non-constexpr bit_cast implementation.
  struct bitcast_construct_t {};
  explicit constexpr Int2Padded(bitcast_construct_t, int8_t rep) : rep_(rep) {}

  // The canonical representation should be sign-extended to 8 bits, i.e.
  // same signed integer value for int4 and int8.
  // This is _not_ guaranteed due to memcpy / reinterpret_cast.
  // All operations defined here will produce the canonical form.
  int8_t rep_;
};

/// Returns the absolute value of `x`.
///
/// \membergroup Basic operations
/// \relates Int2Padded
/// \id Int2Padded
inline Int2Padded abs(Int2Padded x) {
  x.rep_ = internal::SignedTrunc2(::std::abs(x.rep_));
  return x;
}

/// Raises a number to the given power (:math:`x^y`).
///
/// \membergroup Power functions
/// \relates Int2Padded
/// \id Int2Padded
inline Int2Padded pow(Int2Padded x, Int2Padded y) {
  return Int2Padded(std::pow(static_cast<int8_t>(x), static_cast<int8_t>(y)));
}

}  // namespace tensorstore

namespace std {
template <>
struct numeric_limits<tensorstore::Int2Padded> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = true;
  static constexpr bool is_exact = true;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = false;
  static constexpr bool has_signaling_NaN = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = true;
  static constexpr int digits = 1;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 0;
  static constexpr int radix = 2;

  static constexpr tensorstore::Int2Padded min() {
    return tensorstore::Int2Padded(
        tensorstore::Int2Padded::bitcast_construct_t{}, int8_t{-2});
  }

  static constexpr tensorstore::Int2Padded lowest() {
    return tensorstore::Int2Padded(
        tensorstore::Int2Padded::bitcast_construct_t{}, int8_t{-2});
  }

  static constexpr tensorstore::Int2Padded max() {
    return tensorstore::Int2Padded(
        tensorstore::Int2Padded::bitcast_construct_t{}, int8_t{1});
  }
};

}  // namespace std

#endif  // TENSORSTORE_UTIL_INT2_H_
