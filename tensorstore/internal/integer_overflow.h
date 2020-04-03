// Copyright 2020 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_INTEGER_OVERFLOW_H_
#define TENSORSTORE_INTERNAL_INTEGER_OVERFLOW_H_

#include <type_traits>

#include "absl/base/attributes.h"
#include "absl/meta/type_traits.h"

namespace tensorstore {
namespace internal {
namespace wrap_on_overflow {

#if ABSL_HAVE_ATTRIBUTE(no_sanitize) && defined(__clang__)
#define TENSORSTORE_ATTRIBUTE_NO_SANITIZE_UNSIGNED_INTEGER_OVERFLOW \
  __attribute__((no_sanitize("unsigned-integer-overflow")))
#else
#define TENSORSTORE_ATTRIBUTE_NO_SANITIZE_UNSIGNED_INTEGER_OVERFLOW
#endif

/// Defines a function `NAME` for integral types equivalent to operator `OP`
/// except that it has well-defined wrap-on-overflow behavior.
///
/// In C++, signed integer overflow is undefined behavior, but unsigned integers
/// are guaranteed to wrap on overflow.
///
/// Therefore, the wrap-on-overflow behavior is implemented by casting the
/// arguments to the corresponding unsigned type, performing `OP`, and then
/// casting back to the original type.
///
/// For unsigned argument types, this is exactly equivalent to the corresponding
/// builtin operator, but is still useful as documentation that wrapping is
/// expected, and suppresses errors from Undefined Behavior Sanitizer when the
/// non-default unsigned-integer-overflow detection has been enabled.
#define TENSORSTORE_INTERNAL_DEFINE_WRAP_ON_OVERFLOW_OP(OP, NAME) \
  template <typename T>                                           \
  TENSORSTORE_ATTRIBUTE_NO_SANITIZE_UNSIGNED_INTEGER_OVERFLOW     \
      absl::enable_if_t<std::is_integral<T>::value, T>            \
      NAME(T a, T b) {                                            \
    using UnsignedT = absl::make_unsigned_t<T>;                   \
    return static_cast<T>(static_cast<UnsignedT>(                 \
        static_cast<UnsignedT>(a) OP static_cast<UnsignedT>(b))); \
  }                                                               \
  /**/
TENSORSTORE_INTERNAL_DEFINE_WRAP_ON_OVERFLOW_OP(+, Add)
TENSORSTORE_INTERNAL_DEFINE_WRAP_ON_OVERFLOW_OP(-, Subtract)
TENSORSTORE_INTERNAL_DEFINE_WRAP_ON_OVERFLOW_OP(*, Multiply)

#undef TENSORSTORE_INTERNAL_DEFINE_WRAP_ON_OVERFLOW_OP

/// Returns the inner product of `a` and `b`, wrapping on overflow.
///
/// \tparam AccumType The type to use for accumulating the result.
/// \params n The length.
/// \params a Pointer to an array of length `n`.
/// \params b Pointer to an array of length `n`.
template <typename AccumType, typename T0, typename T1>
inline AccumType InnerProduct(std::ptrdiff_t n, const T0* a, const T1* b) {
  AccumType sum = 0;
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    sum = Add(sum, Multiply(static_cast<AccumType>(a[i]),
                            static_cast<AccumType>(b[i])));
  }
  return sum;
}

}  // namespace wrap_on_overflow

/// Sets `*result` to the result of adding `a` and `b` with infinite precision,
/// and returns `true` if the stored value does not equal the infinite precision
/// result.
template <typename T>
constexpr bool AddOverflow(T a, T b, T* result) {
#if defined(__clang__) || !defined(_MSC_VER)
  return __builtin_add_overflow(a, b, result);
#else
  *result = wrap_on_overflow::Add(a, b);
  return (a > 0 && (b > std::numeric_limits<T>::max() - a)) ||
         (a < 0 && (b < std::numeric_limits<T>::min() - a));
#endif
}

/// Sets `*result` to the result of subtracting `a` and `b` with infinite
/// precision, and returns `true` if the stored value does not equal the
/// infinite precision result.
template <typename T>
constexpr bool SubOverflow(T a, T b, T* result) {
#if defined(__clang__) || !defined(_MSC_VER)
  return __builtin_sub_overflow(a, b, result);
#else
  *result = wrap_on_overflow::Subtract(a, b);
  return (b < 0 && (a > std::numeric_limits<T>::max() + b)) ||
         (b > 0 && (a < std::numeric_limits<T>::min() + b));
#endif
}

/// Sets `*result` to the result of multiplying `a` and `b` with infinite
/// precision, and returns `true` if the stored value does not equal the
/// infinite precision result.
template <typename T>
constexpr bool MulOverflow(T a, T b, T* result) {
#if defined(__clang__) || !defined(_MSC_VER)
  return __builtin_mul_overflow(a, b, result);
#else
  const T r = *result = wrap_on_overflow::Multiply(a, b);
  return b && (r / b) != a;
#endif
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_INTEGER_OVERFLOW_H_
