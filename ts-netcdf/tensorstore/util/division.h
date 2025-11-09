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

#ifndef TENSORSTORE_UTIL_DIVISION_H_
#define TENSORSTORE_UTIL_DIVISION_H_

// Some of the code below is derived from TensorFlow:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/math/math_util.h
//
// Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#include <cassert>
#include <limits>
#include <type_traits>

namespace tensorstore {

template <typename IntegralType>
constexpr IntegralType RoundUpTo(IntegralType input,
                                 IntegralType rounding_value) {
  static_assert(std::is_integral<IntegralType>::value,
                "IntegralType must be an integral type.");
  assert(input >= 0 && rounding_value > 0);
  return ((input + rounding_value - 1) / rounding_value) * rounding_value;
}

template <typename IntegralType, bool ceil>
constexpr IntegralType CeilOrFloorOfRatio(IntegralType numerator,
                                          IntegralType denominator);

// ----------------------------------------------------------------------
// CeilOfRatio<IntegralType>
// FloorOfRatio<IntegralType>
//   Returns the ceil (resp. floor) of the ratio of two integers.
//
//  * IntegralType: any integral type, whether signed or not.
//  * numerator: any integer: positive, negative, or zero.
//  * denominator: a non-zero integer, positive or negative.
//
// This implementation is correct, meaning there is never any precision loss,
// and there is never an overflow. However, if the type is signed, having
// numerator == std::numeric_limits<IntegralType>::min() and denominator == -1
// is not a valid input, because min() has a greater absolute value than max().
//
// Invalid inputs raise SIGFPE.
//
// This method has been designed and tested so that it should always be
// preferred to alternatives. Indeed, there exist popular recipes to compute
// the result, such as casting to double, but they are in general incorrect.
// In cases where an alternative technique is correct, performance measurement
// showed the provided implementation is faster.
template <typename IntegralType>
constexpr IntegralType CeilOfRatio(IntegralType numerator,
                                   IntegralType denominator) {
  return CeilOrFloorOfRatio<IntegralType, true>(numerator, denominator);
}
template <typename IntegralType>
constexpr IntegralType FloorOfRatio(IntegralType numerator,
                                    IntegralType denominator) {
  return CeilOrFloorOfRatio<IntegralType, false>(numerator, denominator);
}

// ---- CeilOrFloorOfRatio ----
// This is a branching-free, cast-to-double-free implementation.
//
// Casting to double is in general incorrect because of loss of precision
// when casting an int64 into a double.
//
// There's a bunch of 'recipes' to compute a integer ceil (or floor) on the web,
// and most of them are incorrect.
template <typename IntegralType, bool ceil>
constexpr IntegralType CeilOrFloorOfRatio(IntegralType numerator,
                                          IntegralType denominator) {
  const IntegralType rounded_toward_zero = numerator / denominator;
  const IntegralType intermediate_product = rounded_toward_zero * denominator;

  if constexpr (ceil) {  // Compile-time condition: not an actual branching
    // When rounded_toward_zero is negative, then an adjustment is never needed:
    // the real ratio is negative, and so rounded toward zero is the ceil.
    // When rounded_toward_zero is non-negative, an adjustment is needed if the
    // sign of the difference numerator - intermediate_product is the same as
    // the sign of the denominator.
    //
    //
    // Using a bool and then a static_cast to IntegralType is not strictly
    // necessary, but it makes the code clear, and anyway the compiler should
    // get rid of it.
    const bool needs_adjustment =
        (rounded_toward_zero >= 0) &&
        ((denominator > 0 && numerator > intermediate_product) ||
         (denominator < 0 && numerator < intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType ceil_of_ratio = rounded_toward_zero + adjustment;
    return ceil_of_ratio;
  } else {
    // Floor case: symmetrical to the previous one
    const bool needs_adjustment =
        (rounded_toward_zero <= 0) &&
        ((denominator > 0 && numerator < intermediate_product) ||
         (denominator < 0 && numerator > intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType floor_of_ratio = rounded_toward_zero - adjustment;
    return floor_of_ratio;
  }
}

/// Computes the non-negative remainder of `numerator / denominator`.
template <typename IntegralType>
constexpr IntegralType NonnegativeMod(IntegralType numerator,
                                      IntegralType denominator) {
  assert(denominator > 0);
  IntegralType modulus = numerator % denominator;
  return modulus + (modulus < 0) * denominator;
}

/// Computes the non-negative greater common divisor of `x` and `y` using
/// Euclid's Algorithm.
///
/// GreatestCommonDivisor(0, 0) is undefined.
/// GreatestCommonDivisor(x, 0) == x
/// GreatestCommonDivisor(0, x) == x
template <typename IntegralType>
constexpr IntegralType GreatestCommonDivisor(IntegralType x, IntegralType y) {
  assert(x != 0 || y != 0);
  if (std::is_signed_v<IntegralType> &&
      x == std::numeric_limits<IntegralType>::min()) {
    // If `y == 0`, the mathematical result is `-x`, which overflows.  Instead,
    // this traps due to division by 0.
    x = x % y;
  }
  if (std::is_signed_v<IntegralType> &&
      y == std::numeric_limits<IntegralType>::min()) {
    // If `x == 0`, the mathematical result is `-y`, which overflows.  Instead,
    // this traps due to division by 0.
    y = y % x;
  }
  if (std::is_signed_v<IntegralType> && x < 0) x = -x;
  if (std::is_signed_v<IntegralType> && y < 0) y = -y;
  while (y != 0) {
    IntegralType r = x % y;
    x = y;
    y = r;
  }
  return x;
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_DIVISION_H_
