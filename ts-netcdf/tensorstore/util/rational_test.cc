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

// Some of the following tests are from Boost.Rational, which is subject to the
// following copyright and license:
//
//  (C) Copyright Paul Moore 1999. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or
//  implied warranty, and with no claim as to its suitability for any purpose.

#include "tensorstore/util/rational.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/str_cat.h"

namespace {

constexpr double pi = 3.14159265358979323846264338327950288;

using ::tensorstore::Rational;

TEST(RationalTest, Initialization) {
  static constexpr Rational<int> r1, r2(0), r3(1), r4(-3), r5(7, 2), r6(5, 15),
      r7(14, -21), r8(-4, 6), r9(-14, -70);
  static_assert(r1.numerator() == 0);
  static_assert(r2.numerator() == 0);
  static_assert(r3.numerator() == 1);
  static_assert(r4.numerator() == -3);
  static_assert(r5.numerator() == 7);
  static_assert(r6.numerator() == 1);
  static_assert(r7.numerator() == -2);
  static_assert(r8.numerator() == -2);
  static_assert(r9.numerator() == 1);

  static_assert(r1.denominator() == 1);
  static_assert(r2.denominator() == 1);
  static_assert(r3.denominator() == 1);
  static_assert(r4.denominator() == 1);
  static_assert(r5.denominator() == 2);
  static_assert(r6.denominator() == 3);
  static_assert(r7.denominator() == 3);
  static_assert(r8.denominator() == 3);
  static_assert(r9.denominator() == 5);

  static_assert(Rational<int>(0, 0).is_nan());
  static_assert(Rational<int>(1, std::numeric_limits<int>::min()).is_nan());
  static_assert(!Rational<int>(1, -std::numeric_limits<int>::max()).is_nan());
}

TEST(RationalTest, Compare) {
  static constexpr Rational<int> r1, r2(0), r3(1), r4(-3), r5(7, 2), r6(5, 15),
      r7(14, -21), r8(-4, 6), r9(-14, -70), nan = Rational<int>::nan();

  static_assert(r1 == r2);
  static_assert(r2 != r3);
  static_assert(r4 < r3);
  static_assert(r4 <= r5);
  static_assert(r1 <= r2);
  static_assert(r5 > r6);
  static_assert(r5 >= r6);
  static_assert(r7 >= r8);

  static_assert(!(r3 == r2));
  static_assert(!(r1 != r2));
  static_assert(!(r1 < r2));
  static_assert(!(r5 < r6));
  static_assert(!(r9 <= r2));
  static_assert(!(r8 > r7));
  static_assert(!(r8 > r2));
  static_assert(!(r4 >= r6));

  static_assert(r1 == 0);
  static_assert(r2 != -1);
  static_assert(r3 < 2);
  static_assert(r4 <= -3);
  static_assert(r5 > 3);
  static_assert(r6 >= 0);

  static_assert(0 == r2);
  static_assert(0 != r7);
  static_assert(-1 < r8);
  static_assert(-2 <= r9);
  static_assert(1 > r1);
  static_assert(1 >= r3);

  // Extra tests with values close in continued-fraction notation
  static constexpr Rational<int> x1(9, 4);
  static constexpr Rational<int> x2(61, 27);
  static constexpr Rational<int> x3(52, 23);
  static constexpr Rational<int> x4(70, 31);

  static_assert(x1 < x2);
  static_assert(!(x1 < x1));
  static_assert(!(x2 < x2));
  static_assert(!(x2 < x1));
  static_assert(x2 < x3);
  static_assert(x4 < x2);
  static_assert(!(x3 < x4));
  static_assert(r7 < x1);  // not actually close; wanted -ve v. +ve instead
  static_assert(!(x2 < r7));

  static_assert(!(nan < nan));
  static_assert(!(nan <= nan));
  static_assert(!(nan == nan));
  static_assert(nan != nan);
  static_assert(!(nan > nan));
  static_assert(!(nan >= nan));

  static_assert(!(nan < r1));
  static_assert(!(nan == r1));
  static_assert(nan != r1);
  static_assert(!(nan <= r1));
  static_assert(!(nan > r1));
  static_assert(!(nan >= r1));

  static_assert(!(r1 < nan));
  static_assert(!(r1 <= nan));
  static_assert(!(r1 == nan));
  static_assert(r1 != nan);
  static_assert(!(r1 > nan));
  static_assert(!(r1 >= nan));

  static_assert(!(nan < 0));
  static_assert(!(nan == 0));
  static_assert(nan != 0);
  static_assert(!(nan <= 0));
  static_assert(!(nan > 0));
  static_assert(!(nan >= 0));

  static_assert(!(0 < nan));
  static_assert(!(0 <= nan));
  static_assert(!(0 == nan));
  static_assert(0 != nan);
  static_assert(!(0 > nan));
  static_assert(!(0 >= nan));
}

TEST(RationalTest, Increment) {
  Rational<int> r1, r2(0), r3(1), r7(14, -21), r8(-4, 6);

  EXPECT_EQ(r1++, r2);
  EXPECT_NE(r1, r2);
  EXPECT_EQ(r1, r3);
  EXPECT_EQ(--r1, r2);
  EXPECT_EQ(r8--, r7);
  EXPECT_NE(r8, r7);
  EXPECT_EQ(++r8, r7);

  Rational<int> x1 = std::numeric_limits<int>::max();
  EXPECT_FALSE(x1.is_nan());
  ++x1;
  EXPECT_TRUE(x1.is_nan());

  Rational<int> x2 = std::numeric_limits<int>::min();
  EXPECT_FALSE(x2.is_nan());
  --x2;
  EXPECT_TRUE(x2.is_nan());
}

TEST(RationalTest, UnaryOperators) {
  static constexpr Rational<int> r2(0), r3(1), r4(-3), r5(7, 2);

  static_assert(+r5 == r5);
  static_assert(-r3 != r3);
  static_assert(-(-r3) == r3);
  static_assert(-r4 == 3);
  static_assert(!r2);
  static_assert(!!r3);
  static_assert(r3);
}

TEST(RationalTest, Addition) {
  using T = int;
  using rational_type = Rational<T>;
  EXPECT_EQ(rational_type(1, 2) + rational_type(1, 2), static_cast<T>(1));
  EXPECT_EQ(rational_type(11, 3) + rational_type(1, 2), rational_type(25, 6));
  EXPECT_EQ(rational_type(-8, 3) + rational_type(1, 5), rational_type(-37, 15));
  EXPECT_EQ(rational_type(-7, 6) + rational_type(1, 7),
            rational_type(1, 7) - rational_type(7, 6));
  EXPECT_EQ(rational_type(13, 5) - rational_type(1, 2), rational_type(21, 10));
  EXPECT_EQ(rational_type(22, 3) + static_cast<T>(1), rational_type(25, 3));
  EXPECT_EQ(rational_type(12, 7) - static_cast<T>(2), rational_type(-2, 7));
  EXPECT_EQ(static_cast<T>(3) + rational_type(4, 5), rational_type(19, 5));
  EXPECT_EQ(static_cast<T>(4) - rational_type(9, 2), rational_type(-1, 2));

  rational_type r(11);

  r -= rational_type(20, 3);
  EXPECT_EQ(r, rational_type(13, 3));

  r += rational_type(1, 2);
  EXPECT_EQ(r, rational_type(29, 6));

  r -= static_cast<T>(5);
  EXPECT_EQ(r, rational_type(1, -6));

  r += rational_type(1, 5);
  EXPECT_EQ(r, rational_type(1, 30));

  r += static_cast<T>(2);
  EXPECT_EQ(r, rational_type(61, 30));
}

TEST(RationalTest, Multiplication) {
  using T = int;
  using rational_type = Rational<T>;
  EXPECT_EQ(rational_type(1, 3) * rational_type(-3, 4), rational_type(-1, 4));
  EXPECT_EQ(rational_type(2, 5) * static_cast<T>(7), rational_type(14, 5));
  EXPECT_EQ(static_cast<T>(-2) * rational_type(1, 6), rational_type(-1, 3));

  rational_type r = rational_type(3, 7);

  r *= static_cast<T>(14);
  EXPECT_EQ(r, static_cast<T>(6));

  r *= rational_type(3, 8);
  EXPECT_EQ(r, rational_type(9, 4));
}

TEST(RationalTest, Division) {
  using T = int;
  using rational_type = Rational<T>;
  EXPECT_EQ(rational_type(-1, 20) / rational_type(4, 5), rational_type(-1, 16));
  EXPECT_EQ(rational_type(5, 6) / static_cast<T>(7), rational_type(5, 42));
  EXPECT_EQ(static_cast<T>(8) / rational_type(2, 7), static_cast<T>(28));

  EXPECT_TRUE((rational_type(23, 17) / rational_type()).is_nan());
  EXPECT_TRUE((rational_type(4, 15) / static_cast<T>(0)).is_nan());

  rational_type r = rational_type(4, 3);

  r /= rational_type(5, 4);
  EXPECT_EQ(r, rational_type(16, 15));

  r /= static_cast<T>(4);
  EXPECT_EQ(r, rational_type(4, 15));

  EXPECT_TRUE((r /= rational_type()).is_nan());
  EXPECT_TRUE((r /= static_cast<T>(0)).is_nan());

  EXPECT_EQ(rational_type(-1) / rational_type(-3), rational_type(1, 3));
}

TEST(RationalTest, AssignArithmetic) {
  using T = int;
  using rational_type = Rational<T>;

  rational_type r = rational_type(4, 3);

  r += r;
  EXPECT_EQ(r, rational_type(8, 3));

  r *= r;
  EXPECT_EQ(r, rational_type(64, 9));

  rational_type s = r;  //  avoid -Wno-self-assign
  r /= s;
  EXPECT_EQ(r, rational_type(1, 1));

  s = r;
  r -= s;
  EXPECT_EQ(r, rational_type(0, 1));

  s = r;
  EXPECT_TRUE((r /= s).is_nan());
}

TEST(RationalTest, Ostream) {
  EXPECT_EQ("nan", tensorstore::StrCat(Rational<int>::nan()));
  EXPECT_EQ("5", tensorstore::StrCat(Rational<int>(5)));
  EXPECT_EQ("22/7", tensorstore::StrCat(Rational<int>(44, 14)));
}

TEST(RationalTest, Overflow) {
  using R = Rational<int32_t>;
  {
    R r = R(2147483647) + R(1);
    EXPECT_TRUE(r.is_nan());
  }

  {
    R r = R(2147483647) - R(-1);
    EXPECT_TRUE(r.is_nan());
  }

  {
    R r = R(2147483647) * R(2);
    EXPECT_TRUE(r.is_nan());
  }

  EXPECT_EQ(R(2147483647, 2), R(2147483647) / R(2));

  {
    R r = R(2147483647, 2) * R(3);
    EXPECT_TRUE(r.is_nan());
  }

  {
    R r = R(2147483647, 2) / R(1, 3);
    EXPECT_TRUE(r.is_nan());
  }
}

TEST(UnifyDenominatorsTest, Overflow) {
  using R = Rational<int32_t>;

  int32_t num0, num1, den;
  EXPECT_FALSE(
      R::UnifyDenominators({1, 2147483647}, {1, 2147483646}, num0, num1, den));

  EXPECT_FALSE(R::UnifyDenominators(R::nan(), 1, num0, num1, den));
  EXPECT_FALSE(R::UnifyDenominators(1, R::nan(), num0, num1, den));
  EXPECT_FALSE(R::UnifyDenominators(R::nan(), R::nan(), num0, num1, den));
}

TEST(UnifyDenominatorsTest, NoOverflow) {
  using R = Rational<int32_t>;

  R r0(1, 3);
  R r1(1, 2);
  int32_t num0, num1, den;
  EXPECT_TRUE(R::UnifyDenominators(r0, r1, num0, num1, den));

  EXPECT_EQ(2, num0);
  EXPECT_EQ(3, num1);
  EXPECT_EQ(6, den);
}

TEST(FromDoubleTest, Simple) {
  using R = Rational<int64_t>;
  EXPECT_EQ(R(0), R::FromDouble(0));
  EXPECT_EQ(R(1, 2), R::FromDouble(0.5));
  EXPECT_EQ(R(1, 4), R::FromDouble(0.25));
  EXPECT_EQ(R(1, 8), R::FromDouble(0.125));
  EXPECT_EQ(R(-1), R::FromDouble(-1));
  EXPECT_EQ(R(1), R::FromDouble(1));
  EXPECT_EQ(R(5404319552844595, 18014398509481984), R::FromDouble(0.3));
  EXPECT_EQ(R(-5404319552844595, 18014398509481984), R::FromDouble(-0.3));
  for (int i = 1; i <= 62; ++i) {
    SCOPED_TRACE(tensorstore::StrCat("i=", i));
    EXPECT_EQ(R(1, static_cast<int64_t>(1) << i),
              R::FromDouble(std::ldexp(1.0, -i)));
    EXPECT_EQ(R(-1, static_cast<int64_t>(1) << i),
              R::FromDouble(std::ldexp(-1.0, -i)));
    EXPECT_EQ(R(static_cast<int64_t>(1) << i),
              R::FromDouble(std::ldexp(1.0, i)));
    EXPECT_EQ(R(static_cast<int64_t>(-1) << i),
              R::FromDouble(std::ldexp(-1.0, i)));
  }
  EXPECT_EQ(R(1, static_cast<int64_t>(1) << 53),
            R::FromDouble(0x1.0000000000000p-53));
  EXPECT_EQ(R(0), R::FromDouble(0x1.0000000000000p-63));
  EXPECT_EQ(R(884279719003555, 281474976710656), R::FromDouble(pi));
}

TEST(ApproximateTest, Simple) {
  using R = Rational<int64_t>;
  EXPECT_EQ(R(1), R(1).Approximate(100));
  EXPECT_EQ(R(-1), R(-1).Approximate(100));
  EXPECT_EQ(R(-100), R(-100).Approximate(100));
  EXPECT_EQ(R(1, 3),
            R::FromDouble(0.33333333333333333333333).Approximate(1000000));
  EXPECT_EQ(R(-1, 3),
            R::FromDouble(-0.33333333333333333333333).Approximate(1000000));
  EXPECT_EQ(R(3, 10), R::FromDouble(0.3).Approximate(1000000));
  EXPECT_EQ(R(1, 5), R::FromDouble(1.0 / 5.0).Approximate(1000000));
  EXPECT_EQ(R(22, 7), R::FromDouble(pi).Approximate(10));
  EXPECT_EQ(R(311, 99), R::FromDouble(pi).Approximate(100));
  EXPECT_EQ(R(355, 113), R::FromDouble(pi).Approximate(1000));
  EXPECT_EQ(R(312689, 99532), R::FromDouble(pi).Approximate(100000));
}

}  // namespace
