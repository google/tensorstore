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

// This implementation is based on Boost.Rational, which is subject to the
// following copyright and license:
//
//  (C) Copyright Paul Moore 1999. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or
//  implied warranty, and with no claim as to its suitability for any purpose.

#ifndef TENSORSTORE_UTIL_RATIONAL_H_
#define TENSORSTORE_UTIL_RATIONAL_H_

#include <climits>
#include <cmath>
#include <ostream>

#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/division.h"

namespace tensorstore {

/// Represents a rational number (numerator / denominator), with explicit NaN
/// state.
///
/// This allows precise representation of relative scales for MultiscaleStore.
///
/// This is based on the Boost.Rational library, but differs in that it has an
/// explicit NaN state (denominator == 0) that is used to signal invalid inputs
/// or overflow.
///
/// The internal representation is always normalized: the numerator and
/// denominator are relatively prime and the denominator is positive (except to
/// indicate NaN).
///
/// \tparam I The base integer type, must be signed.
/// \ingroup utilities
template <typename I>
class Rational {
  static_assert(std::numeric_limits<I>::is_specialized &&
                std::numeric_limits<I>::is_integer &&
                std::numeric_limits<I>::is_signed);

 public:
  /// The base integer type.
  using int_type = I;

  /// Constructs from an integer.
  ///
  /// \id integer
  constexpr Rational(I value = 0) : n_(value), d_(1) {}

  /// Constructs `n / d`.
  ///
  /// \id n, d
  constexpr Rational(I n, I d) {
    if (d != 0) {
      I gcd = tensorstore::GreatestCommonDivisor(n, d);
      n /= gcd;
      d /= gcd;
      if (n == 0) {
        d = 1;
      } else if (d < 0) {
        if (n == std::numeric_limits<I>::min() ||
            d == std::numeric_limits<I>::min()) {
          d = 0;
        } else {
          d = -d;
          n = -n;
        }
      }
    }
    n_ = n;
    d_ = d;
  }

  /// Assigns to an integer.
  constexpr Rational& operator=(I value) {
    *this = Rational<I>(value);
    return *this;
  }

  /// Returns the normalized numerator.
  ///
  /// `numerator()` and `denominator()` are always relatively prime (unless
  /// `is_nan() == true`).
  constexpr I numerator() const { return n_; }

  /// Returns the normalized denominator.
  constexpr I denominator() const { return d_; }

  /// Indicates whether this is an invalid number.
  constexpr bool is_nan() const { return d_ == 0; }

  /// Returns a special not-a-number value.  All comparisons involving `nan()`
  /// return `false`.
  static constexpr Rational nan() {
    Rational r;
    r.d_ = 0;
    return r;
  }

  /// Equivalent to `*this != 0`.
  constexpr explicit operator bool() const { return d_ == 0 || n_ != 0; }

  /// Compares two rational numbers for equality. As with floating point
  /// numbers, NaN values always compare unequal.
  friend constexpr bool operator==(Rational t, Rational r) {
    return t.d_ != 0 && t.n_ == r.n_ && t.d_ == r.d_;
  }

  friend constexpr bool operator==(Rational t, I i) {
    return t.d_ == 1 && t.n_ == i;
  }

  friend constexpr bool operator==(I i, Rational t) { return (t == i); }

  /// Compares two rational numbers for inequality. As with floating point
  /// numbers, NaN values always compare unequal.
  friend constexpr bool operator!=(Rational t, Rational r) { return !(t == r); }

  friend constexpr bool operator!=(Rational t, I i) { return !(t == i); }

  friend constexpr bool operator!=(I i, Rational t) { return !(t == i); }

  /// Formats a rational number to an ostream.
  friend std::ostream& operator<<(std::ostream& os, Rational x) {
    if (x.is_nan()) return os << "nan";
    if (x.d_ == 1) return os << x.n_;
    return os << x.n_ << '/' << x.d_;
  }

  /// No-op.
  constexpr Rational operator+() const { return *this; }

  /// Negates a rational number.
  ///
  /// Returns `nan()` in the case of overflow.
  constexpr Rational operator-() const {
    Rational r;
    if (n_ == std::numeric_limits<I>::min()) return nan();
    r.n_ = -n_;
    r.d_ = d_;
    return r;
  }

  /// Adds two rational numbers.
  ///
  /// Returns `nan()` if either input is `nan()` or overflow occurs.
  friend constexpr Rational operator+(Rational t, Rational r) {
    if (t.is_nan() || r.is_nan()) return nan();
    // This calculation avoids overflow, and minimises the number of expensive
    // calculations. Thanks to Nickolay Mladenov for this algorithm.
    //
    // Proof:
    // We have to compute a/b + c/d, where gcd(a,b)=1 and gcd(b,c)=1.
    // Let g = gcd(b,d), and b = b1*g, d=d1*g. Then gcd(b1,d1)=1
    //
    // The result is (a*d1 + c*b1) / (b1*d1*g).
    // Now we have to normalize this ratio.
    // Let's assume h | gcd((a*d1 + c*b1), (b1*d1*g)), and h > 1
    // If h | b1 then gcd(h,d1)=1 and hence h|(a*d1+c*b1) => h|a.
    // But since gcd(a,b1)=1 we have h=1.
    // Similarly h|d1 leads to h=1.
    // So we have that h | gcd((a*d1 + c*b1) , (b1*d1*g)) => h|g
    // Finally we have gcd((a*d1 + c*b1), (b1*d1*g)) = gcd((a*d1 + c*b1), g)
    // Which proves that instead of normalizing the result, it is better to
    // divide num and den by gcd((a*d1 + c*b1), g)

    I g = GreatestCommonDivisor(t.d_, r.d_);
    t.d_ /= g;  // = b1 from the calculations above

    // Compute: t.n_ = t.n_ * (r.d_ / g) + r.n_ * t.d_;
    if (I temp; internal::MulOverflow(t.n_, r.d_ / g, &t.n_) ||
                internal::MulOverflow(r.n_, t.d_, &temp) ||
                internal::AddOverflow(t.n_, temp, &t.n_)) {
      return nan();
    }
    g = GreatestCommonDivisor(t.n_, g);
    t.n_ /= g;
    // Compute: t.d_ *= r.d_ / g;
    if (internal::MulOverflow(t.d_, r.d_ / g, &t.d_)) return nan();
    return t;
  }

  friend constexpr Rational operator+(Rational t, I i) {
    if (internal::MulOverflow(i, t.d_, &i) ||
        internal::AddOverflow(t.n_, i, &t.n_)) {
      return nan();
    }
    return t;
  }

  friend constexpr Rational operator+(I i, Rational t) { return t + i; }

  constexpr Rational& operator+=(Rational r) { return *this = *this + r; }
  constexpr Rational& operator+=(I i) { return *this = *this + i; }

  constexpr Rational& operator++() { return *this += 1; }

  constexpr Rational operator++(int) {
    Rational r = *this;
    *this += 1;
    return r;
  }

  /// Subtracts two rational numbers.
  ///
  /// Returns `nan()` if either input is `nan()` or overflow occurs.
  friend constexpr Rational operator-(Rational t, Rational r) {
    if (t.is_nan() || r.is_nan()) return nan();
    I g = GreatestCommonDivisor(t.d_, r.d_);
    t.d_ /= g;  // = b1 from the calculations above

    // Compute: t.n_ = t.n_ * (r.d_ / g) - r.n_ * t.d_;
    if (I temp; internal::MulOverflow(t.n_, r.d_ / g, &t.n_) ||
                internal::MulOverflow(r.n_, t.d_, &temp) ||
                internal::SubOverflow(t.n_, temp, &t.n_)) {
      return nan();
    }
    g = GreatestCommonDivisor(t.n_, g);
    t.n_ /= g;
    // Compute: t.d_ *= r.d_ / g;
    if (internal::MulOverflow(t.d_, r.d_ / g, &t.d_)) return nan();
    return t;
  }

  friend constexpr Rational operator-(Rational t, I i) {
    if (internal::MulOverflow(i, t.d_, &i) ||
        internal::SubOverflow(t.n_, i, &t.n_)) {
      return nan();
    }
    return t;
  }

  friend constexpr Rational operator-(I i, Rational r) {
    if (internal::MulOverflow(i, r.d_, &i) ||
        internal::SubOverflow(i, r.n_, &r.n_)) {
      return nan();
    }
    return r;
  }

  constexpr Rational& operator-=(Rational r) { return *this = *this - r; }
  constexpr Rational& operator-=(I i) { return *this = *this - i; }

  constexpr Rational& operator--() { return *this -= 1; }

  constexpr Rational operator--(int) {
    Rational r = *this;
    *this -= 1;
    return r;
  }

  /// Multiplies two rational numbers.
  ///
  /// Returns `nan()` if either input is `nan()` or overflow occurs.
  friend constexpr Rational operator*(Rational t, Rational r) {
    if (t.is_nan() || r.is_nan()) return nan();
    I gcd1 = GreatestCommonDivisor(t.n_, r.d_);
    I gcd2 = GreatestCommonDivisor(r.n_, t.d_);
    // Compute: t.n_ = (t.n_ / gcd1) * (r.n_ / gcd2)
    // Compute: t.d_ = (t.d_ / gcd2) * (r.d_ / gcd1)
    if (internal::MulOverflow(t.n_ / gcd1, r.n_ / gcd2, &t.n_) ||
        internal::MulOverflow(t.d_ / gcd2, r.d_ / gcd1, &t.d_)) {
      return nan();
    }
    return t;
  }

  friend constexpr Rational operator*(Rational t, I i) {
    if (t.is_nan()) return nan();
    I gcd = GreatestCommonDivisor(i, t.d_);
    if (internal::MulOverflow(t.n_, i / gcd, &t.n_)) return nan();
    t.d_ /= gcd;
    return t;
  }

  friend constexpr Rational operator*(I i, Rational t) { return t * i; }

  constexpr Rational& operator*=(Rational r) { return *this = *this * r; }
  constexpr Rational& operator*=(I i) { return *this = *this * i; }

  /// Divides two rational numbers.
  ///
  /// Returns `nan()` if either input is `nan()`, if overflow occurs, or if the
  /// divisor is 0.
  friend constexpr Rational operator/(Rational t, Rational r) {
    if (t.is_nan() || r.is_nan() || r.n_ == 0) return nan();
    I gcd1 = GreatestCommonDivisor(t.n_, r.n_);
    I gcd2 = GreatestCommonDivisor(r.d_, t.d_);
    // Compute: t.n_ = (t.n_ / gcd1) * (r.d_ / gcd2)
    // Compute: t.d_ = (t.d_ / gcd2) * (r.n_ / gcd1)
    if (internal::MulOverflow(t.n_ / gcd1, r.d_ / gcd2, &t.n_) ||
        internal::MulOverflow(t.d_ / gcd2, r.n_ / gcd1, &t.d_)) {
      return nan();
    }

    if (t.d_ < 0) {
      if (t.d_ == std::numeric_limits<I>::min() ||
          t.n_ == std::numeric_limits<I>::min()) {
        return nan();
      }
      t.n_ = -t.n_;
      t.d_ = -t.d_;
    }
    return t;
  }

  friend constexpr Rational operator/(Rational t, I i) {
    if (t.is_nan() || i == 0) return nan();
    I gcd = GreatestCommonDivisor(i, t.n_);
    t.n_ /= gcd;
    if (internal::MulOverflow(t.d_, i / gcd, &t.d_)) return nan();
    return t;
  }

  friend constexpr Rational operator/(I i, Rational r) {
    if (r.is_nan() || r.n_ == 0) return nan();
    I gcd1 = GreatestCommonDivisor(i, r.n_);
    Rational t;
    // Compute: t.n_ = (i / gcd1) * r.d_
    if (internal::MulOverflow(i / gcd1, r.d_, &t.n_)) return nan();
    t.d_ = r.n_ / gcd1;
    if (t.d_ < 0) {
      if (t.d_ == std::numeric_limits<I>::min() ||
          t.n_ == std::numeric_limits<I>::min()) {
        return nan();
      }
      t.n_ = -t.n_;
      t.d_ = -t.d_;
    }
    return t;
  }

  constexpr Rational& operator/=(Rational r) { return *this = *this / r; }
  constexpr Rational& operator/=(I i) { return *this = *this / i; }

  friend constexpr bool operator<(Rational t, Rational r) {
    if (t.is_nan() || r.is_nan()) return false;

    // Fast path if denominators are the same.
    if (t.d_ == r.d_) return t.n_ < r.n_;

    // Fast path if one input is an integer.
    if (t.d_ == 1) return t.n_ < r;
    if (r.d_ == 1) return t < r.n_;

    // Determine relative order by expanding each value to its simple continued
    // fraction representation using the Euclidian GCD algorithm.

    ContinuedFraction ts{t}, rs{r};
    bool reverse = false;

    // Loop through and compare each variable's continued-fraction components
    while (true) {
      // The quotients of the current cycle are the continued-fraction
      // components.  Comparing two c.f. is comparing their sequences,
      // stopping at the first difference.
      if (ts.q != rs.q) {
        // Since reciprocation changes the relative order of two variables,
        // and c.f. use reciprocals, the less/greater-than test reverses
        // after each index.  (Start w/ non-reversed @ whole-number place.)
        return reverse ? ts.q > rs.q : ts.q < rs.q;
      }

      // Prepare the next cycle
      reverse = !reverse;

      if (ts.r == 0 || rs.r == 0) {
        // At least one variable's c.f. expansion has ended
        break;
      }

      ts.next();
      rs.next();
    }

    // Compare infinity-valued components for otherwise equal sequences
    if (ts.r == rs.r) {
      // Both remainders are zero, so the next (and subsequent) c.f.
      // components for both sequences are infinity.  Therefore, the sequences
      // and their corresponding values are equal.
      return false;
    } else {
      // Exactly one of the remainders is zero, so all following c.f.
      // components of that variable are infinity, while the other variable
      // has a finite next c.f. component.  So that other variable has the
      // lesser value (modulo the reversal flag!).
      return (ts.r != 0) != reverse;
    }
  }

  friend constexpr bool operator<(Rational t, I i) {
    if (t.is_nan()) return false;
    // Break value into mixed-fraction form, w/ always-nonnegative remainder
    I q = t.n_ / t.d_, r = t.n_ % t.d_;
    if (r < 0) {
      r += t.d_;
      --q;
    }

    // Compare with just the quotient, since the remainder always bumps the
    // value up.  [Since q = floor(n/d), and if n/d < i then q < i, if n/d == i
    // then q == i, if n/d == i + r/d then q == i, and if n/d >= i + 1 then
    // q >= i + 1 > i; therefore n/d < i iff q < i.]
    return q < i;
  }

  friend constexpr bool operator<(I i, Rational t) {
    if (t.is_nan()) return false;
    // Break value into mixed-fraction form, w/ always-nonpositive remainder
    I q = t.n_ / t.d_, r = t.n_ % t.d_;
    if (r > 0) {
      r -= t.d_;
      ++q;
    }
    return q > i;
  }

  friend constexpr bool operator>(Rational t, Rational r) { return r < t; }

  friend constexpr bool operator>(I i, Rational t) { return t < i; }

  friend constexpr bool operator>(Rational t, I i) { return i < t; }

  friend constexpr bool operator<=(Rational t, Rational r) {
    if (t.is_nan() || r.is_nan()) return false;
    return !(r < t);
  }

  friend constexpr bool operator<=(Rational t, I r) {
    if (t.is_nan()) return false;
    return !(r < t);
  }

  friend constexpr bool operator<=(I r, Rational t) {
    if (t.is_nan()) return false;
    return !(t < r);
  }

  friend constexpr bool operator>=(Rational t, Rational r) {
    if (t.is_nan() || r.is_nan()) return false;
    return !(t < r);
  }

  friend constexpr bool operator>=(I t, Rational r) {
    if (r.is_nan()) return false;
    return !(t < r);
  }

  friend constexpr bool operator>=(Rational t, I r) {
    if (t.is_nan()) return false;
    return !(t < r);
  }

  /// Attempts to unify the denominators of `a` and `b`.
  ///
  /// \param a Input rational.
  /// \param b Other input rational.
  /// \param numerator_a[out] Set to the new numerator of `a`.
  /// \param numerator_b[out] Set to the new numerator of `b`.
  /// \param denominator[out] Set to the new denominator of `a` and `b`.
  /// \returns `true` on success (i.e. no overflow occurs), `false` if overflow
  ///     occurs or either `a` or `b` is NaN.
  [[nodiscard]] static constexpr bool UnifyDenominators(Rational a, Rational b,
                                                        I& numerator_a,
                                                        I& numerator_b,
                                                        I& denominator) {
    if (a.is_nan() || b.is_nan()) return false;
    I g = GreatestCommonDivisor(a.denominator(), b.denominator());
    I a_factor = b.denominator() / g;
    I b_factor = a.denominator() / g;
    return !(internal::MulOverflow(b_factor, b.denominator(), &denominator) ||
             internal::MulOverflow(a.numerator(), a_factor, &numerator_a) ||
             internal::MulOverflow(b.numerator(), b_factor, &numerator_b));
  }

  /// Converts a floating-point number to a rational representation.
  ///
  /// Non-finite and out-of-range values are converted to `nan()`.
  ///
  /// Values that cannot be represented exactly are rounded.
  ///
  /// To obtain an approximation with a smaller denominator, call `Approximate`
  /// on the result of this function.
  static Rational FromDouble(double value) {
    if (!std::isfinite(value)) return nan();
    constexpr int max_exponent = sizeof(I) * CHAR_BIT - 2;
    int exponent;
    double mantissa = std::frexp(value, &exponent);
    if (exponent > max_exponent + 1) return nan();
    if (exponent <= -max_exponent) return I(0);

    // Multiply `mantissa` by `2**n`, where `n` is chosen to be as large as
    // possible subject to:
    //
    // 0 <= n <= max_exponent + 1
    // 0 <= n - exponent <= max_exponent
    int n = std::min(max_exponent + 1, max_exponent + exponent);

    I numerator = static_cast<I>(std::round(std::ldexp(mantissa, n)));
    I denominator = I(1) << (n - exponent);
    return {numerator, denominator};
  }

  /// Approximates as the closest rational number with a denominator at most
  /// `max_denominator`.
  ///
  /// This is particularly useful in conjunction with `FromDouble`.
  constexpr Rational Approximate(I max_denominator) const {
    assert(max_denominator >= I(1));
    if (d_ <= max_denominator) return *this;
    using U = std::make_unsigned_t<I>;
    // This follows the approach from:
    //
    // https://en.wikipedia.org/wiki/Continued_fraction#Best_rational_approximations

    // Compute the continued fraction terms.
    U p0 = 0, q0 = 1, p1 = 1, q1 = 0;
    bool negative = false;
    U n = 0, d = d_;
    if (n_ < I(0)) {
      negative = true;
      n = ~static_cast<U>(n_) + U(1);
    } else {
      n = static_cast<U>(n_);
    }
    while (true) {
      U a = n / d;
      U r = n % d;
      U q2 = q0 + a * q1;
      if (q2 >= max_denominator) {
        // Final term.
        //
        // Determine by how much we should reduce the final term.
        U x = (max_denominator - q0) / q1;
        auto result = (x * 2 >= a)
                          ? FromReduced(static_cast<I>(p0 + x * p1),
                                        static_cast<I>(q0 + x * q1))
                          : FromReduced(static_cast<I>(p1), static_cast<I>(q1));
        if (negative) {
          result.n_ *= -1;
        }
        return result;
      }
      U p2 = p0 + a * p1;
      p0 = p1;
      q0 = q1;
      p1 = p2;
      q1 = q2;
      n = d;
      d = r;
    }
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.n_, x.d_);
  };

 private:
  // Constructs from already-reduced numerator and denominator.
  //
  // This saves the cost of the GCD computation when the factors are already
  // known to be incorrect.
  static constexpr Rational FromReduced(I n, I d) {
    Rational r;
    r.n_ = n;
    r.d_ = d;
    return r;
  }

  struct ContinuedFraction {
    constexpr explicit ContinuedFraction(Rational x)
        : n(x.n_), d(x.d_), q(x.n_ / x.d_), r(x.n_ % x.d_) {
      // Normalize negative moduli by repeatedly adding the (positive)
      // denominator and decrementing the quotient.  Later cycles should have
      // all positive values, so this only has to be done for the first cycle.
      // (The rules of C++ require a nonnegative quotient & remainder for a
      // nonnegative dividend & positive divisor.)
      if (r < 0) {
        r += d;
        --q;
      }
    }

    constexpr void next() {
      n = d;
      d = r;
      q = n / d;
      r = n % d;
    }

    I n, d, q, r;
  };

  I n_ = 0;
  I d_ = 0;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_RATIONAL_H_
