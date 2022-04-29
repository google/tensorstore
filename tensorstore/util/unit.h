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

#ifndef TENSORSTORE_UTIL_UNIT_H_
#define TENSORSTORE_UTIL_UNIT_H_

#include <iosfwd>
#include <string>
#include <string_view>

namespace tensorstore {

/// Represents a physical unit, such as "nm" or "3nm" or "3 m/s".
///
/// A unit is defined by a numeric `multiplier`, represented as a double, and a
/// `base_unit`, represented as a string.  An empty string as the `base_unit`
/// indicates a dimensionless quantity.
///
/// TensorStore does not itself interpret the base unit, but it is recommended
/// to follow the syntax used by the udunits library:
///
/// https://www.unidata.ucar.edu/software/udunits/udunits-2.0.4/udunits2lib.html#Syntax
///
/// For example: "nm" or "nanometers" or "m/s" as the `base_unit`.
///
/// TensorStore does support a syntax for specifying a combined multiplier and
/// base unit as a single string.  Refer to the `Unit(std::string_view unit)`
/// constructor documentation below for details.
///
/// \relates Schema
struct Unit {
  /// Constructs a unit with an empty string as the base unit and a multiplier
  /// of 1.
  ///
  /// \id default
  Unit() = default;

  /// Parses a combined multiplier and base unit from `unit`.
  ///
  /// First, whitespace is stripped from the beginning and end of `unit`.  Then,
  /// if there is a leading number (in JSON decimal number syntax), it is
  /// removed and used as the multiplier.  The remainder (or the entire string
  /// in the case that there is no leading number) is used as the `base_unit`.
  ///
  /// For example::
  ///
  ///     EXPECT_EQ(Unit(4, "nm"), Unit("4nm"));
  ///     EXPECT_EQ(Unit(4e-3, "nm"), Unit("4e-3nm"));
  ///     EXPECT_EQ(Unit(4e-3, "nm"), Unit("+4e-3nm"));
  ///     EXPECT_EQ(Unit(-4e-3, "nm"), Unit("-4e-3nm"));
  ///     EXPECT_EQ(Unit(4.5, "nm"), Unit("4.5nm"));
  ///     EXPECT_EQ(Unit(1, "nm"), Unit("nm"));
  ///     EXPECT_EQ(Unit(4, ""), Unit("4"));
  ///     EXPECT_EQ(Unit(1, ""), Unit(""));
  ///
  /// \id string
  Unit(std::string_view unit);
  Unit(const char* unit) : Unit(std::string_view(unit)) {}
  Unit(const std::string& unit) : Unit(std::string_view(unit)) {}

  /// Constructs from a multiplier and base unit.
  ///
  /// \id multiplier, base_unit
  Unit(double multiplier, std::string base_unit)
      : multiplier(multiplier), base_unit(std::move(base_unit)) {}

  /// Multiplier relative to the `base_unit`.
  double multiplier = 1;

  /// Base unit specification.
  std::string base_unit;

  /// Prints a string representation to an `std::ostream`.
  friend std::ostream& operator<<(std::ostream& os, const Unit& unit);

  /// Compares two units for equality.
  friend bool operator==(const Unit& a, const Unit& b);
  friend bool operator!=(const Unit& a, const Unit& b) { return !(a == b); }

  /// Multiplies the `multiplier`.
  friend Unit operator*(Unit u, double x) {
    u.multiplier *= x;
    return u;
  }
  friend Unit operator*(double x, Unit u) {
    u.multiplier *= x;
    return u;
  }
  friend Unit& operator*=(Unit& u, double x) {
    u.multiplier *= x;
    return u;
  }

  /// Divides the `multiplier`.
  friend Unit operator/(Unit u, double x) {
    u.multiplier /= x;
    return u;
  }
  friend Unit& operator/=(Unit& u, double x) {
    u.multiplier /= x;
    return u;
  }

  // Reflection support.
  static constexpr auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.multiplier, x.base_unit);
  };
};

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_UNIT_H_
