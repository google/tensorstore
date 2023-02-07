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

#ifndef TENSORSTORE_KVSTORE_KEY_RANGE_H_
#define TENSORSTORE_KVSTORE_KEY_RANGE_H_

#include <iosfwd>
#include <string>
#include <string_view>

namespace tensorstore {

/// Specifies a range of keys according to their unsigned lexicographical
/// order.
///
/// The range includes all keys ``x`` satisfying
/// ``inclusive_min <= x < exclusive_max``.
///
/// As a special case, if `exclusive_max.empty()`, the range has no upper bound,
/// i.e. it contains all keys ``x`` satisfying ``inclusive_min <= x``.
///
/// Due to the special interpretation of `exclusive_max.empty()`, it is not
/// possible to represent the empty range that starts with (but does not
/// include) the empty string.  An empty range starting at (but not including)
/// any other string ``x`` can be represented with
/// ``inclusive_min = inclusive_max = x``, but empty ranges are not normally
/// needed.
///
/// \ingroup kvstore
class KeyRange {
 public:
  /// Constructs the range containing all keys.
  ///
  /// \id default
  KeyRange() = default;

  /// Returns a range that contains no keys.
  static KeyRange EmptyRange() {
    return KeyRange(std::string(1, '\0'), std::string(1, '\0'));
  }

  /// Constructs the specified range.
  ///
  /// \id inclusive_min, exclusive_max
  explicit KeyRange(std::string inclusive_min, std::string exclusive_max)
      : inclusive_min(std::move(inclusive_min)),
        exclusive_max(std::move(exclusive_max)) {}

  /// Returns the range that contains all keys that start with `prefix`.
  static KeyRange Prefix(std::string prefix);

  /// Adds a prefix to both `inclusive_min` and `exclusive_max`.
  static KeyRange AddPrefix(std::string_view prefix, KeyRange range);

  /// Returns the range corresponding to the set of keys ``k`` for which
  /// ``prefix + k`` is a member of `range`.
  ///
  /// For example::
  ///
  ///     RemovePrefix("a/", KeyRange("a/b", "a/d")) == KeyRange("b", "d")
  ///     RemovePrefix("a/b", KeyRange("a/b", "a/d")) == KeyRange()
  ///     RemovePrefix("a/d", KeyRange("a/b", "a/d")) == EmptyRange()
  ///     RemovePrefix("a/bc", KeyRange("a/b", "a/bb")) == KeyRange("", "b")
  static KeyRange RemovePrefix(std::string_view prefix, KeyRange range);

  /// Returns the key that occurs immediately after `key`.
  ///
  /// This is equal to `key` with a `0` (NUL) byte appended.
  static std::string Successor(std::string_view key);

  /// Returns the `exclusive_max` value representing the upper bound for keys
  /// that start with `prefix`.
  static std::string PrefixExclusiveMax(std::string prefix);

  /// Returns the three-way comparison result between a key and an exclusive max
  /// bound.
  static int CompareKeyAndExclusiveMax(std::string_view key,
                                       std::string_view bound);
  static int CompareExclusiveMaxAndKey(std::string_view bound,
                                       std::string_view key) {
    return -CompareKeyAndExclusiveMax(key, bound);
  }

  /// Returns the three-way comparison result between two `exclusive_max`
  /// values.
  static int CompareExclusiveMax(std::string_view a, std::string_view b);

  /// Returns the minimum of `a` and `b`, interpreting them as `exclusive_max`
  /// values.
  static std::string_view MinExclusiveMax(std::string_view a,
                                          std::string_view b);

  /// Returns `true` if the range contains no keys.
  bool empty() const {
    return !exclusive_max.empty() && inclusive_min >= exclusive_max;
  }

  /// Returns `true` if the range contains all keys.
  bool full() const { return exclusive_max.empty() && inclusive_min.empty(); }

  /// Compares two ranges for equality.
  friend bool operator==(const KeyRange& a, const KeyRange& b) {
    return a.inclusive_min == b.inclusive_min &&
           a.exclusive_max == b.exclusive_max;
  }
  friend bool operator!=(const KeyRange& a, const KeyRange& b) {
    return !(a == b);
  }

  /// Prints a debugging string representation to an `std::ostream`.
  friend std::ostream& operator<<(std::ostream& os, const KeyRange& range);

  // Reflection support.
  static constexpr auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.inclusive_min, x.exclusive_max);
  };

  /// Inclusive lower bound of the range.
  std::string inclusive_min;

  /// Exclusive upper bound of the range, or an empty string to indicate no
  /// upper bound.
  std::string exclusive_max;
};

/// Returns `true` if `haystack` contains the key `needle`.
///
/// \relates KeyRange
/// \id KeyRange, string
bool Contains(const KeyRange& haystack, std::string_view needle);

/// Returns `true` if `haystack` fully contains the range `needle`.
///
/// \relates KeyRange
/// \id KeyRange, KeyRange
bool Contains(const KeyRange& haystack, const KeyRange& needle);

/// Returns `Contains(haystack, KeyRange::Prefix(prefix))`.
///
/// \relates KeyRange
/// \id KeyRange, string
bool ContainsPrefix(const KeyRange& haystack, std::string_view prefix);

/// Returns the intersection of `a` and `b`.
///
/// \relates KeyRange
/// \id KeyRange
KeyRange Intersect(const KeyRange& a, const KeyRange& b);

/// Returns `!Intersect(a, b).empty()`.
///
/// \relates KeyRange
/// \id KeyRange
bool Intersects(const KeyRange& a, const KeyRange& b);

/// Returns `Intersects(a, KeyRange::Prefix(prefix))`.
///
/// \relates KeyRange
/// \id KeyRange, string
bool IntersectsPrefix(const KeyRange& a, std::string_view prefix);

/// Returns the longest string ``prefix`` that satisfies
/// ``Contains(range, KeyRange::Prefix(prefix))``.
///
/// \relates KeyRange
/// \id KeyRange
std::string_view LongestPrefix(const KeyRange& range);

}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_KEY_RANGE_H_
