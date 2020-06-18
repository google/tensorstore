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

#include "absl/strings/string_view.h"

namespace tensorstore {

/// Specifies a range of keys according to their unsigned lexicographical
/// order.
///
/// The range includes all keys `x` satisfying
/// `inclusive_min <= x < exclusive_max`.
///
/// As a special case, if `exclusive_max.empty()`, the range has no upper
/// bound, i.e. it contains all keys `x` satisfying `inclusive_min <= x`.
///
/// Due to the special interpretation of `exclusive_max.empty()`, it is not
/// possible to represent the empty range that starts with (but does not
/// include) the empty string.  An empty range starting at (but not including)
/// any other string `x` can be represented with
/// `inclusive_min = inclusive_max = x`, but empty ranges are not normally
/// needed.
class KeyRange {
 public:
  /// Constructs the range containing all keys.
  KeyRange() = default;

  /// Constructs the specified range.
  explicit KeyRange(std::string inclusive_min, std::string exclusive_max)
      : inclusive_min(std::move(inclusive_min)),
        exclusive_max(std::move(exclusive_max)) {}

  /// Returns the range that contains all keys that start with `prefix`.
  static KeyRange Prefix(std::string prefix);

  /// Returns the `exclusive_max` value representing the upper bound for keys
  /// that start with `prefix`.
  static std::string PrefixExclusiveMax(std::string prefix);

  /// Returns `true` if `a <= b`, interpreting them as `exclusive_max` values,
  /// where an empty string is not less than any string.
  static bool ExclusiveMaxLessEqual(absl::string_view a, absl::string_view b);

  /// Returns the minimum of `a` and `b`, interpreting them as `exclusive_max`
  /// values.
  static absl::string_view MinExclusiveMax(absl::string_view a,
                                           absl::string_view b);

  /// Returns `true` if the range contains no keys.
  bool empty() const {
    return !exclusive_max.empty() && inclusive_min >= exclusive_max;
  }

  friend bool operator==(const KeyRange& a, const KeyRange& b) {
    return a.inclusive_min == b.inclusive_min &&
           a.exclusive_max == b.exclusive_max;
  }

  friend bool operator!=(const KeyRange& a, const KeyRange& b) {
    return !(a == b);
  }

  friend std::ostream& operator<<(std::ostream& os, const KeyRange& range);

  std::string inclusive_min;
  std::string exclusive_max;
};

/// Returns `true` if `haystack` contains the key `needle`.
bool Contains(const KeyRange& haystack, absl::string_view needle);

/// Returns `true` if `haystack` fully contains the range `needle`.
bool Contains(const KeyRange& haystack, const KeyRange& needle);

/// Returns `Contains(haystack, KeyRange::Prefix(prefix))`.
bool ContainsPrefix(const KeyRange& haystack, absl::string_view prefix);

/// Returns the intersection of `a` and `b`.
KeyRange Intersect(const KeyRange& a, const KeyRange& b);

/// Returns `!Intersect(a, b).empty()`.
bool Intersects(const KeyRange& a, const KeyRange& b);

/// Returns `Intersects(a, KeyRange::Prefix(b))`.
bool IntersectsPrefix(const KeyRange& a, absl::string_view prefix);

/// Returns the longest string `prefix` that satisfies
/// `Contains(range, KeyRange::Prefix(prefix))`.
absl::string_view LongestPrefix(const KeyRange& range);

}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_KEY_RANGE_H_
