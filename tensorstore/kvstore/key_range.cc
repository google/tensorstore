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

#include "tensorstore/kvstore/key_range.h"

#include <stddef.h>

#include <algorithm>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/strings/match.h"
#include "absl/types/compare.h"
#include "tensorstore/internal/compare.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace {

// returns a part of the prefix; to construct a prefix the
// std::string_view::back() should be incremented.
std::string_view PartialPrefix(std::string_view prefix) {
  while (!prefix.empty() && prefix.back() == '\xff') {
    prefix.remove_suffix(1);
  }
  return prefix;
}

// Returns the minimum of `a` and `b`, interpreting them as `exclusive_max`
// values.
std::string_view MinExclusiveMax(std::string_view a, std::string_view b) {
  return KeyRange::CompareExclusiveMax(a, b) < 0 ? a : b;
}

}  // namespace

KeyRange KeyRange::Prefix(std::string prefix) {
  KeyRange range;
  range.exclusive_max = PrefixExclusiveMax(prefix);
  range.inclusive_min = std::move(prefix);
  return range;
}

std::string KeyRange::Successor(std::string_view key) {
  std::string successor;
  successor.reserve(key.size() + 1);
  successor.append(key);
  successor += '\x00';
  return successor;
}

KeyRange KeyRange::Singleton(std::string key) {
  auto exclusive_max = Successor(key);
  return KeyRange(std::move(key), std::move(exclusive_max));
}

bool KeyRange::is_singleton() const {
  return exclusive_max.size() == (inclusive_min.size() + 1) &&
         exclusive_max.back() == '\x00' &&
         std::string_view(exclusive_max).substr(0, inclusive_min.size()) ==
             inclusive_min;
}

bool KeyRange::is_non_empty_prefix() const {
  std::string_view prefix = PartialPrefix(inclusive_min);
  return !full() && exclusive_max.size() == prefix.size() &&
         (prefix.empty() ||
          (exclusive_max.back() == (prefix.back() + 1) &&
           std::string_view(exclusive_max).substr(0, prefix.size() - 1) ==
               prefix.substr(0, prefix.size() - 1)));
}

std::string KeyRange::PrefixExclusiveMax(std::string_view prefix) {
  std::string prefix_copy(PartialPrefix(prefix));
  if (!prefix_copy.empty()) {
    auto& last_byte = prefix_copy.back();
    last_byte = static_cast<unsigned char>(last_byte) + 1;
  }
  return prefix_copy;
}

absl::weak_ordering KeyRange::CompareKeyAndExclusiveMax(
    std::string_view key, std::string_view bound) {
  return bound.empty()
             ? absl::weak_ordering::less
             : internal::CompareResultAsWeakOrdering(key.compare(bound));
}

absl::weak_ordering KeyRange::CompareExclusiveMax(std::string_view a,
                                                  std::string_view b) {
  return a.empty() != b.empty()
             ? (a.empty() ? absl::weak_ordering::greater
                          : absl::weak_ordering::less)
             : internal::CompareResultAsWeakOrdering(a.compare(b));
}

bool Contains(const KeyRange& haystack, std::string_view needle) {
  return haystack.inclusive_min <= needle &&
         KeyRange::CompareKeyAndExclusiveMax(needle, haystack.exclusive_max) <
             0;
}

KeyRange Intersect(const KeyRange& a, const KeyRange& b) {
  const auto* a_ptr = &a;
  const auto* b_ptr = &b;
  if (a_ptr->inclusive_min > b_ptr->inclusive_min) {
    std::swap(a_ptr, b_ptr);
  }
  KeyRange result;
  result.inclusive_min = b_ptr->inclusive_min;
  result.exclusive_max =
      std::string(MinExclusiveMax(a_ptr->exclusive_max, b_ptr->exclusive_max));
  if (result.empty()) {
    result.exclusive_max = result.inclusive_min;
  }
  return result;
}

bool Intersects(const KeyRange& a, const KeyRange& b) {
  return !Intersect(a, b).empty();
}

bool Contains(const KeyRange& haystack, const KeyRange& needle) {
  return haystack.inclusive_min <= needle.inclusive_min &&
         KeyRange::CompareExclusiveMax(needle.exclusive_max,
                                       haystack.exclusive_max) <= 0;
}

std::string_view LongestPrefix(const KeyRange& range) {
  std::string_view inclusive_min = range.inclusive_min;
  std::string_view exclusive_max = range.exclusive_max;
  size_t i = 0;
  if (exclusive_max.empty()) {
    // The range has no upper bound.  The common prefix is the longest prefix of
    // `inclusive_min` that is equal to 255.
    while (i < inclusive_min.size() && inclusive_min[i] == '\xff') {
      ++i;
    }
  } else {
    // The range has an upper bound.  First, find the longest common prefix of
    // `inclusive_min` and `exclusive_max`, which is also a *prefix* of the
    // longest prefix of the range.
    size_t min_length = std::min(inclusive_min.size(), exclusive_max.size());
    while (i < min_length && inclusive_min[i] == exclusive_max[i]) {
      ++i;
    }
    // If the common prefix of `inclusive_min` and `exclusive_max` includes all
    // but the last byte `i` of `exclusive_max`, and
    // `exclusive_max[i] == inclusive_min[i] + 1`, then the longest prefix of
    // the range additionally includes position `i`.
    if (i + 1 == min_length && inclusive_min[i] != '\xff' &&
        static_cast<unsigned char>(inclusive_min[i]) + 1 ==
            static_cast<unsigned char>(exclusive_max[i])) {
      ++i;
      // Any subsequent bytes of `inclusive_min` equal to 255 are also part of
      // the longest prefix.
      while (i < inclusive_min.size() && inclusive_min[i] == '\xff') {
        ++i;
      }
    }
  }
  return inclusive_min.substr(0, i);
}

bool ContainsPrefix(const KeyRange& haystack, std::string_view prefix) {
  return tensorstore::Contains(haystack, KeyRange::Prefix(std::string(prefix)));
}

bool IntersectsPrefix(const KeyRange& a, std::string_view prefix) {
  return tensorstore::Intersects(a, KeyRange::Prefix(std::string(prefix)));
}

std::ostream& operator<<(std::ostream& os, const KeyRange& range) {
  return os << "[" << tensorstore::QuoteString(range.inclusive_min) << ", "
            << tensorstore::QuoteString(range.exclusive_max) << ")";
}

KeyRange KeyRange::AddPrefix(std::string_view prefix, KeyRange range) {
  if (prefix.empty()) return range;
  range.inclusive_min.insert(0, prefix);
  if (range.exclusive_max.empty()) {
    range.exclusive_max = KeyRange::PrefixExclusiveMax(std::string(prefix));
  } else {
    range.exclusive_max.insert(0, prefix);
  }
  return range;
}

KeyRange KeyRange::RemovePrefix(std::string_view prefix, KeyRange range) {
  if (prefix.empty()) return range;
  if (prefix >= range.inclusive_min) {
    range.inclusive_min.clear();
  } else {
    if (!absl::StartsWith(range.inclusive_min, prefix)) return EmptyRange();
    range.inclusive_min.erase(0, prefix.size());
  }
  const auto c = CompareKeyAndExclusiveMax(prefix, range.exclusive_max);
  if (c < 0) {
    if (absl::StartsWith(range.exclusive_max, prefix)) {
      range.exclusive_max.erase(0, prefix.size());
    } else {
      range.exclusive_max.clear();
    }
  } else {
    return EmptyRange();
  }
  return range;
}

KeyRange KeyRange::RemovePrefixLength(size_t n, const KeyRange& range) {
  std::string_view inclusive_min(range.inclusive_min);
  if (n < inclusive_min.size()) {
    inclusive_min.remove_prefix(n);
  } else {
    inclusive_min = {};
  }
  std::string_view exclusive_max(range.exclusive_max);
  if (n < exclusive_max.size()) {
    exclusive_max.remove_prefix(n);
  } else {
    exclusive_max = {};
  }
  return KeyRange(std::string(inclusive_min), std::string(exclusive_max));
}

}  // namespace tensorstore
