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

#include <ostream>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {

KeyRange KeyRange::Prefix(std::string prefix) {
  KeyRange range;
  range.exclusive_max = PrefixExclusiveMax(prefix);
  range.inclusive_min = std::move(prefix);
  return range;
}

std::string KeyRange::PrefixExclusiveMax(std::string prefix) {
  while (!prefix.empty()) {
    auto& last_byte = prefix.back();
    if (last_byte == '\xff') {
      prefix.resize(prefix.size() - 1);
    } else {
      last_byte = static_cast<unsigned char>(last_byte) + 1;
      break;
    }
  }
  return prefix;
}

bool KeyRange::ExclusiveMaxLessEqual(absl::string_view a, absl::string_view b) {
  if (b.empty()) return true;
  if (a.empty()) return false;
  return a <= b;
}

absl::string_view KeyRange::MinExclusiveMax(absl::string_view a,
                                            absl::string_view b) {
  return ExclusiveMaxLessEqual(a, b) ? a : b;
}

bool Contains(const KeyRange& haystack, absl::string_view needle) {
  return haystack.inclusive_min <= needle &&
         (haystack.exclusive_max.empty() || needle < haystack.exclusive_max);
}

KeyRange Intersect(const KeyRange& a, const KeyRange& b) {
  const auto* a_ptr = &a;
  const auto* b_ptr = &b;
  if (a_ptr->inclusive_min > b_ptr->inclusive_min) {
    std::swap(a_ptr, b_ptr);
  }
  KeyRange result;
  result.inclusive_min = b_ptr->inclusive_min;
  result.exclusive_max = std::string(
      KeyRange::MinExclusiveMax(a_ptr->exclusive_max, b_ptr->exclusive_max));
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
         KeyRange::ExclusiveMaxLessEqual(needle.exclusive_max,
                                         haystack.exclusive_max);
}

absl::string_view LongestPrefix(const KeyRange& range) {
  absl::string_view inclusive_min = range.inclusive_min;
  absl::string_view exclusive_max = range.exclusive_max;
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

bool ContainsPrefix(const KeyRange& haystack, absl::string_view prefix) {
  return tensorstore::Contains(haystack, KeyRange::Prefix(std::string(prefix)));
}

bool IntersectsPrefix(const KeyRange& a, absl::string_view prefix) {
  return tensorstore::Intersects(a, KeyRange::Prefix(std::string(prefix)));
}

std::ostream& operator<<(std::ostream& os, const KeyRange& range) {
  return os << "[" << tensorstore::QuoteString(range.inclusive_min) << ", "
            << tensorstore::QuoteString(range.exclusive_max) << ")";
}

}  // namespace tensorstore
