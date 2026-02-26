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

#ifndef TENSORSTORE_INTERNAL_URI_ASCII_SET_H_
#define TENSORSTORE_INTERNAL_URI_ASCII_SET_H_

#include <stdint.h>

#include <cassert>
#include <string_view>

namespace tensorstore {
namespace internal_uri {

/// Set of ASCII characters (0-127) represented as a bit vector.
class AsciiSet {
 public:
  /// Constructs an empty set.
  constexpr AsciiSet() : bitvec_{0, 0} {}

  /// Constructs a set of the characters in `s`.
  constexpr AsciiSet(std::string_view s) : bitvec_{0, 0} {
    for (char c : s) {
      Set(c);
    }
  }

  /// Returns `true` if `c` is in the set.
  constexpr bool Test(char c) const {
    auto uc = static_cast<unsigned char>(c);
    if (uc >= 128) return false;
    return (bitvec_[(uc & 64) ? 1 : 0] >> (uc & 63)) & 1;
  }

  constexpr bool operator()(char c) const { return Test(c); }

  friend constexpr AsciiSet operator|(AsciiSet a, AsciiSet b) {
    a.bitvec_[0] |= b.bitvec_[0];
    a.bitvec_[1] |= b.bitvec_[1];
    return a;
  }
  friend constexpr AsciiSet operator&(AsciiSet a, AsciiSet b) {
    a.bitvec_[0] &= b.bitvec_[0];
    a.bitvec_[1] &= b.bitvec_[1];
    return a;
  }
  friend constexpr AsciiSet operator^(AsciiSet a, AsciiSet b) {
    a.bitvec_[0] ^= b.bitvec_[0];
    a.bitvec_[1] ^= b.bitvec_[1];
    return a;
  }
  friend constexpr AsciiSet operator~(AsciiSet a) {
    a.bitvec_[0] ^= 0xFFFFFFFFFFFFFFFF;
    a.bitvec_[1] ^= 0xFFFFFFFFFFFFFFFF;
    return a;
  }

 private:
  // Adds a character to the set.
  constexpr void Set(char c) {
    auto uc = static_cast<unsigned char>(c);
    bitvec_[(uc & 64) ? 1 : 0] |= static_cast<uint64_t>(1) << (uc & 63);
  }

  uint64_t bitvec_[2];
};

}  // namespace internal_uri
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_URI_ASCII_SET_H_
