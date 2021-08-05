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

#ifndef TENSORSTORE_UTIL_SMALL_BIT_SET_H_
#define TENSORSTORE_UTIL_SMALL_BIT_SET_H_

#include <cstdint>
#include <ostream>

#include "tensorstore/internal/integer_types.h"
#include "tensorstore/util/bit_span.h"

namespace tensorstore {

/// Bit set that fits in a single unsigned integer.
///
/// This is similar to `std::bitset<N>`, but supports iterators, and for
/// simplicity is limited to a single unsigned integer.
template <size_t N>
class SmallBitSet {
 public:
  using Bits = typename internal::uint_type<N>::type;

  using value_type = bool;
  using reference = BitRef<Bits>;
  using const_reference = BitRef<const Bits>;
  using iterator = BitIterator<Bits>;
  using const_iterator = BitIterator<const Bits>;
  using size_type = size_t;
  using difference_type = size_t;

  /// Constructs an all-zero vector.
  constexpr SmallBitSet() : bits_(0) {}

  /// Constructs a vector with all bits set to `value`.
  template <typename T,
            // Prevent narrowing conversions to `bool`.
            typename = std::enable_if_t<std::is_same_v<T, bool>>>
  constexpr SmallBitSet(T value) : bits_(value * ~Bits(0)) {}

  /// Constructs a vector from the specified bool array.
  ///
  /// Can be invoked with a braced list, e.g. `SmallBitSet<8>({0, 1, 1, 0})`.
  template <size_t NumBits, typename = std::enable_if_t<(NumBits <= N)>>
  constexpr SmallBitSet(const bool (&bits)[NumBits]) {
    Bits v = 0;
    for (size_t i = 0; i < NumBits; ++i) {
      v |= Bits(bits[i]) << i;
    }
    bits_ = v;
  }

  /// Constructs a vector from an unsigned integer.
  static constexpr SmallBitSet FromBits(Bits bits) {
    SmallBitSet v;
    v.bits_ = bits;
    return v;
  }

  /// Sets all bits to the specified value.
  constexpr SmallBitSet& operator=(bool value) {
    bits_ = ~Bits(0) * value;
    return *this;
  }

  constexpr iterator begin() { return iterator(&bits_, 0); }
  constexpr iterator end() { return iterator(&bits_, N); }
  constexpr const_iterator begin() const { return const_iterator(&bits_, 0); }
  constexpr const_iterator end() const { return const_iterator(&bits_, N); }

  constexpr size_t size() const { return N; }

  constexpr reference operator[](size_t offset) {
    assert(offset >= 0 && offset < N);
    return reference(&bits_, offset);
  }
  constexpr const_reference operator[](size_t offset) const {
    assert(offset >= 0 && offset < N);
    return const_reference(&bits_, offset);
  }

  /// Returns `true` if any bit is set.
  explicit operator bool() const { return static_cast<bool>(bits_); }

  friend constexpr SmallBitSet operator~(SmallBitSet v) {
    return SmallBitSet::FromBits(~v.bits_);
  }
  friend constexpr SmallBitSet operator&(SmallBitSet a, SmallBitSet b) {
    return SmallBitSet::FromBits(a.bits_ & b.bits_);
  }
  friend constexpr SmallBitSet operator^(SmallBitSet a, SmallBitSet b) {
    return SmallBitSet::FromBits(a.bits_ ^ b.bits_);
  }
  friend constexpr SmallBitSet operator|(SmallBitSet a, SmallBitSet b) {
    return SmallBitSet::FromBits(a.bits_ | b.bits_);
  }
  friend constexpr SmallBitSet& operator&=(SmallBitSet& a, SmallBitSet b) {
    a.bits_ &= b.bits_;
    return a;
  }
  friend constexpr SmallBitSet& operator^=(SmallBitSet& a, SmallBitSet b) {
    a.bits_ ^= b.bits_;
    return a;
  }
  friend constexpr SmallBitSet& operator|=(SmallBitSet& a, SmallBitSet b) {
    a.bits_ |= b.bits_;
    return a;
  }
  friend constexpr bool operator==(SmallBitSet a, SmallBitSet b) {
    return a.bits_ == b.bits_;
  }
  friend constexpr bool operator!=(SmallBitSet a, SmallBitSet b) {
    return !(a == b);
  }

  /// Returns the bitvector as an unsigned integer.
  Bits bits() const { return bits_; }

  friend std::ostream& operator<<(std::ostream& os, SmallBitSet v) {
    for (size_t i = 0; i < N; ++i) {
      os << (static_cast<bool>(v[i]) ? '1' : '0');
    }
    return os;
  }

 private:
  Bits bits_;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_SMALL_BIT_SET_H_
