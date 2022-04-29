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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ostream>
#include <type_traits>

#include "tensorstore/internal/attributes.h"
#include "tensorstore/internal/integer_types.h"

namespace tensorstore {

/// Mutable or constant reference to a single bit of a packed bit sequence.
///
/// `BitRef` is bound to a bit location by the constructor, and cannot be
/// rebound.
///
/// \tparam T The unsigned integer "block" type used to store the packed bits.
///     A mutable reference is indicated by an unqualified type,
///     e.g. `BitRef<uint32_t>`, while a ``const`` reference is indicated by
///     a ``const``-qualified type, e.g. `BitRef<const uint32_t>`.
/// \requires `std::is_unsigned_v<T>`.
/// \relates SmallBitSet
template <typename T>
class BitRef {
  static_assert(std::is_unsigned_v<T>, "Storage type T must be unsigned.");

 public:
  friend class BitRef<const T>;
  /// Block type used to represent the bits.
  using block_type = T;

  /// Element type of the reference.
  using value_type = bool;
  using element_type = bool;

  /// Number of bits stored per `T` value.
  constexpr static ptrdiff_t kBitsPerBlock = sizeof(T) * 8;

  /// Binds to bit `offset % kBitsPerBlock` of `*block`.
  constexpr BitRef(T* block TENSORSTORE_LIFETIME_BOUND, ptrdiff_t offset)
      : block_(block), mask_(static_cast<T>(1) << (offset % kBitsPerBlock)) {
    assert(offset >= 0);
  }

  /// Returns the value of the bound bit.
  constexpr operator bool() const { return *block_ & mask_; }

  /// Sets the bound bit to `value`, leaving all other bits unchanged.
  ///
  /// \id bool
  const BitRef& operator=(bool value) const {
    *block_ = value ? (*block_ | mask_) : (*block_ & ~mask_);
    return *this;
  }

  /// Equivalent to `*this = static_cast<bool>(value)`.
  ///
  /// .. note::
  ///
  ///    This does not rebind the `BitRef`.
  /// \id BitRef
  const BitRef& operator=(BitRef value) const {
    return (*this = static_cast<bool>(value));
  }

  /// Swaps the referenced bit with a `bool` value.
  friend void swap(BitRef a, bool& x) {
    bool temp = a;
    a = x;
    x = temp;
  }
  friend void swap(bool& x, BitRef a) {
    bool temp = a;
    a = x;
    x = temp;
  }

 private:
  T* block_;
  T mask_;
};

/// Swaps the contents of the bit to which `a` refers with the contents of the
/// bit to which `b` refers (does not rebind `a` or `b`).
///
/// \relates BitRef
template <typename T, typename U>
std::enable_if_t<(!std::is_const_v<T> && !std::is_const_v<U>)> swap(
    BitRef<T> a, BitRef<U> b) {
  bool temp = a;
  a = b;
  b = temp;
}

// Additional overload to serve as a better match than `std::swap` when both
// Block types are the same.
template <typename T>
std::enable_if_t<(!std::is_const_v<T>)> swap(BitRef<T> a, BitRef<T> b) {
  bool temp = a;
  a = b;
  b = temp;
}

// Specialization for const reference case.
template <typename T>
class BitRef<const T> {
  static_assert(std::is_unsigned_v<T>, "Storage type T must be unsigned.");

 public:
  using block_type = const T;
  using value_type = bool;
  using element_type = bool;
  constexpr static ptrdiff_t kBitsPerBlock = sizeof(T) * 8;

  /// Binds to bit `offset % kBitsPerBlock` of `*base`.
  constexpr BitRef(const T* block TENSORSTORE_LIFETIME_BOUND, ptrdiff_t offset)
      : block_(block), mask_(static_cast<T>(1) << (offset % kBitsPerBlock)) {
    assert(offset >= 0);
  }

  constexpr BitRef(BitRef<T> other)
      : block_(other.block_), mask_(other.mask_) {}

  constexpr operator bool() const { return *block_ & mask_; }

  BitRef& operator=(BitRef) = delete;

 private:
  const T* block_;
  T mask_;
};

/// Iterator within a packed bit sequence.
///
/// An iterator is represented by a `base` pointer (of type `T*`) to the packed
/// bit storage and a non-negative `offset` (of type `ptrdiff_t`) in bits, and
/// corresponds to bit `offset % kBitsPerBlock` of
/// `base[offset / kBitsPerBlock]`.
///
/// Advancing the iterator changes the `offset` while leaving the `base` pointer
/// unchanged.  Only two iterators with the same `base` pointer may be compared.
///
/// \tparam T The unsigned integer "block" type used to store the packed bits.
///     If ``const``-qualified, the iterator cannot be used to modify the
///     sequence.
/// \requires `std::is_unsigned_v<T>`.
/// \relates SmallBitSet
template <typename T>
class BitIterator {
  static_assert(std::is_unsigned_v<T>, "Storage type T must be unsigned.");

 public:
  /// Proxy pointer type.
  using pointer = BitIterator<T>;
  using const_pointer = BitIterator<const T>;

  /// Proxy reference type.
  using reference = BitRef<T>;
  using const_reference = BitRef<const T>;

  /// Difference type.
  using difference_type = ptrdiff_t;

  /// Value type of iterator.
  using value_type = bool;

  /// Iterator category.
  using iterator_category = std::random_access_iterator_tag;

  /// Number of bits stored per block of type `T`.
  constexpr static ptrdiff_t kBitsPerBlock = sizeof(T) * 8;

  /// Constructs an invalid iterator.
  ///
  /// \id default
  constexpr BitIterator() : base_(nullptr), offset_(0) {}

  /// Constructs from a base pointer and offset.
  ///
  /// \id base, offset
  constexpr BitIterator(T* base TENSORSTORE_LIFETIME_BOUND, ptrdiff_t offset)
      : base_(base), offset_(offset) {}

  /// Converts from a non-``const`` iterator.
  ///
  /// \id convert
  template <typename U, std::enable_if_t<std::is_same_v<const U, T>>* = nullptr>
  constexpr BitIterator(BitIterator<U> other)
      : base_(other.base()), offset_(other.offset()) {}

  /// Returns the base pointer.
  constexpr T* base() const { return base_; }

  /// Returns the bit offset relative to `base`.
  constexpr ptrdiff_t offset() const { return offset_; }

  /// Returns a proxy reference to the bit referenced by this iterator.
  constexpr BitRef<T> operator*() const {
    return BitRef<T>(base() + offset() / kBitsPerBlock, offset());
  }

  /// Returns a proxy reference to the bit at the specified `offset`.
  constexpr BitRef<T> operator[](ptrdiff_t offset) const {
    return *(*this + offset);
  }

  /// Pre-increment operator.
  ///
  /// \membergroup Arithmetic operations
  BitIterator& operator++() {
    ++offset_;
    return *this;
  }

  /// Pre-decrement operator.
  ///
  /// \membergroup Arithmetic operations
  BitIterator& operator--() {
    --offset_;
    return *this;
  }

  /// Post-increment operator.
  ///
  /// \membergroup Arithmetic operations
  BitIterator operator++(int) {
    BitIterator temp = *this;
    ++offset_;
    return temp;
  }

  /// Post-decrement operator.
  ///
  /// \membergroup Arithmetic operations
  BitIterator operator--(int) {
    BitIterator temp = *this;
    --offset_;
    return temp;
  }

  /// Adds an offset to an iterator.
  ///
  /// \membergroup Arithmetic operations
  friend BitIterator operator+(BitIterator it, ptrdiff_t offset) {
    it += offset;
    return it;
  }
  friend BitIterator operator+(ptrdiff_t offset, BitIterator it) {
    it += offset;
    return it;
  }
  BitIterator& operator+=(ptrdiff_t x) {
    offset_ += x;
    return *this;
  }

  /// Subtracts an offset from an iterator.
  ///
  /// \membergroup Arithmetic operations
  /// \id offset
  friend BitIterator operator-(BitIterator it, ptrdiff_t offset) {
    it -= offset;
    return it;
  }
  BitIterator& operator-=(ptrdiff_t x) {
    offset_ -= x;
    return *this;
  }

  /// Returns the distance from `b` to `a`.
  ///
  /// \dchecks `a.base() == b.base()`.
  /// \id iterator
  friend constexpr ptrdiff_t operator-(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() - b.offset();
  }

  /// Compares the positions of two iterators.
  ///
  /// \dchecks `a.base() == b.base()`.
  friend constexpr bool operator==(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() == b.offset();
  }
  friend constexpr bool operator!=(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() != b.offset();
  }
  friend constexpr bool operator<(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() < b.offset();
  }
  friend constexpr bool operator<=(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() <= b.offset();
  }
  friend constexpr bool operator>(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() > b.offset();
  }
  friend constexpr bool operator>=(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() >= b.offset();
  }

 private:
  T* base_;
  ptrdiff_t offset_;
};

/// Bit set that fits in a single unsigned integer.
///
/// This is similar to `std::bitset<N>`, but supports iterators, and for
/// simplicity is limited to a single unsigned integer.
///
/// \ingroup Utilities
template <size_t N>
class SmallBitSet {
 public:
  /// `N`-bit unsigned integer type used to represent the bit set.
  using Bits = typename internal::uint_type<N>::type;

  /// Container value type.
  using value_type = bool;

  /// Proxy reference type.
  using reference = BitRef<Bits>;
  using const_reference = BitRef<const Bits>;

  /// Iterator type.
  using iterator = BitIterator<Bits>;
  using const_iterator = BitIterator<const Bits>;

  /// Constructs an all-zero vector.
  ///
  /// \id default
  constexpr SmallBitSet() : bits_(0) {}

  /// Constructs a vector with all bits set to `value`.
  ///
  /// \id bool
  template <typename T,
            // Prevent narrowing conversions to `bool`.
            typename = std::enable_if_t<std::is_same_v<T, bool>>>
  constexpr SmallBitSet(T value) : bits_(value * ~Bits(0)) {}

  /// Constructs a vector from the specified bool array.
  ///
  /// Can be invoked with a braced list, e.g. `SmallBitSet<8>({0, 1, 1, 0})`.
  ///
  /// \id array
  template <size_t NumBits, typename = std::enable_if_t<(NumBits <= N)>>
  constexpr SmallBitSet(const bool (&bits)[NumBits]) {
    Bits v = 0;
    for (size_t i = 0; i < NumBits; ++i) {
      v |= Bits(bits[i]) << i;
    }
    bits_ = v;
  }

  /// Constructs a vector from an unsigned integer.
  ///
  /// \membergroup constructors
  static constexpr SmallBitSet FromBits(Bits bits) {
    SmallBitSet v;
    v.bits_ = bits;
    return v;
  }

  /// Constructs from a range of `bool` values.
  ///
  /// \dchecks Size of `range` is not greater than `N`
  /// \membergroup constructors
  template <typename Range>
  static constexpr SmallBitSet FromRange(Range&& range) {
    SmallBitSet set;
    size_t i = 0;
    for (bool value : range) {
      set[i++] = value;
    }
    return set;
  }

  /// Sets all bits to the specified value.
  constexpr SmallBitSet& operator=(bool value) {
    bits_ = ~Bits(0) * value;
    return *this;
  }

  /// Returns the begin/end iterators.
  constexpr iterator begin() { return iterator(&bits_, 0); }
  constexpr iterator end() { return iterator(&bits_, N); }
  constexpr const_iterator begin() const { return const_iterator(&bits_, 0); }
  constexpr const_iterator end() const { return const_iterator(&bits_, N); }

  /// Returns the static size, `N`.
  constexpr static size_t size() { return N; }

  /// Returns a reference to an individual bit.
  ///
  /// \dchecks `offset >= 0 && offset < N`
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

  /// Computes the complement of the set.
  ///
  /// \membergroup Set operations
  friend constexpr SmallBitSet operator~(SmallBitSet v) {
    return SmallBitSet::FromBits(~v.bits_);
  }

  /// Computes the intersection of two sets.
  ///
  /// \membergroup Set operations
  friend constexpr SmallBitSet operator&(SmallBitSet a, SmallBitSet b) {
    return SmallBitSet::FromBits(a.bits_ & b.bits_);
  }
  friend constexpr SmallBitSet& operator&=(SmallBitSet& a, SmallBitSet b) {
    a.bits_ &= b.bits_;
    return a;
  }

  /// Computes the exclusive OR of two sets.
  ///
  /// \membergroup Set operations
  friend constexpr SmallBitSet operator^(SmallBitSet a, SmallBitSet b) {
    return SmallBitSet::FromBits(a.bits_ ^ b.bits_);
  }
  friend constexpr SmallBitSet& operator^=(SmallBitSet& a, SmallBitSet b) {
    a.bits_ ^= b.bits_;
    return a;
  }

  /// Computes the union of two sets.
  ///
  /// \membergroup Set operations
  friend constexpr SmallBitSet operator|(SmallBitSet a, SmallBitSet b) {
    return SmallBitSet::FromBits(a.bits_ | b.bits_);
  }
  friend constexpr SmallBitSet& operator|=(SmallBitSet& a, SmallBitSet b) {
    a.bits_ |= b.bits_;
    return a;
  }

  /// Compares two sets for equality.
  friend constexpr bool operator==(SmallBitSet a, SmallBitSet b) {
    return a.bits_ == b.bits_;
  }
  friend constexpr bool operator!=(SmallBitSet a, SmallBitSet b) {
    return !(a == b);
  }

  /// Returns the bitvector as an unsigned integer.
  Bits bits() const { return bits_; }

  /// Returns the set ``[0, k)``.
  ///
  /// \dchecks `k <= N`
  /// \membergroup Constructors
  static constexpr SmallBitSet UpTo(size_t k) {
    assert(k <= N);
    return k == 0 ? SmallBitSet() : FromBits(~Bits(0) << (N - k) >> (N - k));
  }

  /// Prints to an output stream.
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
