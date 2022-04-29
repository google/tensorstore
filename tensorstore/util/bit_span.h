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

#ifndef TENSORSTORE_UTIL_BIT_SPAN_H_
#define TENSORSTORE_UTIL_BIT_SPAN_H_

/// \file
/// Defines BitSpan, a view of a packed bit sequence.
///
/// TODO(jbms): Use the fact that all 64-bit architectures use only the low 48
/// bits of pointers in order to use a single 64-bit value to store a
/// BitRef/BitIterator.

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include "tensorstore/internal/attributes.h"
#include "tensorstore/util/small_bit_set.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

namespace internal_bit_span {
template <bool FillValue, typename T>
void FillBits(T* base, std::ptrdiff_t offset, std::ptrdiff_t size) {
  constexpr std::ptrdiff_t kBitsPerBlock = sizeof(T) * 8;
  constexpr const T kAllOnes = ~static_cast<T>(0);
  assert(offset >= 0);
  std::ptrdiff_t end;
  for (base += offset / kBitsPerBlock, offset %= kBitsPerBlock,
       end = size + offset;
       end >= kBitsPerBlock; ++base, offset = 0, end -= kBitsPerBlock) {
    const T mask = kAllOnes << offset;
    if (FillValue) {
      *base |= mask;
    } else {
      *base &= ~mask;
    }
  }
  if (end) {
    const T mask = (kAllOnes << offset) ^ (kAllOnes << (end % kBitsPerBlock));
    if (FillValue) {
      *base |= mask;
    } else {
      *base &= ~mask;
    }
  }
}

template <typename T, typename U>
void CopyBits(const U* source, std::ptrdiff_t source_offset, T* dest,
              std::ptrdiff_t dest_offset, std::ptrdiff_t size) {
  // TODO(jbms): implement more efficient copy
  std::copy(BitIterator<const U>(source, source_offset),
            BitIterator<const U>(source, source_offset + size),
            BitIterator<T>(dest, dest_offset));
}
}  // namespace internal_bit_span

/// Unowned view of a packed bit sequence.
///
/// The packed bit sequence is represented by a `base` pointer (of type `T*`) to
/// the packed bit storage, a non-negative bit `offset` (of type
/// `std::ptrdiff_t`), and a `size` (of type `std::ptrdiff_t`).  Bit `i` (where
/// `0 <= i < size`) of the bit sequence corresponds to bit
/// `(offset + i) % kBitsPerBlock` of `base[(offset + i) / kBitsPerBlock]`.
///
/// \tparam T The unsigned integer "block" type used to store the packed bits.
///     If not `const`-qualified, the view is mutable.
/// \tparam Extent The static extent of the sequence, or `dynamic_extent` to
///     specify the extent at run time.
/// \requires `std::is_unsigned_v<T>`.
/// \requires `Extent >= 0 || Extent == dynamic_extent`.
/// \ingroup Utilities
template <typename T, std::ptrdiff_t Extent = dynamic_extent>
class BitSpan {
  static_assert(std::is_unsigned_v<T>, "Storage type T must be unsigned.");
  static_assert(Extent == dynamic_extent || Extent >= 0,
                "Extent must be dynamic_extent or >= 0.");

 public:
  using ExtentType =
      std::conditional_t<Extent == dynamic_extent, std::ptrdiff_t,
                         std::integral_constant<std::ptrdiff_t, Extent>>;

  using size_type = std::ptrdiff_t;
  using difference_type = std::ptrdiff_t;
  using iterator = BitIterator<T>;
  using const_iterator = BitIterator<const T>;
  using pointer = BitIterator<T>;
  using const_pointer = BitIterator<T>;
  using value_type = bool;
  using reference = BitRef<T>;
  using base_type = T;
  using element_type = std::conditional_t<std::is_const_v<T>, const bool, bool>;
  constexpr static std::ptrdiff_t kBitsPerBlock = sizeof(T) * 8;
  constexpr static std::ptrdiff_t static_extent = Extent;

  /// Constructs from a base pointer, a bit offset, and a size.
  ///
  /// \param base Base pointer to the stored blocks containing the packed bits.
  /// \param offset The offset in bits from the first (least significant) bit of
  ///     `*base`.  It is valid to specify `offset > kBitsPerBlock`.
  /// \param size Number of bits in the sequence.
  /// \dchecks `offset >= 0`.
  /// \dchecks `size >= 0`.
  /// \dchecks `Extent == dynamic_extent || Extent == size`.
  constexpr BitSpan(T* base TENSORSTORE_LIFETIME_BOUND, std::ptrdiff_t offset,
                    std::ptrdiff_t size)
      : BitSpan(BitIterator<T>(base, offset), size) {}

  /// Constructs from an iterator and size.
  ///
  /// \param begin Iterator pointing to first bit.
  /// \param size Number of bits in the sequence.
  /// \dchecks `Extent == dynamic_extent || Extent == size`.
  constexpr BitSpan(BitIterator<T> begin, std::ptrdiff_t size) : begin_(begin) {
    if constexpr (Extent == dynamic_extent) {
      assert(size >= 0);
      size_ = size;
    } else {
      assert(size == Extent);
    }
  }

  /// Constructs from a compatible existing BitSpan.
  ///
  /// \requires `T` is `U` or `const U`.
  /// \requires `E == Extent || Extent == dynamic_extent`.
  template <
      typename U, std::ptrdiff_t E,
      std::enable_if_t<((std::is_same_v<T, U> || std::is_same_v<T, const U>)&&(
          E == Extent || Extent == dynamic_extent))>* = nullptr>
  constexpr BitSpan(BitSpan<U, E> other)
      : begin_(other.begin()), size_(other.size()) {}

  /// Returns the base pointer to the packed bit representation.
  constexpr T* base() const { return begin().base(); }

  /// Returns the offset in bits from `base()` to the first element.
  constexpr std::ptrdiff_t offset() const { return begin().offset(); }

  /// Returns the size in bits.
  constexpr ExtentType size() const { return size_; }

  BitIterator<T> begin() const { return begin_; }
  BitIterator<T> end() const { return begin_ + size_; }

  /// Returns a `BitRef` to bit `i`.
  ///
  /// \dchecks `i >=0 && i < size()`.
  constexpr BitRef<T> operator[](std::ptrdiff_t i) const {
    assert(i >= 0 && i <= size());
    return *(begin() + i);
  }

  /// Sets all bits to `FillValue`.
  ///
  /// \requires `!std::is_const_v<T>`.
  template <bool FillValue, int&... ExplicitArgumentBarrier, typename X = T>
  std::enable_if_t<!std::is_const_v<X>> fill() const {
    internal_bit_span::FillBits<FillValue>(base(), offset(), size());
  }

  /// Sets all bits to `value`.
  ///
  /// \requires `!std::is_const_v<T>`.
  template <int&... ExplicitArgumentBarrier, typename X = T>
  std::enable_if_t<!std::is_const_v<X>> fill(bool value) const {
    if (value) {
      fill<true>();
    } else {
      fill<false>();
    }
  }

  /// Copies the contents of `other` to the memory referenced by `*this`.
  ///
  /// \requires `!std::is_const_v<T>`.
  /// \dchecks `size() == other.size()`.
  template <typename U, std::ptrdiff_t E, int&... ExplicitArgumentBarrier,
            typename X = T>
  std::enable_if_t<!std::is_const_v<X> &&
                   (E == Extent || Extent == dynamic_extent ||
                    E == dynamic_extent)>
  DeepAssign(BitSpan<U, E> other) {
    assert(other.size() == size());
    internal_bit_span::CopyBits(other.base(), other.offset(), base(), offset(),
                                size());
  }

 private:
  BitIterator<T> begin_;
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS ExtentType size_;
};

/// Returns the number of elements of type `Block` required to store a bitvector
/// of the specified `length`.
///
/// \tparam Block The unsigned integer block type.
/// \relates BitSpan
template <typename Block>
inline constexpr std::ptrdiff_t BitVectorSizeInBlocks(std::ptrdiff_t length) {
  return (length + sizeof(Block) * 8 - 1) / (sizeof(Block) * 8);
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_BIT_SPAN_H_
