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

#ifndef TENSORSTORE_UTIL_BIT_VEC_H_
#define TENSORSTORE_UTIL_BIT_VEC_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/meta/type_traits.h"
#include "tensorstore/util/bit_span.h"
#include "tensorstore/util/bit_vec_impl.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

/// Bit vector of static or dynamic extent.
///
/// \tparam Extent Static extent (in bits), or `dynamic_extent` to specify the
///     extent at run time.  If `dynamic_extent` is specified, the bit vector is
///     stored in heap-allocated memory if the size exceeds 64 bits.  Otherwise,
///     it is stored within the BitVec object.
template <std::ptrdiff_t Extent = dynamic_extent>
class BitVec {
  using Storage = internal_bitvec::BitVecStorage<Extent>;

 public:
  using Block = internal_bitvec::Block;
  using value_type = bool;
  using difference_type = std::ptrdiff_t;
  using size_type = std::ptrdiff_t;
  using reference = BitRef<Block>;
  using const_reference = BitRef<const Block>;
  using iterator = BitIterator<Block>;
  using const_iterator = BitIterator<const Block>;
  static constexpr std::ptrdiff_t static_extent = Extent;
  static constexpr std::ptrdiff_t static_block_extent =
      Extent == dynamic_extent ? dynamic_extent
                               : BitVectorSizeInBlocks<Block>(Extent);
  using ExtentType = typename Storage::ExtentType;
  using BlockExtentType = typename Storage::BlockExtentType;

  /// Constructs with a size of `ExtentType()` and sets all bits to 0.
  ///
  /// \post If `Extent == dynamic_extent`, the size is 0.  Otherwise, the size
  ///     is `Extent`.
  BitVec() : BitVec(ExtentType{}) {}

  /// Constructs from a braced list of bool values.
  ///
  /// \requires `OtherExtent` is compatible with `Extent`.
  template <std::ptrdiff_t OtherExtent,
            typename = std::enable_if_t<(OtherExtent == Extent ||
                                         Extent == dynamic_extent)> >
  BitVec(const bool (&arr)[OtherExtent])
      : storage_(std::integral_constant<std::ptrdiff_t, OtherExtent>{}) {
    std::copy(arr, arr + OtherExtent, begin());
  }

  /// Constructs from an existing BitSpan.
  ///
  /// \requires `OtherExtent` is compatible with `Extent`.
  template <typename OtherBlock, std::ptrdiff_t OtherExtent,
            typename = std::enable_if_t<(OtherExtent == Extent ||
                                         Extent == dynamic_extent)> >
  explicit BitVec(BitSpan<OtherBlock, OtherExtent> other)
      : storage_(other.size()) {
    this->bit_span().DeepAssign(other);
  }

  /// Copy constructs from an existing BitVec.
  ///
  /// \requires `OtherExtent` is compatible with `Extent`.
  template <std::ptrdiff_t OtherExtent,
            std::enable_if_t<(OtherExtent == Extent ||
                              Extent == dynamic_extent)>* = nullptr>
  BitVec(const BitVec<OtherExtent>& other) : storage_(other.size()) {
    this->bit_span().DeepAssign(other.bit_span());
  }

  /// Constructs with a size of `extent` and all bits set to `value`.
  explicit BitVec(ExtentType extent, bool value = false) : storage_(extent) {
    fill(value);
  }

  /// Resizes to `new_size` bits.
  ///
  /// Bits `[0, min(size(), new_size))` are retained.  If `new_size > size()`,
  /// bits `[size(), new_size)` are set to `value`.
  ///
  /// The existing contents are not retained.
  void resize(ExtentType new_size, bool value = false) {
    storage_.resize(new_size, value);
  }

  /// Assigns all bits to `value`.
  void fill(bool value) {
    std::memset(
        storage_.data(),
        value ? ~static_cast<unsigned char>(0) : static_cast<unsigned char>(0),
        storage_.num_blocks() * sizeof(Block));
  }

  /// Returns a view of the `Block` array representation of the bit vector.
  span<const Block, static_block_extent> blocks() const {
    return {storage_.data(), storage_.num_blocks()};
  }
  span<Block, static_block_extent> blocks() {
    return {storage_.data(), storage_.num_blocks()};
  }

  /// Returns the length in bits.
  ExtentType size() const { return storage_.size(); }

  bool empty() const { return size() == 0; }

  template <std::ptrdiff_t OtherExtent,
            std::enable_if_t<(OtherExtent == Extent ||
                              OtherExtent == dynamic_extent)>* = nullptr>
  operator BitSpan<const Block, OtherExtent>() const {
    return {storage_.data(), 0, size()};
  }

  template <std::ptrdiff_t OtherExtent,
            std::enable_if_t<(OtherExtent == Extent ||
                              OtherExtent == dynamic_extent)>* = nullptr>
  operator BitSpan<Block, OtherExtent>() {
    return {storage_.data(), 0, size()};
  }

  /// Returns a `BitSpan` that references this bit vector.
  BitSpan<const Block, Extent> bit_span() const { return *this; }

  /// Returns a mutable `BitSpan` that references this bit vector.
  BitSpan<Block, Extent> bit_span() { return *this; }

  /// Returns a const iterator to the start of the bit vector.
  ABSL_ATTRIBUTE_ALWAYS_INLINE BitIterator<const Block> begin() const {
    return {storage_.data(), 0};
  }

  /// Returns a const iterator to the start of the bit vector.
  ABSL_ATTRIBUTE_ALWAYS_INLINE BitIterator<const Block> cbegin() const {
    return {storage_.data(), 0};
  }

  /// Returns a const iterator to one past the end of the bit vector.
  ABSL_ATTRIBUTE_ALWAYS_INLINE BitIterator<const Block> end() const {
    return {storage_.data(), storage_.size()};
  }

  /// Returns a const iterator to one past the end of the bit vector.
  ABSL_ATTRIBUTE_ALWAYS_INLINE BitIterator<const Block> cend() const {
    return {storage_.data(), storage_.size()};
  }

  /// Returns a non-const iterator to the start of the bit vector.
  ABSL_ATTRIBUTE_ALWAYS_INLINE BitIterator<Block> begin() {
    return {storage_.data(), 0};
  }

  /// Returns a non-const iterator to one past the end of the bit vector.
  ABSL_ATTRIBUTE_ALWAYS_INLINE BitIterator<Block> end() {
    return {storage_.data(), storage_.size()};
  }

  /// Returns a proxy const reference to bit `i`.
  /// \dchecks `0 <= i && i < size()`
  ABSL_ATTRIBUTE_ALWAYS_INLINE BitRef<const Block> operator[](
      std::ptrdiff_t i) const {
    return ABSL_ASSERT(i >= 0 && i <= size()), *(begin() + i);
  }

  /// Returns a proxy reference to bit `i`.
  /// \dchecks `0 <= i && i < size()`
  ABSL_ATTRIBUTE_ALWAYS_INLINE BitRef<Block> operator[](std::ptrdiff_t i) {
    return ABSL_ASSERT(i >= 0 && i <= size()), *(begin() + i);
  }

  /// Compares two bit vectors for equality.
  friend bool operator==(const BitVec& a, const BitVec& b) {
    const std::ptrdiff_t size = a.size();
    if (size != b.size()) return false;
    const std::ptrdiff_t full_blocks = size / (sizeof(Block) * 8);
    const Block* a_data = a.storage_.data();
    const Block* b_data = b.storage_.data();
    if (!std::equal(a_data, a_data + full_blocks, b_data)) {
      return false;
    }
    const Block final_mask =
        (static_cast<Block>(1) << (size % (sizeof(Block) * 8))) - 1;
    return (a_data[full_blocks] & final_mask) ==
           (b_data[full_blocks] & final_mask);
  }

  friend bool operator!=(const BitVec& a, const BitVec& b) { return !(a == b); }

 private:
  Storage storage_;
};

template <std::ptrdiff_t Extent>
BitVec(const bool (&arr)[Extent])->BitVec<Extent>;

template <typename Block, std::ptrdiff_t Extent>
BitVec(BitSpan<Block, Extent>)->BitVec<Extent>;

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_BIT_VEC_H_
