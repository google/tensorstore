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

#ifndef TENSORSTORE_UTIL_BIT_VEC_IMPL_H_
#define TENSORSTORE_UTIL_BIT_VEC_IMPL_H_

/// \file
/// Implementation details for bit_vec.h.

// IWYU pragma: private, include "third_party/tensorstore/util/bit_vec.h"

#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

#include "absl/base/macros.h"
#include "tensorstore/util/bit_span.h"

namespace tensorstore {

namespace internal_bitvec {
using Block = std::uint64_t;

template <std::ptrdiff_t Extent = dynamic_extent>
class BitVecStorage {
  static_assert(Extent >= 0, "Extent must be non-negative.");

 public:
  using ExtentType = std::integral_constant<std::ptrdiff_t, Extent>;
  constexpr static std::ptrdiff_t kNumBlocks =
      BitVectorSizeInBlocks<Block>(Extent);
  using BlockExtentType = std::integral_constant<std::ptrdiff_t, kNumBlocks>;

  constexpr BitVecStorage(ExtentType size) {}
  constexpr static ExtentType size() { return {}; }
  constexpr static BlockExtentType num_blocks() { return {}; }
  void resize(ExtentType size, bool value) {}
  Block* data() { return data_; }
  const Block* data() const { return data_; }

 private:
  Block data_[kNumBlocks];
};

template <>
class BitVecStorage<dynamic_extent> {
  constexpr static std::ptrdiff_t kMaxInlineSize = sizeof(Block) * 8;

 public:
  using BlockExtentType = std::ptrdiff_t;
  using ExtentType = std::ptrdiff_t;

  BitVecStorage(ExtentType size) : size_((ABSL_ASSERT(size >= 0), size)) {
    if (size > kMaxInlineSize) {
      data_.ptr = new Block[num_blocks()];
    }
  }

  BitVecStorage(const BitVecStorage& other) : BitVecStorage(other.size()) {
    std::memcpy(data(), other.data(), num_blocks() * sizeof(Block));
  }

  BitVecStorage(BitVecStorage&& other) noexcept {
    data_ = other.data_;
    size_ = other.size_;
    other.size_ = 0;
  }

  BitVecStorage& operator=(const BitVecStorage& other) {
    if (num_blocks() != other.num_blocks()) {
      BitVecStorage(other).swap(*this);
    } else {
      size_ = other.size();
      std::memcpy(data(), other.data(), num_blocks() * sizeof(Block));
    }
    return *this;
  }

  BitVecStorage& operator=(BitVecStorage&& other) noexcept {
    std::swap(size_, other.size_);
    std::swap(data_, other.data_);
    return *this;
  }

  void swap(BitVecStorage& other) {
    std::swap(other.data_, data_);
    std::swap(other.size_, size_);
  }

  void resize(std::ptrdiff_t new_size, bool value);

  ExtentType size() const { return size_; }
  std::ptrdiff_t num_blocks() const {
    return BitVectorSizeInBlocks<Block>(size_);
  }

  Block* data() {
    return size_ <= kMaxInlineSize ? &data_.inline_data : data_.ptr;
  }
  const Block* data() const { return const_cast<BitVecStorage*>(this)->data(); }

  ~BitVecStorage() {
    if (size_ > kMaxInlineSize) delete[] data_.ptr;
  }

 private:
  std::ptrdiff_t size_;
  union Data {
    Block inline_data;
    Block* ptr;
  };
  Data data_;
};
}  // namespace internal_bitvec

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_BIT_VEC_IMPL_H_
