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

#include "tensorstore/internal/nditerable_array.h"

#include <stddef.h>

#include <cassert>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_array_util.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/internal/unique_with_intrusive_allocator.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

// Uncomment the following line to disable specializing `StridedIteratorImpl`
// for small ranks.
//
// #define TENSORSTORE_NDITERABLE_DISABLE_ARRAY_OPTIMIZE

namespace tensorstore {
namespace internal {

namespace {

/// Computes the `byte_strides` for an iterator based on the original byte
/// strides and iteration layout.
Index ComputeIteratorBaseOffsetAndByteStrides(
    NDIterable::IterationLayoutView layout, span<const Index> orig_byte_strides,
    Index* byte_strides) {
  assert(layout.full_rank() == orig_byte_strides.size());
  Index base_offset = 0;
  for (DimensionIndex dim = 0; dim < layout.full_rank(); ++dim) {
    const int dir = layout.directions[dim];
    if (dir == -1) {
      base_offset = wrap_on_overflow::Add(
          base_offset, wrap_on_overflow::Multiply(layout.shape[dim] - 1,
                                                  orig_byte_strides[dim]));
    }
  }
  for (DimensionIndex i = 0; i < layout.iteration_rank(); ++i) {
    const DimensionIndex dim = layout.iteration_dimensions[i];
    if (dim == -1) {
      byte_strides[i] = 0;
    } else {
      byte_strides[i] = orig_byte_strides[dim] * layout.directions[dim];
    }
  }
  return base_offset;
}

template <DimensionIndex Rank>
class StridedIteratorImpl;

template <DimensionIndex Rank = -1>
class StridedIteratorImplBase
    : public NDIterator::Base<StridedIteratorImpl<Rank>> {
 public:
  explicit StridedIteratorImplBase(DimensionIndex rank,
                                   ArenaAllocator<> allocator)
      : allocator_(allocator) {}

  ArenaAllocator<> get_allocator() const override { return allocator_; }

 protected:
  ArenaAllocator<> allocator_;
  std::array<Index, Rank> byte_strides_;
};

template <>
class StridedIteratorImplBase<-1>
    : public NDIterator::Base<StridedIteratorImpl<-1>> {
 public:
  explicit StridedIteratorImplBase(DimensionIndex rank,
                                   ArenaAllocator<> allocator)
      : byte_strides_(rank, allocator) {}

  ArenaAllocator<> get_allocator() const override {
    return byte_strides_.get_allocator();
  }

 protected:
  std::vector<Index, ArenaAllocator<Index>> byte_strides_;
};

template <DimensionIndex Rank = -1>
class StridedIteratorImpl : public StridedIteratorImplBase<Rank> {
  using Base = StridedIteratorImplBase<Rank>;

  using Base::byte_strides_;

 public:
  StridedIteratorImpl(ByteStridedPointer<void> data,
                      span<const Index> orig_byte_strides,
                      NDIterable::IterationLayoutView layout,
                      ArenaAllocator<> allocator)
      : Base(layout.iteration_rank(), allocator) {
    data_ = data + ComputeIteratorBaseOffsetAndByteStrides(
                       layout, orig_byte_strides, byte_strides_.data());
  }

  bool GetBlock(span<const Index> indices, IterationBufferShape block_shape,
                IterationBufferPointer* pointer,
                absl::Status* status) override {
    Index offset;
    if constexpr (Rank == -1) {
      offset = IndexInnerProduct(indices.size(), byte_strides_.data(),
                                 indices.data());
    } else {
      offset = IndexInnerProduct<Rank>(byte_strides_.data(), indices.data());
    }
    *pointer = IterationBufferPointer{data_ + offset,
                                      byte_strides_[byte_strides_.size() - 2],
                                      byte_strides_[byte_strides_.size() - 1]};
    return true;
  }

 private:
  ByteStridedPointer<void> data_;
};

class IndexedIteratorImpl : public NDIterator::Base<IndexedIteratorImpl> {
 public:
  IndexedIteratorImpl(ByteStridedPointer<void> data,
                      span<const Index> orig_byte_strides,
                      NDIterable::IterationBufferLayoutView layout,
                      ArenaAllocator<> allocator)
      : block_inner_size_(layout.block_shape[1]),
        buffer_(layout.iteration_rank() +
                    layout.block_shape[0] * layout.block_shape[1],
                allocator) {
    data_ = data + ComputeIteratorBaseOffsetAndByteStrides(
                       layout, orig_byte_strides, buffer_.data());
    FillOffsetsArrayFromStride(buffer_[layout.iteration_rank() - 2],
                               buffer_[layout.iteration_rank() - 1],
                               layout.block_shape[0], layout.block_shape[1],
                               buffer_.data() + layout.iteration_rank());
  }

  ArenaAllocator<> get_allocator() const override {
    return buffer_.get_allocator();
  }

  bool GetBlock(span<const Index> indices, IterationBufferShape block_shape,
                IterationBufferPointer* pointer,
                absl::Status* status) override {
    *pointer = IterationBufferPointer{
        data_ +
            IndexInnerProduct(indices.size(), buffer_.data(), indices.data()),
        block_inner_size_, buffer_.data() + indices.size()};
    return true;
  }

 private:
  ByteStridedPointer<void> data_;
  Index block_inner_size_;
  std::vector<Index, ArenaAllocator<Index>> buffer_;
};

class ArrayIterableImpl : public NDIterable::Base<ArrayIterableImpl> {
 public:
  ArrayIterableImpl(SharedOffsetArrayView<const void> array,
                    ArenaAllocator<> allocator)
      : dtype_(array.dtype()),
        byte_strides_(array.byte_strides().begin(), array.byte_strides().end(),
                      allocator) {
    void* origin_pointer =
        const_cast<void*>(array.byte_strided_origin_pointer().get());
    data_ = std::shared_ptr<void>(std::move(array.pointer()), origin_pointer);
  }
  ArenaAllocator<> get_allocator() const override {
    return byte_strides_.get_allocator();
  }

  int GetDimensionOrder(DimensionIndex dim_i,
                        DimensionIndex dim_j) const override {
    return GetDimensionOrderFromByteStrides(byte_strides_[dim_i],
                                            byte_strides_[dim_j]);
  }

  void UpdateDirectionPrefs(NDIterable::DirectionPref* prefs) const override {
    UpdateDirectionPrefsFromByteStrides(byte_strides_, prefs);
  }

  bool CanCombineDimensions(DimensionIndex dim_i, int dir_i,
                            DimensionIndex dim_j, int dir_j,
                            Index size_j) const override {
    return CanCombineStridedArrayDimensions(
        byte_strides_[dim_i], dir_i, byte_strides_[dim_j], dir_j, size_j);
  }

  DataType dtype() const override { return dtype_; }

  IterationBufferConstraint GetIterationBufferConstraint(
      IterationLayoutView layout) const override {
    const DimensionIndex last_dim = layout.iteration_dimensions.back();
    return {(last_dim == -1 ||
             (byte_strides_[last_dim] * layout.directions[last_dim] ==
              dtype_->size))
                ? IterationBufferKind::kContiguous
                : IterationBufferKind::kStrided,
            /*.external=*/false};
  }

  std::ptrdiff_t GetWorkingMemoryBytesPerElement(
      IterationLayoutView layout,
      IterationBufferKind buffer_kind) const override {
    return buffer_kind == IterationBufferKind::kIndexed ? sizeof(Index) : 0;
  }

  NDIterator::Ptr GetIterator(
      IterationBufferKindLayoutView layout) const override {
    if (layout.buffer_kind == IterationBufferKind::kIndexed) {
      return MakeUniqueWithVirtualIntrusiveAllocator<IndexedIteratorImpl>(
          get_allocator(), data_.get(), byte_strides_, layout);
    }
    const auto make_strided_iterator = [&](auto rank) {
      return MakeUniqueWithVirtualIntrusiveAllocator<
          StridedIteratorImpl<decltype(rank)::value>>(
          get_allocator(), data_.get(), byte_strides_, layout);
    };
    switch (layout.iteration_rank()) {
#ifndef TENSORSTORE_NDITERABLE_DISABLE_ARRAY_OPTIMIZE
      case 2:
        return make_strided_iterator(
            std::integral_constant<DimensionIndex, 2>{});
      case 3:
        return make_strided_iterator(
            std::integral_constant<DimensionIndex, 3>{});
#endif  // !defined(TENSORSTORE_NDITERABLE_DISABLE_ARRAY_OPTIMIZE)
      default:
        assert(layout.iteration_rank() > 1);
        return make_strided_iterator(
            std::integral_constant<DimensionIndex, -1>{});
    }
  }

 private:
  std::shared_ptr<void> data_;
  DataType dtype_;
  std::vector<Index, ArenaAllocator<Index>> byte_strides_;
};

}  // namespace

NDIterable::Ptr GetArrayNDIterable(SharedOffsetArrayView<const void> array,
                                   Arena* arena) {
  return MakeUniqueWithVirtualIntrusiveAllocator<ArrayIterableImpl>(
      ArenaAllocator<>(arena), std::move(array));
}

}  // namespace internal
}  // namespace tensorstore
