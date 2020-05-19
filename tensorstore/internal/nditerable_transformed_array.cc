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

#include "tensorstore/internal/nditerable_transformed_array.h"

#include <cassert>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/iterate_impl.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_array.h"
#include "tensorstore/internal/nditerable_array_util.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/internal/unique_with_intrusive_allocator.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

namespace input_dim_iter_flags =
    internal_index_space::input_dimension_iteration_flags;

namespace {

/// NIterable implementation for TransformedArray.
///
/// After the preprocessing performed by `InitializeSingleArrayIterationState`,
/// the value for a given position within a transformed array is accessed as:
///
///   base_pointer[
///      Dot(input, input_byte_strides) +
///      Sum_{index array i}
///         index_array_output_byte_strides[i] *
///         index_array_pointers[i][Dot(input, index_array_byte_strides[i])]
///      ]
///
/// Therefore, ordering and dimension combining must take into account both the
/// `input_byte_strides` as well as the `index_array_byte_strides` of all index
/// arrays.
class IterableImpl : public NDIterable::Base<IterableImpl> {
 public:
  IterableImpl(IndexTransform<> transform, allocator_type allocator)
      : transform_(std::move(transform)),
        state_(transform_.input_rank(), transform_.output_rank()),
        input_dimension_flags_(transform_.input_rank(),
                               input_dim_iter_flags::can_skip, allocator) {}

  allocator_type get_allocator() const override {
    return input_dimension_flags_.get_allocator();
  }

  int GetDimensionOrder(DimensionIndex dim_i,
                        DimensionIndex dim_j) const override {
    // Dimensions on which at least one index array depends (which we will call
    // "index array input dimensions") should always come before other
    // dimensions, because (1) we want all the index array input dimensions
    // grouped consecutively in order to potentially combine them, and (2) it is
    // much more expensive to have an index array input dimension as the final
    // dimension (inner loop dimension).
    auto flags_i = input_dimension_flags_[dim_i];
    if ((flags_i & input_dim_iter_flags::array_indexed) !=
        (input_dimension_flags_[dim_j] & input_dim_iter_flags::array_indexed)) {
      // Only one of the two dimensions is an "index array input dimension".
      return (flags_i & input_dim_iter_flags::array_indexed) ? -2 : 2;
    }
    if (flags_i & input_dim_iter_flags::array_indexed) {
      // Both dimensions are "index array input dimensions".  Order by the first
      // non-zero byte stride within the index array.
      for (DimensionIndex i = 0; i < state_.num_array_indexed_output_dimensions;
           ++i) {
        const int order = GetDimensionOrderFromByteStrides(
            state_.index_array_byte_strides[i][dim_i],
            state_.index_array_byte_strides[i][dim_j]);
        if (order != 0) return order;
      }
    }
    // Either neither dimension is an "index array input dimension", or all
    // index array byte strides were identical.  Order by the direct array byte
    // strides, to allow the non-index array input dimensions to be potentially
    // combined.
    return GetDimensionOrderFromByteStrides(state_.input_byte_strides[dim_i],
                                            state_.input_byte_strides[dim_j]);
  }

  void UpdateDirectionPrefs(NDIterable::DirectionPref* prefs) const override {
    // Direction prefs are based on all of the index arrays as well as the
    // direct array byte strides.
    const DimensionIndex input_rank = state_.input_byte_strides.size();
    for (DimensionIndex i = 0; i < state_.num_array_indexed_output_dimensions;
         ++i) {
      UpdateDirectionPrefsFromByteStrides(
          span(state_.index_array_byte_strides[i], input_rank), prefs);
    }
    UpdateDirectionPrefsFromByteStrides(
        span(state_.input_byte_strides.data(), input_rank), prefs);
  }

  bool CanCombineDimensions(DimensionIndex dim_i, int dir_i,
                            DimensionIndex dim_j, int dir_j,
                            Index size_j) const override {
    // Two dimensions may be combined if they can be combined within all of the
    // index arrays as well as within the direct array byte strides.
    auto flags_i = input_dimension_flags_[dim_i];
    if ((flags_i & input_dim_iter_flags::array_indexed) !=
        (input_dimension_flags_[dim_j] & input_dim_iter_flags::array_indexed)) {
      // If only one of the two dimensions is an "index array input dimension",
      // they cannot be combined.
      return false;
    }
    if (flags_i & input_dim_iter_flags::array_indexed) {
      // Both dimensions are "index array input dimensions".  Check if any index
      // array prevents them from being combined.
      for (DimensionIndex i = 0; i < state_.num_array_indexed_output_dimensions;
           ++i) {
        if (!CanCombineStridedArrayDimensions(
                state_.index_array_byte_strides[i][dim_i], dir_i,
                state_.index_array_byte_strides[i][dim_j], dir_j, size_j)) {
          return false;
        }
      }
    }
    // Check if the two dimensions can be combined according to the direct array
    // byte strides.
    return CanCombineStridedArrayDimensions(
        state_.input_byte_strides[dim_i], dir_i,
        state_.input_byte_strides[dim_j], dir_j, size_j);
  }

  DataType data_type() const override { return data_type_; }

  IterationBufferConstraint GetIterationBufferConstraint(
      IterationLayoutView layout) const override {
    const DimensionIndex last_dim = layout.iteration_dimensions.back();
    if (last_dim == -1 || (input_dimension_flags_[last_dim] &
                           input_dim_iter_flags::array_indexed) == 0) {
      return {(last_dim == -1 || state_.input_byte_strides[last_dim] *
                                         layout.directions[last_dim] ==
                                     this->data_type_->size)
                  ? IterationBufferKind::kContiguous
                  : IterationBufferKind::kStrided,
              /*.external=*/false};
    } else {
      return {IterationBufferKind::kIndexed, /*.external=*/false};
    }
  }

  std::ptrdiff_t GetWorkingMemoryBytesPerElement(
      IterationLayoutView layout,
      IterationBufferKind buffer_kind) const override {
    return buffer_kind == IterationBufferKind::kIndexed ? sizeof(Index) : 0;
  }

  NDIterator::Ptr GetIterator(
      NDIterable::IterationBufferKindLayoutView layout) const override {
    return MakeUniqueWithVirtualIntrusiveAllocator<IteratorImpl>(
        get_allocator(), this, layout);
  }

  /// Iterator implementation.
  ///
  /// Uses an arena-allocated `buffer_` of `Index` values with the following
  /// contents:
  ///
  /// 1. The first `num_array_indexed_output_dimensions` elements constitute the
  ///    `remapped_index_arrays` array, which specifies the base pointers for
  ///    each of the index arrays, adjusted as necessary to account for negative
  ///    iteration directions.  These are `reinterpret_cast` to `Index` for
  ///    simplicity.
  ///
  /// 2. The next
  ///    `layout.iteration_rank() * (1 + num_array_indexed_output_dimensions)`
  ///    elements constitutes a C-order
  ///    `[1 + num_array_indexed_output_dimensions, layout.iteration_rank()]`
  ///    array `remapped_byte_strides` with the contents:
  ///
  ///    For `0 <= i < layout.iteration_rank()`:
  ///
  ///      remapped_byte_strides[0, i] =
  ///        input_byte_strides[layout.iteration_dimensions[i]] *
  ///        layout.iteration_directions[i]
  ///
  ///      For `0 <= j < num_array_indexed_output_dimensions`:
  ///
  ///        remapped_byte_strides[1 + j, i] =
  ///          index_array_byte_strides[j][layout.iteration_dimensions[i]] *
  ///           layout.iteration_directions[i]
  ///
  ///    This saves the cost of doing the additional mapping on each call to
  ///    `GetBlock`.
  ///
  /// 3. If `layout.buffer_kind == kIndexed`, an additional `layout.block_size`
  ///    elements in `buffer_` (following the `remapped_byte_strides` table) is
  ///    used for the offset array.  If `layout.iteration_dimensions.back()` is
  ///    an index array input dimension, then the offset array is recomputed on
  ///    every call to `GetBlock`.  Otherwise, it is precomputed within the
  ///    constructor via `FillOffsetsArrayFromStride`.
  class IteratorImpl : public NDIterator::Base<IteratorImpl> {
   public:
    IteratorImpl(const IterableImpl* iterable,
                 NDIterable::IterationBufferKindLayoutView layout,
                 allocator_type allocator)
        : num_index_arrays_(
              iterable->state_.num_array_indexed_output_dimensions),
          // Will be set to the minimum value such that for
          // `i >= num_index_array_iteration_dims_`,
          // `layout.iteration_dimensions[i]` is not an index array input
          // dimension.  If
          // `num_index_array_iteration_dims_ < layout.iteration_rank()`, then
          // we do not necessarily require an offset array, and if one is
          // requested, it can be precomputed in the constructor.
          num_index_array_iteration_dims_(0),
          iterable_(iterable),
          buffer_(
              // Size of `remapped_index_arrays`.
              num_index_arrays_ +
                  // Size of `remapped_byte_strides`.
                  layout.iteration_rank() * (num_index_arrays_ + 1) +
                  // Size of the offset array if required.
                  ((layout.buffer_kind == IterationBufferKind::kIndexed)
                       ? layout.block_size
                       : 0),
              allocator) {
      // Ensure that we can `reinterpret_cast` pointers as `Index` values.
      static_assert(sizeof(Index) >= sizeof(void*));
      // Compute `remapped_index_arrays` (to account for reversed dimensions).
      for (DimensionIndex j = 0; j < num_index_arrays_; ++j) {
        ByteStridedPointer<const Index> index_array_pointer =
            iterable->state_.index_array_pointers[j].get();
        for (DimensionIndex dim = 0; dim < layout.full_rank(); ++dim) {
          if (layout.directions[dim] != -1) continue;
          const Index size_minus_1 = layout.shape[dim] - 1;
          const Index index_array_byte_stride =
              iterable->state_.index_array_byte_strides[j][dim];
          index_array_pointer +=
              wrap_on_overflow::Multiply(index_array_byte_stride, size_minus_1);
        }
        buffer_[j] = reinterpret_cast<Index>(index_array_pointer.get());
      }
      // Compute the adjusted base pointer (to account for reversed dimensions).
      Index base_offset = 0;
      for (DimensionIndex dim = 0; dim < layout.full_rank(); ++dim) {
        if (layout.directions[dim] != -1) continue;
        const Index size_minus_1 = layout.shape[dim] - 1;
        const Index input_byte_stride =
            iterable->state_.input_byte_strides[dim];
        base_offset = wrap_on_overflow::Add(
            base_offset,
            wrap_on_overflow::Multiply(input_byte_stride, size_minus_1));
      }
      // Compute `remapped_byte_strides`.
      for (DimensionIndex i = 0; i < layout.iteration_rank(); ++i) {
        const DimensionIndex dim = layout.iteration_dimensions[i];
        if (dim == -1) {
          // Dummy dimension, just assign all-zero strides.
          for (DimensionIndex j = 0; j < num_index_arrays_ + 1; ++j) {
            buffer_[num_index_arrays_ + layout.iteration_rank() * j + i] = 0;
          }
        } else {
          // Compute `remapped_byte_strides[:, i]`.
          const Index dir = layout.directions[dim];
          const Index input_byte_stride =
              iterable->state_.input_byte_strides[dim];
          buffer_[num_index_arrays_ + i] =
              wrap_on_overflow::Multiply(input_byte_stride, dir);
          if (iterable->input_dimension_flags_[dim] &
              input_dim_iter_flags::array_indexed) {
            num_index_array_iteration_dims_ = i + 1;
            for (DimensionIndex j = 0; j < num_index_arrays_; ++j) {
              const Index index_array_byte_stride =
                  iterable->state_.index_array_byte_strides[j][dim];
              buffer_[num_index_arrays_ + layout.iteration_rank() * (j + 1) +
                      i] =
                  wrap_on_overflow::Multiply(index_array_byte_stride, dir);
            }
          }
        }
      }
      if (layout.buffer_kind == IterationBufferKind::kIndexed) {
        Index* offsets_array =
            buffer_.data() + num_index_arrays_ +
            layout.iteration_rank() * (num_index_arrays_ + 1);
        pointer_ = IterationBufferPointer{
            iterable->state_.base_pointer + base_offset, offsets_array};
        if (num_index_array_iteration_dims_ < layout.iteration_rank()) {
          // The last iteration dimension is not an index array input dimension.
          // Precomputed the offset array.
          FillOffsetsArrayFromStride(
              buffer_[num_index_arrays_ + layout.iteration_rank() - 1],
              span(offsets_array, layout.block_size));
        }
      } else {
        assert(num_index_array_iteration_dims_ < layout.iteration_rank());
        pointer_ = IterationBufferPointer{
            iterable->state_.base_pointer + base_offset,
            buffer_[num_index_arrays_ + layout.iteration_rank() - 1]};
      }
    }

    allocator_type get_allocator() const override {
      return buffer_.get_allocator();
    }

    Index GetBlock(span<const Index> indices, Index block_size,
                   IterationBufferPointer* pointer, Status* status) override {
      IterationBufferPointer block_pointer = pointer_;
      // Add the contribution from the direct array byte strides (corresponding
      // to `single_input_dimension` output index maps).
      block_pointer.pointer += IndexInnerProduct(
          indices.size(), indices.data(), buffer_.data() + num_index_arrays_);
      if (num_index_array_iteration_dims_ < indices.size()) {
        // The last (inner loop) iteration dimension is not an index array input
        // dimension.  Therefore, the index array output dimension maps are
        // already fully determined, and their contribution can be added here.
        // This is the more efficient case.
        for (DimensionIndex j = 0; j < num_index_arrays_; ++j) {
          const Index index = ByteStridedPointer<const Index>(
              reinterpret_cast<const Index*>(buffer_[j]))[IndexInnerProduct(
              num_index_array_iteration_dims_, indices.data(),
              buffer_.data() + num_index_arrays_ + indices.size() * (j + 1))];
          block_pointer.pointer += wrap_on_overflow::Multiply(
              iterable_->state_.index_array_output_byte_strides[j], index);
        }
      } else {
        // The last (inner loop) iteration is an index array input dimension.
        // Initialize the offset array from the last direct array byte stride.
        Index* offsets_array = const_cast<Index*>(block_pointer.byte_offsets);
        FillOffsetsArrayFromStride(
            buffer_[num_index_arrays_ + indices.size() - 1],
            span(offsets_array, block_size));
        for (DimensionIndex j = 0; j < num_index_arrays_; ++j) {
          const Index* index_array_byte_strides =
              buffer_.data() + num_index_arrays_ + indices.size() * (j + 1);
          ByteStridedPointer<const Index> index_array_pointer =
              ByteStridedPointer<const Index>(
                  reinterpret_cast<const Index*>(buffer_[j])) +
              IndexInnerProduct(indices.size() - 1, indices.data(),
                                index_array_byte_strides);
          const Index output_byte_stride =
              iterable_->state_.index_array_output_byte_strides[j];
          const Index last_index_array_byte_stride =
              index_array_byte_strides[indices.size() - 1];
          if (last_index_array_byte_stride == 0) {
            // Index array does not depend on last iteration dimension.
            // Incorporate it into the base pointer rather than into
            // `offsets_array`.
            block_pointer.pointer += wrap_on_overflow::Multiply(
                output_byte_stride, *index_array_pointer);
          } else {
            // Index array depends on the last iteration dimension.
            // Incorporate it into `offsets_array`.
            for (Index i = 0; i < block_size; ++i) {
              const Index cur_contribution = wrap_on_overflow::Multiply(
                  output_byte_stride,
                  index_array_pointer[wrap_on_overflow::Multiply(
                      i, last_index_array_byte_stride)]);
              offsets_array[i] =
                  wrap_on_overflow::Add(offsets_array[i], cur_contribution);
            }
          }
        }
      }
      *pointer = block_pointer;
      return block_size;
    }

   private:
    DimensionIndex num_index_arrays_;
    DimensionIndex num_index_array_iteration_dims_;
    const IterableImpl* iterable_;
    IterationBufferPointer pointer_;
    std::vector<Index, ArenaAllocator<Index>> buffer_;
  };

  /// Maintains ownership of the array data.
  std::shared_ptr<const void> data_owner_;
  IndexTransform<> transform_;
  // TODO(jbms): Use arena allocator for SingleArrayIterationState as well.
  internal_index_space::SingleArrayIterationState state_;
  DataType data_type_;
  std::vector<input_dim_iter_flags::Bitmask,
              ArenaAllocator<input_dim_iter_flags::Bitmask>>
      input_dimension_flags_;
};

Result<NDIterable::Ptr> MaybeConvertToArrayNDIterable(
    std::unique_ptr<IterableImpl, VirtualDestroyDeleter> impl, Arena* arena) {
  if (impl->state_.num_array_indexed_output_dimensions == 0) {
    return GetArrayNDIterable(
        SharedOffsetArrayView<const void>(
            SharedElementPointer<const void>(
                std::shared_ptr<const void>(std::move(impl->data_owner_),
                                            impl->state_.base_pointer),
                impl->data_type_),
            StridedLayoutView<>(impl->transform_.input_rank(),
                                impl->transform_.input_shape().data(),
                                impl->state_.input_byte_strides.data())),
        arena);
  }
  return impl;
}

}  // namespace

Result<NDIterable::Ptr> GetTransformedArrayNDIterable(
    TransformedArrayView<Shared<const void>> array, Arena* arena) {
  if (!array.has_transform()) {
    return GetArrayNDIterable({std::move(array.element_pointer()),
                               array.untransformed_strided_layout()},
                              arena);
  }

  auto impl = MakeUniqueWithVirtualIntrusiveAllocator<IterableImpl>(
      ArenaAllocator<>(arena), array.transform());
  TENSORSTORE_RETURN_IF_ERROR(InitializeSingleArrayIterationState(
      array.base_array(),
      internal_index_space::TransformAccess::rep(impl->transform_),
      impl->transform_.input_origin().data(),
      impl->transform_.input_shape().data(), &impl->state_,
      impl->input_dimension_flags_.data()));
  impl->data_type_ = array.data_type();
  impl->data_owner_ = std::move(array.element_pointer().pointer());
  return MaybeConvertToArrayNDIterable(std::move(impl), arena);
}

Result<NDIterable::Ptr> GetNormalizedTransformedArrayNDIterable(
    NormalizedTransformedArray<Shared<const void>> array, Arena* arena) {
  auto impl = MakeUniqueWithVirtualIntrusiveAllocator<IterableImpl>(
      ArenaAllocator<>(arena), std::move(array.transform()));
  TENSORSTORE_RETURN_IF_ERROR(InitializeSingleArrayIterationState(
      ElementPointer<const void>(array.element_pointer()),
      internal_index_space::TransformAccess::rep(impl->transform_),
      impl->transform_.input_origin().data(),
      impl->transform_.input_shape().data(), &impl->state_,
      impl->input_dimension_flags_.data()));
  impl->data_type_ = array.data_type();
  impl->data_owner_ = std::move(array.element_pointer().pointer());
  return MaybeConvertToArrayNDIterable(std::move(impl), arena);
}

}  // namespace internal
}  // namespace tensorstore
