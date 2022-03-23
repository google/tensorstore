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

#include "tensorstore/index_space/internal/transform_array.h"

#include "absl/status/status.h"
#include "tensorstore/index_space/internal/iterate_impl.h"
#include "tensorstore/index_space/internal/propagate_bounds.h"
#include "tensorstore/index_space/internal/transform_rep_impl.h"

namespace tensorstore {
namespace internal_index_space {

Result<SharedElementPointer<const void>> TransformArraySubRegion(
    const SharedArrayView<const void, dynamic_rank, offset_origin>& array,
    TransformRep* transform, const Index* result_origin,
    const Index* result_shape, Index* result_byte_strides,
    TransformArrayConstraints constraints) {
  const DimensionIndex output_rank = array.rank();
  const DimensionIndex input_rank =
      transform ? transform->input_rank : array.rank();

  // Early exit if result has zero elements.  This is not purely an
  // optimization: the iteration logic below does not correctly handle zero-size
  // dimensions.
  for (DimensionIndex i = 0; i < input_rank; ++i) {
    if (result_shape[i] == 0) {
      std::fill_n(result_byte_strides, input_rank, 0);
      return SharedElementPointer<const void>(std::shared_ptr<const void>(),
                                              array.dtype());
    }
  }

  namespace flags = input_dimension_iteration_flags;

  absl::FixedArray<flags::Bitmask, internal::kNumInlinedDims>
      input_dimension_flags(
          input_rank,
          flags::GetDefaultBitmask(constraints.repeated_elements_constraint()));

  std::array<std::optional<SingleArrayIterationState>, 2> single_array_states;
  single_array_states[0].emplace(input_rank, output_rank);
  TENSORSTORE_RETURN_IF_ERROR(
      internal_index_space::InitializeSingleArrayIterationState(
          /*array=*/array,
          /*transform=*/transform,
          /*iteration_origin=*/result_origin,
          /*iteration_shape=*/result_shape, &*single_array_states[0],
          input_dimension_flags.data()));

  if (single_array_states[0]->num_array_indexed_output_dimensions == 0) {
    // No index arrays are actually needed.
    if (constraints.allocate_constraint() != must_allocate) {
      // We can just return a view of the existing array.
      std::copy_n(single_array_states[0]->input_byte_strides.data(), input_rank,
                  result_byte_strides);
      return SharedElementPointer<void>(
          std::shared_ptr<void>(array.pointer(),
                                single_array_states[0]->base_pointer),
          array.element_pointer().dtype());
    }
    const StridedLayoutView<> source_layout(
        input_rank, result_shape,
        single_array_states[0]->input_byte_strides.data());
    const StridedLayoutView<> new_layout(input_rank, result_shape,
                                         result_byte_strides);
    auto element_pointer = internal::AllocateArrayLike(
        array.element_pointer().dtype(), source_layout, result_byte_strides,
        constraints.iteration_constraints(), default_init);
    CopyArray(ArrayView<const void>(
                  ElementPointer<void>(single_array_states[0]->base_pointer,
                                       array.element_pointer().dtype()),
                  source_layout),
              ArrayView<void>(element_pointer, new_layout));
    return element_pointer;
  }

  MarkSingletonDimsAsSkippable(span(result_shape, input_rank),
                               input_dimension_flags.data());

  SharedElementPointer<void> new_element_pointer;

  single_array_states[1].emplace(input_rank, output_rank);

  if (constraints.order_constraint()) {
    absl::FixedArray<Index, internal::kNumInlinedDims> new_shape(input_rank);
    for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
      new_shape[input_dim] = input_dimension_flags[input_dim] == flags::can_skip
                                 ? 1
                                 : result_shape[input_dim];
    }
    ComputeStrides(constraints.order_constraint().order(), array.dtype()->size,
                   new_shape, span(result_byte_strides, input_rank));
    for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
      if (new_shape[input_dim] <= 1) result_byte_strides[input_dim] = 0;
    }

    const Index new_origin_offset =
        IndexInnerProduct(input_rank, result_byte_strides, result_origin);

    new_element_pointer = internal::AllocateAndConstructSharedElements(
        ProductOfExtents(span(new_shape)), default_init, array.dtype());
    const absl::Status init_status =
        internal_index_space::InitializeSingleArrayIterationState(
            ArrayView<void, dynamic_rank, offset_origin>(
                AddByteOffset(ElementPointer<void>(new_element_pointer),
                              -new_origin_offset),
                StridedLayoutView<dynamic_rank, offset_origin>(
                    input_rank, result_origin, new_shape.data(),
                    result_byte_strides)),
            /*transform=*/nullptr,
            /*iteration_origin=*/result_origin,
            /*iteration_shape=*/result_shape, &*single_array_states[1],
            input_dimension_flags.data());
    assert(init_status.ok());
  }
  DimensionIterationOrder base_layout =
      constraints.order_constraint()
          ? ComputeDimensionIterationOrder<2>(single_array_states,
                                              input_dimension_flags,
                                              /*order_constraint=*/{})
          : ComputeDimensionIterationOrder<1>({&single_array_states[0], 1},
                                              input_dimension_flags,
                                              /*order_constraint=*/{});
  if (!constraints.order_constraint()) {
    absl::FixedArray<Index, internal::kNumInlinedDims> new_shape(
        base_layout.pure_strided_end_dim);
    absl::FixedArray<Index, internal::kNumInlinedDims> new_byte_strides(
        base_layout.pure_strided_end_dim);
    for (DimensionIndex i = 0; i < base_layout.pure_strided_end_dim; ++i) {
      const DimensionIndex input_dim = base_layout.input_dimension_order[i];
      new_shape[i] = result_shape[input_dim];
    }
    std::fill_n(result_byte_strides, input_rank, 0);
    ComputeStrides(ContiguousLayoutOrder::c, array.dtype()->size, new_shape,
                   new_byte_strides);
    for (DimensionIndex i = 0; i < base_layout.pure_strided_end_dim; ++i) {
      const DimensionIndex input_dim = base_layout.input_dimension_order[i];
      result_byte_strides[input_dim] = new_byte_strides[i];
    }
    new_element_pointer = internal::AllocateAndConstructSharedElements(
        ProductOfExtents(span(new_shape)), default_init, array.dtype());
    const Index new_origin_offset =
        IndexInnerProduct(input_rank, result_byte_strides, result_origin);
    const absl::Status init_status =
        internal_index_space::InitializeSingleArrayIterationState(
            ArrayView<void, dynamic_rank, offset_origin>(
                AddByteOffset(ElementPointer<void>(new_element_pointer),
                              -new_origin_offset),
                StridedLayoutView<dynamic_rank, offset_origin>(
                    input_rank, result_origin, new_shape.data(),
                    result_byte_strides)),
            /*transform=*/nullptr,
            /*iteration_origin=*/result_origin,
            /*iteration_shape=*/result_shape, &*single_array_states[1],
            input_dimension_flags.data());
    assert(init_status.ok());
  }

  SimplifiedDimensionIterationOrder layout = SimplifyDimensionIterationOrder<2>(
      base_layout, span(result_shape, input_rank), single_array_states);

  const std::array<std::ptrdiff_t, 2> element_sizes{array.dtype()->size,
                                                    array.dtype()->size};

  [[maybe_unused]] const bool success =
      IterateUsingSimplifiedLayout<2>(layout, span(result_shape, input_rank),
                                      {&array.dtype()->copy_assign, nullptr},
                                      /*status=*/nullptr, single_array_states,
                                      element_sizes)
          .success;
  assert(success);

  return new_element_pointer;
}

Result<SharedElementPointer<const void>> TransformArrayPreservingOrigin(
    SharedArrayView<const void, dynamic_rank, offset_origin> array,
    TransformRep* transform, Index* result_origin, Index* result_shape,
    Index* result_byte_strides, TransformArrayConstraints constraints) {
  const DimensionIndex input_rank =
      transform ? transform->input_rank : array.rank();
  TENSORSTORE_RETURN_IF_ERROR(PropagateExplicitBounds(
      /*b=*/array.domain(),
      /*a_to_b=*/transform,
      /*a=*/MutableBoxView<>(input_rank, result_origin, result_shape)));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto element_pointer,
      TransformArraySubRegion(array, transform, result_origin, result_shape,
                              result_byte_strides, constraints));
  return AddByteOffset(std::move(element_pointer),
                       -IndexInnerProduct(transform->input_rank,
                                          result_byte_strides, result_origin));
}

Result<SharedElementPointer<const void>> TransformArrayDiscardingOrigin(
    SharedArrayView<const void, dynamic_rank, offset_origin> array,
    TransformRep* transform, Index* result_shape, Index* result_byte_strides,
    TransformArrayConstraints constraints) {
  const DimensionIndex input_rank =
      transform ? transform->input_rank : array.rank();
  absl::FixedArray<Index, internal::kNumInlinedDims> result_origin(input_rank);
  TENSORSTORE_RETURN_IF_ERROR(PropagateExplicitBounds(
      /*b=*/array.domain(),
      /*a_to_b=*/transform,
      /*a=*/MutableBoxView<>(input_rank, result_origin.data(), result_shape)));
  return TransformArraySubRegion(array, transform, result_origin.data(),
                                 result_shape, result_byte_strides,
                                 constraints);
}

}  // namespace internal_index_space
}  // namespace tensorstore
