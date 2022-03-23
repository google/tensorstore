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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_ITERATE_IMPL_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_ITERATE_IMPL_H_

#include "absl/container/fixed_array.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_index_space {

namespace input_dimension_iteration_flags {
/// Bitmask that specifies what types of dependencies, if any, there are on a
/// given input dimension by any output dimension.  This is used to optimize
/// iteration.
using Bitmask = std::uint8_t;

/// Indicates that no output dimension depends on this input dimension.
constexpr Bitmask can_skip = 0;

/// Indicates that at least one output dimension depends on this input dimension
/// via a stride value (i.e. a single_input_dimension output index map map).
constexpr Bitmask strided = 1;

/// Indicates that at least one output dimension depends on this input dimension
/// via an index array (i.e. an array output index map with a non-zero
/// byte_strides value for this input dimension).
///
/// Specifically, this flag is set if, and only if, for some `array` there
/// exists an `output_dim` such that:
///
///     `array.byte_strides()[output_dim] != 0`, and
///     `maps[output_dim].method() == array`, and
///     `maps[output_dim].index_array_data().byte_strides[input_dim] != 0`,
///     and `result_input_shape[input_dim] > 1`.
constexpr Bitmask array_indexed = 2;

/// Returns the default bitmask value to use based on the
/// RepeatedElementsConstraint.
///
/// If `constraint == include_repeated_elements`, all dimensions are given the
/// strided flag to ensure they are included in the iteration if they have an
/// extent > 1.
inline Bitmask GetDefaultBitmask(RepeatedElementsConstraint constraint) {
  return constraint == skip_repeated_elements ? can_skip : strided;
}
}  // namespace input_dimension_iteration_flags

struct SingleArrayIterationState {
  explicit SingleArrayIterationState(DimensionIndex input_rank,
                                     DimensionIndex output_rank)
      : index_array_pointers(output_rank),
        index_array_byte_strides(output_rank),
        index_array_output_byte_strides(output_rank),
        input_byte_strides(input_rank, 0) {}

  span<const Index* const> index_array_byte_strides_span() const {
    return {index_array_byte_strides.data(),
            num_array_indexed_output_dimensions};
  }

  span<const Index> index_array_output_byte_strides_span() const {
    return {index_array_output_byte_strides.data(),
            num_array_indexed_output_dimensions};
  }

  /// `index_array_pointers[i]` specifies the adjusted index array base pointer
  /// corresponding to output dimension `array_indexed_output_dimensions[i]`.
  /// Only indices in `[0, num_array_indexed_output_dimensions)` are valid.
  absl::FixedArray<ByteStridedPointer<const Index>, internal::kNumInlinedDims>
      index_array_pointers;

  /// `index_array_byte_strides[i]` is a pointer to the byte strides member of
  /// the IndexArrayData struct corresponding to output dimension
  /// `array_indexed_output_dimensions[i]`.  Only indices in `[0,
  /// num_array_indexed_output_dimensions)` are valid.
  absl::FixedArray<const Index*, internal::kNumInlinedDims>
      index_array_byte_strides;

  /// Adjusted base pointer for the array, that includes all of the offsets.
  ByteStridedPointer<void> base_pointer;

  /// `index_array_output_byte_strides[i]` is the byte stride by which indices
  /// in the index array corresponding to output dimension
  /// `array_indexed_output_dimensions[i]` should be multiplied.  Only indices
  /// in `[0, num_array_indexed_output_dimensions)` are valid.
  absl::FixedArray<Index, internal::kNumInlinedDims>
      index_array_output_byte_strides;

  /// Byte strides into the array `base_pointer` with respect to the input
  /// dimensions.  These are used in addition to the index arrays.
  absl::FixedArray<Index, internal::kNumInlinedDims> input_byte_strides;

  /// Number of output dimensions that require array indexing.
  DimensionIndex num_array_indexed_output_dimensions = 0;
};

/// Computes the SingleArrayIterationState data for a given `array` of rank
/// `output_rank` and `transform` from `input_rank` to `output_rank`.
///
/// The SingleArrayIterationState data is computed for the subset of the input
/// domain specified by `result_input_origin` and `result_input_shape`, which
/// must be contained within the input domain of the `transform`.
///
/// Checks the bounds of any specified index arrays.
///
/// \param array The array to transform.  Must be a valid array reference.
/// \param transform[in] Specifies the transform.  If `nullptr`, indicates an
///     identity transform, and the `input_rank` and `output_rank` are
///     implicitly equal to `array.rank()`.
/// \param result_input_origin Non-null pointer to array of length `input_rank`.
/// \param result_input_shape Non-null pointer to array of length `input_rank`.
/// \param single_array_state[out] Must be non-null.  Pointer to
///     SingleArrayIterationState structure to be filled.
/// \param input_dimension_flags[in,out] Non-null pointer to array of length
///     `input_rank`.  The flags `strided` and `array_indexed` are ORd into
///     `input_dimension_flags[input_dim]` to indicate that an output dimension
///     depends on `input_dim` via a `single_input_dimension` or `array` output
///     index map, respectively.
/// \error `absl::StatusCode::kOutOfRange` the range of the transform is not
///     contained within the domain of the array.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
///     computing the result.
absl::Status InitializeSingleArrayIterationState(
    OffsetArrayView<const void> array, TransformRep* transform,
    const Index* iteration_origin, const Index* iteration_shape,
    SingleArrayIterationState* single_array_state,
    input_dimension_iteration_flags::Bitmask* input_dimension_flags);

/// Equivalent to calling the overload above with an `array` with an element
/// pointer of `element_pointer`, an unbounded domain, and all 1 byte strides.
absl::Status InitializeSingleArrayIterationState(
    ElementPointer<const void> element_pointer, TransformRep* transform,
    const Index* iteration_origin, const Index* iteration_shape,
    SingleArrayIterationState* single_array_state,
    input_dimension_iteration_flags::Bitmask* input_dimension_flags);

/// Marks singleton dimensions as skippable.
///
/// Specifically, sets `input_dimension_flags[i] = can_skip` if `input_shape[i]
/// <= 1`.
///
/// \param input_shape The extents of the input space.  All extents must be
///     non-negative.
/// \param input_dimension_flags[in,out] Pointer to array of length
///     `input_shape.size()`.
void MarkSingletonDimsAsSkippable(
    span<const Index> input_shape,
    input_dimension_iteration_flags::Bitmask* input_dimension_flags);

struct DimensionIterationOrder {
  explicit DimensionIterationOrder(DimensionIndex input_rank)
      : input_dimension_order(input_rank) {}

  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims>
      input_dimension_order;
  DimensionIndex pure_strided_start_dim;
  DimensionIndex pure_strided_end_dim;
};

struct SimplifiedDimensionIterationOrder : public DimensionIterationOrder {
  explicit SimplifiedDimensionIterationOrder(DimensionIndex input_rank)
      : DimensionIterationOrder(input_rank), simplified_shape(input_rank) {}
  absl::FixedArray<Index, internal::kNumInlinedDims> simplified_shape;
};

template <std::size_t Arity>
struct OrderTransformedArrayDimensionsByStrides {
  span<const std::optional<SingleArrayIterationState>, Arity>
      single_array_states;

  bool operator()(DimensionIndex input_dim_a,
                  DimensionIndex input_dim_b) const {
    for (std::size_t i = 0; i < Arity; ++i) {
      const auto& single_array_state = *single_array_states[i];
      // Compare index array byte strides
      for (const Index* cur_byte_strides :
           single_array_state.index_array_byte_strides_span()) {
        const auto byte_stride_a = std::abs(cur_byte_strides[input_dim_a]);
        const auto byte_stride_b = std::abs(cur_byte_strides[input_dim_b]);
        if (byte_stride_a > byte_stride_b) return true;
        if (byte_stride_a < byte_stride_b) return false;
      }
      // Compare direct byte strides
      {
        const auto* cur_byte_strides =
            single_array_state.input_byte_strides.data();
        const auto byte_stride_a = std::abs(cur_byte_strides[input_dim_a]);
        const auto byte_stride_b = std::abs(cur_byte_strides[input_dim_b]);
        if (byte_stride_a > byte_stride_b) return true;
        if (byte_stride_a < byte_stride_b) return false;
      }
    }
    return false;
  }
};

/// Computes a good iteration order subject to constraints.
template <typename OrderDimensions>
DimensionIterationOrder ComputeDimensionIterationOrder(
    span<const input_dimension_iteration_flags::Bitmask> input_dimension_flags,
    LayoutOrderConstraint order_constraint, OrderDimensions order_dimensions) {
  const DimensionIndex input_rank = input_dimension_flags.size();
  DimensionIterationOrder result(input_rank);

  if (order_constraint) {
    result.pure_strided_end_dim = 0;
    if (order_constraint.order() == ContiguousLayoutOrder::c) {
      for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
        if (input_dimension_flags[input_dim] !=
            input_dimension_iteration_flags::can_skip) {
          result.input_dimension_order[result.pure_strided_end_dim++] =
              input_dim;
        }
      }
    } else {
      for (DimensionIndex input_dim = input_rank - 1; input_dim >= 0;
           --input_dim) {
        if (input_dimension_flags[input_dim] !=
            input_dimension_iteration_flags::can_skip) {
          result.input_dimension_order[result.pure_strided_end_dim++] =
              input_dim;
        }
      }
    }
    // Find last reordered dimension index that requires array indexing.
    for (result.pure_strided_start_dim = result.pure_strided_end_dim;
         result.pure_strided_start_dim > 0 &&
         input_dimension_flags[result.input_dimension_order
                                   [result.pure_strided_start_dim - 1]] ==
             input_dimension_iteration_flags::strided;
         --result.pure_strided_start_dim) {
      continue;
    }
  } else {
    // We can reorder dimensions freely.

    // We first move all non-array-indexed dimensions to end, and then sort
    // array-indexed dimensions by the full sequence of byte strides in
    // lexicographical order, where the sequence is:
    //
    //   for each transformed input array
    //     for each index array used by that transform
    //     direct byte strides
    result.pure_strided_start_dim = 0;
    for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
      if (input_dimension_flags[input_dim] &
          input_dimension_iteration_flags::array_indexed) {
        result.input_dimension_order[result.pure_strided_start_dim++] =
            input_dim;
      }
    }
    result.pure_strided_end_dim = result.pure_strided_start_dim;
    for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
      if (input_dimension_flags[input_dim] ==
          input_dimension_iteration_flags::strided) {
        result.input_dimension_order[result.pure_strided_end_dim++] = input_dim;
      }
    }

    std::sort(
        result.input_dimension_order.data(),
        result.input_dimension_order.data() + result.pure_strided_start_dim,
        order_dimensions);
    // TODO(jbms): do a simpler sort for [result.pure_strided_start_dim,
    // result.pure_strided_end_dim).
    std::sort(
        result.input_dimension_order.data() + result.pure_strided_start_dim,
        result.input_dimension_order.data() + result.pure_strided_end_dim,
        order_dimensions);
  }

  // It is not possible for us to have eliminated all of the array indexed
  // dimensions, since array indexed dimensions cannot be skipped.
  assert(result.pure_strided_start_dim > 0);

  return result;
}

template <std::size_t Arity>
DimensionIterationOrder ComputeDimensionIterationOrder(
    span<const std::optional<SingleArrayIterationState>, Arity>
        single_array_states,
    span<const input_dimension_iteration_flags::Bitmask> input_dimension_flags,
    LayoutOrderConstraint order_constraint) {
  return ComputeDimensionIterationOrder(
      input_dimension_flags, order_constraint,
      OrderTransformedArrayDimensionsByStrides<Arity>{single_array_states});
}

template <typename CanCombineDimensions>
SimplifiedDimensionIterationOrder SimplifyDimensionIterationOrder(
    const DimensionIterationOrder& original_layout,
    span<const Index> input_shape,
    CanCombineDimensions can_combine_dimensions) {
  assert(original_layout.pure_strided_start_dim > 0);

  SimplifiedDimensionIterationOrder result(
      original_layout.pure_strided_end_dim);
  result.pure_strided_start_dim = 1;
  DimensionIndex prev_input_dim = original_layout.input_dimension_order[0];
  result.simplified_shape[0] = input_shape[prev_input_dim];
  result.input_dimension_order[0] = prev_input_dim;
  for (DimensionIndex reordered_input_dim = 1;
       reordered_input_dim < original_layout.pure_strided_start_dim;
       ++reordered_input_dim) {
    DimensionIndex input_dim =
        original_layout.input_dimension_order[reordered_input_dim];
    Index size = input_shape[input_dim];
    if (can_combine_dimensions(prev_input_dim, input_dim, size)) {
      --result.pure_strided_start_dim;
      size *= result.simplified_shape[result.pure_strided_start_dim];
    }
    result.simplified_shape[result.pure_strided_start_dim] = size;
    result.input_dimension_order[result.pure_strided_start_dim] = input_dim;
    ++result.pure_strided_start_dim;
    prev_input_dim = input_dim;
  }

  result.pure_strided_end_dim = result.pure_strided_start_dim;
  for (DimensionIndex i = original_layout.pure_strided_start_dim;
       i < original_layout.pure_strided_end_dim; ++i) {
    const DimensionIndex input_dim = original_layout.input_dimension_order[i];
    result.input_dimension_order[result.pure_strided_end_dim] = input_dim;
    result.simplified_shape[result.pure_strided_end_dim] =
        input_shape[input_dim];
    ++result.pure_strided_end_dim;
  }
  return result;
}

/// Predicate for use with SimplifyDimensionIterationOrder.
template <std::size_t Arity>
struct CanCombineTransformedArrayDimensions {
  span<const std::optional<SingleArrayIterationState>, Arity>
      single_array_states;

  bool operator()(DimensionIndex prev_input_dim, DimensionIndex input_dim,
                  Index size) const {
    for (std::size_t i = 0; i < Arity; ++i) {
      // Compare direct byte strides
      {
        const auto* cur_byte_strides =
            single_array_states[i]->input_byte_strides.data();
        if (cur_byte_strides[prev_input_dim] !=
            cur_byte_strides[input_dim] * size)
          return false;
      }

      // Compare index array byte strides
      for (const Index* cur_byte_strides :
           single_array_states[i]->index_array_byte_strides_span()) {
        if (cur_byte_strides[prev_input_dim] !=
            cur_byte_strides[input_dim] * size)
          return false;
      }
    }
    return true;
  }
};

template <std::size_t Arity>
SimplifiedDimensionIterationOrder SimplifyDimensionIterationOrder(
    const DimensionIterationOrder& original_layout,
    span<const Index> input_shape,
    span<const std::optional<SingleArrayIterationState>, Arity>
        single_array_states) {
  return SimplifyDimensionIterationOrder(
      original_layout, input_shape,
      CanCombineTransformedArrayDimensions<Arity>{single_array_states});
}

template <std::size_t Arity>
ArrayIterateResult IterateUsingSimplifiedLayout(
    const SimplifiedDimensionIterationOrder& layout,
    span<const Index> input_shape,
    internal::ElementwiseClosure<Arity, absl::Status*> closure,
    absl::Status* status,
    span<std::optional<SingleArrayIterationState>, Arity> single_array_states,
    std::array<std::ptrdiff_t, Arity> element_sizes);

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_ITERATE_IMPL_H_
