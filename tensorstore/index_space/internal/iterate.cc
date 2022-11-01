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

#include "tensorstore/util/internal/iterate.h"

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/iterate_impl.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/index_space/internal/transform_rep_impl.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/util/internal/iterate_impl.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_index_space {

constexpr Index temp_index_buffer_size = 1024;

void MarkSingletonDimsAsSkippable(
    span<const Index> input_shape,
    input_dimension_iteration_flags::Bitmask* input_dimension_flags) {
  for (DimensionIndex i = 0; i < input_shape.size(); ++i) {
    if (input_shape[i] == 1) {
      input_dimension_flags[i] = input_dimension_iteration_flags::can_skip;
    }
  }
}

namespace {
/// Common implementation of the two variants of
/// InitializeSingleArrayIterationState.
///
/// If `UseStridedLayout == true`, the range of `transform` is checked against
/// the domain of `array`; `transform` may be `nullptr`, in which case it is
/// treated as an identity transform of rank `array.rank()`.
///
/// If `UseStridedLayout == false`, `transform` must not be `nullptr`, and
/// `(array.element_pointer(), transform)` is assumed to be a valid
/// `TransformedArray`.
template <bool UseStridedLayout>
absl::Status InitializeSingleArrayIterationStateImpl(
    OffsetArrayView<const void, (UseStridedLayout ? dynamic_rank : 0)> array,
    TransformRep* transform, const Index* iteration_origin,
    const Index* iteration_shape, SingleArrayIterationState* single_array_state,
    input_dimension_iteration_flags::Bitmask* input_dimension_flags) {
  if constexpr (!UseStridedLayout) {
    assert(transform != nullptr);
  }
  const DimensionIndex output_rank =
      UseStridedLayout ? array.rank() : transform->output_rank;

  single_array_state->base_pointer = const_cast<void*>(array.data());

  if constexpr (UseStridedLayout) {
    if (!transform) {
      // Handle identity transform case.
      for (DimensionIndex output_dim = 0; output_dim < output_rank;
           ++output_dim) {
        const DimensionIndex input_dim = output_dim;
        const Index byte_stride = array.byte_strides()[output_dim];
        single_array_state->input_byte_strides[input_dim] = byte_stride;
        if (iteration_shape[input_dim] != 1) {
          input_dimension_flags[input_dim] |=
              input_dimension_iteration_flags::strided;
        }
        single_array_state->base_pointer +=
            internal::wrap_on_overflow::Multiply(iteration_origin[input_dim],
                                                 byte_stride);
      }
      return absl::OkStatus();
    }
  }

  assert(output_rank == transform->output_rank);

  const DimensionIndex input_rank = transform->input_rank;

  span<OutputIndexMap> maps = transform->output_index_maps().first(output_rank);

  // Updates `single_array_state->input_byte_strides`,
  // `single_array_state->base_pointer`,
  // `single_array_state->index_array_pointer`,
  // `single_array_state->index_array_byte_strides`, and `input_dimension_flags`
  // based on the output index maps.
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const Index byte_stride =
        UseStridedLayout ? array.byte_strides()[output_dim] : 1;
    if (byte_stride == 0) continue;
    const auto& map = maps[output_dim];
    const Index output_offset = map.offset();
    const Index output_stride = map.stride();
    single_array_state->base_pointer +=
        internal::wrap_on_overflow::Multiply(output_offset, byte_stride);
    if (output_stride == 0 || map.method() == OutputIndexMethod::constant) {
      if constexpr (UseStridedLayout) {
        if (!Contains(array.domain()[output_dim], output_offset)) {
          return MaybeAnnotateStatus(
              CheckContains(array.domain()[output_dim], output_offset),
              tensorstore::StrCat(
                  "Checking bounds of constant output index map for dimension ",
                  output_dim));
        }
      } else {
        // `output_offset` is assumed to be valid in the normalized
        // representation.
      }
    } else if (map.method() == OutputIndexMethod::single_input_dimension) {
      const DimensionIndex input_dim = map.input_dimension();
      assert(input_dim >= 0 && input_dim < input_rank);
      if constexpr (UseStridedLayout) {
        TENSORSTORE_ASSIGN_OR_RETURN(
            IndexInterval range,
            GetAffineTransformRange(
                IndexInterval::UncheckedSized(iteration_origin[input_dim],
                                              iteration_shape[input_dim]),
                output_offset, output_stride),
            MaybeAnnotateStatus(
                _, tensorstore::StrCat(
                       "Checking bounds of output index map for dimension ",
                       output_dim)));
        if (!Contains(array.domain()[output_dim], range)) {
          return absl::OutOfRangeError(tensorstore::StrCat(
              "Output dimension ", output_dim, " range of ", range,
              " is not contained within array domain of ",
              array.domain()[output_dim]));
        }
      }
      single_array_state->base_pointer += internal::wrap_on_overflow::Multiply(
          byte_stride, internal::wrap_on_overflow::Multiply(
                           output_stride, iteration_origin[input_dim]));
      // The `input_byte_strides` value for `input_dim` is equal to the sum of
      // the `stride`-adjusted array `byte_stride` values for all output
      // dimensions that depend on `input_dim` via a `single_input_dimension`
      // output index map.
      single_array_state->input_byte_strides[input_dim] =
          internal::wrap_on_overflow::Add(
              single_array_state->input_byte_strides[input_dim],
              internal::wrap_on_overflow::Multiply(byte_stride, output_stride));
      input_dimension_flags[input_dim] |=
          input_dimension_iteration_flags::strided;
    } else {
      const auto& index_array_data = map.index_array_data();
      assert(index_array_data.rank_capacity >= input_rank);
      IndexInterval index_bounds = index_array_data.index_range;

      if constexpr (UseStridedLayout) {
        // propagate bounds back to the index array
        TENSORSTORE_ASSIGN_OR_RETURN(
            IndexInterval propagated_index_bounds,
            GetAffineTransformDomain(array.domain()[output_dim], output_offset,
                                     output_stride),
            MaybeAnnotateStatus(
                _, tensorstore::StrCat(
                       "Propagating bounds from intermediate dimension ",
                       output_dim, ".")));
        index_bounds = Intersect(propagated_index_bounds, index_bounds);
      }

      ByteStridedPointer<const Index> index_array_pointer =
          index_array_data.element_pointer.data();
      // Specifies whether the index array (reduced to the specified input
      // domain) is a singleton array (has only a single distinct value).
      bool has_one_element = true;
      // Adjust `index_array_pointer` and `input_dimension_flags`.
      for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
        const Index index_array_byte_stride =
            index_array_data.byte_strides[input_dim];
        index_array_pointer += internal::wrap_on_overflow::Multiply(
            iteration_origin[input_dim], index_array_byte_stride);
        if (index_array_byte_stride != 0 && iteration_shape[input_dim] != 1) {
          input_dimension_flags[input_dim] |=
              input_dimension_iteration_flags::array_indexed;
          has_one_element = false;
        }
      }
      if (has_one_element) {
        // The index array has only a single distinct value; therefore, we treat
        // it as a constant output index map.
        const Index index = *index_array_pointer;
        TENSORSTORE_RETURN_IF_ERROR(
            CheckContains(index_bounds, index),
            MaybeAnnotateStatus(
                _,
                tensorstore::StrCat("In index array map for output dimension ",
                                    output_dim)));
        single_array_state->base_pointer +=
            internal::wrap_on_overflow::Multiply(
                byte_stride,
                internal::wrap_on_overflow::Multiply(output_stride, index));
      } else {
        // The index array has more than a single distinct value; therefore, we
        // add it as an index array.
        DimensionIndex index_array_num =
            single_array_state->num_array_indexed_output_dimensions++;
        single_array_state->index_array_byte_strides[index_array_num] =
            index_array_data.byte_strides;
        single_array_state->index_array_pointers[index_array_num] =
            index_array_pointer;
        single_array_state->index_array_output_byte_strides[index_array_num] =
            internal::wrap_on_overflow::Multiply(byte_stride, output_stride);

        TENSORSTORE_RETURN_IF_ERROR(
            ValidateIndexArrayBounds(
                index_bounds,
                ArrayView<const Index>(index_array_pointer.get(),
                                       StridedLayoutView<dynamic_rank>(
                                           input_rank, iteration_shape,
                                           index_array_data.byte_strides))),
            MaybeAnnotateStatus(
                _,
                tensorstore::StrCat("In index array map for output dimension ",
                                    output_dim)));
      }
    }
  }
  return {};
}

}  // namespace

absl::Status InitializeSingleArrayIterationState(
    OffsetArrayView<const void> array, TransformRep* transform,
    const Index* iteration_origin, const Index* iteration_shape,
    SingleArrayIterationState* single_array_state,
    input_dimension_iteration_flags::Bitmask* input_dimension_flags) {
  return InitializeSingleArrayIterationStateImpl<true>(
      array, transform, iteration_origin, iteration_shape, single_array_state,
      input_dimension_flags);
}

absl::Status InitializeSingleArrayIterationState(
    ElementPointer<const void> element_pointer, TransformRep* transform,
    const Index* iteration_origin, const Index* iteration_shape,
    SingleArrayIterationState* single_array_state,
    input_dimension_iteration_flags::Bitmask* input_dimension_flags) {
  return InitializeSingleArrayIterationStateImpl<false>(
      element_pointer, transform, iteration_origin, iteration_shape,
      single_array_state, input_dimension_flags);
}

Index IndirectInnerProduct(span<const Index> indices,
                           const DimensionIndex* dimension_order,
                           const Index* byte_strides) {
  Index result = 0;
  for (DimensionIndex i = 0; i < indices.size(); ++i) {
    result = internal::wrap_on_overflow::Add(
        internal::wrap_on_overflow::Multiply(indices[i],
                                             byte_strides[dimension_order[i]]),
        result);
  }
  return result;
}

void FillOffsetsArray(span<Index> offsets, span<const Index> position,
                      const DimensionIndex* input_dimension_order,
                      const SingleArrayIterationState& single_array_state,
                      Index final_input_dim_byte_stride,
                      Index final_input_dim_start_position) {
  std::memset(offsets.data(), 0, sizeof(Index) * offsets.size());
  for (DimensionIndex
           j = 0,
           num_array_indexed_output_dimensions =
               single_array_state.num_array_indexed_output_dimensions;
       j < num_array_indexed_output_dimensions; ++j) {
    ByteStridedPointer<const Index> index_data_pointer =
        single_array_state.index_array_pointers[j];
    const Index* cur_byte_strides =
        single_array_state.index_array_byte_strides[j];
    index_data_pointer += internal_index_space::IndirectInnerProduct(
        position, input_dimension_order, cur_byte_strides);
    const auto final_byte_stride =
        cur_byte_strides[input_dimension_order[position.size()]];
    const Index output_dim_byte_stride =
        single_array_state.index_array_output_byte_strides[j];
    if (final_byte_stride == 0) {
      const Index index_value = *index_data_pointer;
      for (Index j = 0; j < offsets.size(); ++j) {
        offsets[j] = internal::wrap_on_overflow::Add(
            offsets[j], internal::wrap_on_overflow::Multiply(
                            index_value, output_dim_byte_stride));
      }
    } else {
      index_data_pointer += internal::wrap_on_overflow::Multiply(
          final_byte_stride, final_input_dim_start_position);
      for (Index j = 0; j < offsets.size(); ++j) {
        offsets[j] = internal::wrap_on_overflow::Add(
            offsets[j], internal::wrap_on_overflow::Multiply(
                            *index_data_pointer, output_dim_byte_stride));
        index_data_pointer += final_byte_stride;
      }
    }
  }
  if (final_input_dim_byte_stride != 0) {
    for (Index j = 0; j < offsets.size(); ++j) {
      offsets[j] = internal::wrap_on_overflow::Add(
          offsets[j],
          internal::wrap_on_overflow::Multiply(
              final_input_dim_byte_stride, j + final_input_dim_start_position));
    }
  }
}

template <std::size_t Arity>
ArrayIterateResult IterateUsingSimplifiedLayout(
    const SimplifiedDimensionIterationOrder& layout,
    span<const Index> input_shape,
    internal::ElementwiseClosure<Arity, absl::Status*> closure,
    absl::Status* status,
    span<std::optional<SingleArrayIterationState>, Arity> single_array_states,
    std::array<std::ptrdiff_t, Arity> element_sizes) {
  const Index final_indexed_dim_size =
      layout.simplified_shape[layout.pure_strided_start_dim - 1];

  std::array<const Index*, Arity> strides;
  for (std::size_t i = 0; i < Arity; ++i) {
    strides[i] = single_array_states[i]->input_byte_strides.data();
  }

  internal::StridedLayoutFunctionApplyer<Arity> strided_applyer(
      input_shape.data(),
      span(layout.input_dimension_order.data() + layout.pure_strided_start_dim,
           layout.input_dimension_order.data() + layout.pure_strided_end_dim),
      strides, closure, element_sizes);

  struct SingleArrayOffsetsBuffer {
    Index offsets[temp_index_buffer_size];
  };

  const DimensionIndex last_indexed_dim = layout.pure_strided_start_dim - 1;

  ArrayIterateResult outer_result;
  outer_result.count = 0;

  // Iterate over all but the last array-indexed dimension.  We handle the last
  // array-indexed dimension specially for efficiency.
  outer_result.success = IterateOverIndexRange(
      span(layout.simplified_shape.data(), last_indexed_dim),
      [&](span<const Index> position) {
        std::array<SingleArrayOffsetsBuffer, Arity> single_array_offset_buffers;
        std::array<ByteStridedPointer<void>, Arity> pointers;
        std::array<Index, Arity> final_indexed_dim_byte_strides;
        for (std::size_t i = 0; i < Arity; ++i) {
          const auto& single_array_state = *single_array_states[i];
          pointers[i] = single_array_state.base_pointer +
                        internal_index_space::IndirectInnerProduct(
                            position, layout.input_dimension_order.data(),
                            single_array_state.input_byte_strides.data());
          final_indexed_dim_byte_strides[i] =
              single_array_state
                  .input_byte_strides[layout.input_dimension_order
                                          [layout.pure_strided_start_dim - 1]];
        }
        for (Index final_indexed_dim_start_position = 0;
             final_indexed_dim_start_position < final_indexed_dim_size;
             final_indexed_dim_start_position += temp_index_buffer_size) {
          const Index block_size = std::min(
              final_indexed_dim_size - final_indexed_dim_start_position,
              temp_index_buffer_size);
          for (std::size_t i = 0; i < Arity; ++i) {
            Index* offsets = single_array_offset_buffers[i].offsets;
            FillOffsetsArray(span(offsets, block_size), position,
                             layout.input_dimension_order.data(),
                             *single_array_states[i],
                             final_indexed_dim_byte_strides[i],
                             final_indexed_dim_start_position);
          }
          if (strided_applyer.inner_size() == 1) {
            std::array<internal::IterationBufferPointer, Arity>
                pointers_with_offset_arrays;
            for (std::size_t i = 0; i < Arity; ++i) {
              pointers_with_offset_arrays[i] = internal::IterationBufferPointer{
                  pointers[i], single_array_offset_buffers[i].offsets};
            }
            Index cur_count = internal::InvokeElementwiseClosure(
                closure, internal::IterationBufferKind::kIndexed, block_size,
                pointers_with_offset_arrays, status);
            outer_result.count += cur_count;
            if (cur_count != block_size) return false;
          } else {
            for (Index j = 0; j < block_size; ++j) {
              auto cur_pointers = pointers;
              for (std::size_t i = 0; i < Arity; ++i) {
                cur_pointers[i] += single_array_offset_buffers[i].offsets[j];
              }
              auto inner_result = strided_applyer(cur_pointers, status);
              outer_result.count += inner_result.count;
              if (!inner_result.success) return false;
            }
          }
        }

        return true;
      });
  return outer_result;
}

// TODO(jbms): Consider making this a static method of a class template to
// simplify the explicit instantiation.
#define TENSORSTORE_INTERNAL_DO_INSTANTIATE_ITERATE_USING_SIMPLIFIED_LAYOUT( \
    Arity)                                                                   \
  template ArrayIterateResult IterateUsingSimplifiedLayout<Arity>(           \
      const SimplifiedDimensionIterationOrder& layout,                       \
      span<const Index> input_shape,                                         \
      internal::ElementwiseClosure<Arity, absl::Status*> closure,            \
      absl::Status* status,                                                  \
      span<std::optional<SingleArrayIterationState>, Arity>                  \
          single_array_states,                                               \
      std::array<std::ptrdiff_t, Arity> element_sizes);
TENSORSTORE_INTERNAL_FOR_EACH_ARITY(
    TENSORSTORE_INTERNAL_DO_INSTANTIATE_ITERATE_USING_SIMPLIFIED_LAYOUT)
#undef TENSORSTORE_INTERNAL_DO_INSTANTIATE_ITERATE_USING_SIMPLIFIED_LAYOUT

}  // namespace internal_index_space

absl::Status ValidateIndexArrayBounds(
    IndexInterval bounds,
    ArrayView<const Index, dynamic_rank, offset_origin> index_array) {
  const auto finite_bounds = FiniteSubset(bounds);
  const Index inclusive_min = finite_bounds.inclusive_min();
  const Index exclusive_max = finite_bounds.exclusive_max();
  Index bad_index;
  if (!IterateOverArrays(
          [&](const Index* value) {
            if (ABSL_PREDICT_FALSE(*value < inclusive_min ||
                                   *value >= exclusive_max)) {
              bad_index = *value;
              return false;
            }
            return true;
          },
          skip_repeated_elements, index_array)) {
    return CheckContains(bounds, bad_index);
  }
  return absl::OkStatus();
}

}  // namespace tensorstore
