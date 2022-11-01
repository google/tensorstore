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

#include "tensorstore/index_space/transformed_array.h"

#include "absl/status/status.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/index_space/internal/iterate_impl.h"
#include "tensorstore/index_space/internal/transform_rep_impl.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/util/internal/iterate_impl.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_index_space {

std::string DescribeTransformedArrayForCast(DataType dtype,
                                            DimensionIndex rank) {
  return tensorstore::StrCat(
      "transformed array with ", StaticCastTraits<DataType>::Describe(dtype),
      " and ", StaticCastTraits<DimensionIndex>::Describe(rank));
}

namespace {
void MultiplyByteStridesIntoOutputIndexMaps(TransformRep* transform,
                                            span<const Index> byte_strides) {
  const span<OutputIndexMap> output_maps = transform->output_index_maps();
  assert(byte_strides.size() == output_maps.size());
  for (DimensionIndex i = 0; i < byte_strides.size(); ++i) {
    auto& map = output_maps[i];
    const Index byte_stride = byte_strides[i];
    const Index stride =
        internal::wrap_on_overflow::Multiply(map.stride(), byte_stride);
    if (stride == 0) {
      map.SetConstant();
    }
    map.stride() = stride;
    // Overflow is valid for `map.offset()` because handling of non-zero array
    // origins relies on mod-2**64 arithmetic.
    map.offset() =
        internal::wrap_on_overflow::Multiply(map.offset(), byte_stride);
  }
}
}  // namespace

absl::Status CopyTransformedArrayImpl(TransformedArrayView<const void> source,
                                      TransformedArrayView<void> dest) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto r, internal::GetDataTypeConverterOrError(
                                           source.dtype(), dest.dtype()));
  absl::Status status;
  using TA = TransformedArrayView<const void>;
  TENSORSTORE_ASSIGN_OR_RETURN(auto iterate_result,
                               internal::IterateOverTransformedArrays<2>(
                                   r.closure, &status, skip_repeated_elements,
                                   span<const TA, 2>({source, TA(dest)})));
  if (!iterate_result.success) {
    return internal::GetElementCopyErrorStatus(std::move(status));
  }
  return status;
}

TransformRep::Ptr<> MakeTransformFromStridedLayout(
    StridedLayoutView<dynamic_rank, offset_origin> layout) {
  auto result = MakeIdentityTransform(layout.domain());
  MultiplyByteStridesIntoOutputIndexMaps(result.get(), layout.byte_strides());
  internal_index_space::DebugCheckInvariants(result.get());
  return result;
}

Result<TransformRep::Ptr<>> MakeTransformFromStridedLayoutAndTransform(
    StridedLayoutView<dynamic_rank, offset_origin> layout,
    TransformRep::Ptr<> transform) {
  if (!transform) return MakeTransformFromStridedLayout(layout);
  if (transform->output_rank != layout.rank()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Transform output rank (", transform->output_rank,
        ") does not equal array rank (", layout.rank(), ")"));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      transform, PropagateExplicitBoundsToTransform(layout.domain(),
                                                    std::move(transform)));
  MultiplyByteStridesIntoOutputIndexMaps(transform.get(),
                                         layout.byte_strides());
  internal_index_space::DebugCheckInvariants(transform.get());
  return transform;
}

StridedLayoutView<dynamic_rank, offset_origin> GetUnboundedLayout(
    DimensionIndex rank) {
  return StridedLayoutView<dynamic_rank, offset_origin>(
      rank, GetConstantVector<Index, -kInfIndex>(rank).data(),
      GetConstantVector<Index, kInfSize>(rank).data(),
      GetConstantVector<Index, 1>(rank).data());
}

}  // namespace internal_index_space

namespace internal {

template <std::size_t Arity>
Result<ArrayIterateResult> IterateOverTransformedArrays(
    ElementwiseClosure<Arity, absl::Status*> closure, absl::Status* status,
    IterationConstraints constraints,
    span<const TransformedArrayView<const void>, Arity> transformed_arrays) {
  if (Arity == 0) return ArrayIterateResult{/*.success=*/true, /*.count=*/0};

  const DimensionIndex input_rank = transformed_arrays[0].rank();

  namespace flags = internal_index_space::input_dimension_iteration_flags;

  absl::FixedArray<flags::Bitmask, kNumInlinedDims> input_dimension_flags(
      input_rank,
      flags::GetDefaultBitmask(constraints.repeated_elements_constraint()));
  std::array<std::optional<internal_index_space::SingleArrayIterationState>,
             Arity>
      single_array_states;

  Box<dynamic_rank(kNumInlinedDims)> input_bounds(input_rank);

  // Validate domain ranks
  bool failed = false;
  for (std::size_t i = 0; i < Arity; ++i) {
    if (transformed_arrays[i].domain().rank() != input_rank) {
      failed = true;
    }
  }
  if (failed) {
    absl::FixedArray<DimensionIndex, internal::kNumInlinedDims>
        transformed_ranks(Arity);
    for (std::size_t i = 0; i < Arity; ++i) {
      transformed_ranks[i] = transformed_arrays[i].domain().rank();
    }
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Transformed array input ranks ",
                            span(transformed_ranks), " do not all match"));
  }

  // Compute input_bounds.
  for (std::size_t i = 0; i < Arity; ++i) {
    const BoxView<> domain = transformed_arrays[i].domain().box();
    TENSORSTORE_RETURN_IF_ERROR(
        internal_index_space::ValidateAndIntersectBounds(
            domain, input_bounds, [](IndexInterval a, IndexInterval b) {
              return AreCompatibleOrUnbounded(a, b);
            }));
  }

  // Early return if the domain is empty.  The iteration code below cannot
  // handle this case correctly.
  for (DimensionIndex i = 0; i < input_rank; ++i) {
    if (input_bounds.shape()[i] == 0) {
      return ArrayIterateResult{/*.success=*/true, /*.count=*/0};
    }
  }

  bool has_array_indexed_output_dimensions = false;

  for (std::size_t i = 0; i < Arity; ++i) {
    const auto& ta = transformed_arrays[i];
    single_array_states[i].emplace(input_rank, ta.transform().output_rank());
    auto& single_array_state = single_array_states[i];
    TENSORSTORE_RETURN_IF_ERROR(
        internal_index_space::InitializeSingleArrayIterationState(
            ta.element_pointer(),
            internal_index_space::TransformAccess::rep(ta.transform()),
            input_bounds.origin().data(), input_bounds.shape().data(),
            &*single_array_state, input_dimension_flags.data()));
    if (single_array_state->num_array_indexed_output_dimensions) {
      has_array_indexed_output_dimensions = true;
    }
  }

  std::array<std::ptrdiff_t, Arity> element_sizes;
  for (std::size_t i = 0; i < Arity; ++i) {
    element_sizes[i] = transformed_arrays[i].dtype()->size;
  }
  if (!has_array_indexed_output_dimensions) {
    // This reduces to just a regular strided layout iteration.
    std::array<ByteStridedPointer<void>, Arity> pointers;
    std::array<const Index*, Arity> strides;
    for (std::size_t i = 0; i < Arity; ++i) {
      pointers[i] = single_array_states[i]->base_pointer;
      strides[i] = single_array_states[i]->input_byte_strides.data();
    }
    return IterateOverStridedLayouts<Arity>(
        closure, status, input_bounds.shape(), pointers, strides, constraints,
        element_sizes);
  }
  internal_index_space::MarkSingletonDimsAsSkippable(
      input_bounds.shape(), input_dimension_flags.data());

  internal_index_space::SimplifiedDimensionIterationOrder layout =
      internal_index_space::SimplifyDimensionIterationOrder<Arity>(
          internal_index_space::ComputeDimensionIterationOrder<Arity>(
              single_array_states, input_dimension_flags,
              constraints.order_constraint()),
          input_bounds.shape(), single_array_states);
  return internal_index_space::IterateUsingSimplifiedLayout<Arity>(
      layout, input_bounds.shape(), closure, status, single_array_states,
      element_sizes);
}

#define TENSORSTORE_DO_INSTANTIATE_ITERATE_OVER_TRANSFORMED_ARRAYS(Arity)      \
  template Result<ArrayIterateResult> IterateOverTransformedArrays<Arity>(     \
      ElementwiseClosure<Arity, absl::Status*> closure, absl::Status * status, \
      IterationConstraints constraints,                                        \
      span<const TransformedArrayView<const void>, Arity> transformed_arrays); \
  /**/
TENSORSTORE_INTERNAL_FOR_EACH_ARITY(
    TENSORSTORE_DO_INSTANTIATE_ITERATE_OVER_TRANSFORMED_ARRAYS)
#undef TENSORSTORE_DO_INSTANTIATE_ITERATE_OVER_TRANSFORMED_ARRAYS

}  // namespace internal
}  // namespace tensorstore
