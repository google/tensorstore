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

#include "tensorstore/index_space/internal/transpose_op.h"

#include <cassert>
#include <numeric>

#include "absl/container/fixed_array.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/index_transform.h"

namespace tensorstore {
namespace internal_index_space {
namespace {

Status MakePermutationFromMoveDimsTarget(DimensionIndexBuffer* dimensions,
                                         DimensionIndex target,
                                         span<DimensionIndex> permutation) {
  if (dimensions->empty()) {
    std::iota(permutation.begin(), permutation.end(),
              static_cast<DimensionIndex>(0));
    return absl::OkStatus();
  }
  const DimensionIndex input_rank = permutation.size();
  const DimensionIndex num_dims = dimensions->size();
  TENSORSTORE_ASSIGN_OR_RETURN(
      target, NormalizeDimensionIndex(target, input_rank - num_dims + 1));
  std::fill(permutation.begin(), permutation.end(),
            static_cast<DimensionIndex>(-1));
  absl::FixedArray<bool, internal::kNumInlinedDims> moved_dims(input_rank,
                                                               false);
  for (DimensionIndex i = 0; i < num_dims; ++i) {
    DimensionIndex& input_dim = (*dimensions)[i];
    moved_dims[input_dim] = true;
    permutation[target + i] = input_dim;
    input_dim = target + i;
  }
  for (DimensionIndex i = 0, orig_input_dim = 0; i < input_rank; ++i) {
    if (permutation[i] != -1) continue;
    while (moved_dims[orig_input_dim]) ++orig_input_dim;
    permutation[i] = orig_input_dim++;
  }
  return absl::OkStatus();
}

/// Permutes `orig_array` and stores the result in `new_array`.
/// \dchecks `orig_array.size() == new_array.size()`
/// \dchecks `orig_array.size() == new_to_orig_map.size()`
/// \dchecks `0 <= new_to_orig_map[i] && new_to_orig_map[i] < orig_array.size()`
template <typename Source, typename Dest>
void PermuteArray(Source orig_array, Dest new_array,
                  span<const DimensionIndex> new_to_orig_map) {
  assert(orig_array.size() == new_array.size());
  assert(orig_array.size() == new_to_orig_map.size());
  for (std::ptrdiff_t i = 0; i < orig_array.size(); ++i) {
    new_array[i] = orig_array[new_to_orig_map[i]];
  }
}

/// Permutes `array` in place, using the specified temporary array.
/// \dchecks `array.size() == temp_array.size()`
/// \dchecks `array.size() == new_to_orig_map.size()`.
/// \dchecks `0 <= new_to_orig_map[i] && new_to_orig_map[i] < orig_array.size()`
template <typename Source, typename Temp>
void PermuteArrayInPlace(Source array, Temp temp_array,
                         span<const DimensionIndex> new_to_orig_map) {
  assert(array.size() == temp_array.size());
  assert(array.size() == new_to_orig_map.size());
  for (DimensionIndex i = 0; i < array.size(); ++i) {
    temp_array[i] = array[i];
  }
  PermuteArray(temp_array, array, new_to_orig_map);
}

TransformRep::Ptr<> PermuteDimsOutOfPlace(
    TransformRep* original, span<const DimensionIndex> permutation) {
  const DimensionIndex input_rank = original->input_rank;
  const DimensionIndex output_rank = original->output_rank;
  assert(permutation.size() == input_rank);

  auto result =
      TransformRep::Allocate(original->input_rank, original->output_rank);
  result->input_rank = input_rank;
  result->output_rank = output_rank;

  // Maps original input dimension indices to new input dimension indices.
  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims>
      inverse_dimension_map(input_rank);

  // Compute the `input_origin` and `input_shape` of `result`.  Also set
  // `inverse_dimension_map` to be the inverse of `permutation`, which is needed
  // to compute the output index maps of the `result` transform.
  for (DimensionIndex new_input_dim = 0; new_input_dim < input_rank;
       ++new_input_dim) {
    const DimensionIndex orig_input_dim = permutation[new_input_dim];
    ABSL_ASSERT(orig_input_dim >= 0 && orig_input_dim < input_rank);
    result->input_dimension(new_input_dim) =
        original->input_dimension(orig_input_dim);
    inverse_dimension_map[orig_input_dim] = new_input_dim;
  }

  // Compute the output index maps of the `result` transform.
  span<const OutputIndexMap> original_maps =
      original->output_index_maps().first(output_rank);
  span<OutputIndexMap> result_maps =
      result->output_index_maps().first(output_rank);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    auto& result_map = result_maps[output_dim];
    const auto& orig_map = original_maps[output_dim];
    result_map.offset() = orig_map.offset();
    result_map.stride() = orig_map.stride();
    switch (orig_map.method()) {
      case OutputIndexMethod::constant:
        result_map.SetConstant();
        break;
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex orig_input_dim = orig_map.input_dimension();
        ABSL_ASSERT(orig_input_dim >= 0 && orig_input_dim < input_rank);
        const DimensionIndex new_input_dim =
            inverse_dimension_map[orig_input_dim];
        ABSL_ASSERT(new_input_dim >= 0 && new_input_dim < input_rank);
        result_map.SetSingleInputDimension(new_input_dim);
        break;
      }
      case OutputIndexMethod::array: {
        auto& result_index_array_data = result_map.SetArrayIndexing(input_rank);
        const auto& orig_index_array_data = orig_map.index_array_data();
        result_index_array_data.element_pointer =
            orig_index_array_data.element_pointer;
        result_index_array_data.index_range = orig_index_array_data.index_range;
        PermuteArray(span(orig_index_array_data.byte_strides, input_rank),
                     span(result_index_array_data.byte_strides, input_rank),
                     permutation);
        break;
      }
    }
  }
  return result;
}

TransformRep::Ptr<> PermuteDimsInplace(TransformRep::Ptr<> rep,
                                       span<const DimensionIndex> permutation) {
  const DimensionIndex input_rank = rep->input_rank;
  const DimensionIndex output_rank = rep->output_rank;
  assert(permutation.size() == input_rank);

  // Maps original input dimension indices to new input dimension indices.
  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims>
      inverse_dimension_map(input_rank);

  // Set `inverse_dimension_map` to be the inverse of `permutation`.
  for (DimensionIndex new_input_dim = 0; new_input_dim < input_rank;
       ++new_input_dim) {
    const DimensionIndex orig_input_dim = permutation[new_input_dim];
    inverse_dimension_map[orig_input_dim] = new_input_dim;
  }

  // Permute the input dimensions.
  {
    absl::FixedArray<IndexDomainDimension<container>, internal::kNumInlinedDims>
        temp_array(input_rank);
    PermuteArrayInPlace(rep->all_input_dimensions(input_rank), span(temp_array),
                        permutation);
  }

  // Update the output index maps of the transform.
  {
    const span<OutputIndexMap> maps =
        rep->output_index_maps().first(output_rank);
    absl::FixedArray<DimensionIndex, internal::kNumInlinedDims>
        temp_index_array(input_rank);
    for (DimensionIndex output_dim = 0; output_dim < output_rank;
         ++output_dim) {
      auto& map = maps[output_dim];
      switch (map.method()) {
        case OutputIndexMethod::constant:
          break;
        case OutputIndexMethod::single_input_dimension:
          map.SetSingleInputDimension(
              inverse_dimension_map[map.input_dimension()]);
          break;
        case OutputIndexMethod::array: {
          auto& index_array_data = map.index_array_data();
          PermuteArrayInPlace(span(index_array_data.byte_strides, input_rank),
                              span(temp_index_array), permutation);
          break;
        }
      }
    }
  }
  return rep;
}

/// Permutes the input dimension order of `rep`.
///
/// \param permutation Specifies the old dimension index corresponding to each
///     new dimension: `permutation[i]` is the old dimension index corresponding
///     to new dimension `i`.
/// \pre `rep != nullptr`
/// \pre `permutation.size() == rep->input_rank`
TransformRep::Ptr<> PermuteDims(TransformRep::Ptr<> rep,
                                span<const DimensionIndex> permutation) {
  if (rep->is_unique()) {
    return PermuteDimsInplace(std::move(rep), permutation);
  } else {
    return PermuteDimsOutOfPlace(rep.get(), permutation);
  }
}

}  // namespace

Result<IndexTransform<>> ApplyMoveDimsTo(IndexTransform<> transform,
                                         DimensionIndexBuffer* dimensions,
                                         DimensionIndex target) {
  const DimensionIndex input_rank = transform.input_rank();
  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims> permutation(
      input_rank);
  TENSORSTORE_RETURN_IF_ERROR(
      MakePermutationFromMoveDimsTarget(dimensions, target, permutation));
  return TransformAccess::Make<IndexTransform<>>(PermuteDims(
      TransformAccess::rep_ptr<container>(std::move(transform)), permutation));
}

Result<IndexTransform<>> ApplyTranspose(IndexTransform<> transform,
                                        DimensionIndexBuffer* dimensions) {
  if (static_cast<DimensionIndex>(dimensions->size()) !=
      transform.input_rank()) {
    return absl::InvalidArgumentError(
        StrCat("Number of dimensions (", dimensions->size(),
               ") must equal input_rank (", transform.input_rank(), ")."));
  }
  TransformRep::Ptr<> rep = PermuteDims(
      TransformAccess::rep_ptr<container>(std::move(transform)), *dimensions);
  std::iota(dimensions->begin(), dimensions->end(),
            static_cast<DimensionIndex>(0));
  return TransformAccess::Make<IndexTransform<>>(std::move(rep));
}

Result<IndexTransform<>> ApplyTransposeTo(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions,
    span<const DimensionIndex> target_dimensions) {
  const DimensionIndex input_rank = transform.input_rank();
  if (static_cast<DimensionIndex>(dimensions->size()) !=
      target_dimensions.size()) {
    return absl::InvalidArgumentError(
        StrCat("Number of selected dimensions (", dimensions->size(),
               ") must equal number of target dimensions (",
               target_dimensions.size(), ")"));
  }
  // Specifies whether a given existing dimension index occurs in `*dimensions`.
  absl::FixedArray<bool, internal::kNumInlinedDims> seen_existing_dim(
      input_rank, false);
  // Maps each new dimension index to the corresponding existing dimension
  // index.
  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims> permutation(
      input_rank, -1);
  for (DimensionIndex i = 0; i < target_dimensions.size(); ++i) {
    DimensionIndex& orig_dim = (*dimensions)[i];
    TENSORSTORE_ASSIGN_OR_RETURN(
        const DimensionIndex target_dim,
        NormalizeDimensionIndex(target_dimensions[i], input_rank));
    if (permutation[target_dim] != -1) {
      return absl::InvalidArgumentError(
          StrCat("Target dimension ", target_dim, " occurs more than once"));
    }
    seen_existing_dim[orig_dim] = true;
    permutation[target_dim] = orig_dim;
    orig_dim = target_dim;
  }
  // Fill in remaining dimensions.
  for (DimensionIndex orig_dim = 0, target_dim = 0; orig_dim < input_rank;
       ++orig_dim) {
    if (seen_existing_dim[orig_dim]) continue;
    while (permutation[target_dim] != -1) ++target_dim;
    permutation[target_dim] = orig_dim;
  }
  return TransformAccess::Make<IndexTransform<>>(PermuteDims(
      TransformAccess::rep_ptr<container>(std::move(transform)), permutation));
}

}  // namespace internal_index_space
}  // namespace tensorstore
