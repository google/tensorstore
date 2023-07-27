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

#include "tensorstore/index_space/internal/transpose.h"

#include <cassert>

#include "absl/container/fixed_array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_permutation.h"
#include "tensorstore/rank.h"

namespace tensorstore {
namespace internal_index_space {
namespace {

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
    TransformRep* original, span<const DimensionIndex> permutation,
    bool domain_only) {
  const DimensionIndex input_rank = original->input_rank;
  const DimensionIndex output_rank = domain_only ? 0 : original->output_rank;
  assert(permutation.size() == input_rank);

  auto result = TransformRep::Allocate(input_rank, output_rank);
  result->input_rank = input_rank;
  result->output_rank = output_rank;

  // Maps original input dimension indices to new input dimension indices.
  DimensionIndex inverse_dimension_map[kMaxRank];
  assert(IsValidPermutation(permutation));

  // Compute the `input_origin` and `input_shape` of `result`.  Also set
  // `inverse_dimension_map` to be the inverse of `permutation`, which is needed
  // to compute the output index maps of the `result` transform.
  for (DimensionIndex new_input_dim = 0; new_input_dim < input_rank;
       ++new_input_dim) {
    const DimensionIndex orig_input_dim = permutation[new_input_dim];
    assert(orig_input_dim >= 0 && orig_input_dim < input_rank);
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
        assert(orig_input_dim >= 0 && orig_input_dim < input_rank);
        const DimensionIndex new_input_dim =
            inverse_dimension_map[orig_input_dim];
        assert(new_input_dim >= 0 && new_input_dim < input_rank);
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
  internal_index_space::DebugCheckInvariants(result.get());
  return result;
}

TransformRep::Ptr<> PermuteDimsInplace(TransformRep::Ptr<> rep,
                                       span<const DimensionIndex> permutation,
                                       bool domain_only) {
  if (domain_only) {
    ResetOutputIndexMaps(rep.get());
  }
  const DimensionIndex input_rank = rep->input_rank;
  const DimensionIndex output_rank = rep->output_rank;
  assert(permutation.size() == input_rank);

  // Maps original input dimension indices to new input dimension indices.
  DimensionIndex inverse_dimension_map[kMaxRank];
  InvertPermutation(input_rank, permutation.data(), inverse_dimension_map);

  // Permute the input dimensions.
  {
    absl::FixedArray<IndexDomainDimension<container>, kMaxRank> temp_array(
        input_rank);
    PermuteArrayInPlace(rep->all_input_dimensions(input_rank), span(temp_array),
                        permutation);
  }

  // Update the output index maps of the transform.
  {
    const span<OutputIndexMap> maps =
        rep->output_index_maps().first(output_rank);
    DimensionIndex temp_index_array[kMaxRank];
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
          PermuteArrayInPlace(
              span(index_array_data.byte_strides, input_rank),
              span<DimensionIndex>(&temp_index_array[0], input_rank),
              permutation);
          break;
        }
      }
    }
  }
  internal_index_space::DebugCheckInvariants(rep.get());
  return rep;
}

TransformRep::Ptr<> PermuteOutputDimsInplace(
    TransformRep::Ptr<> rep, span<const DimensionIndex> permutation) {
  const DimensionIndex output_rank = rep->output_rank;
  assert(permutation.size() == output_rank);

  alignas(OutputIndexMap) char temp_buf[sizeof(OutputIndexMap) * kMaxRank];
  OutputIndexMap* output_index_maps = rep->output_index_maps().data();
  OutputIndexMap* temp_index_maps = reinterpret_cast<OutputIndexMap*>(temp_buf);
  memcpy(static_cast<void*>(temp_index_maps),
         static_cast<const void*>(output_index_maps),
         sizeof(OutputIndexMap) * output_rank);
  for (DimensionIndex new_output_dim = 0; new_output_dim < output_rank;
       ++new_output_dim) {
    const DimensionIndex old_output_dim = permutation[new_output_dim];
    memcpy(static_cast<void*>(&output_index_maps[new_output_dim]),
           static_cast<const void*>(&temp_index_maps[old_output_dim]),
           sizeof(OutputIndexMap));
  }
  internal_index_space::DebugCheckInvariants(rep.get());
  return rep;
}

TransformRep::Ptr<> PermuteOutputDimsOutOfPlace(
    TransformRep* original, span<const DimensionIndex> permutation) {
  const DimensionIndex input_rank = original->input_rank;
  const DimensionIndex output_rank = original->output_rank;
  assert(permutation.size() == output_rank);

  auto result = TransformRep::Allocate(input_rank, output_rank);
  result->input_rank = input_rank;
  result->output_rank = output_rank;
  internal_index_space::CopyTransformRepDomain(original, result.get());
  const OutputIndexMap* old_output_index_maps =
      original->output_index_maps().data();
  OutputIndexMap* new_output_index_maps = result->output_index_maps().data();
  for (DimensionIndex new_output_dim = 0; new_output_dim < output_rank;
       ++new_output_dim) {
    const DimensionIndex old_output_dim = permutation[new_output_dim];
    new_output_index_maps[new_output_dim].Assign(
        input_rank, old_output_index_maps[old_output_dim]);
  }
  return result;
}

}  // namespace

TransformRep::Ptr<> TransposeInputDimensions(
    TransformRep::Ptr<> transform, span<const DimensionIndex> permutation,
    bool domain_only) {
  if (!transform) return {};
  if (transform->is_unique()) {
    return PermuteDimsInplace(std::move(transform), permutation, domain_only);
  } else {
    return PermuteDimsOutOfPlace(transform.get(), permutation, domain_only);
  }
}

TransformRep::Ptr<> TransposeInputDimensions(TransformRep::Ptr<> transform,
                                             bool domain_only) {
  if (!transform) return {};
  DimensionIndex permutation[kMaxRank];
  const DimensionIndex rank = transform->input_rank;
  for (DimensionIndex i = 0; i < rank; ++i) {
    permutation[i] = rank - i - 1;
  }
  return TransposeInputDimensions(
      std::move(transform), span<const DimensionIndex>(&permutation[0], rank),
      domain_only);
}

TransformRep::Ptr<> TransposeOutputDimensions(
    TransformRep::Ptr<> transform, span<const DimensionIndex> permutation) {
  if (!transform) return {};
  if (transform->is_unique()) {
    return PermuteOutputDimsInplace(std::move(transform), permutation);
  } else {
    return PermuteOutputDimsOutOfPlace(transform.get(), permutation);
  }
}

TransformRep::Ptr<> TransposeOutputDimensions(TransformRep::Ptr<> transform) {
  if (!transform) return {};
  DimensionIndex permutation[kMaxRank];
  const DimensionIndex rank = transform->output_rank;
  for (DimensionIndex i = 0; i < rank; ++i) {
    permutation[i] = rank - i - 1;
  }
  return TransposeOutputDimensions(
      std::move(transform), span<const DimensionIndex>(&permutation[0], rank));
}

}  // namespace internal_index_space
}  // namespace tensorstore
