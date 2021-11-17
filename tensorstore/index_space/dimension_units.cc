// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/index_space/dimension_units.h"

#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

Result<DimensionUnitsVector> TransformInputDimensionUnits(
    IndexTransformView<> transform, DimensionUnitsVector input_units) {
  if (!transform.valid()) return input_units;
  const DimensionIndex input_rank = transform.input_rank(),
                       output_rank = transform.output_rank();
  assert(input_units.size() == input_rank);
  std::optional<Unit> output_units[kMaxRank];
  DimensionSet seen_input_dims;
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const auto map = transform.output_index_maps()[output_dim];
    if (map.method() != OutputIndexMethod::single_input_dimension) continue;
    const Index stride = map.stride();
    if (stride == 0) continue;
    const DimensionIndex input_dim = map.input_dimension();
    const auto& input_unit = input_units[input_dim];
    if (!input_unit) continue;
    seen_input_dims[input_dim] = true;
    auto& output_unit = output_units[output_dim];
    output_unit = input_unit;
    *output_unit /= std::abs(static_cast<double>(stride));
  }
  for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
    if (!input_units[input_dim] || seen_input_dims[input_dim]) continue;
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "No output dimension corresponds to input dimension ", input_dim,
        " with unit ", *input_units[input_dim]));
  }
  // Copy `output_units` to `input_units`.
  input_units.resize(output_rank);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    input_units[output_dim] = std::move(output_units[output_dim]);
  }
  return input_units;
}

DimensionUnitsVector TransformOutputDimensionUnits(
    IndexTransformView<> transform, DimensionUnitsVector output_units) {
  if (!transform.valid()) return output_units;
  const DimensionIndex input_rank = transform.input_rank(),
                       output_rank = transform.output_rank();
  assert(output_units.size() == output_rank);
  DimensionSet one_to_one_input_dims =
      internal::GetOneToOneInputDimensions(transform).one_to_one;
  std::optional<Unit> input_units[kMaxRank];
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const auto& output_unit = output_units[output_dim];
    if (!output_unit) continue;
    const auto map = transform.output_index_maps()[output_dim];
    if (map.method() != OutputIndexMethod::single_input_dimension) continue;
    const Index stride = map.stride();
    if (stride == 0) continue;
    const DimensionIndex input_dim = map.input_dimension();
    if (!one_to_one_input_dims[input_dim]) continue;
    auto& input_unit = input_units[input_dim];
    input_unit = output_unit;
    *input_unit *= std::abs(static_cast<double>(stride));
  }
  // Copy `input_units` to `output_units`.
  output_units.resize(input_rank);
  for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
    output_units[input_dim] = std::move(input_units[input_dim]);
  }
  return output_units;
}

absl::Status MergeDimensionUnits(DimensionUnitsVector& existing_units,
                                 span<const std::optional<Unit>> new_units) {
  assert(existing_units.empty() || existing_units.size() == new_units.size());
  existing_units.resize(new_units.size());
  // First, check for conflicts.
  for (size_t i = 0; i < new_units.size(); ++i) {
    auto& existing_unit = existing_units[i];
    auto& new_unit = new_units[i];
    if (!new_unit) continue;
    if (existing_unit && existing_unit != new_unit) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Cannot merge dimension units ", DimensionUnitsToString(new_units),
          " and ", DimensionUnitsToString(existing_units)));
    }
  }
  // Apply changes.
  for (size_t i = 0; i < new_units.size(); ++i) {
    auto& existing_unit = existing_units[i];
    auto& new_unit = new_units[i];
    if (!new_unit || existing_unit) continue;
    existing_unit = new_unit;
  }
  return absl::OkStatus();
}

std::string DimensionUnitsToString(span<const std::optional<Unit>> u) {
  std::string result = "[";
  std::string_view sep = "";
  for (const auto& unit : u) {
    result += sep;
    sep = ", ";
    if (!unit) {
      result += "null";
    } else {
      result += tensorstore::QuoteString(tensorstore::StrCat(*unit));
    }
  }
  result += "]";
  return result;
}

}  // namespace tensorstore
