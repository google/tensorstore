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

#include "tensorstore/index_space/index_transform_builder.h"

#include "tensorstore/internal/integer_overflow.h"

namespace tensorstore {
namespace internal_index_space {

void InitializeTransformRepForBuilder(TransformRep* data) {
  ABSL_ASSERT(data != nullptr);
  const DimensionIndex output_rank = data->output_rank;
  span<OutputIndexMap> maps = data->output_index_maps().first(output_rank);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    auto& map = maps[output_dim];
    map.stride() = 0;
    map.offset() = 0;
  }
}

Status SetOutputIndexMapsAndValidateTransformRep(
    TransformRep* data, span<const OutputIndexMapInitializer> output_index_maps,
    IntervalForm interval_form, BuilderFlags flags) {
  const DimensionIndex input_rank = data->input_rank;
  const DimensionIndex output_rank = data->output_rank;
  ABSL_ASSERT(output_index_maps.size() == output_rank);

  span<Index> input_origin = data->input_origin().first(input_rank);
  span<Index> input_shape = data->input_shape().first(input_rank);
  const auto implicit_lower_bounds = data->implicit_lower_bounds(input_rank);
  const auto implicit_upper_bounds = data->implicit_upper_bounds(input_rank);

  if (!(flags & kSetLower)) {
    std::fill(input_origin.begin(), input_origin.end(),
              ((flags & kSetUpper) && interval_form == IntervalForm::sized)
                  ? 0
                  : -kInfIndex);
  }

  if (!(flags & kSetUpper)) {
    interval_form = IntervalForm::half_open;
    std::fill(input_shape.begin(), input_shape.end(), kInfIndex + 1);
  }

  if (!(flags & kSetImplicitLower)) {
    implicit_lower_bounds.fill(!(flags & kSetLower) &&
                               interval_form != IntervalForm::sized);
  }

  if (!(flags & kSetImplicitUpper)) {
    implicit_upper_bounds.fill(!(flags & kSetUpper));
  }

  TENSORSTORE_RETURN_IF_ERROR(
      ValidateLabelsAreUnique(data->input_labels().first(input_rank)));

  span<OutputIndexMap> maps = data->output_index_maps().first(output_rank);

  switch (interval_form) {
    case IntervalForm::sized:
      for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
        Index& size = input_shape[input_dim];
        if (size == kInfSize) {
          size = kInfIndex + 1 - input_origin[input_dim];
        }
        TENSORSTORE_RETURN_IF_ERROR(
            IndexInterval::Sized(input_origin[input_dim], size));
      }
      break;
    case IntervalForm::closed:
      for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto interval, IndexInterval::Closed(input_origin[input_dim],
                                                 input_shape[input_dim]));
        input_shape[input_dim] = interval.size();
      }
      break;
    case IntervalForm::half_open:
      for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto interval, IndexInterval::HalfOpen(input_origin[input_dim],
                                                   input_shape[input_dim]));
        input_shape[input_dim] = interval.size();
      }
      break;
    default:
      TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
  }

  // Initialize and validate the output index maps.
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const auto& initializer = output_index_maps[output_dim];
    auto& map = maps[output_dim];
    if (initializer.index_array.valid()) {
      if (!initializer.index_array_bounds) {
        return initializer.index_array_bounds.status();
      }
      span<const Index> shape = initializer.index_array.shape();
      const Index* byte_strides = initializer.index_array.byte_strides().data();
      if (shape.size() != input_rank) {
        return Status(
            absl::StatusCode::kInvalidArgument,
            StrCat("Index array for dimension ", output_dim, " has rank ",
                   shape.size(), " but must have rank ", input_rank, "."));
      }
      auto& index_array_data = map.SetArrayIndexing(shape.size());
      // Check that the index array shape is broadcast-compatible with
      // `input_shape`.
      for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
        if (shape[input_dim] == 1) {
          index_array_data.byte_strides[input_dim] = 0;
        } else if (shape[input_dim] != input_shape[input_dim]) {
          return Status(
              absl::StatusCode::kInvalidArgument,
              StrCat("Index array for dimension ", output_dim, " has shape ",
                     shape, " which does not match input_shape ", input_shape,
                     "."));
        } else if (implicit_lower_bounds[input_dim] ||
                   implicit_upper_bounds[input_dim]) {
          return absl::InvalidArgumentError(
              StrCat("Index array depends on input dimension ", input_dim,
                     " with implicit bounds."));
        } else {
          index_array_data.byte_strides[input_dim] = byte_strides[input_dim];
        }
      }
      index_array_data.index_range = *initializer.index_array_bounds;
      index_array_data.element_pointer = AddByteOffset(
          initializer.index_array.element_pointer(),
          internal::wrap_on_overflow::Subtract(
              initializer.index_array.layout().origin_byte_offset(),
              IndexInnerProduct(input_rank, input_origin.data(),
                                index_array_data.byte_strides)));
    } else if (initializer.input_dimension) {
      const DimensionIndex input_dim = *initializer.input_dimension;
      if (input_dim < 0 || input_dim >= input_rank) {
        return absl::InvalidArgumentError(StrCat(
            "Input dimension ", input_dim, " specified for output dimension ",
            output_dim, " is outside valid range [0, ", input_rank, ")."));
      }
      if (map.stride() == 0) {
        map.SetConstant();
      } else {
        map.SetSingleInputDimension(input_dim);
      }
    } else {
      map.SetConstant();
      map.stride() = 0;
    }
  }
  return absl::OkStatus();
}

}  // namespace internal_index_space
}  // namespace tensorstore
