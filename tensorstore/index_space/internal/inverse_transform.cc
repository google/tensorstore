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

#include "tensorstore/index_space/internal/inverse_transform.h"

#include <limits>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_index_space {

Result<TransformRep::Ptr<>> InverseTransform(TransformRep* transform) {
  if (!transform) {
    return TransformRep::Ptr<>();
  }

  const DimensionIndex input_rank = transform->input_rank;
  const DimensionIndex output_rank = transform->output_rank;

  auto new_transform = TransformRep::Allocate(output_rank, input_rank);
  new_transform->input_rank = output_rank;
  new_transform->output_rank = input_rank;
  new_transform->implicit_lower_bounds = false;
  new_transform->implicit_upper_bounds = false;

  const auto maps = transform->output_index_maps().first(output_rank);
  const auto new_maps = new_transform->output_index_maps().first(input_rank);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const auto& map = maps[output_dim];
    const auto new_d = new_transform->input_dimension(output_dim);
    switch (map.method()) {
      case OutputIndexMethod::array:
        return absl::InvalidArgumentError(
            absl::StrFormat("Transform is not invertible due to index array "
                            "map for output dimension %d",
                            output_dim));
      case OutputIndexMethod::constant: {
        if (!IsFiniteIndex(map.offset())) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Transform is not invertible due to offset %d"
              " outside valid range %v for output dimension %d",
              map.offset(), IndexInterval::FiniteRange(), output_dim));
        }
        new_d.domain() = IndexInterval::UncheckedSized(map.offset(), 1);
        new_d.implicit_lower_bound() = false;
        new_d.implicit_upper_bound() = false;
        break;
      }
      case OutputIndexMethod::single_input_dimension: {
        if (map.stride() != 1 && map.stride() != -1) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Transform is not invertible due to "
                              "stride of %d for output dimension %d",
                              map.stride(), output_dim));
        }
        const DimensionIndex input_dim = map.input_dimension();
        auto& new_map = new_maps[input_dim];
        if (new_map.method() == OutputIndexMethod::single_input_dimension) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Transform is not invertible because input dimension %d"
              " maps to output dimensions %d and %d",
              input_dim, new_map.input_dimension(), output_dim));
        }
        new_map.SetSingleInputDimension(output_dim);
        auto new_domain_result = GetAffineTransformRange(
            transform->input_dimension(input_dim).optionally_implicit_domain(),
            map.offset(), map.stride());
        if (!new_domain_result.ok()) {
          return MaybeAnnotateStatus(
              new_domain_result.status(),
              absl::StrFormat("Error inverting map from input dimension %d -> "
                              "output dimension %d",
                              input_dim, output_dim));
        }
        if (map.offset() == std::numeric_limits<Index>::min()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Integer overflow occurred while inverting map from "
              "input dimension %d -> output dimension %d",
              input_dim, output_dim));
        }
        new_map.offset() = -map.offset() * map.stride();
        new_map.stride() = map.stride();
        new_d.domain() = new_domain_result->interval();
        new_d.label() = transform->input_dimension(input_dim).label();
        new_d.implicit_lower_bound() = new_domain_result->implicit_lower();
        new_d.implicit_upper_bound() = new_domain_result->implicit_upper();
        break;
      }
    }
  }
  for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
    auto& new_map = new_maps[input_dim];
    if (new_map.method() == OutputIndexMethod::single_input_dimension) {
      // This input dimension is referenced by exactly one output index map, and
      // has already been set.
      continue;
    }
    // Otherwise, this input dimension is not referenced by an output index map.
    // If it is a singleton dimension, it can be inverted to a constant output
    // index map.
    auto input_domain =
        transform->input_dimension(input_dim).optionally_implicit_domain();
    if (input_domain.implicit_lower() || input_domain.implicit_upper() ||
        input_domain.size() != 1) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Transform is not invertible due to non-singleton input dimension "
          "%d with domain %v that is not mapped by an output dimension",
          input_dim, input_domain));
    }
    new_map.offset() = input_domain.inclusive_min();
    new_map.stride() = 0;
  }
  internal_index_space::DebugCheckInvariants(new_transform.get());
  return new_transform;
}

}  // namespace internal_index_space
}  // namespace tensorstore
