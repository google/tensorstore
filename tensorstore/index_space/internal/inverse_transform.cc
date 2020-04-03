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

#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_index_space {

Result<TransformRep::Ptr<>> InverseTransform(TransformRep* transform) {
  if (!transform) {
    return TransformRep::Ptr<>();
  }

  const DimensionIndex rank = transform->input_rank;
  if (rank != transform->output_rank) {
    return Status(
        absl::StatusCode::kInvalidArgument,
        StrCat("Transform with input rank (", rank, ") != output rank (",
               transform->output_rank, ") is not invertible"));
  }

  auto new_transform = TransformRep::Allocate(rank, rank);
  new_transform->input_rank = new_transform->output_rank = rank;

  const auto maps = transform->output_index_maps().first(rank);
  const auto new_maps = new_transform->output_index_maps().first(rank);
  for (DimensionIndex output_dim = 0; output_dim < rank; ++output_dim) {
    const auto& map = maps[output_dim];
    if (map.method() != OutputIndexMethod::single_input_dimension) {
      return Status(
          absl::StatusCode::kInvalidArgument,
          StrCat("Transform is not invertible due to "
                 "non-`single_input_dimension` map for output dimension ",
                 output_dim));
    }
    if (map.stride() != 1 && map.stride() != -1) {
      return absl::InvalidArgumentError(
          StrCat("Transform is not invertible due to "
                 "stride of ",
                 map.stride(), " for output dimension ", output_dim));
    }
    const DimensionIndex input_dim = map.input_dimension();
    auto& new_map = new_maps[input_dim];
    if (new_map.method() == OutputIndexMethod::single_input_dimension) {
      return Status(
          absl::StatusCode::kInvalidArgument,
          StrCat("Transform is not invertible because input dimension ",
                 input_dim, " maps to output dimensions ",
                 new_map.input_dimension(), " and ", output_dim));
    }
    new_map.SetSingleInputDimension(output_dim);

    auto new_domain_result = GetAffineTransformRange(
        transform->input_dimension(input_dim).optionally_implicit_domain(),
        map.offset(), map.stride());
    if (!new_domain_result.ok()) {
      return MaybeAnnotateStatus(
          new_domain_result.status(),
          StrCat("Error inverting map from input dimension ", input_dim,
                 " -> output dimension ", output_dim));
    }
    if (map.offset() == std::numeric_limits<Index>::min()) {
      return absl::InvalidArgumentError(
          StrCat("Integer overflow occurred while inverting map from "
                 "input dimension ",
                 input_dim, " -> output dimension ", output_dim));
    }
    new_map.offset() = -map.offset() * map.stride();
    new_map.stride() = map.stride();
    const auto new_d = new_transform->input_dimension(output_dim);
    new_d.domain() = new_domain_result->interval();
    new_d.label() = transform->input_dimension(input_dim).label();
    new_d.implicit_lower_bound() = new_domain_result->implicit_lower();
    new_d.implicit_upper_bound() = new_domain_result->implicit_upper();
  }
  return new_transform;
}

}  // namespace internal_index_space
}  // namespace tensorstore
