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

#include "tensorstore/index_space/internal/identity_transform.h"

namespace tensorstore {
namespace internal_index_space {

void SetToIdentityTransform(span<OutputIndexMap> maps) {
  for (DimensionIndex i = 0; i < maps.size(); ++i) {
    auto& map = maps[i];
    map.SetSingleInputDimension(i);
    map.offset() = 0;
    map.stride() = 1;
  }
}

void SetToIdentityTransform(TransformRep* data, DimensionIndex rank) {
  ABSL_ASSERT(data->input_rank_capacity >= rank &&
              data->output_rank_capacity >= rank);
  data->input_rank = data->output_rank = rank;
  std::fill_n(data->input_origin().begin(), rank, -kInfIndex);
  std::fill_n(data->input_shape().begin(), rank, kInfSize);
  data->implicit_lower_bounds(rank).fill(true);
  data->implicit_upper_bounds(rank).fill(true);
  SetToIdentityTransform(data->output_index_maps().first(rank));
}

TransformRep::Ptr<> MakeIdentityTransform(DimensionIndex rank) {
  auto data = TransformRep::Allocate(rank, rank);
  SetToIdentityTransform(data.get(), rank);
  return data;
}

TransformRep::Ptr<> MakeIdentityTransform(internal::StringLikeSpan labels) {
  const DimensionIndex rank = labels.size();
  auto data = TransformRep::Allocate(rank, rank);
  SetToIdentityTransform(data.get(), rank);
  span<std::string> input_labels = data->input_labels().first(rank);
  for (DimensionIndex i = 0; i < rank; ++i) {
    absl::string_view label = labels[i];
    input_labels[i].assign(label.data(), label.size());
  }
  return data;
}

TransformRep::Ptr<> MakeIdentityTransformLike(TransformRep* data) {
  ABSL_ASSERT(data != nullptr);
  const DimensionIndex rank = data->input_rank;
  auto result = TransformRep::Allocate(rank, rank);
  result->output_rank = rank;
  CopyTransformRepDomain(data, result.get());
  SetToIdentityTransform(result->output_index_maps().first(rank));
  return result;
}

TransformRep::Ptr<> MakeIdentityTransform(span<const Index> shape) {
  const DimensionIndex rank = shape.size();
  auto result = TransformRep::Allocate(rank, rank);
  result->input_rank = result->output_rank = rank;
  std::fill_n(result->input_origin().begin(), rank, 0);
  std::copy_n(shape.begin(), rank, result->input_shape().begin());
  result->implicit_lower_bounds(rank).fill(false);
  result->implicit_upper_bounds(rank).fill(false);
  SetToIdentityTransform(result->output_index_maps().first(rank));
  return result;
}

TransformRep::Ptr<> MakeIdentityTransform(BoxView<> domain) {
  const DimensionIndex rank = domain.rank();
  auto result = TransformRep::Allocate(rank, rank);
  result->input_rank = result->output_rank = rank;
  result->input_domain(rank).DeepAssign(domain);
  result->implicit_lower_bounds(rank).fill(false);
  result->implicit_upper_bounds(rank).fill(false);
  SetToIdentityTransform(result->output_index_maps().first(rank));
  return result;
}

}  // namespace internal_index_space
}  // namespace tensorstore
