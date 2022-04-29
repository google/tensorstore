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

namespace {
void SetUnboundedDomain(TransformRep* data, DimensionIndex rank) {
  assert(data->input_rank_capacity >= rank);
  data->input_rank = rank;
  std::fill_n(data->input_origin().begin(), rank, -kInfIndex);
  std::fill_n(data->input_shape().begin(), rank, kInfSize);
  const auto mask = DimensionSet::UpTo(rank);
  data->implicit_lower_bounds = mask;
  data->implicit_upper_bounds = mask;
}

void SetIdentityOutputOrDomainOnly(TransformRep* data, DimensionIndex rank,
                                   bool domain_only) {
  if (domain_only) {
    data->output_rank = 0;
  } else {
    assert(data->output_rank_capacity >= rank);
    data->output_rank = rank;
    SetToIdentityTransform(data->output_index_maps().first(rank));
  }
}

void SetToIdentityTransform(TransformRep* data, DimensionIndex rank,
                            bool domain_only) {
  SetUnboundedDomain(data, rank);
  SetIdentityOutputOrDomainOnly(data, rank, domain_only);
}

}  // namespace

TransformRep::Ptr<> MakeIdentityTransform(DimensionIndex rank,
                                          bool domain_only) {
  auto data = TransformRep::Allocate(rank, domain_only ? 0 : rank);
  SetToIdentityTransform(data.get(), rank, domain_only);
  internal_index_space::DebugCheckInvariants(data.get());
  return data;
}

TransformRep::Ptr<> MakeIdentityTransform(internal::StringLikeSpan labels,
                                          bool domain_only) {
  const DimensionIndex rank = labels.size();
  auto data = TransformRep::Allocate(rank, domain_only ? 0 : rank);
  SetToIdentityTransform(data.get(), rank, domain_only);
  span<std::string> input_labels = data->input_labels().first(rank);
  for (DimensionIndex i = 0; i < rank; ++i) {
    std::string_view label = labels[i];
    input_labels[i].assign(label.data(), label.size());
  }
  internal_index_space::DebugCheckInvariants(data.get());
  return data;
}

TransformRep::Ptr<> MakeIdentityTransformLike(TransformRep* data,
                                              bool domain_only) {
  assert(data != nullptr);
  const DimensionIndex rank = data->input_rank;
  auto result = TransformRep::Allocate(rank, domain_only ? 0 : rank);
  CopyTransformRepDomain(data, result.get());
  SetIdentityOutputOrDomainOnly(result.get(), rank, domain_only);
  internal_index_space::DebugCheckInvariants(result.get());
  return result;
}

TransformRep::Ptr<> MakeIdentityTransform(span<const Index> shape,
                                          bool domain_only) {
  const DimensionIndex rank = shape.size();
  auto result = TransformRep::Allocate(rank, domain_only ? 0 : rank);
  result->input_rank = rank;
  std::fill_n(result->input_origin().begin(), rank, 0);
  std::copy_n(shape.begin(), rank, result->input_shape().begin());
  result->implicit_lower_bounds = false;
  result->implicit_upper_bounds = false;
  SetIdentityOutputOrDomainOnly(result.get(), rank, domain_only);
  internal_index_space::DebugCheckInvariants(result.get());
  return result;
}

TransformRep::Ptr<> MakeIdentityTransform(BoxView<> domain, bool domain_only) {
  const DimensionIndex rank = domain.rank();
  auto result = TransformRep::Allocate(rank, domain_only ? 0 : rank);
  result->input_rank = rank;
  result->input_domain(rank).DeepAssign(domain);
  result->implicit_lower_bounds = false;
  result->implicit_upper_bounds = false;
  SetIdentityOutputOrDomainOnly(result.get(), rank, domain_only);
  internal_index_space::DebugCheckInvariants(result.get());
  return result;
}

}  // namespace internal_index_space
}  // namespace tensorstore
