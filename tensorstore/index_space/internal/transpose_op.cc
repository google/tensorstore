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

#include "absl/status/status.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_permutation.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/transpose.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_index_space {
namespace {

absl::Status MakePermutationFromMoveDimsTarget(
    DimensionIndexBuffer* dimensions, DimensionIndex target,
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
  DimensionSet moved_dims = false;
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

}  // namespace

Result<IndexTransform<>> ApplyMoveDimsTo(IndexTransform<> transform,
                                         DimensionIndexBuffer* dimensions,
                                         DimensionIndex target,
                                         bool domain_only) {
  const DimensionIndex input_rank = transform.input_rank();
  DimensionIndex permutation[kMaxRank];
  TENSORSTORE_RETURN_IF_ERROR(MakePermutationFromMoveDimsTarget(
      dimensions, target, span<DimensionIndex>(&permutation[0], input_rank)));
  return TransformAccess::Make<IndexTransform<>>(TransposeInputDimensions(
      TransformAccess::rep_ptr<container>(std::move(transform)),
      span<const DimensionIndex>(&permutation[0], input_rank), domain_only));
}

Result<IndexTransform<>> ApplyTranspose(IndexTransform<> transform,
                                        DimensionIndexBuffer* dimensions,
                                        bool domain_only) {
  if (static_cast<DimensionIndex>(dimensions->size()) !=
      transform.input_rank()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Number of dimensions (", dimensions->size(),
        ") must equal input_rank (", transform.input_rank(), ")."));
  }
  TransformRep::Ptr<> rep = TransposeInputDimensions(
      TransformAccess::rep_ptr<container>(std::move(transform)), *dimensions,
      domain_only);
  std::iota(dimensions->begin(), dimensions->end(),
            static_cast<DimensionIndex>(0));
  return TransformAccess::Make<IndexTransform<>>(std::move(rep));
}

Result<IndexTransform<>> ApplyTransposeTo(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions,
    span<const DimensionIndex> target_dimensions, bool domain_only) {
  const DimensionIndex input_rank = transform.input_rank();
  if (static_cast<DimensionIndex>(dimensions->size()) !=
      target_dimensions.size()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Number of selected dimensions (", dimensions->size(),
        ") must equal number of target dimensions (", target_dimensions.size(),
        ")"));
  }
  // Specifies whether a given existing dimension index occurs in `*dimensions`.
  DimensionSet seen_existing_dim = false;
  // Maps each new dimension index to the corresponding existing dimension
  // index.
  DimensionIndex permutation[kMaxRank];
  std::fill_n(permutation, input_rank, -1);
  for (DimensionIndex i = 0; i < target_dimensions.size(); ++i) {
    DimensionIndex& orig_dim = (*dimensions)[i];
    TENSORSTORE_ASSIGN_OR_RETURN(
        const DimensionIndex target_dim,
        NormalizeDimensionIndex(target_dimensions[i], input_rank));
    if (permutation[target_dim] != -1) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Target dimension ", target_dim, " occurs more than once"));
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
  return TransformAccess::Make<IndexTransform<>>(TransposeInputDimensions(
      TransformAccess::rep_ptr<container>(std::move(transform)),
      span<const DimensionIndex>(&permutation[0], input_rank), domain_only));
}

Result<IndexTransform<>> ApplyTransposeToDynamic(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions,
    span<const DynamicDimSpec> target_dim_specs, bool domain_only) {
  if (target_dim_specs.size() == 1) {
    if (auto* target = std::get_if<DimensionIndex>(&target_dim_specs.front())) {
      return ApplyMoveDimsTo(std::move(transform), dimensions, *target,
                             domain_only);
    }
  }
  DimensionIndexBuffer target_dimensions;
  const DimensionIndex input_rank = transform.input_rank();
  for (const auto& s : target_dim_specs) {
    if (auto* index = std::get_if<DimensionIndex>(&s)) {
      target_dimensions.push_back(*index);
    } else if (auto* r = std::get_if<DimRangeSpec>(&s)) {
      TENSORSTORE_RETURN_IF_ERROR(
          NormalizeDimRangeSpec(*r, input_rank, &target_dimensions));
    } else {
      return absl::InvalidArgumentError(
          "Target dimensions cannot be specified by label");
    }
  }
  return ApplyTransposeTo(std::move(transform), dimensions, target_dimensions,
                          domain_only);
}

Result<IndexTransform<>> ApplyTranspose(
    IndexTransform<> transform, span<const DynamicDimSpec> source_dim_specs,
    bool domain_only) {
  DimensionIndexBuffer source_dimensions;
  source_dimensions.reserve(transform.input_rank());
  TENSORSTORE_RETURN_IF_ERROR(NormalizeDynamicDimSpecs(
      source_dim_specs, transform.input_labels(), &source_dimensions));
  if (!IsValidPermutation(source_dimensions)) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Source dimension list ", span(source_dimensions),
                            " is not a valid dimension permutation for rank ",
                            transform.input_rank()));
  }
  return TransformAccess::Make<IndexTransform<>>(TransposeInputDimensions(
      TransformAccess::rep_ptr<container>(std::move(transform)),
      source_dimensions, domain_only));
}

}  // namespace internal_index_space
}  // namespace tensorstore
