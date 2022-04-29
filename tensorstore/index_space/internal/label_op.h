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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_LABEL_OP_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_LABEL_OP_H_

/// \file
/// Implementation of the DimExpression::Label operation.

// IWYU pragma: private, include "third_party/tensorstore/index_space/dim_expression.h"

#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/internal/string_like.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_index_space {

/// Returns a new index transform with the specified dimensions
/// labeled/relabeled.
///
/// \param transform Existing transform.
/// \param dimensions[in] Non-null pointer to the list of indices of the
///     dimensions to relabel.  The value is not modified, since these indices
///     already correspond to the indices of the new dimensions in the result.
/// \param labels The vector of labels corresponding to the specified
///     dimensions.
/// \param domain_only Indicates the output dimensions of `transform` should be
///     ignored, and returned transform should have an output rank of 0.
/// \pre `transform.valid()`
/// \pre Each `index` in `*dimensions` must be unique and satisfy `0 <= index`
///     and `index < transform.input_rank()`.
/// \returns A new index transform.
/// \error `absl::StatusCode::kInvalidArgument` if `labels.size() !=
///     dimensions->size()`.
Result<IndexTransform<>> ApplyLabel(IndexTransform<> transform,
                                    DimensionIndexBuffer* dimensions,
                                    internal::StringLikeSpan labels,
                                    bool domain_only);

/// Type representing a Label operation.
/// \tparam Labels The container type for the label vector, Must be convertible
///     to internal::StringLikeSpan.
template <typename Labels>
struct LabelOp {
  static constexpr DimensionIndex num_required_dims =
      internal::ConstSpanType<Labels>::extent;
  static constexpr bool selected_dimensions_are_new = false;

  constexpr static DimensionIndex GetNewStaticInputRank(
      DimensionIndex input_rank, DimensionIndex num_input_dims) {
    TENSORSTORE_CONSTEXPR_ASSERT(
        (input_rank == dynamic_rank || input_rank >= num_required_dims) &&
        "Number of dimensions must not exceed input rank.");
    return input_rank;
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex num_input_dims) {
    TENSORSTORE_CONSTEXPR_ASSERT(
        RankConstraint::EqualOrUnspecified(num_input_dims, num_required_dims) &&
        "Number of selected dimensions must match number of indices.");
    return num_input_dims == dynamic_rank ? num_required_dims : num_input_dims;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions,
                                 bool domain_only) const {
    return ApplyLabel(std::move(transform), dimensions,
                      internal::StringLikeSpan(labels), domain_only);
  }

  Labels labels;
};

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_LABEL_OP_H_
