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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_SINGLE_INDEX_SLICE_OP_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_SINGLE_INDEX_SLICE_OP_H_

/// \file
/// Implementation of the DimExpression::IndexSlice operation.

// IWYU pragma: private, include "third_party/tensorstore/index_space/dim_expression.h"

#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_vector_or_scalar.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_index_space {

/// Returns a new transform with the specified dimensions indexed by the
/// specified constant indices.
///
/// See DimExpression::IndexSlice for details.
///
/// \param transform Existing transform.
/// \param dimensions[in,out] Must be non-null.  On input, the dimensions from
///     which to extract a single-index slice.  On return, an empty list.
/// \param indices The indices corresponding to the selected dimensions.
/// \pre `transform.valid()`
/// \pre Each `index` in `*dimensions` must be unique and satisfy `0 <= index`
///     and `index < transform.input_rank()`.
/// \param domain_only Indicates the output dimensions of `transform` should be
///     ignored, and returned transform should have an output rank of 0.
/// \returns The new index transform.
/// \error `absl::StatusCode::kInvalidArgument` if `indices.size()` is not
///     compatible with `dimensions->size()`.
/// \error `absl::StatusCode::kOutOfRange` if an invalid index is specified.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
///     computing the new transform.
Result<IndexTransform<>> ApplySingleIndexSlice(IndexTransform<> transform,
                                               DimensionIndexBuffer* dimensions,
                                               IndexVectorOrScalarView indices,
                                               bool domain_only);

/// Type representing the IndexSlice operation.
/// \tparam Indices Container type for the indices vector.  Must satisfy
///     IsIndexVectorOrScalar.
template <typename Indices>
struct SingleIndexSliceOp {
  static constexpr bool selected_dimensions_are_new = false;

  static constexpr DimensionIndex static_selection_rank =
      IsIndexVectorOrScalar<Indices>::extent;

  constexpr static DimensionIndex GetNewStaticInputRank(
      DimensionIndex input_rank, DimensionIndex NumSelectedDims) {
    TENSORSTORE_CONSTEXPR_ASSERT(
        (input_rank == dynamic_rank || input_rank >= static_selection_rank) &&
        "Number of dimensions must not exceed input rank.");
    return RankConstraint::Subtract(
        input_rank,
        RankConstraint::And(NumSelectedDims, static_selection_rank));
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex num_input_dims) {
    TENSORSTORE_CONSTEXPR_ASSERT(
        RankConstraint::EqualOrUnspecified(num_input_dims,
                                           static_selection_rank) &&
        "Number of selected dimensions must match number of indices.");
    return 0;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions,
                                 bool domain_only) const {
    return ApplySingleIndexSlice(std::move(transform), dimensions,
                                 IndexVectorOrScalarView(indices), domain_only);
  }

  Indices indices;
};

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_SINGLE_INDEX_SLICE_OP_H_
