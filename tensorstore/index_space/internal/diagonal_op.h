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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_DIAGONAL_OP_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_DIAGONAL_OP_H_

/// \file
/// Implementation of the DimExpression::Diagonal operation.

// IWYU pragma: private, include "third_party/tensorstore/index_space/dim_expression.h"

#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/string_like.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_index_space {

/// Returns a new index transform with the diagonal of the specified dimensions
/// extracted.
///
/// In the returned index transform, a new dimension corresponding to the
/// diagonal is added as the first dimension, and the dimensions that were
/// diagonalized are removed.  The domain of the diagonalized dimension is equal
/// to the intersection of the specified dimensions.
///
/// \param transform Existing transform.
/// \param dimensions[in,out] The dimensions from which to extract the diagonal.
///     Must be non-null.  On return, `*dimensions` contains the single value
///     `0`, corresponding to the index of the new diagonal dimension.
/// \param domain_only Indicates the output dimensions of `transform` should be
///     ignored, and returned transform should have an output rank of 0.
/// \pre `transform.valid()`
/// \pre Each `index` in `*dimensions` must be unique and satisfy `0 <= index`
///     and `index < transform.input_rank()`.
/// \returns The new index transform.  The new transform is wrapped in a Result
///     for consistency with other operations, but an error Result is never
///     returned.
Result<IndexTransform<>> ApplyDiagonal(IndexTransform<> transform,
                                       DimensionIndexBuffer* dimensions,
                                       bool domain_only);

/// Empty type representing a Diagonal operation.  The dimensions from which to
/// extract the diagonal are specified by the dimension selection.
struct DiagonalOp {
  static constexpr bool selected_dimensions_are_new = false;

  constexpr static DimensionIndex GetNewStaticInputRank(
      DimensionIndex input_rank, DimensionIndex num_input_dims) {
    return RankConstraint::Add(
        RankConstraint::Subtract(input_rank, num_input_dims), 1);
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex num_input_dims) {
    return 1;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions,
                                 bool domain_only) const {
    return ApplyDiagonal(std::move(transform), dimensions, domain_only);
  }
};

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_DIAGONAL_OP_H_
