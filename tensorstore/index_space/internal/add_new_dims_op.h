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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_ADD_NEW_DIMS_OP_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_ADD_NEW_DIMS_OP_H_

/// \file
/// Implementation of the DimExpression::AddNew operation.

// IWYU pragma: private, include "third_party/tensorstore/index_space/dim_expression.h"

#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/string_like.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_index_space {

/// Returns a new index transform with the dimensions specified by `*dimensions`
/// added as new dummy dimensions with domains of [-kInfIndex, kInfIndex].
///
/// \param transform Existing transform.
/// \param dimensions[in] Non-null pointer to the list of indices of the new
///     dimensions.  The value is not modified, since these indices already
///     correspond to the indices of the new dimensions in the result.
/// \pre `transform.valid()`
/// \pre Each `index` in `*dimensions` must be unique and satisfy: `0 <= index`
///     and `index < transform.input_rank() + dimensions->size()`.
/// \returns The new index transform.  The new transform is wrapped in a Result
///     for consistency with other operations, but an error Result is never
///     returned.
Result<IndexTransform<>> ApplyAddNewDims(IndexTransform<> transform,
                                         DimensionIndexBuffer* dimensions);

/// Empty type representing an AddDims operation.  The new dummy dimensions to
/// add are specified by the dimension selection.
struct AddNewDimsOp {
  static constexpr bool selected_dimensions_are_new = true;

  constexpr static DimensionIndex GetNewStaticInputRank(
      DimensionIndex input_rank, DimensionIndex num_input_dims) {
    return AddStaticRanks(input_rank, num_input_dims);
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex num_input_dims) {
    return num_input_dims;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions) const {
    return ApplyAddNewDims(std::move(transform), dimensions);
  }
};

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_ADD_NEW_DIMS_OP_H_
