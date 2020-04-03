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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_MARK_EXPLICIT_OP_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_MARK_EXPLICIT_OP_H_

/// \file
/// Implementation of the
/// DimExpression::{Unsafe,}MarkBounds{Explicit,Implicit} operations.

// IWYU pragma: private, include "third_party/tensorstore/index_space/dim_expression.h"

#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_index_space {

/// Returns a new index transform with the implicit bit of some bounds changed.
///
/// \param transform Existing transform.
/// \param dimensions[in] Non-null pointer to the list of indices of the
///     dimensions for which to change the implicit bit.  The value is not
///     modified, since these indices already correspond to the indices of the
///     new dimensions in the result.
/// \param implicit If `true`, change the implicit state to `true`.  If `false`,
///     change the implicit state to `false`.
/// \param lower If `true`, change the implicit state of the lower bounds of the
///     specified dimensions.
/// \param upper If `true`, change the implicit state of the upper bounds of the
///     specified dimensions.
/// \pre `transform.valid()`
/// \pre Each `index` in `*dimensions` must be unique and satisfy `0 <= index`
///     and `index < transform.input_rank()`.
/// \returns The new index transform.
Result<IndexTransform<>> ApplyChangeImplicitState(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions, bool implicit,
    bool lower, bool upper);

/// Type representing a {Unsafe,}MarkBounds{Explicit,Implicit} operation.
struct ChangeImplicitStateOp {
  static constexpr bool selected_dimensions_are_new = false;

  constexpr static DimensionIndex GetNewStaticInputRank(
      DimensionIndex input_rank, DimensionIndex num_input_dims) {
    return input_rank;
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex num_input_dims) {
    return num_input_dims;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions) const {
    return ApplyChangeImplicitState(std::move(transform), dimensions,
                                    /*implicit=*/implicit, lower, upper);
  }

  bool implicit;
  bool lower;
  bool upper;
};

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_MARK_EXPLICIT_OP_H_
