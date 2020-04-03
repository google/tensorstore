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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSLATE_OP_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSLATE_OP_H_

/// \file
/// Implementation of the DimExpression::TranslateTo and,
/// DimExpression::TranslateBy operations.

// IWYU pragma: private, include "third_party/tensorstore/index_space/dim_expression.h"

#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_vector_or_scalar.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_index_space {

/// Returns a new index transform with the domains of the specified input
/// dimensions shifted by the specified offsets.
///
/// See DimExpression::TranslateBy (for `translate_to == false`) and
/// DimExpression::TranslateTo (for `translate_to == true`) for details.
///
/// \param transform Existing transform.
/// \param dimensions[in] Non-null pointer to the list of indices of the
///     dimensions for which to shift the domains.  The value is not modified,
///     since these indices already correspond to the indices of those
///     dimensions in the result.
/// \param offsets The vector of offsets (if `translate_to == false`) or origins
///     (if `translate_to == true`) corresponding to the specified dimensions.
/// \param translate_to If `false`, the domains of the specified dimensions are
///     shifted by the corresponding offsets.  If `true`, the domains of the
///     specified dimensions are shifted to have origins equal to the
///     corresponding offsets.
/// \pre `transform.valid()`
/// \pre Each `index` in `*dimensions` must be unique and satisfy `0 <= index`
///     and `index < transform.input_rank()`.
/// \returns The new index transform.
/// \error `absl::StatusCode::kInvalidArgument` if `offsets.size()` is not
///     compatible with `dimensions->size()`.
/// \error `absl::StatusCode::kInvalidArgument` if `translate_to == true` and
///     the existing origin of one of the dimensions is `-kInfIndex`.
/// \error `absl::StatusCode::kOutOfRange` if an invalid offset is specified.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
///     computing the new transform.
Result<IndexTransform<>> ApplyTranslate(IndexTransform<> transform,
                                        DimensionIndexBuffer* dimensions,
                                        IndexVectorOrScalar offsets,
                                        bool translate_to);

/// Type representing the DimExpression::TranslateBy and
/// DimExpression::TranslateTo operations.
/// \tparam OffsetOrOriginVector The container type for the offset or origin
///     vector.  Must satisfy IsIndexVectorOrScalar.
/// \tparam To Corresponds to the `translate_to` parameter of ApplyTranslate.
template <typename OffsetOrOriginVector, bool To>
struct TranslateOp {
  static constexpr bool selected_dimensions_are_new = false;

  static constexpr DimensionIndex static_selection_rank =
      IsIndexVectorOrScalar<OffsetOrOriginVector>::extent;

  constexpr static DimensionIndex GetNewStaticInputRank(
      DimensionIndex input_rank, DimensionIndex num_input_dims) {
    TENSORSTORE_CONSTEXPR_ASSERT(
        (input_rank == dynamic_rank || input_rank >= static_selection_rank) &&
        "Number of dimensions must not exceed input rank.");
    return input_rank;
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex num_input_dims) {
    TENSORSTORE_CONSTEXPR_ASSERT(
        IsRankExplicitlyConvertible(num_input_dims, static_selection_rank) &&
        "Number of selected dimensions must match number of offsets.");
    return num_input_dims == dynamic_rank ? static_selection_rank
                                          : num_input_dims;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions) const {
    return ApplyTranslate(std::move(transform), dimensions,
                          IndexVectorOrScalar(offset_or_origin_vector), To);
  }

  OffsetOrOriginVector offset_or_origin_vector;
};

template <typename Offsets>
using TranslateToOp = TranslateOp<Offsets, true>;

template <typename Offsets>
using TranslateByOp = TranslateOp<Offsets, false>;

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSLATE_OP_H_
