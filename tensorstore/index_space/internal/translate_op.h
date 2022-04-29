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
///
/// Implementation of the DimExpression::Translate{To,By,BackwardBy} operations.

// IWYU pragma: private, include "third_party/tensorstore/index_space/dim_expression.h"

#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_vector_or_scalar.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_index_space {

/// Specifies the meaning of the `offsets` vector for `ApplyTranslate`.
enum class TranslateOpKind {
  /// Translate the domain to the origin indicated by the offset.
  kTranslateTo,
  /// Translate the domain forward by the offset.
  kTranslateBy,
  /// Translate the domain backward by the offset.
  kTranslateBackwardBy,
};

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
/// \param offsets The vector of offsets (if `kind != kTranslateTo`) or origins
///     (if `kind == kTranslateTo`) corresponding to the specified dimensions.
/// \param kind Specifies the meaning of `offsets`.
/// \param domain_only Indicates the output dimensions of `transform` should be
///     ignored, and returned transform should have an output rank of 0.
/// \pre `transform.valid()`
/// \pre Each `index` in `*dimensions` must be unique and satisfy `0 <= index`
///     and `index < transform.input_rank()`.
/// \returns The new index transform.
/// \error `absl::StatusCode::kInvalidArgument` if `offsets.size()` is not
///     compatible with `dimensions->size()`.
/// \error `absl::StatusCode::kInvalidArgument` if `kind == kTranslateTo` and
///     the existing origin of one of the dimensions is `-kInfIndex`.
/// \error `absl::StatusCode::kOutOfRange` if an invalid offset is specified.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
///     computing the new transform.
Result<IndexTransform<>> ApplyTranslate(IndexTransform<> transform,
                                        DimensionIndexBuffer* dimensions,
                                        IndexVectorOrScalarView offsets,
                                        TranslateOpKind kind,
                                        bool domain_only = false);

/// Type representing the DimExpression::Translate{To,By,BackwardBy} operations.
/// \tparam OffsetOrOriginVector The container type for the offset or origin
///     vector.  Must satisfy IsIndexVectorOrScalar.
/// \tparam Kind Corresponds to the `kind` parameter of ApplyTranslate.
template <typename OffsetOrOriginVector, TranslateOpKind Kind>
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
        RankConstraint::EqualOrUnspecified(num_input_dims,
                                           static_selection_rank) &&
        "Number of selected dimensions must match number of offsets.");
    return num_input_dims == dynamic_rank ? static_selection_rank
                                          : num_input_dims;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions,
                                 bool domain_only) const {
    return ApplyTranslate(std::move(transform), dimensions,
                          IndexVectorOrScalarView(offset_or_origin_vector),
                          Kind, domain_only);
  }

  OffsetOrOriginVector offset_or_origin_vector;
};

template <typename Offsets>
using TranslateToOp = TranslateOp<Offsets, TranslateOpKind::kTranslateTo>;

template <typename Offsets>
using TranslateByOp = TranslateOp<Offsets, TranslateOpKind::kTranslateBy>;

template <typename Offsets>
using TranslateBackwardByOp =
    TranslateOp<Offsets, TranslateOpKind::kTranslateBackwardBy>;

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSLATE_OP_H_
