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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSPOSE_OP_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSPOSE_OP_H_

/// \file
/// Implementation of the DimExpression::Transpose and
/// DimExpression::MoveTo{,Front,Back} operations.

// IWYU pragma: private, include "third_party/tensorstore/index_space/dim_expression.h"

#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_index_space {

/// Returns a new index transform with the dimensions permuted.
///
/// \param transform Existing transform.
/// \param dimensions[in,out] Must be non-null.  In input, specifies the
///     dimension indices corresponding to the input dimensions of `transform`
///     to be shifted.  On return, set to the corresponding indices of those
///     dimensions in the new transform.
/// \param target_dimensions The target dimension indices for each dimension in
///     `dimensions`.  A negative number `-n` is equivalent to
///     `transform.input_rank() - n`.  Dimensions not in `dimensions` remain in
///     the same relative order.
/// \param domain_only Indicates the output dimensions of `transform` should be
///     ignored, and returned transform should have an output rank of 0.
/// \pre `transform.valid()`
/// \pre Each `index` in `*dimensions` must be unique and satisfy `0 <= index`
///     and `index < transform.input_rank()`.
/// \error `absl::StatusCode::kInvalidArgument` if
///     `target_dimensions.size() != dimensions->size()`.
/// \error `absl::StatusCode::kInvalidArgument` if any `index` in
///     `target_dimensions`, after normalization of negative numbers, is not
///     unique or does not satisfy `0 <= index < transform.input_rank()`.
Result<IndexTransform<>> ApplyTransposeTo(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions,
    span<const DimensionIndex> target_dimensions, bool domain_only);

/// Returns a new index transform with the dimensions permuted according to a
/// dynamic target specification.
///
/// If `target_dimensions` contains a single dimension index (not specified as a
/// `DimRangeSpec`), this calls `ApplyMoveDimsTo`.
///
/// Otherwise, this expands the target dimension list and calls
/// `ApplyTransposeTo`.
///
/// \param transform Existing transform.
/// \param dimensions[in,out] Must be non-null.  In input, specifies the
///     dimension indices corresponding to the input dimensions of `transform`
///     to be shifted.  On return, set to the corresponding indices of those
///     dimensions in the new transform.
/// \param target_dim_specs The target dimension specifiers.  Must not specify
///     any dimensions by label.
/// \param domain_only Indicates the output dimensions of `transform` should be
///     ignored, and returned transform should have an output rank of 0.
/// \pre `transform.valid()`
/// \error `absl::StatusCode::kInvalidArgument` if `target_dimensions` specifies
///     any dimensions by label.
Result<IndexTransform<>> ApplyTransposeToDynamic(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions,
    span<const DynamicDimSpec> target_dim_specs, bool domain_only);

/// Type representing the `DimExpression::Transpose(target_dimensions)`
/// operation.
template <typename Container>
struct TransposeToOp {
  static constexpr bool selected_dimensions_are_new = false;
  static constexpr DimensionIndex static_selection_rank =
      internal::ConstSpanType<Container>::extent;

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
        "Number of selected dimensions must match number of target "
        "dimensions.");
    return num_input_dims == dynamic_rank ? static_selection_rank
                                          : num_input_dims;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions,
                                 bool domain_only) const {
    return ApplyTransposeTo(std::move(transform), dimensions, target_dimensions,
                            domain_only);
  }

  Container target_dimensions;
};

/// Returns a new index transform with the dimensions permuted.
///
/// Equivalent to calling `ApplyTransposeTo` with `target_dimensions` equal to
/// `{0, ..., transform.input_rank()-1}`.
///
/// \param transform Existing transform.
/// \param dimensions[in,out] Must be non-null.  On input, specifies the
///     dimension index of `transform` corresponding to each dimension of the
///     new transform.  On return, set to `0, ..., transform->input_rank()-1`.
/// \param domain_only Indicates the output dimensions of `transform` should be
///     ignored, and returned transform should have an output rank of 0.
/// \pre `transform.valid()`
/// \pre Each `index` in `*dimensions` must be unique and satisfy `0 <= index`
///     and `index < transform.input_rank()`.
/// \returns The new index transform.
/// \error `absl::StatusCode::kInvalidArgument` if
///     `dimensions->size() != transform.input_rank()`.
Result<IndexTransform<>> ApplyTranspose(IndexTransform<> transform,
                                        DimensionIndexBuffer* dimensions,
                                        bool domain_only);

/// Empty type representing the `DimExpression::Transpose()` operation.
struct TransposeOp {
  static constexpr bool selected_dimensions_are_new = false;

  constexpr static DimensionIndex GetNewStaticInputRank(
      DimensionIndex input_rank, DimensionIndex num_input_dims) {
    TENSORSTORE_CONSTEXPR_ASSERT(
        RankConstraint::EqualOrUnspecified(input_rank, num_input_dims) &&
        "Number of selected dimensions must equal input rank.");
    return input_rank == dynamic_rank ? num_input_dims : input_rank;
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex num_input_dims) {
    return num_input_dims;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions,
                                 bool domain_only) const {
    return ApplyTranspose(std::move(transform), dimensions, domain_only);
  }
};

/// Returns a new index transform with the dimensions permuted such that the
/// specified dimensions are consecutive and start or end at the specified
/// target dimension index.
///
/// \param transform[in] Non-null pointer to existing transform.
/// \param dimensions[in,out] Must be non-null.  On input, specifies the
///     dimension indices to be shifted.  On return, set to the corresponding
///     indices of those dimensions in the new transform.
/// \param target The target dimension index, must be in the range
///     `[-transform->input_rank() + dimensions->size() - 1, `
///     ` transform->input_rank() - dimensions->size()]`.  If `target >= 0`,
///     `target` is the new index of the first selected dimension.  If
///     `target < 0`, `target + transform.input_rank()` is the new index of the
///     last selected dimension.
/// \param domain_only Indicates the output dimensions of `transform` should be
///     ignored, and returned transform should have an output rank of 0.
/// \pre Each `index` in `*dimensions` must be unique and satisfy `0 <= index`
///     and `index < transform.input_rank()`.
/// \returns The new index transform.
/// \error `absl::StatusCode::kInvalidArgument` if
///     `dimensions->size() != transform.input_rank()`.
Result<IndexTransform<>> ApplyMoveDimsTo(IndexTransform<> transform,
                                         DimensionIndexBuffer* dimensions,
                                         DimensionIndex target,
                                         bool domain_only);

/// Type representing the DimExpression::MoveTo{,Front,Back} operations.
struct MoveToOp {
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
                                 DimensionIndexBuffer* dimensions,
                                 bool domain_only) const {
    return ApplyMoveDimsTo(std::move(transform), dimensions, target,
                           domain_only);
  }

  DimensionIndex target;
};

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSPOSE_OP_H_
