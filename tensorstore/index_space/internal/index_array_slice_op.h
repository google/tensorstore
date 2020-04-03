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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_INDEX_ARRAY_SLICE_OP_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_INDEX_ARRAY_SLICE_OP_H_

/// \file
/// Implementation of the DimExpression::IndexArraySlice,
/// DimExpression::IndexVectorArraySlice, and
/// DimExpression::OuterIndexArraySlice operations.

// IWYU pragma: private, include "third_party/tensorstore/index_space/dim_expression.h"

#include "tensorstore/array.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_index_space {

/// Returns a new index transform modified by the specified index array mapping.
///
/// See DimExpression::IndexArraySlice and DimExpression::OuterIndexArraySlice
/// for details.
///
/// \param transform Existing transform.
/// \param dimensions[in,out] Non-null pointer to the list of indices of input
///     dimensions to which the specified index arrays map.  On output, set to
///     the list of indices in the new transform corresponding to the input
///     dimensions of the index arrays.
/// \pre `transform.valid()`
/// \pre Each `index` in `*dimensions` must be unique and satisfy `0 <= index`
///     and `index < transform.input_rank()`.
/// \param index_arrays List of index arrays corresponding to the elements of
///     `*dimensions`.
/// \param outer_indexing If `false`, the behavior is as specified for
///     DimExpression::IndexArraySlice.  If `true`, the behavior is as specified
///     for DimExpression::OuterIndexArraySlice.
/// \returns The new index transform.
/// \error `absl::StatusCode::kInvalidArgument` if `index_arrays.size()` is not
///     equal to `dimensions->size()`.
/// \error `absl::StatusCode::kInvalidArgument` if `outer_indexing == false` and
///     `index_arrays.size() == 0` or the index arrays cannot be broadcast to a
///     common shape.
/// \error `absl::StatusCode::kOutOfRange` if an invalid index is specified.
///     Because indices are checked lazily, this error is not guaranteed to
///     occur if an invalid index is specified.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
///     computing the new transform.
Result<IndexTransform<>> ApplyIndexArraySlice(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions,
    span<const SharedArrayView<const Index>> index_arrays, bool outer_indexing);

/// Type representing the IndexArraySlice and OuterIndexArraySlice operations.
/// \tparam OuterIndexing Specifies whether joint or outer indexing is used.
/// \tparam IndexArrayInputRank The number of input dimensions of the new index
///     transform corresponding to the specified index arrays, or `dynamic_rank`
///     if unknown at compile time.  If `OuterIndexing` is `false`, this is
///     equal to the common static rank of all of the index arrays.  If
///     `OuterIndexing` is `true`, this is equal to the sum of the static ranks
///     of all of the index arrays.
/// \tparam IndexArrays The container type of the index arrays.  Must be
///     convertible to `span<const SharedArrayView<const Index>>`.
template <bool OuterIndexing, DimensionIndex IndexArrayInputRank,
          typename IndexArrays>
struct IndexArraySliceOp {
  static constexpr bool selected_dimensions_are_new = false;

  using IndexArrayType = typename IndexArrays::value_type;
  constexpr static DimensionIndex static_selection_rank =
      internal::ConstSpanType<IndexArrays>::extent;

  constexpr static DimensionIndex GetNewStaticInputRank(
      DimensionIndex input_rank, DimensionIndex num_input_dims) {
    TENSORSTORE_CONSTEXPR_ASSERT(
        (input_rank == dynamic_rank || input_rank >= static_selection_rank) &&
        "Number of dimensions must not exceed input rank.");
    return AddStaticRanks(SubtractStaticRanks(input_rank, num_input_dims),
                          IndexArrayInputRank);
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex num_input_dims) {
    TENSORSTORE_CONSTEXPR_ASSERT(
        IsRankExplicitlyConvertible(num_input_dims, static_selection_rank) &&
        "Number of selected dimensions must match number of indices.");
    return IndexArrayInputRank;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions) const {
    return ApplyIndexArraySlice(std::move(transform), dimensions, index_arrays,
                                OuterIndexing);
  }

  IndexArrays index_arrays;
};

/// Returns a new index transform modified by the specified index array mapping.
///
/// See DimExpression::IndexVectorArraySlice for details.
///
/// \param transform Existing transform.
/// \param dimensions[in,out] Non-null pointer to the list of indices of input
///     dimensions corresponding to the `vector_dimension` dimension of
///     `index_vector_array`.  On output, set to
///     `0:index_vector_array.size() - 1`.
/// \param vector_dimension The dimension of `index_vector_array` corresponding
///     to the selected dimensions.  May be a negative value, as supported by
///     NormalizeDimensionIndex.
/// \param index_vector_array The array of index vectors.
/// \pre `transform.valid()`
/// \returns The new index transform.
/// \error `absl::StatusCode::kInvalidArgument` if
///     `index_vector_array.size(vector_dimension) != dimensions->size()`.
/// \error `absl::StatusCode::kOutOfRange` if an invalid index is specified.
/// Because
///     indices are checked lazily, this error is not guaranteed to occur if an
///     invalid index is specified.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
/// computing the new
///     transform.
Result<IndexTransform<>> ApplyIndexVectorArraySlice(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions,
    DimensionIndex vector_dimension,
    const SharedArrayView<const Index>& index_vector_array);

/// Type representing the IndexVectorArraySlice operation.
/// \tparam IndexVectorArrayRank Static rank of the index vector array.
template <DimensionIndex IndexVectorArrayRank>
struct IndexVectorArraySliceOp {
  static_assert(IndexVectorArrayRank >= 1,
                "Index vector array must have rank >= 1.");
  static constexpr bool selected_dimensions_are_new = false;

  constexpr static DimensionIndex GetNewStaticInputRank(
      DimensionIndex input_rank, DimensionIndex num_input_dims) {
    return AddStaticRanks(SubtractStaticRanks(input_rank, num_input_dims),
                          SubtractStaticRanks(IndexVectorArrayRank, 1));
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex num_input_dims) {
    return SubtractStaticRanks(IndexVectorArrayRank, 1);
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions) const {
    return ApplyIndexVectorArraySlice(std::move(transform), dimensions,
                                      vector_dimension, index_vector_array);
  }

  /// The index vector array.
  SharedArrayView<const Index, IndexVectorArrayRank> index_vector_array;

  /// Specifies the dimension of `index_vector_array` that corresponds to the
  /// selected dimensions.
  DimensionIndex vector_dimension;
};

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_INDEX_ARRAY_SLICE_OP_H_
