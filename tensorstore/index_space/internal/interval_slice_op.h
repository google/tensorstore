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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_INTERVAL_SLICE_OP_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_INTERVAL_SLICE_OP_H_

#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_vector_or_scalar.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"

/// Implementation of the
/// DimExpression::{Translate,}{Closed,HalfOpen,Sized}Interval operations.

// IWYU pragma: private, include "third_party/tensorstore/index_space/dim_expression.h"

namespace tensorstore {
namespace internal_index_space {

/// Returns a new index transform with the domains of the specified input
/// dimensions restricted to an interval and optionally strided.
///
/// \param transform Existing transform.
/// \param dimensions[in] Non-null pointer to the list of indices of input
///     dimensions to be sliced.  The value is not modified, since these indices
///     already correspond to the indices of the new dimensions in the result.
/// \param interval_form The form of the interval (sized, closed, or half-open),
///     which specifies the interpretation of `stop_or_size_vector`.
/// \param translate If translate is `false`, the domain for each dimension `i`
///     is exactly that returned by
///
///         ExtractStridedSlice(transform.input_domain()[(*dimensions)[i]],
///                             interval_form, start_vector[i],
///                             stop_or_size_vector[i],
///                             stride_vector[i])
///
///     If translate is `true`, the domain for each sliced dimension is shifted
///     to have an origin of `0` (the output range is not affected).
/// \param start_vector Either a scalar, or a vector of size
///     `dimensions->size()`.
/// \param stop_or_size_vector Either a scalar, or a vector of size
///     `dimensions->size()`.
/// \param stride_vector Either a scalar, or a vector of size
///     `dimensions->size()`.
/// \pre `transform.valid()`
/// \pre Each `index` in `*dimensions` must be unique and satisfy `0 <= index`
///     and `index < transform.input_rank()`.
/// \returns The new index transform.
/// \error `absl::StatusCode::kInvalidArgument` if the size of `start_vector`,
///     `stop_or_size_vector`, or `stride_vector` is not compatible with
///     `dimensions->size()`.
/// \error `absl::StatusCode::kInvalidArgument` if a stride value is `0` or
///     `std::numeric_limits<Index>::max()`.
/// \error `absl::StatusCode::kInvalidArgument` if `translate` is `true` but the
///     computed domain for one of the selected dimensions is unbounded below.
/// \error `absl::StatusCode::kOutOfRange` if the specified interval is invalid.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs while
///     computing the new transform.
Result<IndexTransform<>> ApplyIntervalSliceOp(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions,
    IntervalForm interval_form, bool translate,
    IndexVectorOrScalar start_vector, IndexVectorOrScalar stop_or_size_vector,
    IndexVectorOrScalar stride_vector);

/// Type representing the {Translate,}{Closed,HalfOpen,Sized}Interval
/// operations.
/// \tparam StartVector The container type of start vector, must be explicitly
///     convertible to `IndexVectorOrScalar`.
/// \tparam StopOrSizeVector The container type of stop or size vector, must be
///     explicitly convertible to `IndexVectorOrScalar`.
/// \tparam StrideVector The container type of stride vector, must be explicitly
///     convertible to `IndexVectorOrScalar`.
template <typename StartVector, typename StopOrSizeVector,
          typename StrideVector>
struct IntervalSliceOp {
  static constexpr bool selected_dimensions_are_new = false;

  static constexpr DimensionIndex static_selection_rank =
      MaxStaticRank(IsIndexVectorOrScalar<StartVector>::extent,
                    IsIndexVectorOrScalar<StopOrSizeVector>::extent,
                    IsIndexVectorOrScalar<StrideVector>::extent);

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
        "Number of selected dimensions must match number of indices.");
    return num_input_dims == dynamic_rank ? static_selection_rank
                                          : num_input_dims;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions) const {
    return ApplyIntervalSliceOp(std::move(transform), dimensions, interval_form,
                                translate, IndexVectorOrScalar(start_vector),
                                IndexVectorOrScalar(stop_or_size_vector),
                                IndexVectorOrScalar(stride_vector));
  }

  /// The form of the interval (sized, closed, or half-open), which specifies
  /// the interpretation of `stop_or_size_vector`.
  IntervalForm interval_form;

  /// Whether to shift the origin of the sliced dimensions to `0`.
  bool translate;

  StartVector start_vector;
  StopOrSizeVector stop_or_size_vector;
  StrideVector stride_vector;
};

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_INTERVAL_SLICE_OP_H_
