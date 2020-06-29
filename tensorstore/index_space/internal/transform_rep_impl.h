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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSFORM_REP_IMPL_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSFORM_REP_IMPL_H_

#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_index_space {

/// Copies the `input_labels` vector from `source` to `dest`.
///
/// \param source[in] Non-null pointer to source transform.
/// \param dest[out] Non-null pointer to dest transform.
/// \param can_move If `true`, may move the labels from `source`.
/// \dchecks `dest->input_rank_capacity >= source->input_rank`.
void CopyInputLabels(TransformRep* source, TransformRep* dest, bool can_move);

/// Checks that `predicate(combined[i], inner[i])` is true for `0 <= i < rank`.
///
/// Assigns `combined[i] = Intersect(combined[i], inner[i])`.
///
/// Typically, `predicate` is either `ContainsOrUnbounded` (in which case it
/// checks that all finite bounds on `inner` are contained within the
/// corresponding interval of `combined`) or `AreCompatibleOrUnbounded` (in
/// which case it checks that all finite bounds of the corresponding intervals
/// of `inner` and `combined` match, but corresponding finite and infinite
/// bounds are also allowed).
///
/// \dchecks `inner.rank() == combined.rank()`.
template <typename Predicate>
Status ValidateAndIntersectBounds(BoxView<> inner, MutableBoxView<> combined,
                                  Predicate predicate) {
  ABSL_ASSERT(inner.rank() == combined.rank());
  std::string error;
  for (DimensionIndex dim = 0; dim < inner.rank(); ++dim) {
    IndexIntervalRef outer_bounds = combined[dim];
    auto inner_bounds = inner[dim];
    if (!predicate(outer_bounds, inner_bounds)) {
      StrAppend(&error, error.empty() ? "" : ", ", "in dimension ", dim,
                " bounds ", inner_bounds, " vs. propagated bounds, ",
                outer_bounds);
    } else {
      outer_bounds = Intersect(outer_bounds, inner_bounds);
    }
  }
  if (!error.empty()) {
    return absl::OutOfRangeError(StrCat(
        "Propagated bounds are incompatible with existing bounds ", error));
  }
  return absl::OkStatus();
}

/// Converts a zero-rank index array output index map to a constant output index
/// map.
///
/// \param index The single index value in the zero-rank index array.
/// \param bounds The index_range for the index array.
/// \param output_offset[in,out] Pointer to the existing output offset, which
///     will be modified to account for `index`.
/// \param output_stride[in,out] Pointer to the existing output stride, which
///     will be set to 0.
/// \error `absl::StatusCode::kOutOfRange` if `index` is not contained within
///    `bounds`.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs.
Status ReplaceZeroRankIndexArrayIndexMap(Index index, IndexInterval bounds,
                                         Index* output_offset,
                                         Index* output_stride);

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSFORM_REP_IMPL_H_
