// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/grid_chunk_key_ranges.h"

#include <cassert>
#include <string>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/grid_partition.h"
#include "tensorstore/internal/grid_partition_impl.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/internal/regular_grid.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

absl::Status GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys(
    const internal_grid_partition::IndexTransformGridPartition& grid_partition,
    IndexTransformView<> transform,
    span<const DimensionIndex> grid_output_dimensions,
    internal_grid_partition::OutputToGridCellFn output_to_grid_cell,
    BoxView<> grid_bounds,
    const LexicographicalGridIndexKeyFormatter& key_formatter,
    absl::FunctionRef<absl::Status(std::string key,
                                   span<const Index> grid_indices)>
        handle_key,
    absl::FunctionRef<absl::Status(KeyRange key_range, BoxView<> grid_bounds)>
        handle_key_range) {
  Box<dynamic_rank(kMaxRank)> grid_bounds_copy(grid_bounds);
  assert(grid_output_dimensions.size() == grid_bounds.rank());

  // Cache of last computed value for
  // `min_grid_index_for_lexicographical_order`.  In practice will only be
  // computed for a single dimension.
  DimensionIndex cached_min_grid_index_for_lexicographical_order_dim = -1;
  Index cached_min_grid_index_for_lexicographical_order;

  const auto get_min_grid_index_for_lexicographical_order =
      [&](DimensionIndex dim) {
        if (dim == cached_min_grid_index_for_lexicographical_order_dim) {
          return cached_min_grid_index_for_lexicographical_order;
        }
        cached_min_grid_index_for_lexicographical_order_dim = dim;
        return cached_min_grid_index_for_lexicographical_order =
                   key_formatter.MinGridIndexForLexicographicalOrder(
                       dim, grid_bounds[dim]);
      };

  const auto forward_bounds =
      [&](BoxView<> bounds, DimensionIndex outer_prefix_rank) -> absl::Status {
    if (bounds.num_elements() == 1) {
      return handle_key(key_formatter.FormatKey(bounds.origin()),
                        bounds.origin());
    }
    assert(outer_prefix_rank < bounds.rank());
    if (bounds[outer_prefix_rank] == grid_bounds[outer_prefix_rank]) {
      // Use prefix as key range.
      return handle_key_range(KeyRange::Prefix(key_formatter.FormatKey(
                                  bounds.origin().first(outer_prefix_rank))),
                              bounds);
    }
    DimensionIndex key_dims = outer_prefix_rank + 1;
    Index inclusive_max_indices[kMaxRank];
    for (DimensionIndex i = 0; i < key_dims; ++i) {
      inclusive_max_indices[i] = bounds[i].inclusive_max();
    }
    return handle_key_range(
        KeyRange(key_formatter.FormatKey(bounds.origin().first(key_dims)),
                 KeyRange::PrefixExclusiveMax(key_formatter.FormatKey(
                     span<const Index>(&inclusive_max_indices[0], key_dims)))),
        bounds);
  };

  const auto handle_interval = [&](BoxView<> bounds) -> absl::Status {
    // Find first dimension of `bounds` where size is not 1.
    DimensionIndex outer_prefix_rank = 0;
    while (outer_prefix_rank < bounds.rank() &&
           bounds.shape()[outer_prefix_rank] == 1) {
      ++outer_prefix_rank;
    }

    // Check if `outer_prefix_rank` dimension is unconstrained.
    if (outer_prefix_rank == bounds.rank() ||
        bounds[outer_prefix_rank] == grid_bounds[outer_prefix_rank]) {
      return forward_bounds(bounds, outer_prefix_rank);
    }

    // Keys must be restricted by `inner_interval`.

    // Check if a portion of the indices in `inner_interval` need to be split
    // off individually due to lexicographical order / numerical order mismatch.
    const Index min_index_for_lexicographical_order =
        get_min_grid_index_for_lexicographical_order(outer_prefix_rank);

    if (min_index_for_lexicographical_order <=
        bounds.origin()[outer_prefix_rank]) {
      // Entire box is a single lexicographical range.
      return forward_bounds(bounds, outer_prefix_rank);
    }

    Box<dynamic_rank(kMaxRank)> new_bounds(bounds);
    IndexInterval inner_interval = bounds[outer_prefix_rank];
    while (!inner_interval.empty() && inner_interval.inclusive_min() <
                                          min_index_for_lexicographical_order) {
      // Split off each `inner_interval.inclusive_min()` value into a separate
      // key range.
      new_bounds[outer_prefix_rank] =
          IndexInterval::UncheckedSized(inner_interval.inclusive_min(), 1);
      TENSORSTORE_RETURN_IF_ERROR(
          forward_bounds(new_bounds, outer_prefix_rank + 1));
      inner_interval = IndexInterval::UncheckedClosed(
          inner_interval.inclusive_min() + 1, inner_interval.inclusive_max());
    }
    if (inner_interval.empty()) return absl::OkStatus();

    // The remaining interval has both bounds greater or equal to
    // `min_index_for_lexicographical_order`, and therefore it can be handled
    // with a single key range.
    new_bounds[outer_prefix_rank] = inner_interval;
    return forward_bounds(new_bounds, inner_interval.size() == 1
                                          ? outer_prefix_rank + 1
                                          : outer_prefix_rank);
  };

  return internal_grid_partition::GetGridCellRanges(
      grid_partition, grid_output_dimensions, grid_bounds, output_to_grid_cell,
      transform, handle_interval);
}

}  // namespace internal
}  // namespace tensorstore
