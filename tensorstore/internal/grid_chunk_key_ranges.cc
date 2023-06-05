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

#include <algorithm>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/grid_partition.h"
#include "tensorstore/internal/regular_grid.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

absl::Status GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys(
    IndexTransformView<> transform,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> chunk_shape, BoxView<> grid_bounds,
    char dimension_separator,
    const LexicographicalGridIndexKeyFormatter& key_formatter,
    absl::FunctionRef<absl::Status(std::string key)> handle_key,
    absl::FunctionRef<absl::Status(KeyRange key_range, size_t prefix,
                                   BoxView<> grid_bounds)>
        handle_key_range) {
  Box<dynamic_rank(kMaxRank)> grid_bounds_copy(grid_bounds);
  const DimensionIndex rank = grid_output_dimensions.size();
  assert(rank == chunk_shape.size());
  assert(rank == grid_bounds.rank());
  std::string key;

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

  const auto handle_interval =
      [&](span<const Index> outer_prefix,
          IndexInterval inner_interval) -> absl::Status {
    if (rank == 0) {
      return handle_key("0");
    }

    // Compute the key prefix corresponding to `outer_prefix`.
    key.clear();
    for (DimensionIndex i = 0; i < outer_prefix.size(); ++i) {
      if (i != 0) {
        key += dimension_separator;
      }
      key_formatter.FormatGridIndex(key, i, outer_prefix[i]);
    }
    if (!outer_prefix.empty() && outer_prefix.size() != rank) {
      key += dimension_separator;
    }

    if (!IsFinite(inner_interval)) {
      // All keys starting with `outer_prefix` are valid.
      if (outer_prefix.size() != rank) {
        return handle_key_range(KeyRange::Prefix(key), key.size(),
                                SubBoxView(grid_bounds, outer_prefix.size()));
      } else {
        return handle_key(std::move(key));
      }
    }

    // Keys must be restricted by `inner_interval`.

    // Check if a portion of the indices in `inner_interval` need to be split
    // off individually due to lexicographical order / numerical order mismatch.
    const Index min_index_for_lexicographical_order =
        get_min_grid_index_for_lexicographical_order(outer_prefix.size());

    size_t orig_key_size = key.size();
    while (!inner_interval.empty() && inner_interval.inclusive_min() <
                                          min_index_for_lexicographical_order) {
      // Split off each `inner_interval.inclusive_min()` value into a separate
      // key range.
      key_formatter.FormatGridIndex(key, outer_prefix.size(),
                                    inner_interval.inclusive_min());
      if (outer_prefix.size() + 1 != rank) {
        key += dimension_separator;
        TENSORSTORE_RETURN_IF_ERROR(
            handle_key_range(KeyRange::Prefix(key), key.size(),
                             SubBoxView(grid_bounds, outer_prefix.size() + 1)));
      } else {
        TENSORSTORE_RETURN_IF_ERROR(handle_key(key));
      }
      inner_interval = IndexInterval::UncheckedClosed(
          inner_interval.inclusive_min() + 1, inner_interval.inclusive_max());
      key.resize(orig_key_size);
    }
    if (inner_interval.empty()) return absl::OkStatus();

    // The remaining interval has both bounds greater or equal to
    // `min_index_for_lexicographical_order`, and therefore it can be handled
    // with a single key range.
    IndexIntervalRef interval = grid_bounds_copy[outer_prefix.size()];
    IndexInterval old_interval = interval;
    interval = inner_interval;
    std::string exclusive_max_prefix = key;
    size_t strip_prefix_length = key.size();
    key_formatter.FormatGridIndex(key, outer_prefix.size(),
                                  inner_interval.inclusive_min());
    key_formatter.FormatGridIndex(exclusive_max_prefix, outer_prefix.size(),
                                  inner_interval.inclusive_max());
    auto status = handle_key_range(
        KeyRange{std::move(key),
                 KeyRange::PrefixExclusiveMax(std::move(exclusive_max_prefix))},
        strip_prefix_length, SubBoxView(grid_bounds_copy, outer_prefix.size()));
    interval = old_interval;
    return status;
  };

  return GetGridCellRanges(grid_output_dimensions, grid_bounds,
                           internal_grid_partition::RegularGridRef{chunk_shape},
                           transform, handle_interval);
}

}  // namespace internal
}  // namespace tensorstore
