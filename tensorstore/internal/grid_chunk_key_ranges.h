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

#ifndef TENSORSTORE_INTERNAL_GRID_CHUNK_KEY_RANGES_H_
#define TENSORSTORE_INTERNAL_GRID_CHUNK_KEY_RANGES_H_

#include <string>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/internal/regular_grid.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

namespace internal_grid_partition {
class IndexTransformGridPartition;
}  // namespace internal_grid_partition

namespace internal {

// Computes the set of keys and key ranges specifying the grid cells that cover
// the output range of `transform`.
//
// The key for a chunk of rank `n` with coordinates `grid_indices` must be of
// the form:
//
//     Encode(0, grid_indices[0]) +
//     Encode(1, grid_indices[1]) +
//     ...
//     Encode(n-1, grid_indices[n-1])
//
// where for each `i`, if `get_grid_min_index_for_max_digits(i) <= i < j`, then
// `Encode(i, j) < Encode(i, k)`.  That is, for grid indices greater or equal to
// `min_grid_indices_for_lexicographical_order[i]`, the lexicographical order of
// the keys matches the numeric order of the grid indices.
//
// Args:
//   grid_partition: Must have been previously initialized by a call to
//     `PrePartitionIndexTransformOverGrid` with the same `transform`,
//     `grid_output_dimensions`, and `output_to_grid_cell`.
//   transform: Index transform.
//   grid_output_dimensions: Output dimensions of `transform` corresponding to
//     each grid dimension.
//   output_to_grid_cell: Computes the grid cell corresponding to a given output
//     index.
//   grid_bounds: Range of grid indices along each grid dimension.  Must be the
//     same rank as `grid_output_dimensions`.
//   key_formatter: Specifies the key format.
//   handle_key: Callback invoked for individual chunk keys.  Any error status
//     will be propagated immediately and no further callbacks will be invoked.
//   handle_key_range: Callback invoked for chunk key ranges.  Any error status
//     will be propagated immediately and no further callbacks will be invoked.
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
        handle_key_range);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRID_CHUNK_KEY_RANGES_H_
