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

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "tensorstore/box.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

class LexicographicalGridIndexKeyFormatter;

// Computes the set of keys and key ranges specifying the grid cells that cover
// the output range of `transform`.
//
// The key for a chunk of rank `n` with coordinates `grid_indices` must be of
// the form:
//
//     Encode(0, grid_indices[0]) +
//     dimension_separator +
//     Encode(1, grid_indices[1]) +
//     dimension_separator +
//     ...
//     Encode(n-1, grid_indices[n-1])
//
// where for each `i`, if `get_grid_min_index_for_max_digits(i) <= i < j`, then
// `Encode(i, j) < Encode(i, k)`.  That is, for grid indices greater or equal to
// `min_grid_indices_for_lexicographical_order[i]`, the lexicographical order of
// the keys matches the numeric order of the grid indices.
//
// Args:
//   transform: Index transform.
//   grid_output_dimensions: Output dimensions of `transform` corresponding to
//     each grid dimension.
//   chunk_shape: Chunk size along each grid dimension.  Must be the same length
//     as `grid_output_dimensions`.
//   grid_bounds: Range of grid indices along each grid dimension.  Must be the
//     same rank as `grid_output_dimensions`.
//   dimension_separator: Separator character between encoded grid indices.
//   key_formatter: Specifies the key format.
//   handle_key: Callback invoked for individual chunk keys.  Any error status
//     will be propagated immediately and no further callbacks will be invoked.
//   handle_key_range: Callback invoked for chunk key ranges.  Any error status
//     will be propagated immediately and no further callbacks will be invoked.
absl::Status GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys(
    IndexTransformView<> transform,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> chunk_shape, BoxView<> grid_bounds,
    char dimension_separator,
    const LexicographicalGridIndexKeyFormatter& key_formatter,
    absl::FunctionRef<absl::Status(std::string key)> handle_key,
    absl::FunctionRef<absl::Status(KeyRange key_range, size_t prefix,
                                   BoxView<> grid_bounds)>
        handle_key_range);

// Specifies the key format for
// `GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys`.
class LexicographicalGridIndexKeyFormatter {
 public:
  // Appends to `out` the representation of `grid_index` for dimension `dim`.
  virtual void FormatGridIndex(std::string& out, DimensionIndex dim,
                               Index grid_index) const = 0;

  // Returns the first grid index for dimension `dim` at which the formatted
  // keys are ordered lexicographically.
  virtual Index MinGridIndexForLexicographicalOrder(
      DimensionIndex dim, IndexInterval grid_interval) const = 0;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRID_CHUNK_KEY_RANGES_H_
