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

#ifndef TENSORSTORE_INTERNAL_GRID_STORAGE_STATISTICS_H_
#define TENSORSTORE_INTERNAL_GRID_STORAGE_STATISTICS_H_

#include <stdint.h>

#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/grid_chunk_key_ranges.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

class LexicographicalGridIndexKeyParser;

// Computes array storage statistics for drivers that map each chunk to a
// separate key with certain constraints.
//
// The constraints on keys are the same as specified for
// `GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys`.
//
//
// Args:
//   kvs: Key-value store.
//   transform: Index transform.
//   grid_output_dimensions: Output dimensions of `transform` corresponding to
//     each grid dimension.
//   chunk_shape: Chunk size along each grid dimension.  Must be the same length
//     as `grid_output_dimensions`.
//   grid_bounds: Range of grid indices along each grid dimension.  Must be the
//     same rank as `grid_output_dimensions`.
//   dimension_separator: Separator character between encoded grid indices.
//   key_formatter: Specifies the key format.
//   staleness_bound: Staleness bound to use for kvstore operations.
//   options: Specifies which statistics to compute.
Future<ArrayStorageStatistics>
GetStorageStatisticsForRegularGridWithSemiLexicographicalKeys(
    const KvStore& kvs, IndexTransformView<> transform,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> chunk_shape, BoxView<> grid_bounds,
    char dimension_separator,
    std::unique_ptr<const LexicographicalGridIndexKeyParser> key_formatter,
    absl::Time staleness_bound, GetArrayStorageStatisticsOptions options);

// Specifies the key format for
// `GetStorageStatisticsForRegularGridWithSemiLexicographicalKeys`.
class LexicographicalGridIndexKeyParser
    : public LexicographicalGridIndexKeyFormatter {
 public:
  // Parses a grid index, inverse of
  // `LexicographicalGridIndexKeyFormatter::FormatGridIndex`.
  virtual bool ParseGridIndex(std::string_view key, DimensionIndex dim,
                              Index& grid_index) const = 0;

  virtual ~LexicographicalGridIndexKeyParser();
};

// Same as above, but uses `Base10LexicographicalGridIndexKeyParser` as the
// `key_formatter`.
Future<ArrayStorageStatistics> GetStorageStatisticsForRegularGridWithBase10Keys(
    const KvStore& kvs, IndexTransformView<> transform,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> chunk_shape, span<const Index> shape,
    char dimension_separator, absl::Time staleness_bound,
    GetArrayStorageStatisticsOptions options);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRID_STORAGE_STATISTICS_H_
