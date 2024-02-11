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

#include <memory>

#include "absl/time/time.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/grid_chunk_key_ranges.h"
#include "tensorstore/internal/grid_partition_impl.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/internal/storage_statistics.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

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
//   key_formatter: Specifies the key format.
//   staleness_bound: Staleness bound to use for kvstore operations.
//   options: Specifies which statistics to compute.
Future<ArrayStorageStatistics>
GetStorageStatisticsForRegularGridWithSemiLexicographicalKeys(
    const KvStore& kvs, IndexTransformView<> transform,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> chunk_shape, BoxView<> grid_bounds,
    std::unique_ptr<const LexicographicalGridIndexKeyParser> key_formatter,
    absl::Time staleness_bound, GetArrayStorageStatisticsOptions options);

// Same as above, but uses `Base10LexicographicalGridIndexKeyParser` as the
// `key_formatter`.
Future<ArrayStorageStatistics> GetStorageStatisticsForRegularGridWithBase10Keys(
    const KvStore& kvs, IndexTransformView<> transform,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> chunk_shape, span<const Index> shape,
    char dimension_separator, absl::Time staleness_bound,
    GetArrayStorageStatisticsOptions options);

struct GridStorageStatisticsChunkHandler
    : public internal::AtomicReferenceCount<GridStorageStatisticsChunkHandler> {
  internal::IntrusivePtr<GetStorageStatisticsAsyncOperationState> state;
  internal_grid_partition::IndexTransformGridPartition grid_partition;
  IndexTransform<> full_transform;
  span<const DimensionIndex> grid_output_dimensions;
  span<const Index> chunk_shape;
  const LexicographicalGridIndexKeyParser* key_formatter;

  virtual void ChunkPresent(span<const Index> grid_indices);

  virtual ~GridStorageStatisticsChunkHandler();
};

// Computes array storage statistics for drivers that map each chunk to a
// separate key with certain constraints.
//
// The constraints on keys are the same as specified for
// `GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys`.
//
//
// Args:
//   handler: Must have been initialized.
//   kvs: Key-value store.
//   grid_bounds: Range of grid indices along each grid dimension.  Must be the
//     same rank as `grid_output_dimensions`.
//   staleness_bound: Staleness bound to use for kvstore operations.
void GetStorageStatisticsForRegularGridWithSemiLexicographicalKeys(
    internal::IntrusivePtr<GridStorageStatisticsChunkHandler> handler,
    const KvStore& kvs, BoxView<> grid_bounds, absl::Time staleness_bound);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRID_STORAGE_STATISTICS_H_
