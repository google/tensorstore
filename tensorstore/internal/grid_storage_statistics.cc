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

#include "tensorstore/internal/grid_storage_statistics.h"

#include <stdint.h>

#include <atomic>
#include <cassert>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/grid_chunk_key_ranges.h"
#include "tensorstore/internal/grid_chunk_key_ranges_base10.h"
#include "tensorstore/internal/grid_partition_impl.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/internal/regular_grid.h"
#include "tensorstore/internal/storage_statistics.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

#ifndef TENSORSTORE_INTERNAL_GRID_STORAGE_STATISTICS_DEBUG
#define TENSORSTORE_INTERNAL_GRID_STORAGE_STATISTICS_DEBUG 0
#endif

namespace tensorstore {
namespace internal {
namespace {

using ::tensorstore::kvstore::ListEntry;

template <typename T>
struct MovableAtomic : public std::atomic<T> {
  using std::atomic<T>::atomic;
  MovableAtomic(MovableAtomic&& other) noexcept {
    this->store(other.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
  }
};

struct ListReceiver {
  internal::IntrusivePtr<GridStorageStatisticsChunkHandler> handler;
  Box<> grid_bounds;
  MovableAtomic<int64_t> total_chunks_seen{0};
  FutureCallbackRegistration cancel_registration;

  template <typename Cancel>
  void set_starting(Cancel cancel) {
    cancel_registration =
        handler->state->promise.ExecuteWhenNotNeeded(std::move(cancel));
  }

  void set_stopping() { cancel_registration.Unregister(); }

  void set_done() {
    if (grid_bounds.num_elements() !=
        total_chunks_seen.load(std::memory_order_relaxed)) {
      handler->state->ChunkMissing();
    }
  }

  void set_value(ListEntry entry) {
    Index grid_indices[kMaxRank];
    const DimensionIndex rank = handler->grid_output_dimensions.size();
    span<Index> grid_indices_span(&grid_indices[0], rank);
    if (!handler->key_formatter->ParseKey(entry.key, grid_indices_span) ||
        !Contains(grid_bounds, span<const Index>(grid_indices_span))) {
      return;
    }
    ++total_chunks_seen;

    handler->ChunkPresent(grid_indices_span);
  }

  void set_error(absl::Status error) {
    handler->state->SetError(std::move(error));
  }
};

}  // namespace

GridStorageStatisticsChunkHandler::~GridStorageStatisticsChunkHandler() =
    default;

void GridStorageStatisticsChunkHandler::ChunkPresent(
    span<const Index> grid_indices) {
  state->IncrementChunksPresent();
}

Future<ArrayStorageStatistics>
GetStorageStatisticsForRegularGridWithSemiLexicographicalKeys(
    const KvStore& kvs, IndexTransformView<> transform,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> chunk_shape, BoxView<> grid_bounds,
    std::unique_ptr<const LexicographicalGridIndexKeyParser> key_formatter,
    absl::Time staleness_bound, GetArrayStorageStatisticsOptions options) {
  // TODO(jbms): integrate this with the chunk cache

  struct Handler : public GridStorageStatisticsChunkHandler {
    std::unique_ptr<const LexicographicalGridIndexKeyParser> key_formatter_ptr;
  };

  Future<ArrayStorageStatistics> future;
  auto handler = internal::MakeIntrusivePtr<Handler>();
  // Note: `future` is a output parameter.
  handler->state =
      internal::MakeIntrusivePtr<GetStorageStatisticsAsyncOperationState>(
          future, options);
  handler->full_transform = transform;
  handler->grid_output_dimensions = grid_output_dimensions;
  handler->chunk_shape = chunk_shape;
  handler->key_formatter_ptr = std::move(key_formatter);
  handler->key_formatter = handler->key_formatter_ptr.get();

  // This function calls
  // `GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys` to compute the
  // set of individual chunk keys and chunk key ranges that correspond to the
  // range of `transform`.  The keys and key ranges are computed, and read and
  // list operations are issued, all before this function returns.  The
  // `AsyncOperationState` object handles the asynchronous
  // completion of the read and list operations.

  internal::GetStorageStatisticsForRegularGridWithSemiLexicographicalKeys(
      std::move(handler), std::move(kvs), grid_bounds, staleness_bound);

  return future;
}

void GetStorageStatisticsForRegularGridWithSemiLexicographicalKeys(
    internal::IntrusivePtr<GridStorageStatisticsChunkHandler> handler,
    const KvStore& kvs, BoxView<> grid_bounds, absl::Time staleness_bound) {
  // TODO(jbms): integrate this with the chunk cache

  // This function calls
  // `GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys` to compute the
  // set of individual chunk keys and chunk key ranges that correspond to the
  // range of `transform`.  The keys and key ranges are computed, and read and
  // list operations are issued, all before this function returns.  The
  // `handler` object handles the asynchronous completion of the read and list
  // operations.

  int64_t total_chunks = 0;

  const auto handle_key = [&](std::string key, span<const Index> grid_indices) {
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_GRID_STORAGE_STATISTICS_DEBUG)
        << "key: " << tensorstore::QuoteString(key);
    if (internal::AddOverflow<Index>(total_chunks, 1, &total_chunks)) {
      return absl::OutOfRangeError(
          "Integer overflow computing number of chunks");
    }
    kvstore::ReadOptions read_options;
    read_options.byte_range = OptionalByteRangeRequest(0, 0);
    read_options.staleness_bound = staleness_bound;
    LinkValue(
        [handler, grid_indices = std::vector<Index>(grid_indices.begin(),
                                                    grid_indices.end())](
            Promise<ArrayStorageStatistics> promise,
            ReadyFuture<kvstore::ReadResult> future) {
          auto& read_result = future.value();
          if (!read_result.has_value()) {
            handler->state->ChunkMissing();
          } else {
            handler->ChunkPresent(grid_indices);
          }
        },
        handler->state->promise,
        kvstore::Read(kvs, std::move(key), std::move(read_options)));
    return absl::OkStatus();
  };

  const auto handle_key_range = [&](KeyRange key_range,
                                    BoxView<> grid_bounds) -> absl::Status {
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_GRID_STORAGE_STATISTICS_DEBUG)
        << "key_range: " << key_range << ", grid_bounds=" << grid_bounds;
    Index cur_total_chunks = grid_bounds.num_elements();
    if (cur_total_chunks == std::numeric_limits<Index>::max()) {
      return absl::OutOfRangeError(tensorstore::StrCat(
          "Integer overflow computing number of chunks in ", grid_bounds));
    }
    if (internal::AddOverflow(total_chunks, cur_total_chunks, &total_chunks)) {
      return absl::OutOfRangeError(
          "Integer overflow computing number of chunks");
    }
    kvstore::ListOptions list_options;
    list_options.staleness_bound = staleness_bound;
    list_options.range = std::move(key_range);
    kvstore::List(kvs, std::move(list_options),
                  ListReceiver{handler, Box<>(grid_bounds)});
    return absl::OkStatus();
  };

  internal_grid_partition::RegularGridRef output_to_grid_cell{
      handler->chunk_shape};

  TENSORSTORE_RETURN_IF_ERROR(
      internal_grid_partition::PrePartitionIndexTransformOverGrid(
          handler->full_transform, handler->grid_output_dimensions,
          output_to_grid_cell, handler->grid_partition),
      handler->state->SetError(_));

  TENSORSTORE_RETURN_IF_ERROR(
      internal::GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys(
          handler->grid_partition, handler->full_transform,
          handler->grid_output_dimensions, output_to_grid_cell, grid_bounds,
          *handler->key_formatter, handle_key, handle_key_range),
      handler->state->SetError(_));

  handler->state->total_chunks += total_chunks;
}

Future<ArrayStorageStatistics> GetStorageStatisticsForRegularGridWithBase10Keys(
    const KvStore& kvs, IndexTransformView<> transform,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> chunk_shape, span<const Index> shape,
    char dimension_separator, absl::Time staleness_bound,
    GetArrayStorageStatisticsOptions options) {
  const DimensionIndex rank = grid_output_dimensions.size();
  assert(rank == chunk_shape.size());
  assert(rank == shape.size());
  Box<dynamic_rank(kMaxRank)> grid_bounds(rank);
  for (DimensionIndex i = 0; i < shape.size(); ++i) {
    const Index grid_size = CeilOfRatio(shape[i], chunk_shape[i]);
    grid_bounds[i] = IndexInterval::UncheckedSized(0, grid_size);
  }
  return GetStorageStatisticsForRegularGridWithSemiLexicographicalKeys(
      kvs, transform, grid_output_dimensions, chunk_shape, grid_bounds,
      std::make_unique<Base10LexicographicalGridIndexKeyParser>(
          rank, dimension_separator),
      staleness_bound, std::move(options));
}

}  // namespace internal
}  // namespace tensorstore
