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

#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/str_split.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/grid_chunk_key_ranges.h"
#include "tensorstore/internal/grid_chunk_key_ranges_base10.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

#ifndef TENSORSTORE_INTERNAL_GRID_STORAGE_STATISTICS_DEBUG
#define TENSORSTORE_INTERNAL_GRID_STORAGE_STATISTICS_DEBUG 0
#endif

namespace tensorstore {
namespace internal {

namespace {
struct GetStorageStatisticsAsyncOperationState
    : public internal::AtomicReferenceCount<
          GetStorageStatisticsAsyncOperationState> {
  std::atomic<int64_t> chunks_present{0};
  int64_t total_chunks = 0;
  GetArrayStorageStatisticsOptions options;
  Promise<ArrayStorageStatistics> promise;
  char dimension_separator;
  std::unique_ptr<const LexicographicalGridIndexKeyParser> key_formatter;
  DimensionIndex rank;
  std::atomic<bool> chunk_missing{false};

  // Check if we can stop early.
  void MaybeStopEarly() {
    if (options.mask & ArrayStorageStatistics::query_not_stored) {
      if (chunks_present.load() == 0) {
        // Don't yet know if any data is stored.
        return;
      }
    }

    if (options.mask & ArrayStorageStatistics::query_fully_stored) {
      if (chunk_missing.load() == false) {
        // Don't yet know if any data is missing.
        return;
      }
    }

    // Mark `promise` as `result_not_needed`.  The actual statistics will be set
    // by `~GetStorageStatisticsAsyncOperationState`.
    SetDeferredResult(promise, ArrayStorageStatistics{});
  }

  void IncrementChunksPresent() {
    if (++chunks_present == 1) {
      MaybeStopEarly();
    }
  }

  void ChunkMissing() {
    if (chunk_missing.exchange(true) == false) {
      MaybeStopEarly();
    }
  }

  ~GetStorageStatisticsAsyncOperationState() {
    auto& r = promise.raw_result();
    if (!r.ok()) return;
    r->mask = options.mask;
    int64_t num_present = chunks_present.load(std::memory_order_relaxed);
    if (options.mask & ArrayStorageStatistics::query_not_stored) {
      r->not_stored = (num_present == 0);
    }
    if (options.mask & ArrayStorageStatistics::query_fully_stored) {
      r->fully_stored = num_present == total_chunks;
    }
  }
};

template <typename T>
struct MovableAtomic : public std::atomic<T> {
  using std::atomic<T>::atomic;
  MovableAtomic(MovableAtomic&& other) noexcept {
    this->store(other.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
  }
};

struct ListReceiver {
  internal::IntrusivePtr<GetStorageStatisticsAsyncOperationState> state;
  Box<> grid_bounds;
  MovableAtomic<int64_t> total_chunks_seen{0};
  FutureCallbackRegistration cancel_registration;

  template <typename Cancel>
  void set_starting(Cancel cancel) {
    cancel_registration =
        state->promise.ExecuteWhenNotNeeded(std::move(cancel));
  }

  void set_stopping() { cancel_registration.Unregister(); }

  void set_done() {
    if (grid_bounds.num_elements() !=
        total_chunks_seen.load(std::memory_order_relaxed)) {
      state->ChunkMissing();
    }
  }

  void set_value(std::string key) {
    DimensionIndex base_dim = state->rank - grid_bounds.rank();
    DimensionIndex i = 0;
    assert(grid_bounds.rank() > 0);
    for (std::string_view part : absl::StrSplit(
             key,
             absl::MaxSplits(state->dimension_separator, grid_bounds.rank()))) {
      Index num;
      if (!state->key_formatter->ParseGridIndex(part, base_dim + i, num)) {
        return;
      }
      if (!Contains(grid_bounds[i], num)) return;
      ++i;
    }
    ++total_chunks_seen;
    state->IncrementChunksPresent();
  }

  void set_error(const absl::Status& error) { state->promise.SetResult(error); }
};

}  // namespace

Future<ArrayStorageStatistics>
GetStorageStatisticsForRegularGridWithSemiLexicographicalKeys(
    const KvStore& kvs, IndexTransformView<> transform,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> chunk_shape, BoxView<> grid_bounds,
    char dimension_separator,
    std::unique_ptr<const LexicographicalGridIndexKeyParser> key_formatter,
    absl::Time staleness_bound, GetArrayStorageStatisticsOptions options) {
  // TODO(jbms): integrate this with the chunk cache

  // Initialize the contained `Result` with default-constructed
  // `ArrayStorageStatistics`.  The destructor of
  // `GetStorageStatisticsAsyncOperationState` checks if it has been replaced
  // with an error status, before setting the actual statistics.
  auto [promise, future] =
      PromiseFuturePair<ArrayStorageStatistics>::Make(std::in_place);
  auto state =
      internal::MakeIntrusivePtr<GetStorageStatisticsAsyncOperationState>();
  state->promise = std::move(promise);
  state->options = options;
  auto& key_formatter_ref = *key_formatter;
  state->key_formatter = std::move(key_formatter);
  state->dimension_separator = dimension_separator;
  state->rank = grid_output_dimensions.size();

  // This function calls
  // `GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys` to compute the
  // set of individual chunk keys and chunk key ranges that correspond to the
  // range of `transform`.  The keys and key ranges are computed, and read and
  // list operations are issued, all before this function returns.  The
  // `GetStorageStatisticsAsyncOperationState` object handles the asynchronous
  // completion of the read and list operations.

  const auto handle_key = [&](std::string key) {
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_GRID_STORAGE_STATISTICS_DEBUG)
        << "key: " << tensorstore::QuoteString(key);
    if (internal::AddOverflow<Index>(state->total_chunks, 1,
                                     &state->total_chunks)) {
      return absl::OutOfRangeError(
          "Integer overflow computing number of chunks");
    }
    kvstore::ReadOptions read_options;
    read_options.byte_range = OptionalByteRangeRequest(0, 0);
    read_options.staleness_bound = staleness_bound;
    LinkValue(
        [state](Promise<ArrayStorageStatistics> promise,
                ReadyFuture<kvstore::ReadResult> future) {
          auto& read_result = future.value();
          if (!read_result.has_value()) {
            state->ChunkMissing();
          } else {
            state->IncrementChunksPresent();
          }
        },
        state->promise,
        kvstore::Read(kvs, std::move(key), std::move(read_options)));
    return absl::OkStatus();
  };

  const auto handle_key_range = [&](KeyRange key_range, size_t prefix,
                                    BoxView<> grid_bounds) -> absl::Status {
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_GRID_STORAGE_STATISTICS_DEBUG)
        << "key_range: " << key_range << ", prefix=" << prefix
        << ", grid_bounds=" << grid_bounds;
    Index total_chunks = grid_bounds.num_elements();
    if (total_chunks == 1) {
      // Convert to single-key
      std::string key = std::move(key_range.inclusive_min);
      key.resize(prefix);
      DimensionIndex base_dim =
          grid_output_dimensions.size() - grid_bounds.rank();
      for (DimensionIndex i = 0; i < grid_bounds.rank(); ++i) {
        if (i != 0) {
          key += dimension_separator;
        }
        key_formatter_ref.FormatGridIndex(key, base_dim + i,
                                          grid_bounds[i].inclusive_min());
      }
      return handle_key(std::move(key));
    }
    if (total_chunks == std::numeric_limits<Index>::max()) {
      return absl::OutOfRangeError(tensorstore::StrCat(
          "Integer overflow computing number of chunks in ", grid_bounds));
    }
    if (internal::AddOverflow(state->total_chunks, total_chunks,
                              &state->total_chunks)) {
      return absl::OutOfRangeError(
          "Integer overflow computing number of chunks");
    }
    kvstore::ListOptions list_options;
    list_options.staleness_bound = staleness_bound;
    list_options.range = std::move(key_range);
    list_options.strip_prefix_length = prefix;
    kvstore::List(kvs, std::move(list_options),
                  ListReceiver{state, Box<>(grid_bounds)});
    return absl::OkStatus();
  };

  TENSORSTORE_RETURN_IF_ERROR(
      internal::GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys(
          transform, grid_output_dimensions, chunk_shape, grid_bounds,
          dimension_separator, key_formatter_ref, handle_key,
          handle_key_range));
  return std::move(future);
}

LexicographicalGridIndexKeyParser::~LexicographicalGridIndexKeyParser() =
    default;

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
      dimension_separator,
      std::make_unique<Base10LexicographicalGridIndexKeyParser>(),
      staleness_bound, std::move(options));
}

}  // namespace internal
}  // namespace tensorstore
