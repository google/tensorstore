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

#ifndef TENSORSTORE_INTERNAL_CACHE_CHUNK_CACHE_H_
#define TENSORSTORE_INTERNAL_CACHE_CHUNK_CACHE_H_

/// \file
/// Defines the abstract base class `ChunkCache`, which extends
/// `AsyncCache` for the specific case of representing
/// multi-dimensional arrays where a subset of the dimensions are partitioned
/// according to a regular grid.  In this context, a regular grid means a grid
/// where all cells have the same size.
///
/// `ChunkCache` provides a framework for implementing TensorStore drivers for
/// array storage formats that divide the array into a regular grid of chunks,
/// where each chunk may be read/written independently.  Derived classes must
/// define how to read and writeback individual chunks; based on that, the
/// `ChunkCacheDriver` class provides the core `Read` and `Write` operations for
/// a TensorStore `Driver`.
///
/// In the simplest case, a single `ChunkCache` object corresponds to a single
/// chunked multi-dimensional array.  In general, though, a single `ChunkCache`
/// object may correspond to multiple related chunked multi-dimensional arrays,
/// called "component arrays".  All of the component arrays must be partitioned
/// by the same common chunk grid, and each grid cell corresponds to a single
/// cache entry that holds the data for all of the component arrays.  It is
/// assumed that for a given chunk, the data for all of the component arrays is
/// stored together and may be read/written atomically.  The component arrays
/// may, however, have different data types and even different ranks due to each
/// having additional unchunked dimensions.
///
/// Chunked dimensions of component arrays do not have any explicit lower or
/// upper bounds; enforcing any bounds on those dimensions is the responsibility
/// of higher level code.  Unchunked dimensions have a zero origin and have
/// explicit bounds specified by `ChunkGridSpecification::Component`.

#include <stddef.h>

#include <atomic>
#include <memory>
#include <string_view>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/async_write_array.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/chunk_grid_specification.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

/// Cache for chunked multi-dimensional arrays.
class ChunkCache : public AsyncCache {
 public:
  using ReadData = SharedArray<const void>;

  static SharedArrayView<const void> GetReadComponent(
      const ChunkCache::ReadData* components, size_t component_index) {
    if (!components) return {};
    return components[component_index];
  }

  /// Extends `AsyncCache::Entry` with storage of the data for all
  /// component arrays corresponding to a single grid cell.
  class Entry : public AsyncCache::Entry {
   public:
    using OwningCache = ChunkCache;

    /// Returns the grid cell index vector corresponding to this cache entry.
    span<const Index> cell_indices() {
      return {reinterpret_cast<const Index*>(key().data()),
              static_cast<std::ptrdiff_t>(key().size() / sizeof(Index))};
    }

    span<const ChunkGridSpecification::Component> component_specs() {
      return GetOwningCache(*this).grid().components;
    }

    Future<const void> Delete(internal::OpenTransactionPtr transaction);

    std::size_t ComputeReadDataSizeInBytes(const void* read_data) override;
  };

  class TransactionNode : public AsyncCache::TransactionNode {
   public:
    using OwningCache = ChunkCache;

    explicit TransactionNode(Entry& entry);

    using Component = AsyncWriteArray;

    span<Component> components() { return components_; }

    /// Overwrites all components with the fill value.
    absl::Status Delete();

    std::size_t ComputeWriteStateSizeInBytes() override;

    span<const ChunkGridSpecification::Component> component_specs() {
      return GetOwningCache(*this).grid().components;
    }

    bool IsUnconditional() const {
      return unconditional_.load(std::memory_order_relaxed);
    }
    void SetUnconditional() {
      unconditional_.store(true, std::memory_order_relaxed);
    }

    /// Called after this transaction node is modified.
    ///
    /// By default just returns `absl::OkStatus()`, but may be overridden by a
    /// derived class, e.g. to call `MarkAsTerminal()`.
    virtual absl::Status OnModified();

    void DoApply(ApplyOptions options, ApplyReceiver receiver) override;

    void InvalidateReadState() override;

   private:
    friend class ChunkCache;
    absl::InlinedVector<Component, 1> components_;
    std::atomic<bool> unconditional_{false};

   public:
    bool is_modified{false};
  };

  /// Acquires a snapshot of the chunk data for use by derived class
  /// `DoWriteback` implementations.
  class WritebackSnapshot {
   public:
    /// Obtains a snapshot of the write state, rebased on top of `read_state`.
    ///
    /// \param node The node to snapshot.
    /// \param read_state The read state on which to base the snapshot.  If
    ///     `node.IsUnconditional()`, `read_state` is ignored.  Otherwise, if
    ///     the write state is not already based on
    ///     `read_state.generation().generation`, unmasked positions will be
    ///     copied from `read_state`.  The value of
    ///     `read_state.generation().generation` is used to avoid duplicate work
    ///     in the case that the write state has already been updated for this
    ///     generation.  The special value of `StorageGeneration::Local()`
    ///     serves to indicate a temporary read state with no assigned
    ///     generation.  If specified, the write state will always be rebased
    ///     again on top of `read_state`.
    explicit WritebackSnapshot(TransactionNode& node,
                               AsyncCache::ReadView<ReadData> read_state);

    /// If `true`, all components are equal to the fill value.  If supported by
    /// the driver, writeback can be accomplished by deleting the chunk.
    bool equals_fill_value() const { return !new_read_data_; }

    const std::shared_ptr<ReadData>& new_read_data() const {
      return new_read_data_;
    }
    std::shared_ptr<ReadData>& new_read_data() { return new_read_data_; }

   private:
    std::shared_ptr<ReadData> new_read_data_;
  };

  /// Returns the grid specification.
  virtual const ChunkGridSpecification& grid() const = 0;

  /// Returns the data copy executor.
  virtual const Executor& executor() const = 0;

  /// Implements the behavior of `Driver::Read` for a given component array.
  ///
  /// Each chunk sent to `receiver` corresponds to a single grid cell.
  ///
  /// \param transaction If not null, the read will reflect the uncommitted
  ///     modifications made in `transaction`.  Otherwise, the read will only
  ///     reflect the committed state.  (The read will never reflect uncommitted
  ///     non-transactional modifications.)
  /// \param component_index Component array index in the range
  ///     `[0, grid().components.size())`.
  /// \param transform The transform to apply.
  /// \param staleness Cached data older than `staleness` will not be returned
  ///     without being rechecked.
  /// \param receiver Receiver for the chunks.
  void Read(
      internal::OpenTransactionPtr transaction, std::size_t component_index,
      IndexTransform<> transform, absl::Time staleness,
      AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver);

  /// Implements the behavior of `Driver::Write` for a given component array.
  ///
  /// Each chunk sent to `receiver` corresponds to a single grid cell.
  ///
  /// \param transaction If not null, the modifications will be recorded for
  ///     `transaction`.  If null, fine-grained implicit transactions will be
  ///     used (typically one per chunk, not a single implicit transaction for
  ///     the entire write).
  /// \param component_index Component array index in the range
  ///     `[0, grid().components.size())`.
  /// \param transform The transform to apply.
  /// \param receiver Receiver for the chunks.
  void Write(
      internal::OpenTransactionPtr transaction, std::size_t component_index,
      IndexTransform<> transform,
      AnyFlowReceiver<absl::Status, WriteChunk, IndexTransform<>> receiver);

  Future<const void> DeleteCell(span<const Index> grid_cell_indices,
                                internal::OpenTransactionPtr transaction);
};

class ConcreteChunkCache : public ChunkCache {
 public:
  explicit ConcreteChunkCache(ChunkGridSpecification grid, Executor executor)
      : grid_(std::move(grid)), executor_(std::move(executor)) {}

  const ChunkGridSpecification& grid() const override { return grid_; }
  const Executor& executor() const override { return executor_; }

 private:
  internal::ChunkGridSpecification grid_;
  Executor executor_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_CHUNK_CACHE_H_
