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

#ifndef TENSORSTORE_INTERNAL_CHUNK_CACHE_H_
#define TENSORSTORE_INTERNAL_CHUNK_CACHE_H_

/// \file
/// Defines the abstract base class `ChunkCache`, which extends
/// `AsyncStorageBackedCache` for the specific case of representing
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

#include <atomic>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/async_storage_backed_cache.h"
#include "tensorstore/internal/async_write_array.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/sender.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// Specifies a common chunk grid that divides the component arrays.
///
/// Each component array has its own data type, and all of the component arrays
/// need not have the same number of dimensions.  Some subset of the dimensions
/// of each component array are of fixed size and not broken into chunks, and
/// are called "unchunked dimensions".  The remaining dimensions of each
/// component array are divided into fixed size chunks, and are called "chunked
/// dimensions".
///
/// While the unchunked dimensions of each component are independent, every
/// component must have the same number of chunked dimensions, and the chunked
/// dimensions of all components must correspond.  Specifically, there is a
/// bijection between the chunked dimensions of each component and a common list
/// of chunked dimensions shared by all components of the chunk grid, and the
/// chunk size must be the same.
///
/// The chunked dimensions of one component may, however, be in a different
/// order, and may be interleaved with unchunked dimensions differently, than
/// the chunked dimensions of another component.  The data for each component is
/// always stored within the cache contiguously in C order.  The dimensions of a
/// given component may be permuted to control the effective layout order of
/// each component independently.  Multiple component arrays are not interleaved
/// in memory, i.e. "columnar" storage is used, even if they are interleaved in
/// the underlying persistent storage format.
///
/// It is expected that the most common use will be with just a single
/// component.
///
/// A "cell" corresponds to a vector of integer chunk coordinates, of dimension
/// equal to the number of chunked dimensions.
///
/// For example, to specify a chunk cache with a common 3-d chunk shape of
/// `[25, 50, 30]` and two component arrays, one of data type uint16 with an
/// additional unchunked dimension of extent 2, and one of data type float32
/// with additional unchunked dimensions of extents 3 and 4, the following
/// `GridChunkSpecification` could be used:
///
///     components[0]:
///       data_type: uint16
///       shape: [25, 50, 30, 2]
///       chunked_to_cell_dimensions: [0, 1, 2]
///
///     components[1]:
///       data_type: float32
///       shape: [3, 30, 50, 25, 4]
///       chunked_to_cell_dimensions: [3, 2, 1]
///
///     chunk_shape: [25, 50, 30]
struct ChunkGridSpecification {
  DimensionIndex rank() const {
    return static_cast<DimensionIndex>(chunk_shape.size());
  }

  /// Specification of the data type, unchunked dimensions, and fill value of a
  /// single component array.
  ///
  /// The fill value specifies the default value to use when there is no
  /// existing data for a chunk.  When reading, if a missing chunk is
  /// encountered, the read is satisfied using the fill value.  When writing
  /// back a partially-written chunk for which there is no existing data, the
  /// fill value is substituted at unwritten positions.
  ///
  /// For chunked dimensions, the extent in `fill_value.shape()` must match the
  /// corresponding extent in `chunk_shape`.  For unchunked dimensions, the
  /// extent in `fill_value.shape()` specifies the full extent for that
  /// dimension of the component array.
  ///
  /// For each `chunked_to_cell_dimensions[i]`, it must be the case that
  /// `fill_value.shape()[chunked_to_cell_dimensions[i]] = chunk_shape[i]`,
  /// where `chunk_shape` is from the containing `ChunkGridSpecification`.
  struct Component : public AsyncWriteArray::Spec {
    /// Construct a component specification from a fill value.
    ///
    /// The `chunked_to_cell_dimensions` map is set to an identity map over
    /// `[0, fill_value.rank())`, meaning all dimensions are chunked.
    ///
    /// There are no constraints on the memory layout of `fill_value`.  To more
    /// efficiently represent the `fill_value` if the same value is used for all
    /// positions within a given dimension, you can specify a byte stride of 0
    /// for that dimension.  In particular, if the same value is used for all
    /// positions in the cell, you can specify all zero byte strides.
    Component(SharedArray<const void> fill_value, Box<> component_bounds);

    /// Constructs a component specification with the specified fill value and
    /// set of chunked dimensions.
    Component(SharedArray<const void> fill_value, Box<> component_bounds,
              std::vector<DimensionIndex> chunked_to_cell_dimensions);

    /// Mapping from chunked dimensions (corresponding to components of
    /// `chunk_shape`) to cell dimensions (corresponding to dimensions of
    /// `fill_value`).
    std::vector<DimensionIndex> chunked_to_cell_dimensions;
  };

  using Components = absl::InlinedVector<Component, 1>;

  /// Constructs a grid specification with the specified components.
  ChunkGridSpecification(Components components_arg);

  /// The list of components.
  Components components;

  /// The dimensions that are chunked (must be common to all components).
  std::vector<Index> chunk_shape;

  /// Returns the number of chunked dimensions.
  DimensionIndex grid_rank() const { return chunk_shape.size(); }
};

/// Cache for chunked multi-dimensional arrays.
class ChunkCache : public AsyncStorageBackedCache {
 public:
  /// Extends `AsyncStorageBackedCache::Entry` with storage of the data for all
  /// component arrays corresponding to a single grid cell.
  ///
  /// Derived classes must not define a different nested `Entry` type.
  class Entry : public AsyncStorageBackedCache::Entry {
   public:
    using Cache = ChunkCache;

    /// Returns the grid cell index vector corresponding to this cache entry.
    span<const Index> cell_indices() {
      return {reinterpret_cast<const Index*>(key().data()),
              static_cast<std::ptrdiff_t>(key().size() / sizeof(Index))};
    }

    /// Stores the data for a single component array.
    struct Component : public internal::AsyncWriteArray {
      using internal::AsyncWriteArray::AsyncWriteArray;
    };

    /// Overwrites all components with the fill value.
    /// \pre `data_mutex` is not locked by the current thread.
    Future<const void> Delete();

    absl::InlinedVector<Component, 1> components;
  };

  /// RAII class used by implementations of `DoWriteback` to acquire a snapshot
  /// of the chunk data.
  class WritebackSnapshot {
   public:
    /// Acquires a (shared) read lock on the entry referenced by `receiver`, and
    /// obtains a snapshot of the chunk data.
    explicit WritebackSnapshot(Entry* entry);

    /// Returns the snapshot of the component arrays.
    ///
    /// The size of the returned `span` is equal to the number of component
    /// arrays.
    span<const SharedArrayView<const void>> component_arrays() const {
      return component_arrays_;
    }

    /// If `true`, all components are equal to the fill value.  If supported by
    /// the driver, writeback can be accomplished by deleting the chunk.
    bool equals_fill_value() const { return equals_fill_value_; }

    /// If `true`, the snapshot does not depend on any prior read (because all
    /// positions were overwritten), and should therefore be written back
    /// unconditionally.  If `false`, writeback should be conditioned on the
    /// prior read generation.
    bool unconditional() const { return unconditional_; }

   private:
    absl::InlinedVector<SharedArrayView<const void>, 1> component_arrays_;
    bool equals_fill_value_;
    bool unconditional_;
  };

  /// Constructs a chunk cache with the specified grid.
  explicit ChunkCache(ChunkGridSpecification grid);

  /// Returns the grid specification.
  const ChunkGridSpecification& grid() const { return grid_; }

  /// Implements the behavior of `Driver::Read` for a given component array.
  ///
  /// Each chunk sent to `receiver` corresponds to a single grid cell.
  ///
  /// \param component_index Component array index in the range
  ///     `[0, grid().components.size())`.
  void Read(std::size_t component_index, IndexTransform<> transform,
            StalenessBound staleness,
            AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver);

  /// Implements the behavior of `Driver::Write` for a given component array.
  ///
  /// Each chunk sent to `receiver` corresponds to a single grid cell.
  ///
  /// \param component_index Component array index in the range
  ///     `[0, grid().components.size())`.
  void Write(std::size_t component_index, IndexTransform<> transform,
             AnyFlowReceiver<Status, WriteChunk, IndexTransform<>> receiver);

  void DoInitializeEntry(Cache::Entry* entry) override;
  std::size_t DoGetReadStateSizeInBytes(Cache::Entry* base_entry) override;
  std::size_t DoGetWriteStateSizeInBytes(Cache::Entry* base_entry) override;

  PinnedCacheEntry<ChunkCache> GetEntryForCell(
      span<const Index> grid_cell_indices);

  using AsyncStorageBackedCache::NotifyReadSuccess;
  void NotifyReadSuccess(Cache::Entry* entry, ReadStateWriterLock lock,
                         span<const SharedArrayView<const void>> components);

  void NotifyWritebackSuccess(Cache::Entry* entry,
                              WriteAndReadStateLock lock) override;

  void NotifyWritebackNeedsRead(Cache::Entry* entry, WriteStateLock lock,
                                absl::Time staleness_bound) override;

  void NotifyWritebackError(Cache::Entry* entry, WriteStateLock lock,
                            Status error) override;

 private:
  ChunkGridSpecification grid_;
};

/// Base class that partially implements the TensorStore `Driver` interface
/// based on `ChunkCache`.
///
/// Driver implementations such as ZarrDriver define a `DerivedChunkCache` class
/// that inherits from `ChunkCache` and implements `DoRead` and `DoWriteback`,
/// and also define a `DerivedDriver` class that inherits from
/// `ChunkCacheDriver` and implements the remaining abstract methods of the
/// `Driver` interface.  The `DerivedDriver` class always holds a pointer to a
/// `DerivedChunkCache`.
///
/// A single `ChunkCache` may correspond to multiple component arrays, while a
/// TensorStore `Driver` corresponds to a single array.  Therefore, a
/// `component_index` is stored along with a pointer to the `ChunkCache`.
///
/// This class is simply a thin wrapper around a `ChunkCache` pointer and a
/// `component_index` that provides the core `Read` and `Write` operations of
/// the `Driver` interface.
class ChunkCacheDriver : public Driver {
 public:
  /// Constructs a chunk cache driver for a given component.
  ///
  /// \dchecks `cache != nullptr`
  /// \dchecks `component_index < cache->grid().components.size()`
  explicit ChunkCacheDriver(CachePtr<ChunkCache> cache,
                            std::size_t component_index,
                            StalenessBound data_staleness_bound = {});

  /// Returns `cache->grid().components[component_index()].data_type()`.
  DataType data_type() override;

  DimensionIndex rank() override;

  /// Simply forwards to `ChunkCache::Read`.
  void Read(
      IndexTransform<> transform,
      AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver) override;

  /// Simply forwards to `ChunkCache::Write`.
  void Write(
      IndexTransform<> transform,
      AnyFlowReceiver<Status, WriteChunk, IndexTransform<>> receiver) override;

  std::size_t component_index() const { return component_index_; }

  ChunkCache* cache() const { return cache_.get(); }

  const StalenessBound& data_staleness_bound() const {
    return data_staleness_bound_;
  }

  ~ChunkCacheDriver() override;

 private:
  CachePtr<ChunkCache> cache_;
  std::size_t component_index_;
  StalenessBound data_staleness_bound_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CHUNK_CACHE_H_
