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
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/masked_array.h"
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
  struct Component {
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
    Component(SharedArray<const void> fill_value);

    /// Constructs a component specification with the specified fill value and
    /// set of chunked dimensions.
    Component(SharedArray<const void> fill_value,
              std::vector<DimensionIndex> chunked_to_cell_dimensions);

    /// Returns the rank of this component array.
    DimensionIndex rank() const { return fill_value.rank(); }

    /// Returns the data type of this component array.
    DataType data_type() const { return fill_value.data_type(); }

    /// Shape of the array representing the data of a single grid cell of this
    /// component array.
    span<const Index> cell_shape() const { return fill_value.shape(); }

    /// The fill value of this component array.  Must be non-null.  This also
    /// specifies the shape of the grid cell.
    ///
    /// For chunked dimensions, the extent must match the corresponding extent
    /// in `chunk_shape`.  For unchunked dimensions, the extent specifies the
    /// full extent for that dimension of the component array.
    ///
    /// For each `chunked_to_cell_dimensions[i]`, it must be the case that
    /// `fill_value.shape()[chunked_to_cell_dimensions[i]] = chunk_shape[i]`,
    /// where `chunk_shape` is from the containing `ChunkGridSpecification`.
    SharedArray<const void> fill_value;

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
    struct Component {
      /// Constructs a component of the specified rank.
      ///
      /// Does not actually allocate the `data` or mask arrays.
      Component(DimensionIndex rank);

      /// Never actually invoked, but required by `InlinedVector::reserve`.
      Component(Component&& other) noexcept : Component(0) {
        TENSORSTORE_UNREACHABLE;
      }

      /// Optional pointer to C-order multi-dimensional array of data type and
      /// shape given by the `data_type` and `cell_shape`, respectively, of the
      /// `ChunkGridSpecification::Component`.  If equal to `nullptr`, no data
      /// has been read or written yet, or the contents may be equal to the fill
      /// value.
      std::shared_ptr<void> data;

      /// If `write_mask` is all `true` (`num_masked_elements` is equal to the
      /// total number of elements in the `data` array), `data == nullptr`
      /// represents the same state as `data` containing the fill value.
      MaskData write_mask;

      /// Value of `write_mask` prior to the current in-progress writeback
      /// operation.  If the writeback succeeds, this is discarded.  Otherwise,
      /// it is merged with `write_mask` when the writeback fails (e.g. due to a
      /// generation mismatch).
      MaskData write_mask_prior_to_writeback;

      /// Specifies whether the `data` array is valid at positions for which
      /// `write_mask` is `true`.
      ///
      /// If `true`, all positions in the `data` array are valid (because they
      /// have been filled either from a successful read or from the fill
      /// value).  If `false`, only positions in the `data` array for which
      /// `write_mask` is `true` are valid.
      std::atomic<bool> valid_outside_write_mask{false};
    };

    /// Overwrites all components with the fill value.
    /// \pre `data_mutex` is not locked by the current thread.
    Future<const void> Delete();

    /// Mutex that protects access to all fields of `components`.  A shared
    /// reader lock may be used when reading, while an exclusive lock must be
    /// used for writing.
    Mutex data_mutex;
    absl::InlinedVector<Component, 1> components;
  };

  /// Copyable shared handle used by `DoRead` representing an in-progress read
  /// request.
  struct ReadReceiver {
    Entry* entry() const { return static_cast<Entry*>(receiver_.entry()); }

    struct ComponentsWithGeneration {
      /// Specifies the data for all component arrays.  Must have a length equal
      /// to the number of component arrays, except to indicate that the read
      /// was aborted or the chunk data was not found.
      span<const ArrayView<const void>> components;

      /// Specifies the storage generation corresponding to `components`.
      ///
      /// To indicate that the read was aborted because the existing generation
      /// was already up to date, specify a `generation` of
      /// `StorageGeneration::Unknown()` and an empty `components` array.
      ///
      /// To indicate that the data for the chunk as not found, specify a
      /// `generation` not equal to `StorageGeneration::Unknown()` and an empty
      /// `components` array.
      TimestampedStorageGeneration generation;
    };

    /// Must be called exactly once by the implementation of `DoRead` to
    /// indicate that the read operation has completed (successfully or
    /// unsuccessfully).
    ///
    /// If `components` has an error state, it indicates a failed read that will
    /// result in any pending read operation that depended on this chunk
    /// failing.
    ///
    /// Otherwise, `components` should specify the successful read result.  Note
    /// that if the existing data was already up to date or the chunk data was
    /// not found, that is indicated by the `ComponentsWithGeneration` value,
    /// not by an error status.
    ///
    /// \param components The error result, or component data for the chunk.
    /// \param read_time The latest local machine time as of which the read
    ///     result is known to be up-to-date (typically the time immediately
    ///     before the actual read operation was issued).
    void NotifyDone(Result<ComponentsWithGeneration> components) const;

    // Treat as private
    AsyncStorageBackedCache::ReadReceiver receiver_;
  };

  /// Copyable shared handle used by `DoWriteback` representing an in-progress
  /// writeback request.
  struct WritebackReceiver {
    /// Must be called exactly once by the implementation of `DoWriteback` to
    /// indicate that the writeback operation has completed (successfully or
    /// unsuccessfully).
    ///
    /// Prior to calling `NotifyDone`, a `WritebackSnapshot` object must have
    /// been constructed and destroyed to indicate that writeback started.
    ///
    /// If the request generation does not match, `generation` should have the
    /// error `absl::StatusCode::kAborted`.
    ///
    /// \param generation If writeback is successful, specifies the new storage
    ///     generation (or `StorageGeneration::NoValue()` if the chunk was
    ///     deleted due to the component arrays all matching the fill value) and
    ///     the latest local machine time as of which this storage generation is
    ///     known to be up to date (typically the time immediately before the
    ///     actual write operation was issued).
    void NotifyDone(Result<TimestampedStorageGeneration> generation) const;

    /// Returns the entry associated with this writeback operation.
    Entry* entry() const { return static_cast<Entry*>(receiver_.entry()); }

    // Treat as private
    AsyncStorageBackedCache::WritebackReceiver receiver_;
  };

  /// RAII class used by implementations of `DoWriteback` to acquire a snapshot
  /// of the chunk data.
  class WritebackSnapshot {
   public:
    /// Acquires a (shared) read lock on the entry referenced by `receiver`, and
    /// obtains a snapshot of the chunk data.
    explicit WritebackSnapshot(const WritebackReceiver& receiver);
    WritebackSnapshot(const WritebackSnapshot&) = delete;

    /// Releases the read lock on the entry, and indicates that writeback has
    /// started.
    ~WritebackSnapshot();

    /// Returns the snapshot of the component arrays.
    ///
    /// The size of the returned `span` is equal to the number of component
    /// arrays.
    span<const ArrayView<const void>> component_arrays() const {
      return component_arrays_;
    }

    /// If `true`, all components are equal to the fill value.  If supported by
    /// the driver, writeback can be accomplished by deleting the chunk.
    bool equals_fill_value() const { return equals_fill_value_; }

   private:
    absl::InlinedVector<ArrayView<const void>, 1> component_arrays_;
    absl::InlinedVector<Index, kNumInlinedDims> byte_strides_;
    bool equals_fill_value_;
    const WritebackReceiver& receiver_;
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

  /// Requests initial or updated data from persistent storage for a single grid
  /// cell.
  ///
  /// This is called automatically by the `ChunkCache` implementation either due
  /// to a call to `Read` that cannot be satisfied by the existing cached data,
  /// or due to a requested writeback that requires the existing data.
  ///
  /// Derived classes must implement this method, and implementations must call
  /// methods on `receiver` as specified by the `ReadReceiver` documentation.
  virtual void DoRead(ReadOptions options, ReadReceiver receiver) = 0;

  /// Requests that local modifications recorded for a single grid cell be
  /// written back to persistent storage.
  ///
  /// This is called automatically by the `ChunkCache` implementation when a
  /// writeback is forced, either due to a call to `Force` on a `Future`
  /// returned from `Write`, or due to memory pressure in the containing
  /// `CachePool`.
  ///
  /// Derived classes must implement this method, and implementations must call
  /// methods on `receiver` as specified by the `WritebackReceiver`
  /// documentation.
  virtual void DoWriteback(TimestampedStorageGeneration existing_generation,
                           WritebackReceiver receiver) = 0;

  /// Simply wraps `receiver` in a `ChunkCache::ReadReceiver` and calls `DoRead`
  /// defined above.
  void DoRead(ReadOptions options,
              AsyncStorageBackedCache::ReadReceiver receiver) final;

  /// Simply wraps `receiver` in a `ChunkCache::WritebackReceiver` and calls
  /// `DoWriteback` defined above.
  void DoWriteback(TimestampedStorageGeneration existing_generation,
                   AsyncStorageBackedCache::WritebackReceiver receiver) final;

  void DoDeleteEntry(Cache::Entry* base_entry) final;
  Cache::Entry* DoAllocateEntry() final;
  std::size_t DoGetSizeInBytes(Cache::Entry* base_entry) final;

  PinnedCacheEntry<ChunkCache> GetEntryForCell(
      span<const Index> grid_cell_indices);

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
