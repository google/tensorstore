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

#ifndef TENSORSTORE_INTERNAL_ASYNC_STORAGE_BACKED_CACHE_H_
#define TENSORSTORE_INTERNAL_ASYNC_STORAGE_BACKED_CACHE_H_

/// \file
/// Defines the abstract `AsyncStorageBackedCache` base class that extends the
/// basic `Cache` class with asynchronous read and read-modify-write
/// functionality.

#include "tensorstore/internal/async_storage_backed_cache_impl.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

/// Abstract base class that extends `Cache` with asynchronous read and
/// read-modify-write functionality based on optimistic concurrency using
/// `StorageGeneration`.
///
/// Derived cache classes inherit from this class and define a nested `Entry`
/// class that extends `AsyncStorageBackedCache::Entry` with additional members
/// storing the actual entry data to be contained in the cache.
///
/// Each derived cache Entry is assumed to correspond to some persistent object
/// (i.e. a particular hyperrectangular region of a multi-dimensional array, or
/// the metadata for an array), which has an associated `StorageGeneration`,
/// with the cache entry containing:
///
/// 1. optionally, some representation of the persistent object as of a
///    particular StorageGeneration and local timestamp;
///
/// 2. optionally, a representation of a partial or complete modification to the
///    persistent object, such that it can be applied to any version of the
///    persistent object.  For example, for an entry representing an array
///    region, the modification may be represented as the new value of the array
///    region along with a mask indicating which elements of the region have
///    been modified.
///
/// Typically the "persistent object" is stored as a single key in a
/// `KeyValueStore`, although `AsyncStorageBackedCache` does not directly
/// interface with `KeyValueStore`, but that may be done by a derived class.
///
/// Derived classes must define the virtual `DoRead` and `DoWriteback` methods,
/// described below.  Derived classes should also provide an interface for
/// recording local modifications to the persistent object, which should take
/// care of acquiring any necessary locks, record the modifications, and then
/// call the `Entry::FinishWrite` method to obtain a `Future` that may be used
/// to force writeback and receive notification of when writeback completes.
///
/// For each entry, the `AsyncStorageBackedCache` implementation keeps track of
/// read and writeback requests, and issues read and writeback operations as
/// necessary to satisfy them.  For a single entry at any given time there is at
/// most one in-progress read or writeback operation (a read and writeback
/// operation for a given entry cannot be in progress at the same time).
///
/// `AsyncStorageBackedCache` is extended by `ChunkCache` to provide caching of
/// chunked array data.
///
class AsyncStorageBackedCache : public Cache {
 public:
  enum class WriteFlags {
    /// Normal write.  Writeback is assumed to depend on the existing value.
    kConditionalWriteback = 0,

    /// Writeback does not require a prior read result.
    kUnconditionalWriteback = 1,

    /// Read requests can be satisfied from the cache alone.  Any in-progress
    /// read from the underlying storage will be cancelled.  This implies
    /// `kUnconditionalWriteback`.
    kSupersedesRead = 2,
  };

  /// Base Entry class.  Derived classes must define a nested `Entry` class that
  /// extends this `Entry` class.
  class Entry : public Cache::Entry {
   public:
    using Cache = AsyncStorageBackedCache;

    /// Requests data no older than `staleness_bound`.
    ///
    /// \returns A future that resolves to a success state once data no older
    ///     than `staleness_bound` is available, or to an error state if the
    ///     request failed.
    Future<const void> Read(StalenessBound staleness_bound);

    /// Must be called to indicate that the derived entry has been updated to
    /// reflect a local modification.
    ///
    /// \param size_update Optional size update.  The `size_update.lock` field
    ///     should specify a lock that protects the portion of the derived entry
    ///     state that was updated by the write.
    /// \param flags Specifies flags that control the meaning of the write.
    Future<const void> FinishWrite(SizeUpdate size_update, WriteFlags flags);

    /// Returns a Future that becomes ready when writeback of all prior writes
    /// has completed (successfully or unsuccessfully).
    Future<const void> GetWritebackFuture();

   private:
    internal_async_cache::AsyncEntryData entry_data_;
    friend class internal_async_cache::Access;
  };

  /// Copyable shared handle used by `DoRead` representing an in-progress read
  /// request.
  class ReadReceiver {
   public:
    /// Must be called to indicate that the read attempt has finished
    /// (successfully or with an error).
    ///
    /// To indicate that the read was aborted because the existing generation
    /// was already up to date, specify a `generation` of
    /// `StorageGeneration::Unknown()`.
    void NotifyDone(CacheEntry::SizeUpdate size_update,
                    Result<TimestampedStorageGeneration> generation) const;

    /// Returns the entry associated with this read request.
    Entry* entry() const { return entry_.get(); }

    // Treat as private.
    PinnedCacheEntry<AsyncStorageBackedCache> entry_;
  };

  /// Copyable shared handle used by `DoWriteback` representing an in-progress
  /// writeback request.
  class WritebackReceiver {
   public:
    /// Must be called at least once prior to completing the writeback request.
    ///
    /// Indicates that all current local modifications will be included in the
    /// writeback.
    void NotifyStarted(CacheEntry::SizeUpdate size_update) const;

    /// Must be called to indicate that the writeback attempt has finished
    /// (successfully or with an error).
    ///
    /// If writeback succeeded, specify a non-error `generation` value.
    ///
    /// To indicate the writeback failed due to a generation mismatch, or that
    /// an updated read result is required despite a prior call to `FinishWrite`
    /// with `kUnconditionalWriteback`, specify a `generation` of
    /// `StorageGeneration::Unknown()`.  That will cause another read to be
    /// issued, followed by another writeback.
    ///
    /// To indicate that the writeback was unnecessary, specify a generation of
    /// `absl::StatusCode::kAborted`.
    void NotifyDone(CacheEntry::SizeUpdate size_update,
                    Result<TimestampedStorageGeneration> generation) const;

    /// Returns the entry associated with this writeback request.
    Entry* entry() const { return entry_.get(); }

    // Treat as private.
    PinnedCacheEntry<AsyncStorageBackedCache> entry_;
  };

  struct ReadOptions {
    StorageGeneration existing_generation;
    StalenessBound staleness_bound;
  };

  /// Requests initial or updated data from persistent storage for a single
  /// `Entry`.
  ///
  /// This is called automatically by the `AsyncStorageBackedCache`
  /// implementation either due to a call to `Read` that cannot be satisfied by
  /// the existing cached data, or due to a requested writeback that requires
  /// the existing data.
  ///
  /// Derived classes must implement this method, and implementations must call
  /// methods on `receiver` as specified by the `ReadReceiver` documentation.
  virtual void DoRead(ReadOptions options, ReadReceiver receiver) = 0;

  /// Requests that local modifications recorded in a single `Entry` be written
  /// back to persistent storage.
  ///
  /// This is called automatically by the `AsyncStorageBackedCache`
  /// implementation when a writeback is forced, either due to a call to `Force`
  /// on a `Future` returned from `Entry::FinishWrite`, or due to memory
  /// pressure in the containing `CachePool`.
  ///
  /// Derived classes must implement this method, and implementations must call
  /// methods on `receiver` as specified by the `WritebackReceiver`
  /// documentation.
  virtual void DoWriteback(TimestampedStorageGeneration existing_generation,
                           WritebackReceiver receiver) = 0;

  /// Handles writeback requests triggered by memory pressure in the containing
  /// `CachePool`.
  void DoRequestWriteback(PinnedCacheEntry<Cache> base_entry) final;

  /// Derived classes must override these methods from `Cache`, along with
  /// `DoGetSizeInBytes`.
  Cache::Entry* DoAllocateEntry() override = 0;
  void DoDeleteEntry(Cache::Entry* entry) override = 0;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ASYNC_STORAGE_BACKED_CACHE_H_
