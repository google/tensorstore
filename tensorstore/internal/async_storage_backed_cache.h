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

#include "absl/base/thread_annotations.h"
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
/// read-modify-write functionality based on optimistic concurrency.
///
/// Derived cache classes inherit from this class and define a nested `Entry`
/// class that extends `AsyncStorageBackedCache::Entry` with additional members
/// storing the actual entry data to be contained in the cache.
///
/// Each derived cache Entry is assumed to correspond to some persistent object
/// (i.e. a particular hyperrectangular region of a multi-dimensional array, or
/// the metadata for an array), with the cache entry containing:
///
/// 1. optionally, some representation of the persistent object as of a
///    particular local timestamp;
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
/// interface with `KeyValueStore`, but that may be done by a derived class (see
/// `KeyValueStoreCache`).
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
/// `ChunkCache` extends `AsyncStorageBackedCache` to provide caching of chunked
/// array data.
///
class AsyncStorageBackedCache : public Cache {
 public:
  enum class WriteFlags {
    /// Normal write.  Writeback is assumed to depend on the existing value.
    kConditionalWriteback = 0,

    /// Writeback does not require a prior read result.
    kUnconditionalWriteback = 1,
  };

  class Entry;

  /// Shared lock on the "read state" of an `Entry`.
  class ReadStateReaderLock {
   public:
    ReadStateReaderLock() = default;
    explicit ReadStateReaderLock(Entry* entry)
        ABSL_SHARED_LOCK_FUNCTION(entry->entry_data_.read_mutex)
        : entry_(entry) {
      entry->entry_data_.read_mutex.ReaderLock();
    }
    explicit ReadStateReaderLock(Entry* entry, std::adopt_lock_t)
        : entry_(entry) {}

    Entry* entry() const { return entry_.get(); }

    Entry* release() { return entry_.release(); }

    explicit operator bool() const { return static_cast<bool>(entry_); }

   private:
    struct Unlocker {
      void operator()(Entry* entry) const
          ABSL_UNLOCK_FUNCTION(&entry->entry_data_.read_mutex) {
        entry->entry_data_.read_mutex.ReaderUnlock();
      }
    };
    std::unique_ptr<Entry, Unlocker> entry_;
  };

  /// Exclusive lock on the "read state" of an `Entry`.
  class ReadStateWriterLock {
   public:
    ReadStateWriterLock() = default;
    explicit ReadStateWriterLock(Entry* entry)
        ABSL_EXCLUSIVE_LOCK_FUNCTION(entry->entry_data_.read_mutex)
        : entry_(entry) {
      entry->entry_data_.read_mutex.WriterLock();
    }
    explicit ReadStateWriterLock(Entry* entry, std::adopt_lock_t)
        : entry_(entry) {}

    Entry* entry() const { return entry_.get(); }

    Entry* release() { return entry_.release(); }

    explicit operator bool() const { return static_cast<bool>(entry_); }

   private:
    struct Unlocker {
      void operator()(Entry* entry) const
          ABSL_UNLOCK_FUNCTION(&entry->entry_data_.read_mutex) {
        entry->entry_data_.read_mutex.WriterUnlock();
      }
    };
    std::unique_ptr<Entry, Unlocker> entry_;
  };

  /// Exclusive lock on the "write state" and "read state" of an `Entry`.
  ///
  /// To avoid deadlock, the "write state" lock must always be acquired first.
  class WriteAndReadStateLock {
   public:
    WriteAndReadStateLock() = default;
    explicit WriteAndReadStateLock(Entry* entry)
        ABSL_EXCLUSIVE_LOCK_FUNCTION(entry->entry_data_.write_mutex)
            ABSL_EXCLUSIVE_LOCK_FUNCTION(entry->entry_data_.read_mutex)
        : entry_(entry) {
      entry->entry_data_.write_mutex.WriterLock();
      entry->entry_data_.read_mutex.WriterLock();
    }
    explicit WriteAndReadStateLock(Entry* entry, std::adopt_lock_t)
        : entry_(entry) {}

    Entry* entry() const { return entry_.get(); }

    Entry* release() { return entry_.release(); }

    explicit operator bool() const { return static_cast<bool>(entry_); }

   private:
    struct Unlocker {
      void operator()(Entry* entry) const
          ABSL_UNLOCK_FUNCTION(&entry->entry_data_.read_mutex)
              ABSL_UNLOCK_FUNCTION(&entry->entry_data_.write_mutex) {
        entry->entry_data_.read_mutex.WriterUnlock();
        entry->entry_data_.write_mutex.WriterUnlock();
      }
    };
    std::unique_ptr<Entry, Unlocker> entry_;
  };

  /// Exclusive lock on the "write state" of an `Entry`.
  class WriteStateLock {
   public:
    WriteStateLock() = default;
    explicit WriteStateLock(Entry* entry)
        ABSL_EXCLUSIVE_LOCK_FUNCTION(entry->entry_data_.write_mutex)
        : entry_(entry) {
      entry->entry_data_.write_mutex.WriterLock();
    }
    explicit WriteStateLock(Entry* entry, std::adopt_lock_t) : entry_(entry) {}

    Entry* entry() const { return entry_.get(); }

    Entry* release() { return entry_.release(); }

    /// Upgrades to an exclusive lock on the "write state" and "read state".
    WriteAndReadStateLock Upgrade() && ABSL_NO_THREAD_SAFETY_ANALYSIS {
      assert(entry_);
      entry_->entry_data_.read_mutex.WriterLock();
      return WriteAndReadStateLock(entry_.release(), std::adopt_lock);
    }

    explicit operator bool() const { return static_cast<bool>(entry_); }

   private:
    struct Unlocker {
      void operator()(Entry* entry) const
          ABSL_UNLOCK_FUNCTION(entry->entry_data_.write_mutex) {
        entry->entry_data_.write_mutex.WriterUnlock();
      }
    };
    std::unique_ptr<Entry, Unlocker> entry_;
  };

  /// Base Entry class.  Derived classes must define a nested `Entry` class that
  /// extends this `Entry` class.
  ///
  /// Data members of this class and derived `Entry` classes should be
  /// designated as belonging to one of three groups:
  ///
  /// 1. Read state: immutable snapshot of the last state read or successfully
  ///    written back.  "Read state" members must only be read by external code
  ///    while holding a `ReadStateReaderLock`.  Only the implementation of
  ///    `DoRead` and `DoWriteback` is permitted to modify "read state" members,
  ///    and must do so while holding a `ReadStateWriterLock`.  The `DoRead` and
  ///    `DoWriteback` implementation is permitted to read "read state" members
  ///    without a lock (provided that it does not concurrent modify them),
  ///    since there can be at most a single read or single writeback in
  ///    progress.
  ///
  /// 2. Write state: represents uncommitted modifications.  "Write state"
  ///    members must only be read or written while holding a `WriteStateLock`.
  ///
  /// 3. Read/writeback state: additional state used to track the current read
  ///    or writeback operation.  There is no associated lock, but in the
  ///    unlikely case that the read or writeback operation itself needs to
  ///    access this state concurrently, it will need to provide its own mutex.
  class Entry : public Cache::Entry {
   public:
    using Cache = AsyncStorageBackedCache;

    /// Requests data no older than `staleness_bound`.
    ///
    /// \returns A future that resolves to a success state once data no older
    ///     than `staleness_bound` is available, or to an error state if the
    ///     request failed.
    Future<const void> Read(absl::Time staleness_bound);

    /// Must be called to indicate that the derived entry has been updated to
    /// reflect a local modification.
    ///
    /// \param lock Valid lock on the "write state".
    /// \param flags Specifies flags that control the meaning of the write.
    Future<const void> FinishWrite(WriteStateLock lock, WriteFlags flags);

    /// Must be called if the "write state" size was modified but `FinishWrite`
    /// will not be called before releasing `lock`.
    void AbortWrite(WriteStateLock lock);

    /// Returns a Future that becomes ready when writeback of all prior writes
    /// has completed (successfully or unsuccessfully).
    Future<const void> GetWritebackFuture();

    ReadStateReaderLock AcquireReadStateReaderLock() {
      return ReadStateReaderLock(this);
    }

    ReadStateWriterLock AcquireReadStateWriterLock() {
      return ReadStateWriterLock(this);
    }

    WriteStateLock AcquireWriteStateLock() { return WriteStateLock(this); }

    WriteAndReadStateLock AcquireWriteAndReadStateLock() {
      return WriteAndReadStateLock(this);
    }

    /// Timestamp as of which current "read state" is known to be up to date.
    /// This is part of the "read state" and should be set by the `DoRead` and
    /// `DoWriteback` implementations prior to calling `NotifyReadSuccess` or
    /// `NotifyWritebackSuccess`.
    absl::Time last_read_time = absl::InfinitePast();

   private:
    friend class internal_async_cache::Access;
    friend class AsyncStorageBackedCache;
    internal_async_cache::AsyncEntryData entry_data_;
  };

  /// Derived classes should override this to return the size of the "read
  /// state".
  ///
  /// This method is always called with at least a shared lock on the "read
  /// state".
  virtual size_t DoGetReadStateSizeInBytes(Cache::Entry* entry);

  /// Derived classes should override this to return the combined size of the
  /// "write state" and any "writeback state".
  ///
  /// This method is always called with a lock on the "write state".
  virtual size_t DoGetWriteStateSizeInBytes(Cache::Entry* entry);

  /// Derived classes should override this to return the size of any additional
  /// heap allocations that are unaffected by changes to the read state, write
  /// state, or writeback state.
  ///
  /// Derived implementations should include in the returned sum the result of
  /// calling this base implementation.
  virtual size_t DoGetFixedSizeInBytes(Cache::Entry* entry);

  /// The total size in bytes is equal to
  ///
  ///     DoGetFixedSizeInBytes() +
  ///     DoGetReadStateSizeInBytes(entry) +
  ///     DoGetWriteStateSizeInBytes(entry)
  size_t DoGetSizeInBytes(Cache::Entry* entry) final;

  /// Requests initial or updated data from persistent storage for a single
  /// `Entry`.
  ///
  /// This is called automatically by the `AsyncStorageBackedCache`
  /// implementation either due to a call to `Read` that cannot be satisfied by
  /// the existing cached data, or due to a requested writeback that requires
  /// the existing data.
  ///
  /// Derived classes must implement this method, and implementations must call
  /// (either immediately or asynchronously) `NotifyReadSuccess` or
  /// `NotifyReadError` to signal completion.
  virtual void DoRead(PinnedEntry entry, absl::Time staleness_bound) = 0;

  /// Signals that the read request initiated by the most recent call to
  /// `DoRead` succeeded.
  ///
  /// Implementations of `DoRead` should first acquire a `ReadStateWriterLock`
  /// on `entry`, update `entry->last_read_time` and any other applicable "read
  /// state" members, then call this method.
  ///
  /// Derived classes may override this method, but must ensure this base class
  /// method is called.
  virtual void NotifyReadSuccess(Cache::Entry* entry, ReadStateWriterLock lock);

  /// Signals that the read request initiated by the most recent call to
  /// `DoRead` failed.
  ///
  /// The "read state" of `entry` should not have been modified.
  ///
  /// Derived classes may override this method, but must ensure this base class
  /// method is called.
  virtual void NotifyReadError(Cache::Entry* entry, absl::Status error);

  /// Requests that local modifications recorded in a single `Entry` be written
  /// back to persistent storage.
  ///
  /// This is called automatically by the `AsyncStorageBackedCache`
  /// implementation when a writeback is forced, either due to a call to `Force`
  /// on a `Future` returned from `Entry::FinishWrite`, or due to memory
  /// pressure in the containing `CachePool`.
  ///
  /// Derived classes must implement this method, which should synchronously or
  /// asynchronously perform the following steps:
  ///
  /// 1. Acquire a `WriteStateLock` on `entry`.
  ///
  /// 2. Move the modifications reflected in the "write state" into the
  ///    "writeback state" and call `NotifyWritebackStarted`, transferring
  ///    ownership of the `WriteStateLock`.
  ///
  /// 3. Submit the writeback of the modifications that were moved into the
  ///    "writeback state", possibly conditioned on the existing "read state"
  ///    being current.
  ///
  /// 4. If writeback succeeds:
  ///
  ///    4a. Acquire a `WriteAndReadStateLock` on `entry`.
  ///
  ///    4b. Update the "read state" based on the modifications that were copied
  ///        into the "writeback state" in step 2.
  ///
  ///    4c. Update `entry->last_read_time` to reflect the time just before the
  ///        writeback was submitted.
  ///
  ///    4d. Call `NotifyWritebackSuccess`, transferring ownership of the
  ///        `WriteAndReadStateLock`.
  ///
  /// 5. If writeback fails due to the existing "read state" being out of date:
  ///
  ///    5a. Acquire a `WriteStateLock` on `entry`.
  ///
  ///    5b. Rebase the "write state", which may now include additional
  ///        concurrent modifications since step 2, on top of the modifications
  ///        copied into the "writeback state" in step 2.
  ///
  ///    5c. Call `NotifyWritebackNeedsRead`, transferring ownership of the
  ///        `WriteStateLock` and providing a timestamp as of which the current
  ///        "read state" is known to be out of date.  This will result in a
  ///        call to `DoRead` to obtain an updated "read state" followed by
  ///        another call to `DoWriteback` to retry the writeback after the read
  ///        completes successfully.
  ///
  /// 6. If writeback fails for any other reason:
  ///
  ///    6a. Acquire a `WriteStateLock` on `entry`.
  ///
  ///    6b. Delete the modifications saved in the "writeback state".
  ///
  ///    6c. Call `NotifyWritebackError`, transferring ownership of the
  ///        `WriteStateLock`.  The modifications are lost.
  virtual void DoWriteback(PinnedEntry entry);

  /// Signals that the writeback request initiated by the most recent call to
  /// `DoWriteback` has taken a snapshot of the current modifications to
  /// `entry`.
  ///
  /// This must be called at least once, and may be called more than once
  /// (e.g. if writeback is retried), by the implementation of `DoWriteback`.
  ///
  /// All modifications to the "write state" of `entry` made prior to calling
  /// `NotifyWritebackStarted` are reflected in the writeback, while any
  /// modifications made after the last call to `NotifyWritebackStarted` are not
  /// reflected in the writeback.
  ///
  /// Derived classes may override this method, but must ensure this base class
  /// method is called.
  virtual void NotifyWritebackStarted(Cache::Entry* entry, WriteStateLock lock);

  /// Signals that the writeback request initiated by the most recent call to
  /// `DoWriteback` has completed successfully.
  ///
  /// Derived classes may override this method (e.g. to perform some or all of
  /// the work of step 4b or 4c), but must ensure this base class method is
  /// called.
  virtual void NotifyWritebackSuccess(Cache::Entry* entry,
                                      WriteAndReadStateLock lock);

  /// Signals that the writeback request initiated by the most recent call to
  /// `DoWriteback` failed.
  ///
  /// Derived classes may override this method (e.g. to perform some or all of
  /// the work of step 6b), but must ensure this base class method is called.
  virtual void NotifyWritebackError(Cache::Entry* entry, WriteStateLock lock,
                                    absl::Status error);

  /// Signals that the writeback request initiated by the most recent call to
  /// `DoWriteback` failed due to the "read state" being out of date.
  ///
  /// The `staleness_bound` specifies a time as of which the current "read
  /// state" is known to be out of date.
  ///
  /// Derived classes may override this method (e.g. to perform some or all of
  /// the work of step 5b above), but must ensure this base class method is
  /// called.
  virtual void NotifyWritebackNeedsRead(Cache::Entry* entry,
                                        WriteStateLock lock,
                                        absl::Time staleness_bound);

  /// Handles writeback requests triggered by memory pressure in the containing
  /// `CachePool`.
  void DoRequestWriteback(PinnedEntry base_entry) final;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ASYNC_STORAGE_BACKED_CACHE_H_
