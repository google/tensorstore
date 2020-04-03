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

#ifndef TENSORSTORE_INTERNAL_ASYNC_STORAGE_BACKED_CACHE_IMPL_H_
#define TENSORSTORE_INTERNAL_ASYNC_STORAGE_BACKED_CACHE_IMPL_H_

// IWYU pragma: private, include "third_party/tensorstore/internal/async_storage_backed_cache.h"

#include "absl/time/time.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_async_cache {

/// Identifies a local in-memory write generation (in contrast to
/// StorageGeneration, which identifies a persistent write generation).
using CacheGeneration = std::uint64_t;

/// Per-entry data used by AsyncStorageBackedCache to keep track of reads and
/// writeback requests, and the in-progress read or writeback operation (if
/// there is one).
///
/// For a single entry, there is at most one read or writeback operation (but
/// not both) in progress at a given time.
///
/// An in-progress read operation is indicated by
/// `issued_read.promise.valid() == true` (while `mutex` is unlocked).  An
/// in-progress writeback operation is indicated implicitly by a non-zero value
/// of `requested_writeback_generation` and
/// `issued_read.promise.valid() == false` (while `mutex` is unlocked).
struct AsyncEntryData {
  /// Protects access to all other members.  Note that this does not protect
  /// access to any actual "data" stored by a derived Entry class; the derived
  /// class will define a separate lock for that purpose, in order to avoid
  /// blocking fast operations that depend only on the AsyncEntryData metadata
  /// from being blocked by long read/write operations on the actual derived
  /// Entry data.
  Mutex mutex;

  /// If valid, `ready_read`, `ready_read_generation`, and `ready_read_time`
  /// together specifies the most recent result for resolving read requests.
  /// This is set from the most recent of:
  ///
  /// 1. a successful or failed read;
  ///
  /// 2. a successful (but not failed) writeback, in which case it is always in
  ///    a success state;
  ///
  /// 3. a call to `AsyncStorageBackedCache::Entry::FinishWrite` with
  ///    `flags=kSupersedesRead`, in which case it is always in a success state.
  ReadyFuture<const void> ready_read;

  /// Most recent remote storage generation upon which any cached data stored by
  /// a derived class of `AsyncStorageBackedCache::Entry` is conditioned.  If
  /// equal to `StorageGeneration::Unknown()`, then the state is not conditioned
  /// on any read state and indicates that there have been no successful reads
  /// or writebacks since the entry was added to the cache, or
  /// `AsyncStorageBackedCache::Entry::FinishWrite` was called with
  /// `flags=kSupersedesRead`.
  TimestampedStorageGeneration ready_read_generation;

  /// Promise/future pair corresponding to an in-progress read request, which
  /// was initiated at the local time `issued_read_time`.
  ///
  /// If `issued_read.promise.valid()`, then there is an in-progress read
  /// request.  Additionally, if `issued_read.future.valid()`, then the result
  /// won't be ignored.  Otherwise, it will be ignored because `FinishWrite` was
  /// called with `flags=kSupersedesRead` while the read was in progress.
  ///
  /// \invariant `issued_read.promise.valid() >= issued_read.future.valid()`
  PromiseFuturePair<void> issued_read;

  /// Only used if `issued_read.promise.valid()`.
  absl::Time issued_read_time;

  /// Promise corresponding to a queued read that was requested with a
  /// `queued_read_time` newer than `issued_read_time` while a read was in
  /// progress.  Once the in-progress read completes, another read may be issued
  /// to satisfy this request.
  ///
  /// \invariant `!queued_read.valid() || issued_read.promise.valid()`.
  Promise<void> queued_read;

  /// Staleness bound for next read.
  absl::Time queued_read_time = absl::InfinitePast();

  /// Local modification generation.  Initialized to 0, and incremented each
  /// time `AsyncStorageBackedCache::Entry::FinishWrite` is called.  Once a
  /// writeback completes (successfully or unsuccessfully), this is reset to 0.
  ///
  /// \invariant `write_generation == 0` if, and only if, the `EntryState` is
  ///     `clean_and_in_use` or `clean_and_not_in_use` (or will be set to one of
  ///     those states atomically with the release of `mutex`).
  CacheGeneration write_generation = 0;

  /// Latest value of `write_generation` for which writeback has been requested
  /// (e.g. by calling `Future::Force` on a Future returned from `FinishWrite`).
  ///
  /// A non-zero value indicates that there is an outstanding writeback request,
  /// and a writeback operation will be started as soon as any necessary read
  /// operations (or prior in-progress writeback operations) complete.
  ///
  /// \invariant `requested_writeback_generation <= write_generation`.
  CacheGeneration requested_writeback_generation = 0;

  /// Promises corresponding to `Future` objects returned from calls to
  /// `FinishWrite` that correspond to pending writebacks that have not yet
  /// completed (successfully or unsuccessfully), and may not have even been
  /// requested yet.
  ///
  /// While there may be arbitrarily many calls to `FinishWrite` prior to a
  /// writeback completing, `FinishWrite` returns a `Future` corresponding to an
  /// existing `Promise` when possible, and therefore at most 2 `Promise`
  /// objects must be stored: `writebacks[0]` corresponds to writebacks that
  /// will necessarily be satisfied by the next writeback operation to complete
  /// (which may or may not already be in progress); this promise will be marked
  /// ready when the writeback operation completes.  `writebacks[1]` is used for
  /// writebacks corresponding to calls to `FinishWrite` after a writeback has
  /// already started, i.e. when `issued_writeback_generation` is non-zero.
  /// Unless the writeback is restarted (i.e. due to a generation mismatch),
  /// this promise won't be marked ready when the writeback completes.
  ///
  /// \invariant `writebacks[0].valid() >= writebacks[1].valid()`
  /// \invariant `requested_writeback_generation == 0 || writebacks[0].valid()`
  Promise<void> writebacks[2];

  /// If non-zero, specifies the latest value of `write_generation` reflected in
  /// the current in-progress writeback.
  ///
  /// This is set to the current value of `write_generation` when
  /// `AsyncStorageBackedCache::WritebackReceiver::NotifyStarted` is called.
  ///
  /// If at the time the writeback operation completes this is still equal to
  /// `write_generation`, it indicates that there were no intervening writes
  /// during the writeback operation and the entry can now be considered clean.
  ///
  /// \invariant `issued_writeback_generation <= write_generation`.
  CacheGeneration issued_writeback_generation = 0;

  /// Used by `DoRequestWriteback` to store the Future corresponding to a
  /// writeback request made due to memory pressure in the `CachePool`.
  ///
  /// The Future must be stored to ensures the writeback request is not
  /// considered cancelled.
  Future<const void> writeback_requested_by_cache;

  /// Last value of `write_generation` at which `FinishWrite` was called with
  /// `flags=kSupersedesRead`.  This is reset to 0 once all local modifications
  /// have been written back.
  CacheGeneration supersedes_read_generation = 0;

  /// Last value of `write_generation` at which `FinishWrite` was called with
  /// `flags=kUnconditionalWriteback` or `flags=kSupersedesRead`.  This is reset
  /// to 0 once all local modifications have been written back.
  CacheGeneration unconditional_writeback_generation = 0;
};

/// Friend class of `AsyncStorageBackedCache::Entry` used to access
/// `AsyncEntryData`.
class Access;

}  // namespace internal_async_cache
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ASYNC_STORAGE_BACKED_CACHE_IMPL_H_
