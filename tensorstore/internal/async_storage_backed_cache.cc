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

#include "tensorstore/internal/async_storage_backed_cache.h"

#include <algorithm>
#include <cassert>
#include <mutex>  // NOLINT
#include <optional>
#include <system_error>  // NOLINT
#include <utility>

#include "absl/base/macros.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/utility/utility.h"
#include "tensorstore/internal/async_storage_backed_cache_impl.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace {
// Set to `true` to enable debug logging within this file.
constexpr static bool debug = false;
}  // namespace

namespace tensorstore {

namespace internal_async_cache {
class Access {
 public:
  static AsyncEntryData& entry_data(
      internal::AsyncStorageBackedCache::Entry& entry) {  // NOLINT
    return entry.entry_data_;
  }
};
}  // namespace internal_async_cache

namespace internal {

namespace {
using internal_async_cache::Access;
using internal_async_cache::AsyncEntryData;
using internal_async_cache::CacheGeneration;

/// Returns a `Future` corresponding to `*promise` if possible.  Otherwise,
/// resets `*promise` to a null `Promise` object and returns a null `Future`
/// object.
template <typename T>
Future<T> GetFutureOrInvalidate(Promise<T>* promise) {
  Future<T> future;
  if (promise->valid()) {
    future = promise->future();
    if (!future.valid()) *promise = Promise<T>();
  }
  return future;
}

/// Makes a new read request `Promise`/`Future` pair for `entry`, and starts a
/// read operation.
///
/// After recording the read request but before starting the read operation,
/// updates the entry state with `update`.
///
/// \pre There is no outstanding read request for `entry`.
/// \pre No read operation is in progress for `entry`.
Future<const void> IssueRead(AsyncStorageBackedCache::Entry* entry,
                             CacheEntry::StateUpdate update) {
  TENSORSTORE_DEBUG_LOG(debug, "IssueRead: ", entry);
  auto& ed = Access::entry_data(*entry);
  ABSL_ASSERT(!ed.queued_read.valid());
  ABSL_ASSERT(!ed.issued_read.future.valid());
  ABSL_ASSERT(!ed.issued_read.promise.valid());
  absl::Time staleness_bound = ed.issued_read_time = ed.queued_read_time;
  ed.issued_read = PromiseFuturePair<void>::Make();
  auto future = ed.issued_read.future;
  entry->UpdateState(std::move(update));
  GetOwningCache(entry)->DoRead(
      PinnedCacheEntry<AsyncStorageBackedCache>(entry), staleness_bound);
  return future;
}

/// Starts a previously-requested read or writeback operation.
///
/// This function is called when a read or writeback operation completes, or a
/// new writeback is requested.
///
/// If there is a queued read request that is already satisfied by the current
/// entry state, it is marked completed without actually issuing another read
/// operation.
///
/// \pre No read or writeback operation is in progress.
void MaybeStartReadOrWriteback(AsyncStorageBackedCache::Entry* entry,
                               CacheEntry::StateUpdate update) {
  auto& ed = Access::entry_data(*entry);
  // A read must not be in progress.
  assert(!ed.issued_read.promise.valid());

  // A writeback must not have started (i.e. `NotifyWritebackStarted` called).
  assert(ed.issued_writeback_generation == 0);

  // A writeback must not be in progress even if `NotifyWritebackStarted` has
  // not yet been called, but we cannot check for that condition.

  /// Starts a read if one has been requested.
  /// \returns `true` if a read has been started, in which case `update` has
  ///     been applied.
  auto maybe_start_read = [&]() -> bool {
    if (!ed.queued_read.valid()) {
      TENSORSTORE_DEBUG_LOG(debug, "MaybeStartReadOrWriteback: entry=", entry,
                            " no pending read request");
      // No pending read request.
      return false;
    }
    Promise<void> queued_read_promise = std::move(ed.queued_read);
    ed.issued_read.future = queued_read_promise.future();
    if (!ed.issued_read.future.valid()) {
      TENSORSTORE_DEBUG_LOG(debug, "MaybeStartReadOrWriteback: entry=", entry,
                            " pending read request was cancelled");
      // Pending read request was cancelled.
      return false;
    }

    // Read will be started.
    ed.issued_read.promise = std::move(queued_read_promise);
    ed.issued_read_time = ed.queued_read_time;

    absl::Time staleness_bound = ed.queued_read_time;
    entry->UpdateState(std::move(update));
    TENSORSTORE_DEBUG_LOG(debug, "MaybeStartReadOrWriteback: entry=", entry,
                          " calling DoRead");
    GetOwningCache(entry)->DoRead(
        PinnedCacheEntry<AsyncStorageBackedCache>(entry), staleness_bound);
    return true;
  };

  /// Starts a writeback if one has been requested and does not require a read
  /// first.
  /// \returns `true` if a writeback has been started, in which case `update`
  ///     has been applied.
  auto maybe_start_writeback = [&]() -> bool {
    if (!ed.requested_writeback_generation) {
      TENSORSTORE_DEBUG_LOG(debug, "MaybeStartReadOrWriteback: entry=", entry,
                            " no requested writeback");
      // No requested writeback.
      return false;
    }
    assert(ed.writebacks[0].valid());
    if (ed.writebacks[1].valid() && !ed.writebacks[1].result_needed()) {
      // The second queued writeback promise was cancelled, so we can reset it.
      TENSORSTORE_DEBUG_LOG(debug, "MaybeStartReadOrWriteback: entry=", entry,
                            " second queued writeback was cancelled");
      ed.writebacks[1] = Promise<void>();
    }
    if (!ed.writebacks[0].result_needed()) {
      // The first queued writeback promise was cancelled, so we can pop it from
      // the queue.
      TENSORSTORE_DEBUG_LOG(debug, "MaybeStartReadOrWriteback: entry=", entry,
                            " first queued writeback was cancelled");
      ed.writebacks[0] = std::move(ed.writebacks[1]);
      if (!ed.writebacks[0].valid()) {
        TENSORSTORE_DEBUG_LOG(debug, "MaybeStartReadOrWriteback: entry=", entry,
                              " all queued writebacks were cancelled");
        // All queued writebacks were cancelled.
        ed.requested_writeback_generation = 0;
        update.new_state = CacheEntryQueueState::dirty;
        return false;
      }
    }
    if (!ed.unconditional_writeback_generation &&
        (ed.writeback_needs_read ||
         entry->last_read_time == absl::InfinitePast())) {
      // Writeback can't be started yet because a read must complete first.
      TENSORSTORE_DEBUG_LOG(debug, "MaybeStartReadOrWriteback: entry=", entry,
                            " read required for writeback");
      if (!maybe_start_read()) {
        update.new_state = CacheEntryQueueState::writeback_requested;
        IssueRead(entry, std::move(update));
      }
      return true;
    }
    // Writeback will be started.
    update.new_state = CacheEntryQueueState::writeback_requested;
    entry->UpdateState(std::move(update));
    TENSORSTORE_DEBUG_LOG(debug, "MaybeStartReadOrWriteback: entry=", entry,
                          " calling DoWriteback");
    GetOwningCache(entry)->DoWriteback(
        PinnedCacheEntry<AsyncStorageBackedCache>(entry));
    return true;
  };

  // Starts a writeback or read as required.  Upon return, `update` has been
  // applied.
  auto update_state = [&] {
    if (!maybe_start_writeback() && !maybe_start_read()) {
      entry->UpdateState(std::move(update));
    }
  };

  if (ed.queued_read.valid() && ed.queued_read_time <= entry->last_read_time) {
    // There is a pending read request that is satisfied by the existing read
    // result.  Remove it from the entry state before calling `update_state` to
    // avoid it triggering a new read operation, then mark it as completed
    // (after applying `update`, which also releases `mutex`).
    Result<void> result = ed.ready_read.result();
    Promise<void> queued_read_promise = std::move(ed.queued_read);
    update_state();
    // Mark the read request completed after calling `update_state` (which
    // releases the lock held by `update`).
    queued_read_promise.SetResult(std::move(result));
    return;
  }

  // No already-satisfied pending read request.  Just call `update_state`.
  update_state();
}

/// Returns a new `Future` that will become ready when writeback completes
/// (successfully or unsuccessfully) for all previously-issued writes.
///
/// The returned `Future` corresponds to a pending writeback, but calling this
/// function does not cause writeback to actually start.  To request that
/// writeback actually start, `Force` must be called (directly or indirectly_ on
/// the returned `Future`.
///
/// \pre The entry `writebacks[0]` promise is null, cancelled, or
///     `issued_writeback_generation != write_generation`.
/// \pre The `writebacks[1]` promise is null or cancelled.
/// \pre The entry `mutex` is locked exclusively by the current thread (this
///     function does not release the lock).
Future<const void> GetNewWritebackFuture(
    AsyncStorageBackedCache::Entry* entry) {
  struct Callback {
    void operator()(Promise<void> promise) {
      auto& ed = Access::entry_data(*entry);
      std::unique_lock<Mutex> lock(ed.mutex);
      if (!HaveSameSharedState(ed.writebacks[0], promise) &&
          !HaveSameSharedState(ed.writebacks[1], promise)) {
        TENSORSTORE_DEBUG_LOG(
            debug, "WritebackForce Force: entry=", entry.get(),
            " stale promise, writeback must have just finished");
        // A writeback must have just finished, and we will be marked ready
        // soon.
        return;
      }
      if (absl::exchange(ed.requested_writeback_generation,
                         ed.write_generation)) {
        TENSORSTORE_DEBUG_LOG(debug,
                              "WritebackFuture Force: entry=", entry.get(),
                              " another writeback is in progress");
        // Another writeback operation is in progress.  The writeback we
        // requested will be issued automatically once that completes, if
        // necessary.
        return;
      }
      if (ed.issued_read.promise.valid()) {
        TENSORSTORE_DEBUG_LOG(debug,
                              "WritebackFuture Force: entry=", entry.get(),
                              " another read is in progress");
        entry->UpdateState(
            {/*.SizeUpdate=*/{std::move(lock)},
             /*.new_state=*/CacheEntryQueueState::writeback_requested});
        return;
      }
      TENSORSTORE_DEBUG_LOG(debug, "WritebackFuture Force: entry=", entry.get(),
                            " attempting to start writeback");
      MaybeStartReadOrWriteback(
          entry.get(),
          {/*.SizeUpdate=*/{std::move(lock)},
           /*.new_state=*/CacheEntryQueueState::writeback_requested});
    }
    PinnedCacheEntry<AsyncStorageBackedCache> entry;
  };
  auto pair = PromiseFuturePair<void>::Make();
  pair.promise.ExecuteWhenForced(
      Callback{PinnedCacheEntry<AsyncStorageBackedCache>(entry)});
  auto& ed = Access::entry_data(*entry);
  assert(!ed.writebacks[1].valid());
  (ed.writebacks[0].valid() ? ed.writebacks[1] : ed.writebacks[0]) =
      std::move(pair.promise);
  return std::move(pair.future);
}

/// Returns a `Future` that will become ready when writeback completes
/// (successfully or unsuccessfully) for all previously-issued writes.
///
/// This returns a `Future` corresponding to one of the `writebacks` `Promise`
/// objects if possible, or otherwise calls `GetNewWritebackFuture` to create a
/// new one.
///
/// \pre The entry `mutex` is locked exclusively by the current thread (this
///     function does not release the lock).
Future<const void> GetWritebackFutureWithLock(
    AsyncStorageBackedCache::Entry* entry) {
  auto& ed = Access::entry_data(*entry);
  assert(ed.writebacks[0].valid() == (ed.write_generation != 0));
  if (ed.write_generation == 0) return {};
  Future<const void> future = GetFutureOrInvalidate(&ed.writebacks[1]);
  if (!future.valid() &&
      (ed.issued_writeback_generation == 0 ||
       ed.issued_writeback_generation == ed.write_generation)) {
    future = ed.writebacks[0].future();
  }
  if (future.valid()) return future;
  return GetNewWritebackFuture(entry);
}

template <bool Read, bool Write, typename ExistingLock>
CacheEntry::SizeUpdate GetSizeUpdateImpl(AsyncStorageBackedCache::Entry* entry,
                                         ExistingLock existing_lock) {
  auto& ed = Access::entry_data(*entry);
  CacheEntry::SizeUpdate size_update{
      std::unique_lock<tensorstore::Mutex>(ed.mutex)};
  auto* cache = GetOwningCache(entry);
  if constexpr (Read) {
    ed.read_state_size = cache->DoGetReadStateSizeInBytes(entry);
  }
  if constexpr (Write) {
    ed.write_state_size = cache->DoGetWriteStateSizeInBytes(entry);
  }
  size_update.new_size = cache->DoGetFixedSizeInBytes(entry) +
                         ed.read_state_size + ed.write_state_size;
  return size_update;
}

/// Acquires a lock on `entry->entry_data_.mutex`, and computes an updated size,
/// before releasing `existing_lock`.
///
/// The cached `read_state_size` and/or `write_state_size` are updated depending
/// on the `lock` type.
CacheEntry::SizeUpdate GetSizeUpdate(
    AsyncStorageBackedCache::Entry* entry,
    AsyncStorageBackedCache::ReadStateWriterLock lock) {
  return GetSizeUpdateImpl</*Read=*/true, /*Write=*/false>(entry,
                                                           std::move(lock));
}

CacheEntry::SizeUpdate GetSizeUpdate(
    AsyncStorageBackedCache::Entry* entry,
    AsyncStorageBackedCache::WriteStateLock lock) {
  return GetSizeUpdateImpl</*Read=*/false, /*Write=*/true>(entry,
                                                           std::move(lock));
}

CacheEntry::SizeUpdate GetSizeUpdate(
    AsyncStorageBackedCache::Entry* entry,
    AsyncStorageBackedCache::WriteAndReadStateLock lock) {
  return GetSizeUpdateImpl</*Read=*/true, /*Write=*/true>(entry,
                                                          std::move(lock));
}

/// Marks `issued_read` ready with the specified `status` after calling
/// `MaybeStartReadOrWriteback`.
void ResolveIssuedRead(AsyncStorageBackedCache::Entry* entry,
                       CacheEntry::StateUpdate update, absl::Status status) {
  auto& ed = Access::entry_data(*entry);
  assert(ed.issued_read.promise.valid());
  // Remove `issued_read` from the entry state prior to calling
  // `MaybeStartReadOrWriteback` in order to indicate that a read operation is
  // no longer in progress.
  Promise<void> issued_read_promise = std::move(ed.issued_read.promise);
  ed.issued_read.future = Future<void>();
  MaybeStartReadOrWriteback(entry, std::move(update));
  issued_read_promise.SetResult(MakeResult(status));
}

void ResolveIssuedWriteback(AsyncStorageBackedCache::Entry* entry,
                            CacheEntry::SizeUpdate size_update,
                            internal::CacheEntryQueueState new_state,
                            absl::Status status) {
  TENSORSTORE_DEBUG_LOG(debug, "ResolveIssuedWriteback: ", entry,
                        ", status=", status);
  auto& ed = Access::entry_data(*entry);
  assert(ed.issued_writeback_generation != 0);
  Promise<void> next_writeback = std::move(ed.writebacks[0]);
  ed.writebacks[0] = std::move(ed.writebacks[1]);
  Future<const void> writeback_requested_by_cache;
  Promise<void> queued_writeback;
  if (new_state == CacheEntryQueueState::clean_and_in_use) {
    ed.write_generation = 0;
    ed.requested_writeback_generation = 0;
    writeback_requested_by_cache = std::move(ed.writeback_requested_by_cache);
    queued_writeback = std::move(ed.writebacks[0]);
  } else if (ed.requested_writeback_generation <=
             ed.issued_writeback_generation) {
    // Entry will remain dirty, but writeback was not requested since the last
    // writeback started.  Therefore, mark the entry as not having a writeback
    // request.
    ed.requested_writeback_generation = 0;
  }
  ed.issued_writeback_generation = 0;

  MaybeStartReadOrWriteback(entry, {/*.SizeUpdate=*/std::move(size_update),
                                    /*.new_state=*/new_state});
  // The order in which we call SetResult on these two promises is arbitrary,
  // but the race condition test for a writeback being forced after the entry
  // has been marked clean depends on next_writeback being ready before
  // queued_writeback.
  next_writeback.SetResult(MakeResult(status));
  if (queued_writeback.valid()) {
    queued_writeback.SetResult(MakeResult(status));
  }
}

}  // namespace

size_t AsyncStorageBackedCache::DoGetReadStateSizeInBytes(Cache::Entry* entry) {
  return 0;
}

size_t AsyncStorageBackedCache::DoGetWriteStateSizeInBytes(
    Cache::Entry* entry) {
  return 0;
}

size_t AsyncStorageBackedCache::DoGetFixedSizeInBytes(Cache::Entry* entry) {
  return this->Cache::DoGetSizeInBytes(entry);
}

size_t AsyncStorageBackedCache::DoGetSizeInBytes(Cache::Entry* base_entry) {
  auto* entry = static_cast<Entry*>(base_entry);
  return this->DoGetFixedSizeInBytes(entry) +
         this->DoGetReadStateSizeInBytes(entry) +
         this->DoGetWriteStateSizeInBytes(entry);
}

Future<const void> AsyncStorageBackedCache::Entry::Read(
    absl::Time staleness_bound) {
  TENSORSTORE_DEBUG_LOG(debug, "Read: ", this,
                        ", staleness_bound=", staleness_bound);
  absl::Time last_read_time = [&] {
    auto lock = this->AcquireReadStateReaderLock();
    return this->last_read_time;
  }();
  auto& ed = Access::entry_data(*this);
  std::unique_lock<Mutex> lock(ed.mutex);
  if (ed.ready_read.valid() && last_read_time >= staleness_bound) {
    // `staleness_bound` satisfied by current data.
    return ed.ready_read;
  }

  // `staleness_bound` not satisfied by current data.
  ed.queued_read_time =
      std::max(ed.queued_read_time, staleness_bound == absl::InfiniteFuture()
                                        ? absl::Now()
                                        : staleness_bound);

  if (ed.issued_read.promise.valid() || ed.requested_writeback_generation) {
    // Another read or write operation is in progress.
    if (ed.issued_read.promise.valid() &&
        ed.issued_read_time >= staleness_bound) {
      // Another read is in progress, and `staleness_bound` will be satisfied by
      // it when it completes.
      return ed.issued_read.future;
    }
    // A read or write operation is in progress.  We will wait until it
    // completes, and then may need to issue another read operation to satisfy
    // `staleness_bound`.
    if (ed.queued_read.valid()) {
      auto future = ed.queued_read.future();
      if (future.valid()) return future;
    }
    auto pair = PromiseFuturePair<void>::Make();
    ed.queued_read = std::move(pair.promise);
    return pair.future;
  }
  // No read or write is in progress.  Issue a new read operation.
  return IssueRead(this, {/*.SizeUpdate=*/{std::move(lock)}});
}

Future<const void> AsyncStorageBackedCache::Entry::GetWritebackFuture() {
  auto& ed = Access::entry_data(*this);
  absl::MutexLock lock(&ed.mutex);
  return GetWritebackFutureWithLock(this);
}

Future<const void> AsyncStorageBackedCache::Entry::FinishWrite(
    WriteStateLock lock, WriteFlags flags) {
  auto size_update = GetSizeUpdate(this, std::move(lock));
  auto& ed = Access::entry_data(*this);
  // State is transitioning to dirty, so the previous writeback request, if any,
  // is no longer applicable.  We will invalidate the handle once we have
  // released the mutex (to avoid running any ExecuteWhenNotNeeded callbacks
  // with the mutex held).
  auto writeback_requested_by_cache =
      std::move(ed.writeback_requested_by_cache);
  ++ed.write_generation;
  // Get a suitable existing or new Future to represent the pending writeback.
  Future<const void> future = GetFutureOrInvalidate(&ed.writebacks[1]);
  if (future.valid()) {
    // Reuse pending writeback future.
    TENSORSTORE_DEBUG_LOG(
        debug, "Entry::FinishWrite: entry=", this,
        " new write_generation=", ed.write_generation,
        ", issued_writeback_generation=", ed.issued_writeback_generation,
        ", using pending writeback future 1");
  } else {
    if (ed.writebacks[0].valid() && ed.issued_writeback_generation == 0) {
      future = ed.writebacks[0].future();
      TENSORSTORE_DEBUG_LOG(
          debug, "Entry::FinishWrite: entry=", this,
          " new write_generation=", ed.write_generation,
          ", issued_writeback_generation=", ed.issued_writeback_generation,
          ", using existing writeback future 0");
    }
    if (!future.valid()) {
      future = GetNewWritebackFuture(this);
      TENSORSTORE_DEBUG_LOG(
          debug, "Entry::FinishWrite: entry=", this,
          " new write_generation=", ed.write_generation,
          ", issued_writeback_generation=", ed.issued_writeback_generation,
          ", using new writeback future");
    }
  }

  if (flags >= WriteFlags::kUnconditionalWriteback) {
    // Record that a read result is not required for writeback as of this write
    // generation.
    ed.unconditional_writeback_generation = ed.write_generation;
  }

  UpdateState({/*.SizeUpdate=*/std::move(size_update),
               /*.new_state=*/CacheEntryQueueState::dirty});
  return future;
}

void AsyncStorageBackedCache::Entry::AbortWrite(WriteStateLock lock) {
  UpdateState({/*.SizeUpdate=*/GetSizeUpdate(this, std::move(lock))});
}

void AsyncStorageBackedCache::NotifyReadSuccess(Cache::Entry* base_entry,
                                                ReadStateWriterLock lock) {
  auto* entry = static_cast<Entry*>(base_entry);
  TENSORSTORE_DEBUG_LOG(debug, "NotifyReadSuccess: ", entry);
  assert(entry->last_read_time != absl::InfinitePast());
  auto& ed = Access::entry_data(*entry);
  auto size_update = GetSizeUpdate(entry, std::move(lock));
  assert(ed.issued_read.future.valid());
  ed.ready_read = MakeReadyFuture<void>(MakeResult());
  ed.writeback_needs_read = false;
  ResolveIssuedRead(entry, {/*.SizeUpdate=*/std::move(size_update)},
                    absl::OkStatus());
}

void AsyncStorageBackedCache::NotifyReadError(Cache::Entry* base_entry,
                                              absl::Status error) {
  auto* entry = static_cast<Entry*>(base_entry);
  TENSORSTORE_DEBUG_LOG(debug, "NotifyReadError: ", entry, ", error=", error);
  assert(!error.ok());
  auto& ed = Access::entry_data(*entry);
  CacheEntry::SizeUpdate update{std::unique_lock<tensorstore::Mutex>(ed.mutex)};
  assert(ed.issued_read.future.valid());
  ed.ready_read = MakeReadyFuture<void>(MakeResult(error));
  if (ed.requested_writeback_generation &&
      !ed.unconditional_writeback_generation) {
    // Read failed, and a requested writeback was waiting on this read.
    // Therefore, we treat the writeback as having failed.
    auto queued_writeback = std::move(ed.writebacks[1]);
    auto next_writeback = std::move(ed.writebacks[0]);
    assert(next_writeback.valid());
    auto writeback_requested_by_cache =
        std::move(ed.writeback_requested_by_cache);
    ed.requested_writeback_generation = 0;
    ResolveIssuedRead(entry,
                      {/*.SizeUpdate=*/std::move(update),
                       /*.new_state=*/CacheEntryQueueState::clean_and_in_use},
                      error);
    next_writeback.SetResult(error);
    if (queued_writeback.valid()) {
      queued_writeback.SetResult(error);
    }
  } else {
    ResolveIssuedRead(entry, {/*.SizeUpdate=*/std::move(update)},
                      std::move(error));
  }
}

void AsyncStorageBackedCache::DoWriteback(PinnedEntry entry) {
  TENSORSTORE_UNREACHABLE;
}

void AsyncStorageBackedCache::NotifyWritebackStarted(Cache::Entry* base_entry,
                                                     WriteStateLock lock) {
  auto* entry = static_cast<Entry*>(base_entry);
  TENSORSTORE_DEBUG_LOG(debug, "NotifyWritebackStarted: ", entry);
  auto size_update = GetSizeUpdate(entry, std::move(lock));
  auto& ed = Access::entry_data(*entry);
  ed.issued_writeback_generation = ed.write_generation;
  TENSORSTORE_DEBUG_LOG(debug, "NotifyWritebackStarted: entry=", entry);
  if (ed.writebacks[1].valid()) {
    if (auto future = GetFutureOrInvalidate(&ed.writebacks[0]);
        future.valid()) {
      auto promise = std::move(ed.writebacks[1]);
      entry->UpdateState(
          {std::move(size_update), CacheEntryQueueState::writeback_requested});
      // Set up the link only after releasing the lock, because this could
      // trigger a call to the force callback, which attempts to acquire the
      // lock.
      Link(std::move(promise), std::move(future));
      return;
    } else {
      ed.writebacks[0] = std::move(ed.writebacks[1]);
    }
  }
  entry->UpdateState(
      {std::move(size_update), CacheEntryQueueState::writeback_requested});
}

void AsyncStorageBackedCache::NotifyWritebackSuccess(
    Cache::Entry* base_entry, WriteAndReadStateLock lock) {
  auto* entry = static_cast<Entry*>(base_entry);
  TENSORSTORE_DEBUG_LOG(debug, "NotifyWritebackSuccess: ", entry);
  auto size_update = GetSizeUpdate(entry, std::move(lock));
  auto& ed = Access::entry_data(*entry);

  // Read operation must not be in progress.
  assert(!ed.issued_read.promise.valid());

  // Writeback must have been pending.
  assert(ed.writebacks[0].valid());

  assert(ed.issued_writeback_generation != 0);

  CacheEntryQueueState new_state = CacheEntryQueueState::clean_and_in_use;

  ed.ready_read = MakeReadyFuture<void>(MakeResult());

  if (ed.unconditional_writeback_generation <= ed.issued_writeback_generation) {
    // FinishWrite with `flags=kUnconditionalWriteback` was not called since
    // writeback was issued.  Mark this entry as no longer being in an
    // "unconditional writeback" state.
    ed.unconditional_writeback_generation = 0;
  }

  if (ed.issued_writeback_generation != ed.write_generation) {
    // Additional writes were issued that were not included in the
    // just-completed writeback.  Leave the entry marked dirty.
    new_state = CacheEntryQueueState::dirty;
  }

  ResolveIssuedWriteback(entry, std::move(size_update), new_state,
                         absl::OkStatus());
}

void AsyncStorageBackedCache::NotifyWritebackError(Cache::Entry* base_entry,
                                                   WriteStateLock lock,
                                                   Status error) {
  auto* entry = static_cast<Entry*>(base_entry);
  TENSORSTORE_DEBUG_LOG(debug, "NotifyWritebackError: ", entry,
                        ", error=", error);
  assert(!error.ok());
  auto size_update = GetSizeUpdate(entry, std::move(lock));
  auto& ed = Access::entry_data(*entry);

  // Read operation must not be in progress.
  assert(!ed.issued_read.promise.valid());

  // Writeback must have been pending.
  assert(ed.writebacks[0].valid());

  assert(ed.issued_writeback_generation != 0);

  // If `unconditional_writeback_generation` is non-zero, set it to 1 in order
  // to retain its meaning but remain consistent with `write_generation` being
  // reset to `0`.
  if (ed.unconditional_writeback_generation != 0) {
    ed.unconditional_writeback_generation = 1;
  }

  ResolveIssuedWriteback(entry, std::move(size_update),
                         CacheEntryQueueState::clean_and_in_use,
                         std::move(error));
}

void AsyncStorageBackedCache::NotifyWritebackNeedsRead(
    Cache::Entry* base_entry, WriteStateLock lock, absl::Time staleness_bound) {
  auto* entry = static_cast<Entry*>(base_entry);
  TENSORSTORE_DEBUG_LOG(debug, "NotifyWritebackNeedsRead: ", entry,
                        ", staleness_bound=", staleness_bound);
  auto size_update = GetSizeUpdate(entry, std::move(lock));
  auto& ed = Access::entry_data(*entry);

  // Read operation must not be in progress.
  assert(!ed.issued_read.promise.valid());

  // Writeback must have been pending.
  assert(ed.writebacks[0].valid());

  TENSORSTORE_DEBUG_LOG(debug, "NotifyWritebackNeedsRead: entry=", entry,
                        " writeback requires updated read result to proceed");
  // Writeback requires updated read result to proceed.
  ed.queued_read_time = std::max(ed.queued_read_time, staleness_bound);
  ed.unconditional_writeback_generation = 0;
  ed.issued_writeback_generation = 0;
  ed.writeback_needs_read = true;
  MaybeStartReadOrWriteback(
      entry, {std::move(size_update), CacheEntryQueueState::dirty});
}

void AsyncStorageBackedCache::DoRequestWriteback(PinnedEntry base_entry) {
  auto* entry = static_cast<Entry*>(base_entry.get());
  Future<const void> future;
  {
    auto& ed = Access::entry_data(*entry);
    absl::MutexLock lock(&ed.mutex);
    future = ed.writeback_requested_by_cache =
        GetWritebackFutureWithLock(entry);
  }
  // If `future` is invalid, entry is already clean.
  if (future.valid()) {
    // Request that writeback actually starts.
    future.Force();
  }
}

}  // namespace internal
}  // namespace tensorstore
