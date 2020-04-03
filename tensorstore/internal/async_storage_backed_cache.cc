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
  AsyncStorageBackedCache::ReadOptions options;
  options.existing_generation = ed.ready_read_generation.generation;
  options.staleness_bound = ed.issued_read_time = ed.queued_read_time;
  ed.issued_read = PromiseFuturePair<void>::Make();
  auto future = ed.issued_read.future;
  entry->UpdateState(std::move(update));
  GetOwningCache(entry)->DoRead(
      std::move(options),
      AsyncStorageBackedCache::ReadReceiver{
          PinnedCacheEntry<AsyncStorageBackedCache>(entry)});
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

  // A writeback must not have started (i.e. `WritebackReceiver::NotifyStarted`
  // called).
  assert(ed.issued_writeback_generation == 0);

  // A writeback must not be in progress even if
  // `WritebackReceiver::NotifyStarted` has not yet been called, but we cannot
  // check for that condition.

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

    AsyncStorageBackedCache::ReadOptions options;
    options.existing_generation = ed.ready_read_generation.generation;
    options.staleness_bound = ed.queued_read_time;
    entry->UpdateState(std::move(update));
    TENSORSTORE_DEBUG_LOG(debug, "MaybeStartReadOrWriteback: entry=", entry,
                          " calling DoRead");
    GetOwningCache(entry)->DoRead(
        std::move(options),
        AsyncStorageBackedCache::ReadReceiver{
            PinnedCacheEntry<AsyncStorageBackedCache>(entry)});
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
        StorageGeneration::IsUnknown(ed.ready_read_generation.generation)) {
      // Writeback can't be started yet because a read must complete first.
      TENSORSTORE_DEBUG_LOG(debug, "MaybeStartReadOrWriteback: entry=", entry,
                            " read required for writeback");
      if (!maybe_start_read()) {
        IssueRead(entry, {std::move(update),
                          CacheEntryQueueState::writeback_requested});
      }
      return true;
    }
    // Writeback will be started.
    auto existing_generation = ed.ready_read_generation;
    entry->UpdateState(
        {std::move(update), CacheEntryQueueState::writeback_requested});
    TENSORSTORE_DEBUG_LOG(debug, "MaybeStartReadOrWriteback: entry=", entry,
                          " calling DoWriteback");
    GetOwningCache(entry)->DoWriteback(
        std::move(existing_generation),
        AsyncStorageBackedCache::WritebackReceiver{
            PinnedCacheEntry<AsyncStorageBackedCache>(entry)});
    return true;
  };

  // Starts a writeback or read as required.  Upon return, `update` has been
  // applied.
  auto update_state = [&] {
    if (!maybe_start_writeback() && !maybe_start_read()) {
      entry->UpdateState(std::move(update));
    }
  };

  if (ed.queued_read.valid() &&
      ed.queued_read_time <= ed.ready_read_generation.time) {
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
            {std::move(lock), CacheEntryQueueState::writeback_requested});
        return;
      }
      TENSORSTORE_DEBUG_LOG(debug, "WritebackFuture Force: entry=", entry.get(),
                            " attempting to start writeback");
      MaybeStartReadOrWriteback(
          entry.get(),
          {std::move(lock), CacheEntryQueueState::writeback_requested});
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

}  // namespace

Future<const void> AsyncStorageBackedCache::Entry::Read(
    StalenessBound staleness_bound) {
  auto& ed = Access::entry_data(*this);
  std::unique_lock<Mutex> lock(ed.mutex);
  if (ed.ready_read.valid() &&
      ed.ready_read_generation.time >= staleness_bound) {
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
  return IssueRead(this, {std::move(lock)});
}

void AsyncStorageBackedCache::WritebackReceiver::NotifyStarted(
    CacheEntry::SizeUpdate update) const {
  auto* entry = this->entry();
  auto& ed = Access::entry_data(*entry);
  update.lock = std::unique_lock<Mutex>(ed.mutex);
  ed.issued_writeback_generation = ed.write_generation;
  TENSORSTORE_DEBUG_LOG(debug,
                        "WritebackReceiver::NotifyStarted: entry=", entry);
  if (ed.writebacks[1].valid()) {
    if (auto future = GetFutureOrInvalidate(&ed.writebacks[0]);
        future.valid()) {
      auto promise = std::move(ed.writebacks[1]);
      entry->UpdateState(
          {std::move(update), CacheEntryQueueState::writeback_requested});
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
      {std::move(update), CacheEntryQueueState::writeback_requested});
}

Future<const void> AsyncStorageBackedCache::Entry::GetWritebackFuture() {
  auto& ed = Access::entry_data(*this);
  absl::MutexLock lock(&ed.mutex);
  return GetWritebackFutureWithLock(this);
}

Future<const void> AsyncStorageBackedCache::Entry::FinishWrite(
    SizeUpdate size_update, WriteFlags flags) {
  auto& ed = Access::entry_data(*this);
  // Lock hand-off from existing lock (if any) to lock on `mutex`.
  size_update.lock = std::unique_lock<Mutex>(ed.mutex);
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

  if (flags == WriteFlags::kSupersedesRead &&
      absl::exchange(ed.supersedes_read_generation, ed.write_generation) == 0) {
    // As a result of this write, all read requests can be satisfied by the
    // local modification data alone, making read operations unnecessary.
    // Furthermore, this is the first write since the last completed writeback
    // to cause this to be true.
    ed.ready_read_generation = {StorageGeneration::Unknown(),
                                absl::InfiniteFuture()};
    // If the entry does not already have a successful read result state, mark
    // it as such.
    if (!ed.ready_read.valid() || !ed.ready_read.result()) {
      ed.ready_read = MakeReadyFuture<void>(MakeResult(Status()));
    }
    if (ed.issued_read.future.valid() || ed.queued_read.valid()) {
      // There is a pending read request.
      //
      // Mark pending reads as successfully completed (after
      // calling `UpdateState` to release `mutex`).
      //
      // If a read operation is in progress, the result will be ignored, but any
      // subsequent writeback operation will still wait to start until the read
      // operation completes to avoid multiple concurrent read/write operations.

      // Leave `ed.issued_read.promise` as is; if it is non-null, it must
      // non-null to indicate that a read operation is still in progress.
      auto read_promise = ed.issued_read.promise;
      auto read_future = std::move(ed.issued_read.future);
      auto queued_read = std::move(ed.queued_read);
      UpdateState({std::move(size_update), CacheEntryQueueState::dirty});

      if (read_promise.valid()) {
        read_promise.SetResult(MakeResult(Status()));
      }
      if (queued_read.valid()) {
        queued_read.SetResult(MakeResult(Status()));
      }
      return future;
    }
  }
  UpdateState({std::move(size_update), CacheEntryQueueState::dirty});
  return future;
}

void AsyncStorageBackedCache::ReadReceiver::NotifyDone(
    CacheEntry::SizeUpdate size_update,
    Result<TimestampedStorageGeneration> generation) const {
  auto* entry = this->entry();
  auto& ed = Access::entry_data(*entry);
  size_update.lock = std::unique_lock<Mutex>(ed.mutex);
  if (!ed.issued_read.future.valid()) {
    // The read was cancelled due to a prior call to `FinishWrite` with
    // `flags=kSupersedesRead`.
    TENSORSTORE_DEBUG_LOG(
        debug, "ReadReceiver::NotifyDone: entry=", this,
        "read was cancelled due to prior FinishWrite with kSupersedesRead");
    ed.issued_read.promise = Promise<void>();
    MaybeStartReadOrWriteback(entry, std::move(size_update));
    return;
  }

  if (generation) {
    ed.ready_read_generation.time = generation->time;
    if (StorageGeneration::IsUnknown(generation->generation)) {
      assert(
          !StorageGeneration::IsUnknown(ed.ready_read_generation.generation));
      TENSORSTORE_DEBUG_LOG(debug, "ReadReceiver::NotifyDone: entry=", entry,
                            " existing data is still up to date");
      // The existing data is still up to date.  Keep existing storage
      // generation.
    } else {
      TENSORSTORE_DEBUG_LOG(debug, "ReadReceiver::NotifyDone: entry=", entry,
                            " received updated data");
      ed.ready_read_generation.generation = generation->generation;
    }
  } else {
    TENSORSTORE_DEBUG_LOG(debug, "ReadReceiver::NotifyDone: entry=", entry,
                          " error=", generation.status().ToString());
  }

  ed.ready_read = MakeReadyFuture<void>(MakeResult(GetStatus(generation)));

  /// Marks `issued_read` ready with the specified `status` after calling
  /// `MaybeStartReadOrWriteback`.
  const auto resolve_issued_read = [](AsyncStorageBackedCache::Entry* entry,
                                      CacheEntry::StateUpdate update,
                                      Status status) {
    auto& ed = Access::entry_data(*entry);
    assert(ed.issued_read.promise.valid());
    // Remove `issued_read` from the entry state prior to calling
    // `MaybeStartReadOrWriteback` in order to indicate that a read operation is
    // no longer in progress.
    Promise<void> issued_read_promise = std::move(ed.issued_read.promise);
    ed.issued_read.future = Future<void>();
    MaybeStartReadOrWriteback(entry, std::move(update));
    issued_read_promise.SetResult(MakeResult(status));
  };

  if (!generation && ed.requested_writeback_generation &&
      !ed.unconditional_writeback_generation) {
    // Read failed, and a requested writeback was waiting on this read.
    // Therefore, we treat the writeback as having failed.
    assert(ed.supersedes_read_generation == 0);
    auto queued_writeback = std::move(ed.writebacks[1]);
    auto next_writeback = std::move(ed.writebacks[0]);
    assert(next_writeback.valid());
    auto writeback_requested_by_cache =
        std::move(ed.writeback_requested_by_cache);
    ed.requested_writeback_generation = 0;
    resolve_issued_read(
        entry,
        CacheEntry::StateUpdate(std::move(size_update),
                                CacheEntryQueueState::clean_and_in_use),
        generation.status());
    next_writeback.SetResult(generation.status());
    if (queued_writeback.valid()) {
      queued_writeback.SetResult(generation.status());
    }
  } else {
    // Either read succeeded, or read failed but no requested writeback depends
    // on it.
    resolve_issued_read(entry, std::move(size_update), GetStatus(generation));
  }
}

void AsyncStorageBackedCache::WritebackReceiver::NotifyDone(
    CacheEntry::SizeUpdate size_update,
    Result<TimestampedStorageGeneration> generation) const {
  auto* entry = this->entry();
  auto& ed = Access::entry_data(*entry);
  size_update.lock = std::unique_lock<Mutex>(ed.mutex);

  // Read operation must not be in progress.
  assert(!ed.issued_read.promise.valid());

  // Writeback must have been pending.
  assert(ed.writebacks[0].valid());

  if (generation && StorageGeneration::IsUnknown(generation->generation)) {
    TENSORSTORE_DEBUG_LOG(debug, "WritebackReceiver::NotifyDone: entry=", entry,
                          " writeback requires updated read result to proceed");
    // Writeback requires updated read result to proceed.
    ed.ready_read_generation.generation = StorageGeneration::Unknown();
    ed.queued_read_time = std::max(ed.queued_read_time, generation->time);
    ed.unconditional_writeback_generation = 0;
    ed.issued_writeback_generation = 0;
    MaybeStartReadOrWriteback(
        entry, {std::move(size_update), CacheEntryQueueState::dirty});
    return;
  }
  Promise<void> next_writeback = std::move(ed.writebacks[0]);
  ed.writebacks[0] = std::move(ed.writebacks[1]);
  TENSORSTORE_DEBUG_LOG(
      debug, "WritebackReceiver::NotifyDone: entry=", entry,
      " issued_writeback_generation=", ed.issued_writeback_generation,
      ", write_generation=", ed.write_generation,
      ", unconditional_writeback_generation=",
      ed.unconditional_writeback_generation,
      ", requested_writeback_generation=", ed.requested_writeback_generation,
      ", status=", GetStatus(generation).ToString());
  assert(ed.issued_writeback_generation != 0);

  Status writeback_status;

  CacheEntryQueueState new_state = CacheEntryQueueState::clean_and_in_use;
  if (generation || generation.status().code() == absl::StatusCode::kAborted) {
    // Writeback completed successfully.
    if (ed.supersedes_read_generation <= ed.issued_writeback_generation) {
      // FinishWrite with `flags=kSupersedesRead` was not called since writeback
      // was issued.  Mark this entry as no longer being in a "write supersedes
      // read" state.
      ed.supersedes_read_generation = 0;
      ed.ready_read = MakeReadyFuture<void>(MakeResult(Status()));
      if (generation) {
        ed.ready_read_generation = *generation;
      }
    }

    if (ed.unconditional_writeback_generation <=
        ed.issued_writeback_generation) {
      // FinishWrite with `flags=kSupersedesRead` or
      // `flags=kUnconditionalWriteback` was not called since writeback was
      // issued.  Mark this entry as no longer being in an "unconditional
      // writeback" state.
      ed.unconditional_writeback_generation = 0;
    }

    if (ed.issued_writeback_generation != ed.write_generation) {
      // Additional writes were issued that were not included in the
      // just-completed writeback.  Leave the entry marked dirty.
      new_state = CacheEntryQueueState::dirty;
    }
  } else {
    writeback_status = std::move(generation).status();
    // An error occurred during writeback.  This may lead to losing pending
    // writes, but we still need to maintain a consistent cache state.

    // If `supersedes_read_generation` or `unconditional_writeback_generation`
    // are non-zero, set them to 1 in order to retain their meaning but remain
    // consistent with `write_generation` being reset to `0`.
    if (ed.supersedes_read_generation != 0) {
      ed.supersedes_read_generation = 1;
    }
    if (ed.unconditional_writeback_generation != 0) {
      ed.unconditional_writeback_generation = 1;
    }
  }
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

  MaybeStartReadOrWriteback(entry, {std::move(size_update), new_state});
  // The order in which we call SetResult on these two promises is arbitrary,
  // but the race condition test for a writeback being forced after the entry
  // has been marked clean depends on next_writeback being ready before
  // queued_writeback.
  next_writeback.SetResult(MakeResult(writeback_status));
  if (queued_writeback.valid()) {
    queued_writeback.SetResult(MakeResult(writeback_status));
  }
}

void AsyncStorageBackedCache::DoRequestWriteback(
    PinnedCacheEntry<Cache> base_entry) {
  auto entry = static_pointer_cast<Entry>(base_entry);
  Future<const void> future;
  {
    auto& ed = Access::entry_data(*entry);
    absl::MutexLock lock(&ed.mutex);
    future = ed.writeback_requested_by_cache =
        GetWritebackFutureWithLock(entry.get());
  }
  // If `future` is invalid, entry is already clean.
  if (future.valid()) {
    // Request that writeback actually starts.
    future.Force();
  }
}

}  // namespace internal
}  // namespace tensorstore
