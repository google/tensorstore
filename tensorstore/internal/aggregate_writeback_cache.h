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

#ifndef TENSORSTORE_INTERNAL_AGGREGATE_WRITEBACK_CACHE_H_
#define TENSORSTORE_INTERNAL_AGGREGATE_WRITEBACK_CACHE_H_

/// \file
///
/// Framework for defining writeback caches that aggregate multiple independent
/// atomic write operations.

#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/async_storage_backed_cache.h"
#include "tensorstore/internal/cache.h"

namespace tensorstore {
namespace internal {

/// CRTP base class for defining an `AsyncStorageBackedCache` that aggregates
/// multiple independent atomic write operations.
///
/// This class does not itself implement any specific writeback behavior, but
/// keeps track of the lists of pending and issued `PendingWrite` operations.
///
/// \tparam Derived The derived class type, must define a `PendingWrite` type
///     and optionally override `DoGetPendingWriteSize`.
/// \tparam Parent The base class to inherit from, must inherit from (or equal)
///     `AsyncStorageBackedCache`.
template <typename Derived, typename Parent>
class AggregateWritebackCache : public Parent {
 public:
  class Entry : public Parent::Entry {
   public:
    using Cache = AggregateWritebackCache;
    using PendingWrite = typename Derived::PendingWrite;
    /// Requests that have been enqueued but not yet submitted for writeback.
    std::vector<PendingWrite> pending_writes;

    /// Requests for which writeback is in progress.
    std::vector<PendingWrite> issued_writes;

    /// Latest `request_time` of all requests in `pending_writes`.
    absl::Time last_pending_write_time = absl::InfinitePast();

    /// Latest `request_time` of all requests in `issued_writes`.
    absl::Time last_issued_write_time = absl::InfinitePast();

    /// Adds a `PendingWrite` to the `pending_writes` list, then calls
    /// `FinishWrite`.
    ///
    /// \param pending_write The operation to add.
    /// \param write_flags Flags to pass to `FinishWrite`.
    /// \param request_time The timestamp associated with the request, used to
    ///     update `last_pending_write_time`.
    /// \returns The `Future` returned by `FinishWrite`.  Typically, if
    ///     `pending_write` contains a `Promise`, the caller should call
    ///     `LinkError(promise, future)`, where `future` is the return value of
    ///     `AddPendingWrite` and `promise` is a copy of the `Promise` in
    ///     `pending_write`.
    Future<const void> AddPendingWrite(
        PendingWrite pending_write,
        AsyncStorageBackedCache::WriteFlags write_flags,
        absl::Time request_time = absl::Now()) {
      auto lock = this->AcquireWriteStateLock();
      pending_writes_size +=
          static_cast<Derived*>(GetOwningCache(this))
              ->DoGetPendingWriteSize(
                  static_cast<typename Derived::Entry*>(this), pending_write);
      pending_writes.emplace_back(std::move(pending_write));
      last_pending_write_time = std::max(last_pending_write_time, request_time);
      return this->FinishWrite(std::move(lock), write_flags);
    }

   private:
    friend class AggregateWritebackCache;
    /// Additional heap-allocated memory required by `pending_writes`, not
    /// including the memory allocated directly by the `std::vector`.
    size_t pending_writes_size = 0;

    /// Additional heap-allocated memory required by `issued_writes`, not
    /// including the memory allocated directly by the `std::vector`.
    size_t issued_writes_size = 0;
  };

  using Parent::Parent;

  /// Computes the "write state" size from the cached `issued_writes_size` and
  /// `pending_writes_size` values.
  size_t DoGetWriteStateSizeInBytes(Cache::Entry* base_entry) override {
    auto* entry = static_cast<Entry*>(base_entry);
    return Parent::DoGetWriteStateSizeInBytes(entry) +
           entry->issued_writes_size + entry->pending_writes_size +
           (entry->issued_writes.capacity() +
            entry->pending_writes.capacity()) *
               sizeof(typename Derived::PendingWrite);
  }

  /// Returns the additional heap memory required by a `PendingWrite` object
  /// (should not include `sizeof(PendingWrite)`).
  ///
  /// The `Derived` class should override this method if additional heap memory
  /// is required.
  template <typename PendingWrite>
  size_t DoGetPendingWriteSize(Entry* entry,
                               const PendingWrite& pending_write) {
    return 0;
  }

  /// Extends `NotifyWritebackStarted` to move `pending_writes` to
  /// `issued_writes` and update the associated timestamps and sizes.
  void NotifyWritebackStarted(
      Cache::Entry* base_entry,
      AsyncStorageBackedCache::WriteStateLock lock) override {
    auto* entry = static_cast<Entry*>(base_entry);
    if (entry->issued_writes.empty()) {
      std::swap(entry->issued_writes, entry->pending_writes);
    } else {
      entry->issued_writes.insert(
          entry->issued_writes.end(),
          std::make_move_iterator(entry->pending_writes.begin()),
          std::make_move_iterator(entry->pending_writes.end()));
      entry->pending_writes.clear();
    }
    entry->last_issued_write_time = std::max(
        entry->last_issued_write_time,
        std::exchange(entry->last_pending_write_time, absl::InfinitePast()));
    entry->issued_writes_size += std::exchange(entry->pending_writes_size, 0);
    this->Parent::NotifyWritebackStarted(entry, std::move(lock));
  }

  /// Extends `NotifyWritebackNeedsRead` to re-add all `issued_writes` to
  /// `pending_writes`.
  void NotifyWritebackNeedsRead(Cache::Entry* base_entry,
                                AsyncStorageBackedCache::WriteStateLock lock,
                                absl::Time staleness_bound) override {
    auto* entry = static_cast<Entry*>(base_entry);
    if (!entry->issued_writes.empty()) {
      std::swap(entry->issued_writes, entry->pending_writes);
      entry->pending_writes.insert(
          entry->pending_writes.end(),
          std::make_move_iterator(entry->issued_writes.begin()),
          std::make_move_iterator(entry->issued_writes.end()));
      entry->issued_writes.clear();
      entry->last_pending_write_time = std::max(entry->last_pending_write_time,
                                                entry->last_issued_write_time);
      entry->pending_writes_size += std::exchange(entry->issued_writes_size, 0);
    }
    entry->last_issued_write_time = absl::InfinitePast();

    this->Parent::NotifyWritebackNeedsRead(entry, std::move(lock),
                                           staleness_bound);
  }

  /// Extends `NotifyWritebackError` to clear `issued_writes`.
  ///
  /// Typically, this behavior is not sufficient, and instead derived classes
  /// should override it as follows:
  ///
  ///     std::vector<PendingWrite> issued_writes;
  ///     std::swap(entry->issued_writes, issued_writes);
  ///     Base::NotifyWritebackError(entry, std::move(lock), error);
  ///     // Complete each request in `issued_writes` appropriately.
  void NotifyWritebackError(internal::Cache::Entry* base_entry,
                            AsyncStorageBackedCache::WriteStateLock lock,
                            absl::Status error) override {
    auto* entry = static_cast<Entry*>(base_entry);
    entry->issued_writes.clear();
    entry->issued_writes_size = 0;
    entry->last_issued_write_time = absl::InfinitePast();
    this->Parent::NotifyWritebackError(entry, std::move(lock),
                                       std::move(error));
  }

  /// Extends `NotifyWritebackSuccess` to clear `issued_writes`.
  ///
  /// Typically, this behavior is not sufficient, and instead derived classes
  /// should override it as follows:
  ///
  ///     std::vector<PendingWrite> issued_writes;
  ///     std::swap(entry->issued_writes, issued_writes);
  ///     Base::NotifyWritebackSuccess(entry, std::move(lock));
  ///     // Complete each request in `issued_writes` appropriately.
  void NotifyWritebackSuccess(
      internal::Cache::Entry* base_entry,
      AsyncStorageBackedCache::WriteAndReadStateLock lock) override {
    auto* entry = static_cast<Entry*>(base_entry);
    entry->issued_writes.clear();
    entry->issued_writes_size = 0;
    entry->last_issued_write_time = absl::InfinitePast();
    this->Parent::NotifyWritebackSuccess(entry, std::move(lock));
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_AGGREGATE_WRITEBACK_CACHE_H_
