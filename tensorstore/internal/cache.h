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

#ifndef TENSORSTORE_INTERNAL_CACHE_H_
#define TENSORSTORE_INTERNAL_CACHE_H_

/// \file
/// Defines a writeback cache framework.
///
/// Example usage:
///
///   class MyCache : public CacheBase<MyCache, Cache> {
///    public:
///     class Entry : public Cache::Entry {
///       using Cache = MyCache;
///       Mutex data_mutex;
///       std::string data;
///       std::stsring Read() {
///         absl::ReaderMutexLock lock(&data_mutex);
///         return data;
///       }
///       void Write(std::string value) {
///         std::unique_lock<Mutex> lock(data_mutex);
///         data += value;
///         UpdateState(std::move(lock), CacheEntryQueueState::dirty,
///                     GetOwningCache(this)->DoGetSizeInBytes(this));
///       }
///     };
///
///     std::size_t DoGetSizeInBytes(Cache::Entry* base_entry) override {
///       auto* entry = static_cast<Entry*>(base_entry);
///       return sizeof(Entry) + entry->data.size() +
///           Cache::Entry::DoGetSizeInBytes(entry);
///     }
///
///     void DoRequestWriteback(PinnedCacheEntry<Cache> base_entry) override {
///       PinnedCacheEntry<MyCache> entry =
///           static_pointer_cast<Entry>(std::move(base_entry));
///       // Initiate writeback asynchronously.
///       // When completed, arranges for `MyWritebackCompleted` to be called.
///     }
///
///     void MyWritebackCompleted(PinnedCacheEntry<MyCache> entry) {
///       std::unique_lock<Mutex> lock(entry->data_mutex);
///       // Adjust entry->data to account for writeback completed.
///       // ...
///       entry->UpdateState(std::move(lock),
///                          CacheEntryQueueState::clean_and_in_use,
///                          DoGetSizeInBytes(entry.get()));
///     }
///
///   };
///
///   // Create a pool with a 2MiB size limit, 1MiB limit for pending writes.
///   auto pool = CachePool::Make(CachePool::Limits{2000000, 1000000});
///
///   auto cache = pool.GetCache<MyCache>("cache_key", [&] {
///     return absl::make_unique<MyCache>();
///   });
///
///   auto entry = GetCacheEntry(cache, "entry_a");
///   auto value = entry->Read();
///   entry->Write("value_to_append");

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorstore/internal/cache_impl.h"
#include "tensorstore/internal/cache_pool_limits.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/poly.h"
#include "tensorstore/util/function_view.h"

namespace tensorstore {
namespace internal {

class Cache;

/// Strong pointer to a cache that may be used to obtain new cache entries.
///
/// Holding a strong pointer to a cache ensures the cache is not destroyed even
/// if all of its entries are evicted.
template <typename CacheType>
using CachePtr = internal_cache::CachePtr<CacheType>;

/// A cache pool manages a single logical accounting pool of memory that may be
/// shared by multiple types of caches.
///
/// It is associated with a collection of `Cache` objects, each of which contain
/// a collection of entries.  Each entry has an associated size in bytes, and is
/// in one of several states, as defined by the `CacheEntryQueueState` enum. The
/// cache pool maintains least-recently-used eviction and writeback queues of
/// the entries; once the user-specified `CachePool:Limits` are reached, entries
/// are evicted and/or writeback is requested in order to attempt to free
/// memory.  The limits apply to the aggregate memory usage of all caches
/// managed by the pool, and a single LRU eviction queue and a single LRU
/// writeback queue is used for all managed caches.
class CachePool : private internal_cache::CachePoolImpl {
 public:
  using Limits = CachePoolLimits;

  /// Returns the limits of this cache pool.
  const Limits& limits() const { return limits_; }

  /// Returns a cache of type `CacheType` for the specified `cache_key`.
  ///
  /// If such a cache does not already exist, or `cache_key` is empty,
  /// `make_cache()` is called to obtain a new such cache.
  ///
  /// \tparam CacheType Must be a class that inherits from `Cache`.
  /// \param cache_key Specifies the cache key.
  /// \param make_cache Nullary function that returns an
  ///     `std::unique_ptr<CacheType>`, where `CacheType` as a type that
  ///     inherits from `Cache`.  A `nullptr` may be returned to indicate an
  ///     error creating the cache (any additional error information must be
  ///     communicated via some separate out-of-band channel).
  template <typename CacheType>
  CachePtr<CacheType> GetCache(
      absl::string_view cache_key,
      FunctionView<std::unique_ptr<Cache>()> make_cache) {
    static_assert(std::is_base_of<Cache, CacheType>::value,
                  "CacheType must inherit from Cache.");
    return static_pointer_cast<CacheType>(internal_cache::GetCacheInternal(
        this, typeid(CacheType), cache_key, make_cache));
  }

  class WeakPtr;

  /// Reference-counted pointer to a cache pool that keeps in-use and recently
  /// accessed entries, as well as the caches containing in-use and recently
  /// accessed entries, alive.
  ///
  /// If a `Cache` contained within a `CachePool` directly or indirectly holds a
  /// `StrongPtr` to the same `CachePool`, a memory leak results.  To avoid a
  /// memory leak, the `Cache` must instead hold a `WeakPtr` to the `CachePool`.
  class StrongPtr : private internal_cache::CachePoolStrongPtr {
    using Base = internal_cache::CachePoolStrongPtr;

   public:
    StrongPtr() = default;
    explicit StrongPtr(const WeakPtr& ptr);
    using Base::operator bool;
    using Base::operator->;
    using Base::operator*;
    using Base::get;

   private:
    friend class internal_cache::Access;
  };

  /// Reference-counted pointer to a cache pool that keeps in-use caches, as
  /// well as the in-use and recently accessed entries of in-use caches, alive.
  /// Unlike a `StrongPtr`, this does not keep alive caches that were recently
  /// accessed but are not in-use.
  class WeakPtr : private internal_cache::CachePoolWeakPtr {
    using Base = internal_cache::CachePoolWeakPtr;

   public:
    WeakPtr() = default;
    explicit WeakPtr(const StrongPtr& ptr) : Base(ptr.get()) {}
    using Base::operator bool;
    using Base::operator->;
    using Base::operator*;
    using Base::get;

   private:
    friend class internal_cache::Access;
  };

  /// Returns a handle to a new cache pool with the specified limits.
  static StrongPtr Make(const Limits& limits);

 private:
  using internal_cache::CachePoolImpl::CachePoolImpl;
  friend class internal_cache::Access;
};

/// Represents the queue state of a Cache::Entry object.
enum class CacheEntryQueueState : int {
  /// Clean and has a reference count == 0.  Queued for eviction.
  clean_and_not_in_use,

  /// Clean, but has a reference count > 0.  Not queued for eviction.
  clean_and_in_use,

  /// Has local modifications for which writeback has not yet been requested.
  /// Writeback of some, but not all, local modifications may be in progress.
  dirty,

  /// Writeback of all local modifications has been requested.  If another
  /// local
  /// write happens before writeback completes, the state becomes `dirty`.
  writeback_requested,
};

std::ostream& operator<<(std::ostream& os, CacheEntryQueueState state);

/// Base class for cache entries.
class CacheEntry : private internal_cache::CacheEntryImpl {
 public:
  /// Alias required by the `GetOwningCache` function.  Derived `Entry` classes
  /// must redefine `Cache` to be the derived cache type.
  using Cache = internal::Cache;

  /// Returns the key for this entry.
  absl::string_view key() { return key_; }

  /// Returns the number of references to this cache entry.
  ///
  /// This is intended for testing and debugging.
  std::uint32_t use_count() {
    return reference_count_.load(std::memory_order_acquire);
  }

  /// Returns the current queue state.
  ///
  /// This is intended for testing and debugging, and should not be called
  /// while there may be other concurrent accesses to the same cache pool.
  CacheEntryQueueState queue_state() { return queue_state_; }

  /// Sets whether this entry should be evicted as soon as it is not in use,
  /// regardless of when it was last used or the available memory in the cache
  /// pool.
  ///
  /// This must only be called while the entry is in use, and must not be called
  /// by multiple threads concurrently for the same entry.
  void SetEvictWhenNotInUse(bool value = true) {
    evict_when_not_in_use_ = value;
  }

  /// Specifies an optional size update to be done with an optional lock
  /// hand-off.
  struct SizeUpdate {
    using Lock = Poly<0, /*Copyable=*/false>;

    /// Object whose destructor releases the lock that protects the entry state
    /// "S" corresponding to `new_size`.
    ///
    /// This lock is released after acquiring an exclusive lock on the cache
    /// pool state, in order to ensure that the order in which size updates take
    /// effect is the same order in which modifications to "S" occur.
    Lock lock;

    /// If not `absl::nullopt`, the entry size will be changed to the specified
    /// value.
    absl::optional<std::size_t> new_size;
  };

  /// Extends `SizeUpdate` with an optional state update.
  struct StateUpdate : public SizeUpdate {
    /// If not `absl::nullopt`, the queue state will be changed to the specified
    /// value.  If `new_state` is `clean_and_not_in_use` or `dirty`, the entry
    /// will be moved to the back (most recently used) position of the eviction
    /// or writeback queue, respectively, even if `queue_state` is the same as
    /// the existing queue state.
    absl::optional<CacheEntryQueueState> new_state;
  };

  /// Optionally modifies the entry queue state and/or the entry size.
  ///
  /// This may trigger writeback and/or eviction of entries in any cache that
  /// shares the same cache pool.
  ///
  /// This must be called to mark an entry as dirty before releasing the last
  /// reference to it, and after writeback completes to mark an entry as clean.
  /// It must also be called when the size changes (e.g. due to allocating or
  /// freeing additional heap memory referenced by the entry).
  void UpdateState(StateUpdate update);

  /// Initializes an entry after it is allocated.
  ///
  /// Derived classes may override this method if initialization is required.
  virtual void DoInitialize();

  virtual ~CacheEntry();

 private:
  friend class internal_cache::Access;
};

/// Pointer to a cache entry that prevents it from being evicted.
template <typename CacheType>
using PinnedCacheEntry =
    internal_cache::CacheEntryStrongPtr<typename CacheType::Entry>;

/// Abstract base class from which cache implementations must inherit.
///
/// Derived classes must define a nested `Entry` type that inherits from
/// `CacheEntry`, and override all of the virtual methods.  The nested `Entry`
/// type must define a `Cache` type alias that is equal to the derived class
/// type.
class Cache : private internal_cache::CacheImpl {
 public:
  /// Alias required by the `GetCacheEntry` function.  Derived classes must
  /// redefine `Entry` to be the derived entry type.
  using Entry = CacheEntry;

  using PinnedEntry = PinnedCacheEntry<Cache>;

  Cache();
  virtual ~Cache();

  /// Returns the strong reference count for testing/debugging.
  std::uint32_t use_count() const { return reference_count_.load(); }

  /// Returns the cache identifier.
  ///
  /// If non-empty, requesting a cache of the same type with the same identifier
  /// from the same cache pool while this cache is still alive will return a
  /// pointer to this same cache.
  absl::string_view cache_identifier() const { return cache_identifier_; }

  /// Allocates a new `entry` to be stored in this cache.
  ///
  /// Derived classes must define this method.
  virtual Entry* DoAllocateEntry() = 0;

  /// Destroys `entry`.
  ///
  /// Derived classes must define this method.
  ///
  /// \param entry Non-null pointer to the entry to destroy.
  virtual void DoDeleteEntry(Entry* entry) = 0;

  /// Returns the size in bytes used by `entry`.
  ///
  /// Derived classes that extend the `Entry` type with additional members that
  /// point to heap-allocated memory should override this method to return the
  /// sum of:
  ///
  /// 1. The result of `DoGetSizeofEntry`.
  ///
  /// 2. The result of `Base::DoGetSizeInBytes(entry)`, where `Base` is the
  ///    superclass.
  ///
  /// 3. The additional heap memory required by the additional members of
  ///    `entry`.
  ///
  /// The cache pool implementation only calls this method once during
  /// initialization without holding any locks.  Otherwise, it should only be
  /// called while holding necessary locks to protect the state required to
  /// calculate the size.
  virtual std::size_t DoGetSizeInBytes(Entry* entry);

  /// Returns `sizeof Entry`, where `Entry` is the derived class `Entry` type
  /// allocated by `DoAllocateEntry`.
  ///
  /// Normally it is not necessary to manually define this method, because
  /// `CacheBase` provides a suitable definition automatically.
  virtual std::size_t DoGetSizeofEntry() = 0;

  /// Initiates writeback of `entry`.
  ///
  /// Derived classes must define this method, which is called by the cache pool
  /// when the total number of bytes occupied by entries in the `dirty` state
  /// exceeds the `queued_for_writeback_bytes_limit` and `entry` is the least
  /// recently used entry in the `dirty` state.
  ///
  /// Implementations of this method should hold the `PinnedCacheEntry` pointer
  /// while writeback is in progress.  Once writeback completes,
  /// `CacheEntry::UpdateState` should be called with a `queue_state` of
  /// `clean_and_in_use`.
  ///
  /// \param entry Non-null pointer to the entry contained in this cache for
  ///     which writeback is requested.
  virtual void DoRequestWriteback(PinnedEntry entry) = 0;

 private:
  friend class internal_cache::Access;
};

/// CRTP base class for defining derived `Cache` types.
///
/// This type should be the base class of the most-derived `Cache` class.
///
/// This ensures that `DoAllocateEntry`, `DoDeleteEntry`, and `DoGetSizeofEntry`
/// are defined appropriately for the `Derived::Entry` type.
template <typename Derived, typename Parent>
class CacheBase : public Parent {
  static_assert(std::is_base_of_v<Cache, Parent>);

 public:
  using Parent::Parent;
  Cache::Entry* DoAllocateEntry() override {
    static_assert(std::is_base_of_v<Cache::Entry, typename Derived::Entry>);
    static_assert(
        std::is_base_of_v<typename Parent::Entry, typename Derived::Entry>);
    return new typename Derived::Entry;
  }
  void DoDeleteEntry(Cache::Entry* entry) override {
    delete static_cast<typename Derived::Entry*>(entry);
  }
  std::size_t DoGetSizeofEntry() override {
    return sizeof(typename Derived::Entry);
  }
};

/// Returns a pointer to the cache that contains `entry`.  By default, the
/// returned pointer is only valid at least as long as `entry` is valid, but the
/// lifetime of the cache may be extended by creating a CachePtr from the
/// returned pointer.
template <typename Entry>
inline std::enable_if_t<std::is_base_of<Cache::Entry, Entry>::value,
                        typename Entry::Cache*>
GetOwningCache(Entry* entry) {
  return internal_cache::Access::StaticCast<typename Entry::Cache>(
      internal_cache::Access::StaticCast<internal_cache::CacheEntryImpl>(entry)
          ->cache_);
}

template <typename Entry>
inline std::enable_if_t<std::is_base_of_v<Cache::Entry, Entry>,
                        typename Entry::Cache&>
GetOwningCache(Entry& entry) {
  return *internal_cache::Access::StaticCast<typename Entry::Cache>(
      internal_cache::Access::StaticCast<internal_cache::CacheEntryImpl>(&entry)
          ->cache_);
}

/// Overload for a `PinnedCacheEntry`.
template <typename Entry>
inline std::enable_if_t<std::is_base_of<Cache::Entry, Entry>::value,
                        typename Entry::Cache*>
GetOwningCache(const internal_cache::CacheEntryStrongPtr<Entry>& entry) {
  return GetOwningCache(entry.get());
}

/// Returns the entry of `cache` for the specified `key`.
///
/// If there is no existing entry for `key`, a new entry is created by calling
/// `Cache::DoAllocateEntry`.
template <typename CacheType>
std::enable_if_t<std::is_base_of<Cache, CacheType>::value,
                 PinnedCacheEntry<CacheType>>
GetCacheEntry(CacheType* cache, absl::string_view key) {
  return static_pointer_cast<typename CacheType::Entry>(
      internal_cache::GetCacheEntryInternal(cache, key));
}

/// Overload for a `CachePtr`.
template <typename CacheType>
std::enable_if_t<std::is_base_of<Cache, CacheType>::value,
                 PinnedCacheEntry<CacheType>>
GetCacheEntry(const CachePtr<CacheType>& cache, absl::string_view key) {
  return GetCacheEntry(cache.get(), key);
}

}  // namespace internal
}  // namespace tensorstore

#endif  //  TENSORSTORE_INTERNAL_CACHE_H_
