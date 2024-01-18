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

#ifndef TENSORSTORE_INTERNAL_CACHE_CACHE_H_
#define TENSORSTORE_INTERNAL_CACHE_CACHE_H_

/// \file
/// Defines a writeback cache framework.
///
/// Example usage:
///
///   class MyCache : public CacheBase<MyCache, Cache> {
///    public:
///     class Entry : public Cache::Entry {
///       using OwningCache = MyCache;
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
///     // Implement required virtual interfaces:
///     Entry* DoAllocateEntry() final { return new Entry; }
///     std::size_t DoGetSizeofEntry() final { return sizeof(Entry); }
///   };
///
///   // Create a pool with a 2MiB size limit.
///   auto pool = CachePool::Make(CachePool::Limits{2000000});
///
///   auto cache = GetCache<MyCache>(pool.get(), "cache_key", [&] {
///     return std::make_unique<MyCache>();
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
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/functional/function_ref.h"
#include "tensorstore/internal/cache/cache_impl.h"
#include "tensorstore/internal/cache/cache_pool_limits.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/poly/poly.h"

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
/// cache pool maintains a least-recently-used eviction queue of the entries;
/// once the user-specified `CachePool:Limits` are reached, entries are evicted
/// in order to attempt to free memory.  The limits apply to the aggregate
/// memory usage of all caches managed by the pool, and a single LRU eviction
/// queue is used for all managed caches.
class CachePool : private internal_cache::CachePoolImpl {
 public:
  using Limits = CachePoolLimits;

  /// Returns the limits of this cache pool.
  const Limits& limits() const { return limits_; }

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
    explicit WeakPtr(CachePool* ptr) : Base(ptr) {}
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

/// Returns a cache of type `CacheType` for the specified `cache_key`.
///
/// If such a cache does not already exist, or `cache_key` is empty,
/// `make_cache()` is called to obtain a new such cache.
///
/// \tparam CacheType Must be a class that inherits from `Cache`, or defines a
///     `Cache& cache()` method.
/// \param pool Cache pool, may be `nullptr` to indicate that caching is
///     disabled.
/// \param type_info Additional key used in looking up the cache.  Has no effect
///     if `cache_key` is the empty string or `pool` is `nullptr`.  Set to
///     `typeid(CacheType)` when calling `GetCache`, but can be any arbitrary
///     `std::type_info` object.
/// \param cache_key Specifies the cache key.
/// \param make_cache Nullary function that returns an
///     `std::unique_ptr<CacheType>`, where `CacheType` as a type that inherits
///     from `Cache` or defines a `Cache& cache()` method.  A `nullptr` may be
///     returned to indicate an error creating the cache (any additional error
///     information must be communicated via some separate out-of-band channel).
template <typename CacheType, typename MakeCache>
CachePtr<CacheType> GetCacheWithExplicitTypeInfo(
    CachePool* pool, const std::type_info& type_info,
    std::string_view cache_key, MakeCache&& make_cache) {
  auto cache = internal_cache::GetCacheInternal(
      internal_cache::Access::StaticCast<internal_cache::CachePoolImpl>(pool),
      type_info, cache_key, [&]() -> std::unique_ptr<internal::Cache> {
        std::unique_ptr<CacheType> cache = make_cache();
        if (!cache) return nullptr;
        void* user_ptr = cache.get();
        auto base_ptr = std::unique_ptr<internal::Cache>(
            &internal_cache::GetCacheObject(cache.release()));
        internal_cache::Access::StaticCast<internal_cache::CacheImpl>(
            base_ptr.get())
            ->user_ptr_ = user_ptr;
        return base_ptr;
      });
  if (!cache) return nullptr;
  return CachePtr<CacheType>(
      static_cast<CacheType*>(
          internal_cache::Access::StaticCast<internal_cache::CacheImpl>(
              cache.release())
              ->user_ptr_),
      internal::adopt_object_ref);
}
template <typename CacheType, typename MakeCache>
CachePtr<CacheType> GetCache(CachePool* pool, std::string_view cache_key,
                             MakeCache&& make_cache) {
  return GetCacheWithExplicitTypeInfo<CacheType>(
      pool, typeid(CacheType), cache_key, std::forward<MakeCache>(make_cache));
}

/// Pointer to a cache entry that prevents it from being evicted due to memory
/// pressure, but still permits it to be destroyed if its parent cache is
/// destroyed.
using WeakPinnedCacheEntry = internal_cache::WeakPinnedCacheEntry;

/// Base class for cache entries.
///
/// This class can be used with `tensorstore::UniqueWriterLock`.  Refer to
/// `WriterLock` and `WriterUnlock` for details.
class ABSL_LOCKABLE CacheEntry : private internal_cache::CacheEntryImpl {
 public:
  /// Alias required by the `GetOwningCache` function.  Derived `Entry` classes
  /// must redefine `OwningCache` to be the derived cache type.
  using OwningCache = internal::Cache;

  /// Returns the key for this entry.
  const std::string_view key() const { return key_; }

  /// Returns the number of references to this cache entry.
  ///
  /// This is intended for testing and debugging.
  std::uint32_t use_count() const {
    return reference_count_.load(std::memory_order_acquire) / 2;
  }

  /// Derived classes may use this to protect changes to the "cached data",
  /// whatever that may be, and may also use it to protect any other related
  /// data.  This must be held when the size in bytes of the cached data
  /// changes.
  absl::Mutex& mutex() { return mutex_; }

  /// Acquires a lock on `mutex()`.
  void WriterLock() ABSL_EXCLUSIVE_LOCK_FUNCTION();

  /// Releases a previously-acquired lock on `mutex()`, and updates the size in
  /// the cache pool, if the size is being tracked and `NotifySizeChanged()` was
  /// called.
  void WriterUnlock() ABSL_UNLOCK_FUNCTION();

  void DebugAssertMutexHeld() {
#ifndef NDEBUG
    mutex_.AssertHeld();
#endif
  }

  /// May be called while holding a lock on `mutex()` to indicate that the
  /// result of `GetOwningCache(*this).DoGetSizeInBytes(this)` has changed.
  void NotifySizeChanged() {
    this->DebugAssertMutexHeld();
    flags_ |= kSizeChanged;
  }

  /// Initializes an entry after it is allocated.
  ///
  /// Derived classes may override this method if initialization is required.
  virtual void DoInitialize();

  /// Returns a new weak reference to this entry.
  ///
  /// The caller must hold a strong reference.
  ///
  /// Like a strong reference, a weak reference prevents the entry from being
  /// evicted due to memory pressure.  However, if there are no strong
  /// references to the entry, no strong references to the cache, and no strong
  /// references to the pool (if the cache has a non-empty identifier), then the
  /// cache and all of its entries will be destroyed despite the existence of
  /// weak references to entries.
  WeakPinnedCacheEntry AcquireWeakReference() {
    return internal_cache::AcquireWeakCacheEntryReference(this);
  }

  virtual ~CacheEntry();

 private:
  friend class internal_cache::Access;
};

/// Pointer to a cache entry that prevents it from being evicted due to memory
/// pressure, and also ensures that the entry and its parent cache are not
/// destroyed.
template <typename CacheType>
using PinnedCacheEntry =
    internal_cache::CacheEntryStrongPtr<typename CacheType::Entry>;

/// Abstract base class from which cache implementations must inherit.
///
/// Derived classes must define a nested `Entry` type that inherits from
/// `CacheEntry`, and override all of the virtual methods.  The nested `Entry`
/// type must define a `OwningCache` type alias that is equal to the derived
/// class type.
class Cache : private internal_cache::CacheImpl {
 public:
  /// Alias required by the `GetCacheEntry` function.  Derived classes must
  /// redefine `Entry` to be the derived entry type.
  using Entry = CacheEntry;

  using PinnedEntry = PinnedCacheEntry<Cache>;

  Cache();
  virtual ~Cache();

  /// Returns the associated cache pool, or `nullptr` if using a disabled cache
  /// pool.
  CachePool* pool() const {
    return internal_cache::Access::StaticCast<CachePool>(pool_);
  }

  /// Returns the strong reference count for testing/debugging.
  std::uint32_t use_count() const { return reference_count_.load(); }

  /// Returns the cache identifier.
  ///
  /// If non-empty, requesting a cache of the same type with the same identifier
  /// from the same cache pool while this cache is still alive will return a
  /// pointer to this same cache.
  std::string_view cache_identifier() const { return cache_identifier_; }

  /// Allocates a new `entry` to be stored in this cache.
  ///
  /// Usually this method can be defined as:
  ///
  /// Entry* DoAllocateEntry() final { return new Entry; }
  virtual Entry* DoAllocateEntry() = 0;

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
  /// Usually this method can be defined as:
  ///
  /// std::size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  virtual std::size_t DoGetSizeofEntry() = 0;

 private:
  friend class internal_cache::Access;
};

/// Returns a reference to the cache that contains `entry`.  The reference
/// lifetime is tied to the lifetime of `entry`, but the lifetime of the cache
/// may be extended by creating a CachePtr.
template <typename Entry>
inline std::enable_if_t<std::is_base_of_v<Cache::Entry, Entry>,
                        typename Entry::OwningCache&>
GetOwningCache(Entry& entry) {
  return *internal_cache::Access::StaticCast<typename Entry::OwningCache>(
      internal_cache::Access::StaticCast<internal_cache::CacheEntryImpl>(&entry)
          ->cache_);
}

/// Returns the entry of `cache` for the specified `key`.
///
/// If there is no existing entry for `key`, a new entry is created by calling
/// `Cache::DoAllocateEntry`.
template <typename CacheType>
std::enable_if_t<std::is_base_of<Cache, CacheType>::value,
                 PinnedCacheEntry<CacheType>>
GetCacheEntry(CacheType* cache, std::string_view key) {
  return static_pointer_cast<typename CacheType::Entry>(
      internal_cache::GetCacheEntryInternal(cache, key));
}

/// Overload for a `CachePtr`.
template <typename CacheType>
std::enable_if_t<std::is_base_of<Cache, CacheType>::value,
                 PinnedCacheEntry<CacheType>>
GetCacheEntry(const CachePtr<CacheType>& cache, std::string_view key) {
  return GetCacheEntry(cache.get(), key);
}

}  // namespace internal
}  // namespace tensorstore

#endif  //  TENSORSTORE_INTERNAL_CACHE_CACHE_H_
