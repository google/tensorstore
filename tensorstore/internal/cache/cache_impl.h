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

#ifndef TENSORSTORE_INTERNAL_CACHE_CACHE_IMPL_H_
#define TENSORSTORE_INTERNAL_CACHE_CACHE_IMPL_H_

#ifndef TENSORSTORE_CACHE_REFCOUNT_DEBUG
#define TENSORSTORE_CACHE_REFCOUNT_DEBUG 0
#endif

// IWYU pragma: private, include "third_party/tensorstore/internal/cache/cache.h"

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/cache/cache_pool_limits.h"
#include "tensorstore/internal/container/heterogeneous_container.h"
#include "tensorstore/internal/intrusive_ptr.h"

namespace tensorstore {
namespace internal {
class Cache;
class CacheEntry;
class CachePool;
}  // namespace internal
namespace internal_cache {
using internal::Cache;
using internal::CacheEntry;
using internal::CachePool;
using internal::CachePoolLimits;

#define TENSORSTORE_INTERNAL_CACHE_DEBUG_REFCOUNT(method, p, new_count) \
  ABSL_LOG_IF(INFO, TENSORSTORE_CACHE_REFCOUNT_DEBUG)                   \
      << method << ": " << p << " -> " << (new_count) /**/

class Access;
class CacheImpl;
class CachePoolImpl;

struct LruListNode {
  LruListNode* next;
  LruListNode* prev;
};

class CacheEntryImpl;

// Weak reference state for a cache entry.
//
// This is stored in a separate heap allocation from the entry itself, in order
// to allow weak references to outlive the entry.
struct CacheEntryWeakState {
  // Number of weak references.  When non-zero, the least-significant bit (LSB)
  // of `entry->reference_count_` is set to 1.
  std::atomic<size_t> weak_references;

  // Mutex that protects access to `entry`.  If locked along with the cache
  // pool's mutex, this mutex must be locked first.
  absl::Mutex mutex;

  // Pointer to the entry for which this is a weak reference.
  CacheEntryImpl* entry;

  // Acquires an additional weak reference, assuming at least one is already
  // held.
  friend void intrusive_ptr_increment(CacheEntryWeakState* p) {
    [[maybe_unused]] auto old_count =
        p->weak_references.fetch_add(1, std::memory_order_relaxed);
    TENSORSTORE_INTERNAL_CACHE_DEBUG_REFCOUNT("CacheEntryWeakState:increment",
                                              p, old_count + 1);
  }

  // Releases a weak reference.
  friend void intrusive_ptr_decrement(CacheEntryWeakState* p);
};

using WeakPinnedCacheEntry = internal::IntrusivePtr<CacheEntryWeakState>;

// Acquires a weak reference to a cache entry.
//
// The caller must hold a strong reference already.
WeakPinnedCacheEntry AcquireWeakCacheEntryReference(CacheEntry* e);

class CacheEntryImpl : public internal_cache::LruListNode {
 public:
  CacheImpl* cache_;
  std::string key_;

  // Protects `num_bytes_`, `flags_`, and any other fields that reflect the
  // state of the cached data in derived classes.
  absl::Mutex mutex_;

  // Most recently computed value of `DoGetSizeInBytes` that is reflected in the
  // LRU cache state.
  size_t num_bytes_;

  // Each strong reference adds 2 to the reference count.  The least-significant
  // bit (LSB) indicates if there is at least one weak reference,
  // `weak_state_.load()->reference_count.load() > 0`.
  //
  // When the reference count is non-zero, the entry is considered "in-use" and
  // won't be evicted due to memory pressure.  In particular, `queue_state_` may
  // be `clean_and_not_in_use` if, and only if, `reference_count_` is 0.
  // Conversely, `queue_state_` may be `clean_and_in_use` if, and only if,
  // `reference_count_` is non-zero.
  std::atomic<uint32_t> reference_count_;

  // Guards calls to `DoInitializeEntry`.
  absl::once_flag initialized_;

  using Flags = uint8_t;

  // Bit vector of flags, see below.
  //
  // These must only be changed while holding `mutex_`.
  Flags flags_ = 0;

  // Set if the return value of `DoGetSizeInBytes` may have changed.
  constexpr static Flags kSizeChanged = 1;

  // Initially set to `nullptr`.  Allocated when the first weak reference is
  // obtained, and remains until the entry is destroyed even if all weak
  // references are released.
  std::atomic<CacheEntryWeakState*> weak_state_{nullptr};

  virtual ~CacheEntryImpl() = default;
};

class CacheImpl {
 public:
  CacheImpl();
  virtual ~CacheImpl();
  using Entry = CacheEntryImpl;

  CachePoolImpl* pool_;

  /// Stores the pointer to `this`, cast to the `CacheType` specified in
  /// `GetCache` when this cache was created.  This pointer needs to be stored
  /// because the address may not equal `this` in the case that `CacheType` is a
  /// sibling class rather than a subclass of `CacheImpl`.
  void* user_ptr_;

  /// Specifies the `cache_type_` to be used along with `cache_identifier_` for
  /// looking up this cache in the `caches_` table of the cache pool.  This
  /// should be equal to, or a base class of, the actual dynamic type of `this`.
  ///
  /// This holds `typeid(CacheType)`, where `CacheType` is the type specified to
  /// `GetCache`.  This needs to be stored because the `make_cache` function
  /// supplied to `GetCache` may actually return a derived cache type, and the
  /// same type needs to be used for both the initial lookup and the insertion
  /// into the `caches_` table.
  const std::type_info* cache_type_;

  /// If non-empty, this cache is stored in the `caches_` table of the cache
  /// pool, and should only be destroyed once:
  ///
  /// 1. `reference_count_` becomes zero; and
  ///
  /// 2. `entries_` becomes empty or `pool_->strong_references_` becomes zero.
  ///
  /// If empty, this cache is not stored in the `caches_` table of the cache
  /// pool, and is destroyed as soon as `reference_count_` becomes zero.
  std::string cache_identifier_;

  std::atomic<uint32_t> reference_count_;

  struct ABSL_CACHELINE_ALIGNED Shard {
    absl::Mutex mutex;
    internal::HeterogeneousHashSet<CacheEntryImpl*, std::string_view,
                                   &CacheEntryImpl::key_>
        entries ABSL_GUARDED_BY(mutex);
  };

  Shard shards_[8];

  Shard& ShardForKey(std::string_view key) {
    absl::Hash<std::string_view> h;
    return shards_[h(key) & 7];
  }

  // Key by which a cache may be looked up in a `CachePool`.
  using CacheKey = std::pair<std::type_index, std::string_view>;

  CacheKey cache_key() const { return {*cache_type_, cache_identifier_}; }

  friend class internal::CachePool;
};

class CachePoolImpl {
 public:
  explicit CachePoolImpl(const CachePoolLimits& limits);

  using CacheKey = CacheImpl::CacheKey;

  CachePoolLimits limits_;
  std::atomic<size_t> total_bytes_;

  // Protects access to `eviction_queue_`.  If `lru_mutex_` is held at the same
  // time as `caches_mutex_`, `caches_mutex_` must be acquired first.  If
  // `lru_mutex_` is held at the same time as `entries_mutex_`, `lru_mutex_`
  // must be acquired first.
  absl::Mutex lru_mutex_;

  // next points to the front of the queue, which is the first to be evicted.
  LruListNode eviction_queue_;

  // Protects access to `caches_`.
  absl::Mutex caches_mutex_;
  internal::HeterogeneousHashSet<CacheImpl*, CacheKey, &CacheImpl::cache_key>
      caches_;

  /// Initial strong reference returned when the cache pool is created.
  std::atomic<size_t> strong_references_;
  /// One weak reference is kept until strong_references_ becomes 0.
  std::atomic<size_t> weak_references_;
};

class Access {
 public:
  template <typename T, typename U>
  static T* StaticCast(U* p) {
    return static_cast<T*>(p);
  }
};

template <typename T>
internal::Cache& GetCacheObject(T* p) {
  if constexpr (std::is_base_of_v<internal::Cache, T>) {
    return *p;
  } else {
    return p->cache();
  }
}

struct StrongPtrTraitsCache {
  template <typename T>
  using pointer = T*;

  template <typename U>
  static void increment(U* p) noexcept {
    [[maybe_unused]] auto old_count =
        Access::StaticCast<CacheImpl>(&GetCacheObject(p))
            ->reference_count_.fetch_add(1, std::memory_order_relaxed);
    TENSORSTORE_INTERNAL_CACHE_DEBUG_REFCOUNT("Cache:increment", p,
                                              old_count + 1);
  }
  static void decrement(internal::Cache* p) noexcept;
  template <typename U>
  static void decrement(U* p) noexcept {
    decrement(&GetCacheObject(p));
  }
};

template <typename CacheType>
using CachePtr = internal::IntrusivePtr<CacheType, StrongPtrTraitsCache>;

struct StrongPtrTraitsCacheEntry {
  template <typename Entry>
  using pointer = Entry*;

  // Defined as a template because `internal::CacheEntry` is incomplete here.
  template <typename U = internal::CacheEntry*>
  static void increment(U* p) noexcept {
    [[maybe_unused]] auto old_count =
        Access::StaticCast<CacheEntryImpl>(p)->reference_count_.fetch_add(
            2, std::memory_order_relaxed);
    TENSORSTORE_INTERNAL_CACHE_DEBUG_REFCOUNT("CacheEntry:increment", p,
                                              old_count + 2);
  }

  static void decrement(internal::CacheEntry* p) noexcept;
};

using CacheEntryWeakPtr = internal::IntrusivePtr<CacheEntryWeakState>;

template <typename Entry>
using CacheEntryStrongPtr =
    internal::IntrusivePtr<Entry, StrongPtrTraitsCacheEntry>;

struct StrongPtrTraitsCachePool {
  template <typename>
  using pointer = CachePool*;
  static void increment(CachePool* p) noexcept;
  static void decrement(CachePool* p) noexcept;
};

struct WeakPtrTraitsCachePool {
  template <typename>
  using pointer = CachePool*;
  static void increment(CachePool* p) noexcept;
  static void decrement(CachePool* p) noexcept;
};

using CachePoolStrongPtr =
    internal::IntrusivePtr<CachePool, StrongPtrTraitsCachePool>;

using CachePoolWeakPtr =
    internal::IntrusivePtr<CachePool, WeakPtrTraitsCachePool>;

CachePtr<Cache> GetCacheInternal(
    CachePoolImpl* pool, const std::type_info& cache_type,
    std::string_view cache_key,
    absl::FunctionRef<std::unique_ptr<Cache>()> make_cache);

CacheEntryStrongPtr<CacheEntry> GetCacheEntryInternal(internal::Cache* cache,
                                                      std::string_view key);

inline bool HasLruCache(CachePoolImpl* pool) {
  return pool && pool->limits_.total_bytes_limit != 0;
}

void UpdateTotalBytes(CachePoolImpl& pool, ptrdiff_t change);

}  // namespace internal_cache
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_CACHE_IMPL_H_
