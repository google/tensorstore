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

// IWYU pragma: private, include "third_party/tensorstore/internal/cache/cache.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <typeindex>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/functional/function_ref.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/cache/cache_pool_limits.h"
#include "tensorstore/internal/heterogeneous_container.h"
#include "tensorstore/internal/intrusive_ptr.h"

namespace tensorstore {
namespace internal {
class Cache;
class CacheEntry;
class CachePool;
enum class CacheEntryQueueState : int;
}  // namespace internal
namespace internal_cache {
using internal::Cache;
using internal::CacheEntry;
using internal::CachePool;
using internal::CachePoolLimits;

class Access;
class CacheImpl;
class CachePoolImpl;

using CacheEntryQueueState = internal::CacheEntryQueueState;

struct LruListNode {
  LruListNode* next;
  LruListNode* prev;
};

class CacheEntryImpl : public internal_cache::LruListNode {
 public:
  CacheImpl* cache_;
  std::string key_;
  size_t num_bytes_;
  CacheEntryQueueState queue_state_;
  bool evict_when_not_in_use_ = false;
  std::atomic<std::uint32_t> reference_count_;
  // Guards calls to `DoInitializeEntry`.
  absl::once_flag initialized_;
};

class CacheImpl {
 public:
  CacheImpl();
  virtual ~CacheImpl();
  using Entry = CacheEntryImpl;

  CachePoolImpl* pool_;

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

  std::atomic<std::uint32_t> reference_count_;

  internal::HeterogeneousHashSet<CacheEntryImpl*, std::string_view,
                                 &CacheEntryImpl::key_>
      entries_;

  // Key by which a cache may be looked up in a `CachePool`.
  using CacheKey = std::pair<std::type_index, std::string_view>;

  CacheKey cache_key() const { return {*cache_type_, cache_identifier_}; }
};

class CachePoolImpl {
 public:
  explicit CachePoolImpl(const CachePoolLimits& limits);

  using CacheKey = CacheImpl::CacheKey;

  /// Protects access to `total_bytes_`, `queued_for_writeback_bytes_`,
  /// `writeback_queue_`, `eviction_queue_`, `caches_`, and the `entries_` hash
  /// tables of all caches associated with this pool.
  absl::Mutex mutex_;
  CachePoolLimits limits_;
  size_t total_bytes_;
  size_t queued_for_writeback_bytes_;
  LruListNode writeback_queue_;

  // next points to the front of the queue, which is the first to be evicted.
  LruListNode eviction_queue_;

  internal::HeterogeneousHashSet<CacheImpl*, CacheKey, &CacheImpl::cache_key>
      caches_;

  /// Initial strong reference returned when the cache is created.
  std::atomic<std::size_t> strong_references_;
  /// One weak reference is kept until strong_references_ becomes 0.
  std::atomic<std::size_t> weak_references_;
};

class Access {
 public:
  template <typename T, typename U>
  static T* StaticCast(U* p) {
    return static_cast<T*>(p);
  }
};

struct StrongPtrTraitsCache {
  template <typename T>
  using pointer = T*;

  template <typename U = internal::Cache>
  static void increment(U* p) noexcept {
    Access::StaticCast<CacheImpl>(p)->reference_count_.fetch_add(
        1, std::memory_order_relaxed);
  }
  static void decrement(internal::Cache* p) noexcept;
};

template <typename CacheType>
using CachePtr = internal::IntrusivePtr<CacheType, StrongPtrTraitsCache>;

struct StrongPtrTraitsCacheEntry {
  template <typename Entry>
  using pointer = Entry*;

  // Defined as a template because `internal::CacheEntry` is incomplete here.
  template <typename U = internal::CacheEntry*>
  static void increment(U* p) noexcept {
    Access::StaticCast<CacheEntryImpl>(p)->reference_count_.fetch_add(
        1, std::memory_order_relaxed);
  }

  static void decrement(internal::CacheEntry* p) noexcept;
};

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

}  // namespace internal_cache
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_CACHE_IMPL_H_
