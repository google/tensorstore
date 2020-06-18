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

#ifndef TENSORSTORE_INTERNAL_CACHE_IMPL_H_
#define TENSORSTORE_INTERNAL_CACHE_IMPL_H_

// IWYU pragma: private, include "third_party/tensorstore/internal/cache.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "tensorstore/internal/cache_pool_limits.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/util/function_view.h"

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
  using Cache = CacheImpl;
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
  /// Helper type to support heterogeneous lookup using either a
  /// `CacheEntryImpl*` or an `absl::string_view`.
  class EntryKey : public absl::string_view {
    using Base = absl::string_view;

   public:
    using Base::Base;
    EntryKey(CacheEntryImpl* entry) : Base(entry->key_) {}
    EntryKey(absl::string_view key) : Base(key) {}
  };

  struct EntryKeyHash : public absl::Hash<EntryKey> {
    using is_transparent = void;
  };

  struct EntryKeyEqualTo : public std::equal_to<EntryKey> {
    using is_transparent = void;
  };

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

  absl::flat_hash_set<CacheEntryImpl*, EntryKeyHash, EntryKeyEqualTo> entries_;
};

class CachePoolImpl {
 public:
  explicit CachePoolImpl(const CachePoolLimits& limits);

  /// Key type used for looking up a Cache within a CachePool.
  class CacheKey : public std::pair<std::type_index, absl::string_view> {
    using Base = std::pair<std::type_index, absl::string_view>;

   public:
    using Base::Base;
    CacheKey(const CacheImpl* ptr)
        : Base(*ptr->cache_type_, ptr->cache_identifier_) {}
  };

  struct CacheKeyHash : public absl::Hash<CacheKey> {
    using is_transparent = void;
  };

  struct CacheKeyEqualTo : public std::equal_to<CacheKey> {
    using is_transparent = void;
  };

  /// Protects access to `total_bytes_`, `queued_for_writeback_bytes_`,
  /// `writeback_queue_`, `eviction_queue_`, `caches_`, and the `entries_` hash
  /// tables of all caches associated with this pool.
  Mutex mutex_;
  CachePoolLimits limits_;
  size_t total_bytes_;
  size_t queued_for_writeback_bytes_;
  LruListNode writeback_queue_;

  // next points to the front of the queue, which is the first to be evicted.
  LruListNode eviction_queue_;

  absl::flat_hash_set<CacheImpl*, CacheKeyHash, CacheKeyEqualTo> caches_;

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
    internal::IntrusivePtr<CachePoolImpl, StrongPtrTraitsCachePool>;

using CachePoolWeakPtr =
    internal::IntrusivePtr<CachePoolImpl, WeakPtrTraitsCachePool>;

CachePtr<Cache> GetCacheInternal(
    CachePoolImpl* pool, const std::type_info& cache_type,
    absl::string_view cache_key,
    FunctionView<std::unique_ptr<Cache>()> make_cache);

CacheEntryStrongPtr<CacheEntry> GetCacheEntryInternal(internal::Cache* cache,
                                                      absl::string_view key);

}  // namespace internal_cache
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_IMPL_H_
