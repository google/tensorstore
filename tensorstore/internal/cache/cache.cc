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

#include "tensorstore/internal/cache/cache.h"

#include <atomic>
#include <cassert>
#include <memory>
#include <mutex>  // NOLINT
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <typeindex>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/cache/cache_pool_limits.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/intrusive_linked_list.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/mutex.h"

// A CacheEntry owns a strong reference to the Cache that contains it only if
// its reference count is > 0.
//
// A Cache owns a weak reference to the CachePool that contains it only if its
// reference count is > 0.

namespace tensorstore {
namespace internal_cache {

auto& hit_count = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/cache/hit_count", "Number of cache hits.");
auto& miss_count = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/cache/miss_count", "Number of cache misses.");
auto& evict_count = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/cache/evict_count", "Number of evictions from the cache.");

using ::tensorstore::internal::PinnedCacheEntry;

#if !defined(NDEBUG)
inline void DebugAssertMutexHeld(absl::Mutex* mutex) { mutex->AssertHeld(); }
#else
inline void DebugAssertMutexHeld(absl::Mutex* mutex) {}
#endif

using LruListAccessor =
    internal::intrusive_linked_list::MemberAccessor<LruListNode>;

CachePoolImpl::CachePoolImpl(const CachePool::Limits& limits)
    : limits_(limits),
      total_bytes_(0),
      queued_for_writeback_bytes_(0),
      strong_references_(1),
      weak_references_(1) {
  Initialize(LruListAccessor{}, &writeback_queue_);
  Initialize(LruListAccessor{}, &eviction_queue_);
}

namespace {
inline void AcquireWeakReference(CachePoolImpl* p) {
  p->weak_references_.fetch_add(1, std::memory_order_relaxed);
}
void ReleaseWeakReference(CachePoolImpl* p) {
  if (--p->weak_references_ == 0) {
    delete Access::StaticCast<CachePool>(p);
  }
}

inline CachePtr<Cache> AcquireCacheStrongPtr(CacheImpl* cache_impl) {
  if (cache_impl->reference_count_.fetch_add(1, std::memory_order_acq_rel) ==
      0) {
    // When the first reference to the cache is acquired (either when the cache
    // is created, or when it is retrieved from the caches_ table of the pool
    // after all prior references have been removed), also acquire a weak
    // reference to the pool to ensure the CachePoolImpl object is not destroyed
    // while a cache still references it.
    AcquireWeakReference(cache_impl->pool_);
  }
  return CachePtr<Cache>(Access::StaticCast<Cache>(cache_impl),
                         internal::adopt_object_ref);
}

void SetStateAndSize(CacheEntryImpl* entry, CacheEntryQueueState state,
                     size_t num_bytes) noexcept;

void UnlinkListNode(LruListNode* node) noexcept {
  Remove(LruListAccessor{}, node);
  Initialize(LruListAccessor{}, node);
}

void UnregisterEntryFromPool(CacheEntryImpl* entry,
                             CachePoolImpl* pool) noexcept {
  DebugAssertMutexHeld(&pool->mutex_);
  UnlinkListNode(entry);
  pool->total_bytes_ -= entry->num_bytes_;
  if (entry->queue_state_ == CacheEntryQueueState::dirty) {
    pool->queued_for_writeback_bytes_ -= entry->num_bytes_;
  }
}

void EvictEntry(CacheEntryImpl* entry) noexcept ABSL_NO_THREAD_SAFETY_ANALYSIS {
  evict_count.Increment();
  auto* pool = entry->cache_->pool_;
  DebugAssertMutexHeld(&pool->mutex_);
  UnregisterEntryFromPool(entry, pool);
  // Note: If this is being called from `GetCacheEntryInternal` because an
  // exception was thrown while inserting `entry` into
  // `entry->cache_->entries_`, `entry` won't be in `entry->caches_`.
  auto& entries = entry->cache_->entries_;
  entries.erase(entry);
  {
    // Hold a reference to `cache` before releasing the mutex to ensure `cache`
    // is not destroyed.
    CachePtr<Cache> cache = AcquireCacheStrongPtr(entry->cache_);
    internal::ScopedWriterUnlock unlock(pool->mutex_);
    delete Access::StaticCast<CacheEntry>(entry);
    // Remove reference to cache while mutex is unlocked.  This may cause the
    // cache to be destroyed.
    cache.reset();
  }
}

void EnsureNotOnCleanList(CacheEntryImpl* entry) noexcept {
  DebugAssertMutexHeld(&entry->cache_->pool_->mutex_);
  if (entry->queue_state_ == CacheEntryQueueState::clean_and_not_in_use) {
    UnlinkListNode(entry);
    entry->queue_state_ = CacheEntryQueueState::clean_and_in_use;
  }
}

void AddToEvictionQueue(CachePoolImpl* pool, CacheEntryImpl* entry) noexcept {
  DebugAssertMutexHeld(&pool->mutex_);
  auto* eviction_queue = &pool->eviction_queue_;
  InsertBefore(LruListAccessor{}, eviction_queue, entry);
}

void AddToWritebackQueue(CachePoolImpl* pool, CacheEntryImpl* entry) noexcept {
  DebugAssertMutexHeld(&pool->mutex_);
  auto* queue = &pool->writeback_queue_;
  InsertBefore(LruListAccessor{}, queue, entry);
}

void MaybeEvictEntries(CachePoolImpl* pool) noexcept {
  DebugAssertMutexHeld(&pool->mutex_);
  while (pool->total_bytes_ > pool->limits_.total_bytes_limit) {
    auto* queue = &pool->eviction_queue_;
    if (queue->next == queue) {
      // Queue empty.
      break;
    }
    auto* entry = static_cast<CacheEntryImpl*>(queue->next);
    EvictEntry(entry);
  }
}

void InitializeNewEntry(CacheEntryImpl* entry, CacheImpl* cache) noexcept {
  auto* pool = cache->pool_;
  entry->cache_ = cache;
  entry->reference_count_.store(1, std::memory_order_relaxed);
  entry->num_bytes_ = 0;
  entry->queue_state_ = CacheEntryQueueState::clean_and_in_use;
  pool->total_bytes_ += entry->num_bytes_;
  MaybeEvictEntries(pool);
  Initialize(LruListAccessor{}, entry);
}

void RequestWriteback(CachePoolImpl* pool, CacheEntryImpl* entry) {
  DebugAssertMutexHeld(&pool->mutex_);
  SetStateAndSize(entry, CacheEntryQueueState::writeback_requested,
                  entry->num_bytes_);
  // Acquire a reference to `entry` before releasing the mutex to ensure it
  // remains valid.
  StrongPtrTraitsCacheEntry::increment(Access::StaticCast<CacheEntry>(entry));
  internal::ScopedWriterUnlock unlock(pool->mutex_);
  // Ensure that the reference to `entry` is released while the mutex is not
  // held to avoid deadlock.
  Access::StaticCast<Cache>(entry->cache_)
      ->DoRequestWriteback(PinnedCacheEntry<Cache>(
          Access::StaticCast<Cache::Entry>(entry), internal::adopt_object_ref));
}

void MaybeWritebackEntries(CachePoolImpl* pool) {
  DebugAssertMutexHeld(&pool->mutex_);
  while (pool->queued_for_writeback_bytes_ >
         pool->limits_.queued_for_writeback_bytes_limit) {
    auto* queue = &pool->writeback_queue_;
    assert(queue->next != queue);
    auto* entry = static_cast<CacheEntryImpl*>(queue->next);
    RequestWriteback(pool, entry);
  }
}

void SetStateAndSize(CacheEntryImpl* entry, CacheEntryQueueState state,
                     size_t num_bytes) noexcept {
  CachePoolImpl* pool = entry->cache_->pool_;
  DebugAssertMutexHeld(&pool->mutex_);
  const CacheEntryQueueState old_state = entry->queue_state_;
  const size_t old_num_bytes = entry->num_bytes_;
  if (state == old_state && num_bytes == old_num_bytes) {
    // Nothing to do.
    return;
  }
  // Relies on unsigned overflow to do the right thing.
  pool->total_bytes_ += (num_bytes - old_num_bytes);

  if (old_state == CacheEntryQueueState::dirty) {
    pool->queued_for_writeback_bytes_ -= old_num_bytes;
  }

  UnlinkListNode(entry);
  entry->queue_state_ = state;
  entry->num_bytes_ = num_bytes;

  if (state == CacheEntryQueueState::clean_and_not_in_use) {
    AddToEvictionQueue(pool, entry);
    if (entry->evict_when_not_in_use_) {
      EvictEntry(entry);
    }
  } else if (state == CacheEntryQueueState::dirty) {
    AddToWritebackQueue(pool, entry);
    pool->queued_for_writeback_bytes_ += num_bytes;
    MaybeWritebackEntries(pool);
  }
  MaybeEvictEntries(pool);
}

void DestroyCache(CacheImpl* cache) noexcept {
  for (CacheEntryImpl* entry : cache->entries_) {
    delete Access::StaticCast<Cache::Entry>(entry);
  }
  delete Access::StaticCast<Cache>(cache);
}

/// Decrements `*reference_count` in such a way that it only reaches zero while
/// `mutex` is held.
///
/// If `*reference_count` was decremented to zero, returns a lock on `mutex`.
/// Otherwise, returns an unlocked `UniqueWriterLock`.
template <typename T>
inline UniqueWriterLock<absl::Mutex> DecrementReferenceCountWithLock(
    std::atomic<T>* reference_count, absl::Mutex& mutex) {
  // If the new reference count will be > 0, we can simply decrement it.
  // However, if the reference count will possibly become 0, we must lock the
  // mutex before decrementing it to ensure that another thread doesn't
  // concurrently obtain another reference.
  if (internal::DecrementReferenceCountIfGreaterThanOne(*reference_count)) {
    return {};
  }

  // Handle the case of the reference_count possibly becoming 0.

  UniqueWriterLock lock(mutex);
  // Reference count may have changed between the time at which we last
  // checked it and the time at which we acquired the mutex.
  if (reference_count->fetch_sub(1, std::memory_order_acq_rel) != 1) {
    // Reference count has changed, we didn't bring the count to zero.
    return {};
  }
  return lock;
}

}  // namespace

void StrongPtrTraitsCacheEntry::decrement(CacheEntry* p) noexcept {
  auto* entry = Access::StaticCast<CacheEntryImpl>(p);
  auto* cache = entry->cache_;
  {
    auto lock = DecrementReferenceCountWithLock(&entry->reference_count_,
                                                cache->pool_->mutex_);
    if (!lock) return;
    if (entry->queue_state_ == CacheEntryQueueState::clean_and_in_use) {
      SetStateAndSize(entry, CacheEntryQueueState::clean_and_not_in_use,
                      entry->num_bytes_);
      // `entry` may not be valid at this point.
    }
  }
  StrongPtrTraitsCache::decrement(Access::StaticCast<Cache>(cache));
}

CachePtr<Cache> GetCacheInternal(
    CachePoolImpl* pool, const std::type_info& cache_type,
    std::string_view cache_key,
    absl::FunctionRef<std::unique_ptr<Cache>()> make_cache) {
  CachePoolImpl::CacheKey key(cache_type, cache_key);
  if (!cache_key.empty()) {
    // An non-empty key indicates to look for an existing cache.
    absl::MutexLock lock(&pool->mutex_);
    auto it = pool->caches_.find(key);
    if (it != pool->caches_.end()) {
      return AcquireCacheStrongPtr(*it);
    }
  }
  // No existing cache, create a new one with the pool mutex unlocked.
  std::unique_ptr<Cache> new_cache = make_cache();
  if (!new_cache) return CachePtr<Cache>();
  auto* cache_impl = Access::StaticCast<CacheImpl>(new_cache.get());
  cache_impl->pool_ = pool;
  // An empty key indicates not to store the Cache in the map.
  if (cache_key.empty()) {
    new_cache.release();
    return AcquireCacheStrongPtr(cache_impl);
  }
  cache_impl->cache_type_ = &cache_type;
  cache_impl->cache_identifier_ = std::string(cache_key);
  absl::MutexLock lock(&pool->mutex_);
  auto insert_result = pool->caches_.insert(cache_impl);
  if (insert_result.second) {
    new_cache.release();
  }
  return AcquireCacheStrongPtr(
      Access::StaticCast<CacheImpl>(*insert_result.first));
}

PinnedCacheEntry<Cache> GetCacheEntryInternal(internal::Cache* cache,
                                              std::string_view key) {
  auto* cache_impl = Access::StaticCast<CacheImpl>(cache);
  PinnedCacheEntry<Cache> returned_entry;
  {
    absl::MutexLock lock(&cache_impl->pool_->mutex_);
    auto it = cache_impl->entries_.find(key);
    if (it != cache_impl->entries_.end()) {
      hit_count.Increment();
      auto* entry_impl = *it;
      if (entry_impl->reference_count_.fetch_add(
              1, std::memory_order_acq_rel) == 0) {
        // When the first reference to an entry is acquired, also acquire a
        // strong reference to the cache to be held by the entry.  This ensures
        // the Cache object is not destroyed while any of its entries are
        // referenced.
        StrongPtrTraitsCache::increment(cache);
        EnsureNotOnCleanList(entry_impl);
      }
      // Adopt reference added via `fetch_add` above.
      returned_entry =
          PinnedCacheEntry<Cache>(Access::StaticCast<Cache::Entry>(entry_impl),
                                  internal::adopt_object_ref);
    } else {
      miss_count.Increment();
      std::string temp_key(key);  // May throw, done before allocating entry.
      auto* entry_impl =
          Access::StaticCast<CacheEntryImpl>(cache->DoAllocateEntry());
      entry_impl->key_ = std::move(temp_key);      // noexcept
      InitializeNewEntry(entry_impl, cache_impl);  // noexcept
      struct EvictEntryDeleter {
        void operator()(CacheEntry* entry) const noexcept {
          EvictEntry(Access::StaticCast<CacheEntryImpl>(entry));
        }
      };
      std::unique_ptr<CacheEntry, EvictEntryDeleter> entry(
          Access::StaticCast<CacheEntry>(entry_impl));
      // Add to entries table.  This may throw, in which case the entry will be
      // cleaned up by `EvictEntry`.
      cache_impl->entries_.insert(entry_impl);
      StrongPtrTraitsCache::increment(cache);
      returned_entry =
          PinnedCacheEntry<Cache>(entry.release(), internal::adopt_object_ref);
    }
  }
  absl::call_once(
      Access::StaticCast<CacheEntryImpl>(returned_entry.get())->initialized_,
      [&] {
        returned_entry->DoInitialize();
        // This is the only call to `DoGetSizeInBytes` made by this cache
        // framework directly.  Because `entry` is not yet visible to other
        // threads, it is safe to call `DoGetSizeInBytes` without holding any
        // locks.  All other calls are made by derived classes while holding any
        // relevant locks on the portions of the entry state that the derived
        // implementation of DoGetSizeInBytes may access.
        CacheEntry::StateUpdate state_update;
        state_update.new_size = cache->DoGetSizeInBytes(returned_entry.get());
        returned_entry->UpdateState(std::move(state_update));
      });
  return returned_entry;
}

void StrongPtrTraitsCache::decrement(Cache* p) noexcept {
  auto* cache = Access::StaticCast<CacheImpl>(p);
  auto* pool = cache->pool_;
  auto lock = DecrementReferenceCountWithLock(&cache->reference_count_,
                                              cache->pool_->mutex_);
  if (!lock) return;
  const bool owned_by_pool = !cache->cache_identifier_.empty();

  if (!owned_by_pool ||
      pool->strong_references_.load(std::memory_order_acquire) == 0) {
    if (owned_by_pool) {
      pool->caches_.erase(cache);
    }
    // This cache has no identifier, or the CachePool has no strong references
    // currently.  Destroy it and all of its entries.
    for (CacheEntryImpl* entry : cache->entries_) {
      UnregisterEntryFromPool(entry, pool);
    }

    lock.unlock();
    DestroyCache(cache);
    ReleaseWeakReference(pool);
    return;
  }

  if (cache->entries_.empty()) {
    // The cache contains no entries.  Remove it from the pool's table of
    // caches, and destroy it.
    pool->caches_.erase(cache);
  } else {
    // The cache still contains entries.  We keep it alive until those entries
    // are evicted (at which point this function will be called again when
    // EvictEntry removes the temporary cache reference that it holds).
    cache = nullptr;
  }
  lock.unlock();
  delete cache;
  ReleaseWeakReference(pool);
}

CacheImpl::CacheImpl() : pool_(nullptr), reference_count_(0) {}
CacheImpl::~CacheImpl() = default;

void StrongPtrTraitsCachePool::increment(CachePool* p) noexcept {
  auto* pool = Access::StaticCast<CachePoolImpl>(p);
  if (pool->strong_references_.fetch_add(1, std::memory_order_acq_rel) == 0) {
    AcquireWeakReference(Access::StaticCast<CachePoolImpl>(p));
  }
}

void StrongPtrTraitsCachePool::decrement(CachePool* p) noexcept {
  auto* pool = Access::StaticCast<CachePoolImpl>(p);
  auto lock =
      DecrementReferenceCountWithLock(&pool->strong_references_, pool->mutex_);
  if (!lock) return;
  std::vector<CachePtr<Cache>> caches;
  caches.reserve(pool->caches_.size());
  for (auto* cache : pool->caches_) {
    // Acquire a reference to ensure reference count is non-zero when
    // `strong_references_` becomes zero, since once `strong_references_` is
    // zero a cache reference count of 0 implies the cache should already have
    // been destroyed.
    caches.push_back(AcquireCacheStrongPtr(cache));
  }
  lock.unlock();
  ReleaseWeakReference(pool);
}

void WeakPtrTraitsCachePool::increment(CachePool* p) noexcept {
  AcquireWeakReference(Access::StaticCast<CachePoolImpl>(p));
}

void WeakPtrTraitsCachePool::decrement(CachePool* p) noexcept {
  ReleaseWeakReference(Access::StaticCast<CachePoolImpl>(p));
}

}  // namespace internal_cache

namespace internal {

Cache::Cache() = default;
Cache::~Cache() = default;

std::size_t Cache::DoGetSizeInBytes(Cache::Entry* entry) {
  return ((internal_cache::CacheEntryImpl*)entry)->key_.capacity() +
         this->DoGetSizeofEntry();
}

CacheEntry::~CacheEntry() = default;

void CacheEntry::DoInitialize() {}

void Cache::Entry::UpdateState(StateUpdate update) {
  if (!update.new_state && !update.new_size) return;
  auto* cache_impl =
      internal_cache::Access::StaticCast<internal_cache::CacheImpl>(cache_);
  auto* pool = cache_impl->pool_;
  // Acquire `pool->mutex_` and then release `update.lock`.
  UniqueWriterLock lock(pool->mutex_);
  update.lock = nullptr;
  std::size_t old_num_bytes = num_bytes_;
  std::size_t new_num_bytes = update.new_size.value_or(old_num_bytes);
  if (update.new_state) {
    internal_cache::SetStateAndSize(this, *update.new_state, new_num_bytes);
    return;
  }
  // Just update the size, without affecting the queue position.
  if (old_num_bytes == new_num_bytes) {
    return;
  }
  num_bytes_ = new_num_bytes;
  std::size_t num_bytes_change =
      wrap_on_overflow::Subtract(new_num_bytes, old_num_bytes);
  pool->total_bytes_ += num_bytes_change;
  if (queue_state_ == CacheEntryQueueState::dirty) {
    pool->queued_for_writeback_bytes_ += num_bytes_change;
    if (new_num_bytes > old_num_bytes) {
      internal_cache::MaybeWritebackEntries(pool);
    }
  }
  if (new_num_bytes > old_num_bytes) {
    internal_cache::MaybeEvictEntries(pool);
  }
}

std::ostream& operator<<(std::ostream& os, CacheEntryQueueState state) {
  switch (state) {
    case CacheEntryQueueState::clean_and_not_in_use:
      return os << "clean_and_not_in_use";
    case CacheEntryQueueState::clean_and_in_use:
      return os << "clean_and_in_use";
    case CacheEntryQueueState::dirty:
      return os << "dirty";
    case CacheEntryQueueState::writeback_requested:
      return os << "writeback_requested";
    default:
      return os << "<unknown>";
  }
}

CachePool::StrongPtr CachePool::Make(const CachePool::Limits& cache_limits) {
  CachePool::StrongPtr pool;
  internal_cache::Access::StaticCast<internal_cache::CachePoolStrongPtr>(&pool)
      ->reset(new internal_cache::CachePool(cache_limits), adopt_object_ref);
  return pool;
}

CachePool::StrongPtr::StrongPtr(const CachePool::WeakPtr& ptr)
    : Base(ptr.get(), adopt_object_ref) {
  if (!ptr) return;
  auto* pool =
      internal_cache::Access::StaticCast<internal_cache::CachePoolImpl>(
          ptr.get());
  absl::MutexLock lock(&pool->mutex_);
  if (pool->strong_references_.fetch_add(1, std::memory_order_acq_rel) == 0) {
    internal_cache::AcquireWeakReference(pool);
  }
}

}  // namespace internal

}  // namespace tensorstore
