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

#include <stddef.h>

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/testing/concurrent.h"

namespace {

using ::tensorstore::UniqueWriterLock;
using ::tensorstore::internal::Cache;
using ::tensorstore::internal::CachePool;
using ::tensorstore::internal::CachePtr;
using ::tensorstore::internal::GetCache;
using ::tensorstore::internal::PinnedCacheEntry;
using ::tensorstore::internal::WeakPinnedCacheEntry;
using ::tensorstore::internal_cache::Access;
using ::tensorstore::internal_cache::CacheEntryImpl;
using ::tensorstore::internal_cache::CacheImpl;
using ::tensorstore::internal_cache::CachePoolImpl;
using ::tensorstore::internal_cache::LruListNode;
using ::tensorstore::internal_testing::TestConcurrent;
using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

constexpr CachePool::Limits kSmallCacheLimits{10000000};

CachePoolImpl* GetPoolImpl(const CachePool::StrongPtr& ptr) {
  return Access::StaticCast<CachePoolImpl>(ptr.get());
}

CachePoolImpl* GetPoolImpl(const CachePool::WeakPtr& ptr) {
  return Access::StaticCast<CachePoolImpl>(ptr.get());
}

class TestCache : public Cache {
 public:
  class Entry : public Cache::Entry {
   public:
    using OwningCache = TestCache;

    std::string data;
    size_t size = 1;

    void ChangeSize(size_t new_size) {
      UniqueWriterLock<Cache::Entry> lock(*this);
      size = new_size;
      NotifySizeChanged();
    }

    // Set by some tests.
    WeakPinnedCacheEntry weak_ref;

    ~Entry() override { GetOwningCache(*this).OnDelete(this); }
  };

  struct RequestLog {
    absl::Mutex mutex;
    // Log of calls to DoAllocateEntry.  Only contains the cache key, not the
    // entry key.
    std::deque<std::string> entry_allocate_log;
    // Log of calls to DoDeleteEntry.  Contains the cache key and entry key.
    std::deque<std::pair<std::string, std::string>> entry_destroy_log;
    // Log of calls to GetTestCache (defined below).  ontains the cache key.
    std::deque<std::string> cache_allocate_log;
    // Log of calls to TestCache destructor.  Contains the cache key.
    std::deque<std::string> cache_destroy_log;
  };

  explicit TestCache(std::shared_ptr<RequestLog> log = {}) : log_(log) {}

  ~TestCache() {
    if (log_) {
      absl::MutexLock lock(&log_->mutex);
      log_->cache_destroy_log.emplace_back(cache_identifier());
    }
  }

  size_t DoGetSizeofEntry() override { return sizeof(Entry); }

  Entry* DoAllocateEntry() override {
    if (log_) {
      absl::MutexLock lock(&log_->mutex);
      log_->entry_allocate_log.emplace_back(cache_identifier());
    }
    return new Entry;
  }

  void OnDelete(Entry* entry) {
    if (log_) {
      absl::MutexLock lock(&log_->mutex);
      log_->entry_destroy_log.emplace_back(std::string(cache_identifier()),
                                           std::string(entry->key()));
    }
  }

  size_t DoGetSizeInBytes(Cache::Entry* base_entry) override {
    auto* entry = static_cast<Entry*>(base_entry);
    return entry->size;
  }

  std::shared_ptr<RequestLog> log_;
};

class TestCacheWithCachePool : public TestCache {
 public:
  using TestCache::TestCache;

  CachePool::WeakPtr cache_pool;
};

using EntryIdentifier = std::pair<std::string, void*>;

std::pair<std::string, void*> GetEntryIdentifier(CacheEntryImpl* entry) {
  return {entry->key_, entry};
}

absl::flat_hash_set<EntryIdentifier> GetEntrySet(LruListNode* head) {
  absl::flat_hash_set<EntryIdentifier> entries;
  for (LruListNode* node = head->next; node != head; node = node->next) {
    entries.emplace(
        GetEntryIdentifier(Access::StaticCast<CacheEntryImpl>(node)));
  }
  return entries;
}

// Check the invariants of pool, which should contain the specified caches.
void AssertInvariants(const CachePool::StrongPtr& pool,
                      absl::flat_hash_set<Cache*> expected_caches)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  auto* pool_impl = GetPoolImpl(pool);
  auto eviction_queue_entries = GetEntrySet(&pool_impl->eviction_queue_);

  absl::flat_hash_set<EntryIdentifier> expected_eviction_queue_entries;

  size_t expected_total_bytes = 0;

  // Verify that every cache owned by the pool is in `expected_caches`.
  for (auto* cache : pool_impl->caches_) {
    EXPECT_EQ(pool_impl, cache->pool_);
    EXPECT_NE("", cache->cache_identifier_);
    EXPECT_EQ(1, expected_caches.count(Access::StaticCast<Cache>(cache)));
  }

  EXPECT_EQ(1 + expected_caches.size(), pool_impl->weak_references_.load());

  for (auto* cache : expected_caches) {
    auto* cache_impl = Access::StaticCast<CacheImpl>(cache);
    if (!cache_impl->cache_identifier_.empty()) {
      auto it = pool_impl->caches_.find(cache_impl);
      ASSERT_NE(it, pool_impl->caches_.end());
      EXPECT_EQ(cache_impl, *it);
    }

    if (pool_impl->limits_.total_bytes_limit != 0) {
      for (auto& shard : cache_impl->shards_) {
        // absl::MutexLock lock(&shard.mutex);
        for (CacheEntryImpl* entry : shard.entries) {
          EXPECT_EQ(
              entry->num_bytes_,
              cache->DoGetSizeInBytes(Access::StaticCast<Cache::Entry>(entry)));
          expected_total_bytes += entry->num_bytes_;
          if (entry->reference_count_.load() == 0) {
            expected_eviction_queue_entries.emplace(GetEntryIdentifier(entry));
          }
        }
      }
    }
  }

  EXPECT_EQ(expected_total_bytes, pool_impl->total_bytes_);

  EXPECT_THAT(expected_eviction_queue_entries,
              ::testing::IsSubsetOf(eviction_queue_entries));
}

/// Wrapper around `AssertInvariants` that adds a SCOPED_TRACE to improve error
/// reporting.
#define TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(...) \
  do {                                                    \
    SCOPED_TRACE("");                                     \
    AssertInvariants(__VA_ARGS__);                        \
  } while (false)

template <typename CacheType = TestCache>
CachePtr<CacheType> GetTestCache(
    CachePool* pool, std::string cache_identifier,
    std::shared_ptr<TestCache::RequestLog> log = {}) {
  return GetCache<CacheType>(pool, cache_identifier, [&] {
    if (log) {
      absl::MutexLock lock(&log->mutex);
      log->cache_allocate_log.emplace_back(cache_identifier);
    }
    return std::make_unique<CacheType>(log);
  });
}

// Tests that specifying an empty `cache_key` leads to a cache not included in
// the cache pool's table of caches.
TEST(CachePoolTest, GetCacheEmptyKey) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  auto log = std::make_shared<TestCache::RequestLog>();
  {
    auto test_cache1 = GetTestCache(pool.get(), "", log);
    EXPECT_THAT(log->cache_allocate_log, ElementsAre(""));
    EXPECT_THAT(log->cache_destroy_log, ElementsAre());
    auto test_cache2 = GetTestCache(pool.get(), "", log);
    EXPECT_THAT(log->cache_allocate_log, ElementsAre("", ""));
    EXPECT_THAT(log->cache_destroy_log, ElementsAre());
    EXPECT_NE(test_cache1, test_cache2);
  }
  EXPECT_THAT(log->cache_allocate_log, ElementsAre("", ""));  // No change
  EXPECT_THAT(log->cache_destroy_log, ElementsAre("", ""));
}

TEST(CachePoolTest, GetCacheEmptyKeyCacheDisabled) {
  auto log = std::make_shared<TestCache::RequestLog>();
  {
    auto test_cache1 = GetTestCache(nullptr, "", log);
    EXPECT_THAT(log->cache_allocate_log, ElementsAre(""));
    EXPECT_THAT(log->cache_destroy_log, ElementsAre());
    auto test_cache2 = GetTestCache(nullptr, "", log);
    EXPECT_THAT(log->cache_allocate_log, ElementsAre("", ""));
    EXPECT_THAT(log->cache_destroy_log, ElementsAre());
    EXPECT_NE(test_cache1, test_cache2);
  }
  EXPECT_THAT(log->cache_allocate_log, ElementsAre("", ""));  // No change
  EXPECT_THAT(log->cache_destroy_log, ElementsAre("", ""));
}

// Tests that specifying a non-empty `cache_key` leads to a cache included in
// the cache pool's table of caches.
TEST(CachePoolTest, GetCacheNonEmptyKey) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  auto log = std::make_shared<TestCache::RequestLog>();
  {
    auto test_cache1 = GetTestCache(pool.get(), "x", log);
    EXPECT_THAT(log->cache_allocate_log, ElementsAre("x"));
    auto test_cache2 = GetTestCache(pool.get(), "x", log);
    EXPECT_THAT(log->cache_allocate_log, ElementsAre("x"));  // No change
    EXPECT_EQ(test_cache1, test_cache2);
  }
  EXPECT_THAT(log->cache_destroy_log, ElementsAre("x"));
}

// Tests that getting a cache when using a null pool returns a unique cache.
TEST(CachePoolTest, GetCacheNonEmptyKeyCacheDisabled) {
  auto log = std::make_shared<TestCache::RequestLog>();
  {
    auto test_cache1 = GetTestCache(nullptr, "x", log);
    EXPECT_THAT(log->cache_allocate_log, ElementsAre("x"));
    auto test_cache2 = GetTestCache(nullptr, "x", log);
    EXPECT_THAT(log->cache_allocate_log, ElementsAre("x", "x"));
    EXPECT_NE(test_cache1, test_cache2);
  }
  EXPECT_THAT(log->cache_destroy_log, ElementsAre("", ""));
}

// Tests that if `make_cache` returns `nullptr`, the cache is not retained.
TEST(CachePoolTest, GetCacheNullptr) {
  auto pool = CachePool::Make(CachePool::Limits{10000});
  int make_cache_calls = 0;
  auto make_cache = [&] {
    ++make_cache_calls;
    return nullptr;
  };
  {
    auto cache = GetCache<TestCache>(pool.get(), "x", make_cache);
    EXPECT_EQ(nullptr, cache);
    EXPECT_EQ(1, make_cache_calls);
  }
  {
    auto cache = GetCache<TestCache>(pool.get(), "x", make_cache);
    EXPECT_EQ(nullptr, cache);
    EXPECT_EQ(2, make_cache_calls);
  }
}

TEST(CachePoolTest, GetCacheNonEmptyKeyNoReferences) {
  auto pool = CachePool::Make(CachePool::Limits{});
  auto log = std::make_shared<TestCache::RequestLog>();
  EXPECT_EQ(1, GetPoolImpl(pool)->weak_references_.load());
  EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
  // Tests that copying the CachePool object increments the strong reference
  // count, but not the weak reference count.
  {
    auto pool2 = pool;
    EXPECT_EQ(1, GetPoolImpl(pool)->weak_references_.load());
    EXPECT_EQ(2, GetPoolImpl(pool)->strong_references_.load());
  }
  // Tests that releasing all direct references to a named cache with no entries
  // leads to the cache being destroyed.
  {
    auto test_cache1 = GetTestCache(pool.get(), "x", log);
    EXPECT_THAT(log->cache_allocate_log, ElementsAre("x"));
    EXPECT_EQ(1, GetPoolImpl(pool)->caches_.size());
    EXPECT_EQ(2, GetPoolImpl(pool)->weak_references_.load());
    EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
    EXPECT_EQ(1, test_cache1->use_count());
  }
  EXPECT_THAT(log->cache_destroy_log, ElementsAre("x"));
  EXPECT_EQ(1, GetPoolImpl(pool)->weak_references_.load());
  EXPECT_EQ(0, GetPoolImpl(pool)->caches_.size());
}

TEST(CachePoolTest, StrongToWeakToStrong) {
  CachePool::StrongPtr strong_ptr = CachePool::Make({});
  CachePool::WeakPtr weak_ptr(strong_ptr);
  strong_ptr = CachePool::StrongPtr();
  strong_ptr = CachePool::StrongPtr(weak_ptr);
  weak_ptr = CachePool::WeakPtr();
}

class NamedOrAnonymousCacheTest : public ::testing::TestWithParam<const char*> {
 public:
  std::shared_ptr<TestCache::RequestLog> log =
      std::make_shared<TestCache::RequestLog>();
  std::string cache_key = GetParam();
  CachePtr<TestCache> GetCache(const CachePool::StrongPtr& pool) {
    return GetTestCache(pool.get(), cache_key, log);
  }
};

INSTANTIATE_TEST_SUITE_P(WithoutCacheKey, NamedOrAnonymousCacheTest,
                         ::testing::Values(""));
INSTANTIATE_TEST_SUITE_P(WithCacheKey, NamedOrAnonymousCacheTest,
                         ::testing::Values("k"));

// Tests that a cache entry that outlives the last reference to a cache keeps
// the cache alive.
TEST_P(NamedOrAnonymousCacheTest, CacheEntryKeepsCacheAlive) {
  {
    PinnedCacheEntry<TestCache> entry;
    {
      auto pool = CachePool::Make(CachePool::Limits{});
      auto test_cache = GetCache(pool);
      EXPECT_THAT(log->cache_allocate_log, ElementsAre(cache_key));
      entry = GetCacheEntry(test_cache, "a");
      EXPECT_THAT(log->entry_allocate_log,
                  ElementsAre(cache_key));  // No change
    }
    EXPECT_EQ(1, GetOwningCache(*entry).use_count());
    EXPECT_THAT(log->entry_destroy_log, ElementsAre());
    EXPECT_THAT(log->cache_destroy_log, ElementsAre());
  }
  EXPECT_THAT(log->entry_destroy_log, ElementsAre(Pair(cache_key, "a")));
  EXPECT_THAT(log->cache_destroy_log, ElementsAre(cache_key));
}

// Tests that an unpinned entry is destroyed immediately when using limits of
// `CachePool::Limits{}`.
TEST_P(NamedOrAnonymousCacheTest, GetWithImmediateEvict) {
  auto pool = CachePool::Make(CachePool::Limits{});
  auto test_cache = GetCache(pool);
  EXPECT_EQ(1, test_cache->use_count());
  {
    auto e = GetCacheEntry(test_cache, "a");
    EXPECT_EQ(2, test_cache->use_count());
    EXPECT_THAT(log->entry_allocate_log, ElementsAre(cache_key));
    e->data = "value";
    EXPECT_EQ(1, e->use_count());
    {
      auto e2 = GetCacheEntry(test_cache, "a");
      EXPECT_THAT(log->entry_allocate_log,
                  ElementsAre(cache_key));  // No change
      EXPECT_EQ(2, test_cache->use_count());
      EXPECT_EQ(2, e2->use_count());
      EXPECT_EQ(e, e2);
      TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
    }
    EXPECT_EQ(1, e->use_count());
    EXPECT_EQ(2, test_cache->use_count());
  }
  EXPECT_THAT(log->entry_destroy_log, ElementsAre(Pair(cache_key, "a")));
  EXPECT_EQ(1, test_cache->use_count());
  TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});

  {
    auto e = GetCacheEntry(test_cache, "a");
    EXPECT_THAT(log->entry_allocate_log, ElementsAre(cache_key, cache_key));
    EXPECT_EQ("", e->data);
    TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
  }
  EXPECT_THAT(log->entry_destroy_log,
              ElementsAre(Pair(cache_key, "a"), Pair(cache_key, "a")));
  TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
}

// Tests that an unpinned entry is not destroyed immediately when using the
// default limits.
TEST_P(NamedOrAnonymousCacheTest, GetWithoutImmediateEvict) {
  {
    auto pool = CachePool::Make(kSmallCacheLimits);
    auto test_cache = GetCache(pool);
    {
      auto e = GetCacheEntry(test_cache, "a");
      EXPECT_THAT(log->entry_allocate_log, ElementsAre(cache_key));
      e->data = "value";
      auto e2 = GetCacheEntry(test_cache, "a");
      EXPECT_THAT(log->entry_allocate_log,
                  ElementsAre(cache_key));  // No change
      EXPECT_EQ(e, e2);
      TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
    }
    EXPECT_THAT(log->entry_destroy_log, ElementsAre());
    TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});

    {
      auto e = GetCacheEntry(test_cache, "a");
      EXPECT_THAT(log->entry_allocate_log,
                  ElementsAre(cache_key));  // No change
      EXPECT_EQ("value", e->data);
      TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
    }
    TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});

    {
      auto e1 = GetCacheEntry(test_cache, "a");
      EXPECT_THAT(log->entry_allocate_log,
                  ElementsAre(cache_key));  // No change
      EXPECT_EQ("value", e1->data);

      auto e2 = GetCacheEntry(test_cache, "b");
      EXPECT_THAT(log->entry_allocate_log, ElementsAre(cache_key, cache_key));
      e2->data = "value2";
      TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
    }

    {
      auto e1 = GetCacheEntry(test_cache, "a");
      EXPECT_THAT(log->entry_allocate_log,
                  ElementsAre(cache_key, cache_key));  // No change
      EXPECT_EQ("value", e1->data);
      TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
    }

    {
      auto e2 = GetCacheEntry(test_cache, "b");
      EXPECT_THAT(log->entry_allocate_log,
                  ElementsAre(cache_key, cache_key));  // No change
      EXPECT_EQ("value2", e2->data);
      TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
    }
    EXPECT_THAT(log->entry_destroy_log, ElementsAre());  // No change
  }
  EXPECT_THAT(log->entry_destroy_log,
              UnorderedElementsAre(Pair(cache_key, "a"), Pair(cache_key, "b")));
  EXPECT_THAT(log->cache_destroy_log, ElementsAre(cache_key));
}

// Similar to GetWithoutImmediateEvict test above, but additionally tests the
// retention of named caches by the cache pool after the last reference to the
// cache is released.
TEST(CacheTest, NamedGetWithoutImmediateEvict) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  auto log = std::make_shared<TestCache::RequestLog>();
  EXPECT_EQ(1, GetPoolImpl(pool)->weak_references_);
  EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_);
  {
    auto test_cache = GetTestCache(pool.get(), "cache", log);
    EXPECT_THAT(log->cache_allocate_log, ElementsAre("cache"));
    {
      auto e = GetCacheEntry(test_cache, "a");
      EXPECT_THAT(log->entry_allocate_log, ElementsAre("cache"));
      e->data = "value";
      auto e2 = GetCacheEntry(test_cache, "a");
      EXPECT_THAT(log->entry_allocate_log, ElementsAre("cache"));  // No change
      EXPECT_EQ(e, e2);
      TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
    }
    EXPECT_THAT(log->cache_destroy_log, ElementsAre());
    EXPECT_THAT(log->entry_destroy_log, ElementsAre());
    TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});

    {
      auto e = GetCacheEntry(test_cache, "a");
      EXPECT_THAT(log->entry_allocate_log, ElementsAre("cache"));  // No change
      EXPECT_EQ("value", e->data);
      TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
    }
    TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
  }
  EXPECT_EQ(1, GetPoolImpl(pool)->weak_references_);
  EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_);
  {
    auto test_cache = GetTestCache(pool.get(), "cache");
    EXPECT_THAT(log->entry_allocate_log, ElementsAre("cache"));  // No change
    {
      auto e = GetCacheEntry(test_cache, "a");
      EXPECT_THAT(log->entry_allocate_log, ElementsAre("cache"));  // No change
      EXPECT_EQ("value", e->data);
      TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
    }
  }
  EXPECT_EQ(1, GetPoolImpl(pool)->weak_references_);
  EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_);
}

TEST_P(NamedOrAnonymousCacheTest, UpdateSizeThenEvict) {
  auto pool = CachePool::Make(CachePool::Limits{});
  auto test_cache = GetCache(pool);
  {
    auto entry = GetCacheEntry(test_cache, "a");
    EXPECT_THAT(log->entry_allocate_log, ElementsAre(cache_key));  // No change
    entry->data = "a";
    entry->ChangeSize(5000);
    TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
    EXPECT_THAT(log->entry_destroy_log, ElementsAre());  // No change
  }

  // Test that entry for "a" was evicted.
  EXPECT_THAT(log->entry_destroy_log, ElementsAre(Pair(cache_key, "a")));
  TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
  EXPECT_EQ("", GetCacheEntry(test_cache, "a")->data);
}

TEST_P(NamedOrAnonymousCacheTest, UpdateSizeNoEvict) {
  CachePool::Limits limits;
  limits.total_bytes_limit = 10000;
  auto pool = CachePool::Make(limits);
  auto test_cache = GetCache(pool);
  {
    auto entry = GetCacheEntry(test_cache, "a");
    entry->data = "a";
    // No-op update
    entry->ChangeSize(1);
    // Update size
    entry->ChangeSize(5000);
    // Update size again with same size (no-op).
    entry->ChangeSize(5000);

    // Update size again with same state and size (no-op).
    entry->ChangeSize(5000);
  }
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());  // No change
  TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
  {
    auto entry = GetCacheEntry(test_cache, "b");
    entry->data = "b";
    entry->ChangeSize(5000);
  }

  // Check that no entries were evicted.
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());  // No change
  TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
  EXPECT_EQ("a", GetCacheEntry(test_cache, "a")->data);
  TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
  EXPECT_EQ("b", GetCacheEntry(test_cache, "b")->data);

  // Add one more entry, which should evict "a".
  GetCacheEntry(test_cache, "c")->data = "c";

  EXPECT_THAT(log->entry_destroy_log,
              UnorderedElementsAre(Pair(cache_key, "a")));
  TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
  EXPECT_EQ("", GetCacheEntry(test_cache, "a")->data);
  TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
  EXPECT_EQ("b", GetCacheEntry(test_cache, "b")->data);
  TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
  EXPECT_EQ("c", GetCacheEntry(test_cache, "c")->data);
  TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {test_cache.get()});
  EXPECT_THAT(log->entry_destroy_log,
              UnorderedElementsAre(Pair(cache_key, "a")));  // No change
}

// Tests that having one cache hold a strong pointer to another cache does not
// lead to a circular reference and memory leak (the actual test is done by the
// heap leak checker or sanitizer).
TEST(CacheTest, CacheDependsOnOtherCache) {
  class CacheA : public tensorstore::internal::Cache {
    using Base = tensorstore::internal::Cache;

   public:
    class Entry : public Cache::Entry {};
    using Base::Base;

    Entry* DoAllocateEntry() final { return new Entry; }
    size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  };

  class CacheB : public tensorstore::internal::Cache {
    using Base = tensorstore::internal::Cache;

   public:
    class Entry : public Cache::Entry {};
    using Base::Base;

    Entry* DoAllocateEntry() final { return new Entry; }
    size_t DoGetSizeofEntry() final { return sizeof(Entry); }

    CachePtr<CacheA> cache_a;
  };

  auto pool = CachePool::Make(kSmallCacheLimits);
  auto cache_a = GetCache<CacheA>(pool.get(), "x",
                                  [&] { return std::make_unique<CacheA>(); });
  auto cache_b = GetCache<CacheB>(pool.get(), "x",
                                  [&] { return std::make_unique<CacheB>(); });
  GetCacheEntry(cache_b, "key");
  cache_b->cache_a = cache_a;
  TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool,
                                               {cache_a.get(), cache_b.get()});
}

constexpr static int kDefaultIterations = 100;

TEST(CacheTest, ConcurrentGetCacheEntry) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  auto cache = GetTestCache(pool.get(), "cache");
  PinnedCacheEntry<TestCache> pinned_entries[3];
  TestConcurrent(
      kDefaultIterations,
      /*initialize=*/[] {},
      /*finalize=*/
      [&] {
        TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {cache.get()});
        EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
        EXPECT_EQ(2, GetPoolImpl(pool)->weak_references_.load());
        EXPECT_EQ(2, cache->use_count());
        for (auto& e : pinned_entries) {
          e.reset();
        }
        EXPECT_EQ(1, cache->use_count());
        EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
        EXPECT_EQ(2, GetPoolImpl(pool)->weak_references_.load());
        TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {cache.get()});
      },
      // Concurrent operations:
      [&] { pinned_entries[0] = GetCacheEntry(cache, "a"); },
      [&] { pinned_entries[1] = GetCacheEntry(cache, "a"); },
      [&] { pinned_entries[2] = GetCacheEntry(cache, "a"); });
}

TEST(CacheTest, ConcurrentGetCacheEntryWeakReferenceCacheDisabled) {
  auto cache = GetTestCache(nullptr, "cache");
  PinnedCacheEntry<TestCache> entry;
  TestConcurrent(
      kDefaultIterations,
      /*initialize=*/[&] { entry = GetCacheEntry(cache, "a"); },
      /*finalize=*/
      [&] {},
      // Concurrent operations:
      [&] { entry->AcquireWeakReference(); },
      [&] { entry->AcquireWeakReference(); });
}

TEST(CacheTest,
     ConcurrentDestroyStrongAndWeakCacheEntryReferenceCacheDisabled) {
  auto cache = GetTestCache(nullptr, "cache");
  PinnedCacheEntry<TestCache> entry;
  WeakPinnedCacheEntry weak_ref;
  TestConcurrent(
      kDefaultIterations,
      /*initialize=*/
      [&] {
        entry = GetCacheEntry(cache, "a");
        weak_ref = entry->AcquireWeakReference();
      },
      /*finalize=*/
      [&] {},
      // Concurrent operations:
      [&] { entry = {}; }, [&] { weak_ref = {}; });
}

TEST(CacheTest, ConcurrentGetCache) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  CachePtr<TestCache> caches[3];
  TestConcurrent(
      kDefaultIterations,
      /*initialize=*/[] {},
      /*finalize=*/
      [&] {
        EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
        EXPECT_EQ(2, GetPoolImpl(pool)->weak_references_.load());
        TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(
            pool, {caches[0].get(), caches[1].get(), caches[2].get()});
        size_t use_count = 3;
        for (auto& cache : caches) {
          EXPECT_EQ(use_count, cache->use_count());
          cache.reset();
          --use_count;
        }
        EXPECT_EQ(1, GetPoolImpl(pool)->weak_references_.load());
      },
      // Concurrent operations:
      [&] { caches[0] = GetTestCache(pool.get(), "cache"); },
      [&] { caches[1] = GetTestCache(pool.get(), "cache"); },
      [&] { caches[2] = GetTestCache(pool.get(), "cache"); });
}

TEST(CacheTest, ConcurrentReleaseCache) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  CachePtr<TestCache> caches[3];
  TestConcurrent(
      kDefaultIterations,
      /*initialize=*/
      [&] {
        EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
        EXPECT_EQ(1, GetPoolImpl(pool)->weak_references_.load());
        for (auto& cache : caches) {
          cache = GetTestCache(pool.get(), "cache");
        }
        EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
        EXPECT_EQ(2, GetPoolImpl(pool)->weak_references_.load());
      },
      /*finalize=*/
      [&] {
        EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
        EXPECT_EQ(1, GetPoolImpl(pool)->weak_references_.load());
      },
      // Concurrent operations:
      [&] { caches[0].reset(); }, [&] { caches[1].reset(); },
      [&] { caches[2].reset(); });
}

TEST(CacheTest, ConcurrentGetReleaseCache) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  const auto concurrent_op = [&] {
    auto cache = GetTestCache(pool.get(), "cache");
  };
  TestConcurrent(
      kDefaultIterations,
      /*initialize=*/
      [&] {
        EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
        EXPECT_EQ(1, GetPoolImpl(pool)->weak_references_.load());
      },
      /*finalize=*/[&] {},
      // Concurrent operations:
      concurrent_op, concurrent_op, concurrent_op);
}

TEST(CacheTest, ConcurrentReleaseCacheEntry) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  auto cache = GetTestCache(pool.get(), "cache");
  PinnedCacheEntry<TestCache> pinned_entries[3];
  TestConcurrent(
      kDefaultIterations,
      /*initialize=*/
      [&] {
        EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
        EXPECT_EQ(2, GetPoolImpl(pool)->weak_references_.load());
        for (auto& e : pinned_entries) {
          e = GetCacheEntry(cache, "a");
        }
        EXPECT_EQ(2, cache->use_count());
      },
      /*finalize=*/
      [&] {
        TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {cache.get()});
        EXPECT_EQ(1, cache->use_count());
        EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
        EXPECT_EQ(2, GetPoolImpl(pool)->weak_references_.load());
      },
      // Concurrent operations:
      [&] { pinned_entries[0].reset(); }, [&] { pinned_entries[1].reset(); },
      [&] { pinned_entries[2].reset(); });
}

TEST(CacheTest, ConcurrentGetReleaseCacheEntry) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  auto cache = GetTestCache(pool.get(), "cache");
  const auto concurrent_op = [&] {
    // Get then release cache entry.
    auto entry = GetCacheEntry(cache, "a");
  };
  TestConcurrent(
      kDefaultIterations,
      /*initialize=*/
      [&] {
        EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
        EXPECT_EQ(2, GetPoolImpl(pool)->weak_references_.load());
        EXPECT_EQ(1, cache->use_count());
      },
      /*finalize=*/
      [&] {
        TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {cache.get()});
        EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
        EXPECT_EQ(2, GetPoolImpl(pool)->weak_references_.load());
        EXPECT_EQ(1, cache->use_count());
      },
      // Concurrent operations:
      concurrent_op, concurrent_op, concurrent_op);
}

TEST(CacheTest, EvictEntryDestroyCache) {
  auto log = std::make_shared<TestCache::RequestLog>();
  CachePool::Limits limits;
  limits.total_bytes_limit = 1;
  auto pool = CachePool::Make(limits);
  auto cache_b = GetTestCache(pool.get(), "cache_b", log);
  {
    auto cache_a = GetTestCache(pool.get(), "cache_a", log);
    EXPECT_THAT(log->cache_allocate_log, ElementsAre("cache_b", "cache_a"));
    auto entry_a = GetCacheEntry(cache_a, "entry_a");
    EXPECT_THAT(log->entry_allocate_log, ElementsAre("cache_a"));
    entry_a->data = "entry_a";
  }
  // Verify that entry_a was not evicted.
  EXPECT_THAT(log->cache_destroy_log, ElementsAre());
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  {
    auto cache_a = GetTestCache(pool.get(), "cache_a");
    auto entry_a = GetCacheEntry(cache_a, "entry_a");
    EXPECT_THAT(log->cache_allocate_log,
                ElementsAre("cache_b", "cache_a"));                // No change
    EXPECT_THAT(log->entry_allocate_log, ElementsAre("cache_a"));  // No change
    ASSERT_EQ("entry_a", entry_a->data);
    TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(
        pool, {cache_a.get(), cache_b.get()});
  }
  // Add a new entry to cache_b, which evicts entry_a.
  auto entry_b = GetCacheEntry(cache_b, "entry_b");
  EXPECT_THAT(log->entry_destroy_log, ElementsAre(Pair("cache_a", "entry_a")));
  EXPECT_THAT(log->cache_destroy_log, ElementsAre("cache_a"));
  TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(pool, {cache_b.get()});
  // Verify that entry_a was evicted.
  {
    auto cache_a = GetTestCache(pool.get(), "cache_a");
    auto entry_a = GetCacheEntry(cache_a, "entry_a");
    EXPECT_EQ("", entry_a->data);
    TENSORSTORE_INTERNAL_ASSERT_CACHE_INVARIANTS(
        pool, {cache_a.get(), cache_b.get()});
  }
}

// Tests behavior of `CachePool::WeakPtr` and conversion to/from
// `CachePool::StrongPtr`.
TEST(CacheTest, CachePoolWeakPtr) {
  auto log = std::make_shared<TestCache::RequestLog>();
  auto pool = CachePool::Make(kSmallCacheLimits);
  EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
  EXPECT_EQ(1, GetPoolImpl(pool)->weak_references_.load());

  // Create cache and cache entry.
  auto cache_a = GetTestCache(pool.get(), "cache_a", log);
  EXPECT_THAT(log->cache_allocate_log, ElementsAre("cache_a"));
  auto entry_a = GetCacheEntry(cache_a, "entry_a");
  EXPECT_THAT(log->entry_allocate_log, ElementsAre("cache_a"));
  entry_a->data = "entry_a";

  // `pool` pointer is the only strong reference.
  EXPECT_EQ(1, GetPoolImpl(pool)->strong_references_.load());
  // One weak reference due to strong pointer, additional weak reference from
  // `cache_a`.
  EXPECT_EQ(2, GetPoolImpl(pool)->weak_references_.load());

  // Create another cache and cache entry.
  auto cache_b = GetTestCache(pool.get(), "cache_b", log);
  EXPECT_THAT(log->cache_allocate_log, ElementsAre("cache_a", "cache_b"));
  auto entry_b = GetCacheEntry(cache_b, "entry_b");
  EXPECT_THAT(log->entry_allocate_log, ElementsAre("cache_a", "cache_b"));
  entry_b->data = "entry_b";

  // One weak reference due to strong pointer, one weak reference due to
  // `cache_a`, one weak reference due to `cache_b`.
  EXPECT_EQ(3, GetPoolImpl(pool)->weak_references_.load());

  cache_a.reset();
  entry_a.reset();

  // One weak reference due to strong pointer, one weak reference due to
  // `cache_b`.
  EXPECT_EQ(2, GetPoolImpl(pool)->weak_references_.load());
  EXPECT_THAT(log->cache_destroy_log, ElementsAre());

  CachePool::WeakPtr weak_pool(pool);

  // One strong reference due to `pool`.
  EXPECT_EQ(1, GetPoolImpl(weak_pool)->strong_references_.load());
  // One weak reference due to strong pointer, one weak reference due to
  // `cache_b`, one weak reference from `weak_pool`.
  EXPECT_EQ(3, GetPoolImpl(weak_pool)->weak_references_.load());

  // Tests creating a new strong pointer from a weak pointer when there are
  // existing strong pointers.
  {
    CachePool::StrongPtr strong_pool(pool);
    EXPECT_EQ(2, GetPoolImpl(weak_pool)->strong_references_.load());
    EXPECT_EQ(3, GetPoolImpl(weak_pool)->weak_references_.load());
  }
  EXPECT_EQ(1, GetPoolImpl(weak_pool)->strong_references_.load());
  EXPECT_EQ(3, GetPoolImpl(weak_pool)->weak_references_.load());
  EXPECT_THAT(log->cache_destroy_log, ElementsAre());
  pool = {};

  EXPECT_EQ(0, GetPoolImpl(weak_pool)->strong_references_.load());
  EXPECT_EQ(2, GetPoolImpl(weak_pool)->weak_references_.load());
  // "cache_a" is destroyed because there are no strong references to the cache
  // pool, and there are no strong references to "cache_a".
  EXPECT_THAT(log->cache_destroy_log, ElementsAre("cache_a"));

  // Create a new cache that will be destroyed as soon as there are no
  // references to it, due to there being no strong references to cache pool.
  {
    auto cache_c = GetTestCache(weak_pool.get(), "cache_c", log);
    EXPECT_THAT(log->cache_allocate_log,
                ElementsAre("cache_a", "cache_b", "cache_c"));
    auto entry_c = GetCacheEntry(cache_c, "entry_c");
    EXPECT_THAT(log->entry_allocate_log,
                ElementsAre("cache_a", "cache_b", "cache_c"));
    entry_c->data = "entry_c";
  }

  EXPECT_THAT(log->cache_destroy_log, ElementsAre("cache_a", "cache_c"));

  // Tests creating an new strong pointer from a weak pointer when there aren't
  // existing strong pointers.
  CachePool::StrongPtr strong_pool(weak_pool);
  EXPECT_EQ(1, GetPoolImpl(strong_pool)->strong_references_.load());
  EXPECT_EQ(3, GetPoolImpl(strong_pool)->weak_references_.load());

  // Create a new cache that won't be destroyed after the reference to it is
  // released, due to the cache pool having a strong reference.
  {
    auto cache_d = GetTestCache(strong_pool.get(), "cache_d", log);
    EXPECT_THAT(log->cache_allocate_log,
                ElementsAre("cache_a", "cache_b", "cache_c", "cache_d"));
    auto entry_d = GetCacheEntry(cache_d, "entry_d");
    EXPECT_THAT(log->entry_allocate_log,
                ElementsAre("cache_a", "cache_b", "cache_c", "cache_d"));
    entry_d->data = "entry_d";
  }

  EXPECT_THAT(log->cache_destroy_log, ElementsAre("cache_a", "cache_c"));
}

// Tests that a `Cache` object with a `CachePool::WeakPtr` does not result in a
// memory leak.
TEST(CacheTest, TestCacheWithCachePool) {
  auto log = std::make_shared<TestCache::RequestLog>();
  auto pool = CachePool::Make(kSmallCacheLimits);

  {
    auto cache_a =
        GetTestCache<TestCacheWithCachePool>(pool.get(), "cache_a", log);
    cache_a->cache_pool = CachePool::WeakPtr(pool);
    EXPECT_THAT(log->cache_allocate_log, ElementsAre("cache_a"));
    auto entry_a = GetCacheEntry(cache_a, "entry_a");
    EXPECT_THAT(log->entry_allocate_log, ElementsAre("cache_a"));
    entry_a->data = "entry_a";
  }
}

TEST(CacheTest, EntryWeakReference) {
  auto log = std::make_shared<TestCache::RequestLog>();
  auto pool = CachePool::Make(CachePool::Limits{});
  auto cache = GetTestCache(pool.get(), "cache", log);
  auto entry_a = GetCacheEntry(cache, "a");
  auto weak_ref = entry_a->AcquireWeakReference();
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  pool = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  entry_a = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  weak_ref = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre(Pair("cache", "a")));
}

TEST(CacheTest, EntryWeakReferenceCacheDisabled) {
  auto log = std::make_shared<TestCache::RequestLog>();
  auto cache = GetTestCache(nullptr, "cache", log);
  auto entry_a = GetCacheEntry(cache, "a");
  auto weak_ref = entry_a->AcquireWeakReference();
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  entry_a = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre(Pair("", "a")));
  weak_ref = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre(Pair("", "a")));
}

TEST(CacheTest, EntryWeakReferencesCacheDisabled) {
  auto log = std::make_shared<TestCache::RequestLog>();
  auto cache = GetTestCache(nullptr, "cache", log);
  auto entry_a = GetCacheEntry(cache, "a");
  auto weak_ref = entry_a->AcquireWeakReference();
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  weak_ref = {};
  weak_ref = entry_a->AcquireWeakReference();
  entry_a = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre(Pair("", "a")));
  weak_ref = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre(Pair("", "a")));
}

TEST(CacheTest, EntryWeakReferences) {
  auto log = std::make_shared<TestCache::RequestLog>();
  auto pool = CachePool::Make(CachePool::Limits{});
  auto cache = GetTestCache(pool.get(), "cache", log);
  auto entry_a = GetCacheEntry(cache, "a");
  auto weak_ref = entry_a->AcquireWeakReference();
  auto weak_ref2 = entry_a->AcquireWeakReference();
  auto entry_a2 = GetCacheEntry(cache, "a");
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  pool = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  entry_a = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  weak_ref = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  weak_ref2 = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  entry_a2 = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre(Pair("cache", "a")));
}

TEST(CacheTest, GetStrongEntryReferenceWhileHoldingOnlyWeakReference) {
  auto log = std::make_shared<TestCache::RequestLog>();
  auto pool = CachePool::Make(CachePool::Limits{});
  auto cache = GetTestCache(pool.get(), "cache", log);
  auto entry_a = GetCacheEntry(cache, "a");
  auto weak_ref = entry_a->AcquireWeakReference();
  entry_a = {};
  entry_a = GetCacheEntry(cache, "a");
  entry_a = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  weak_ref = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre(Pair("cache", "a")));
}

TEST(CacheTest,
     GetStrongEntryReferenceWhileHoldingOnlyWeakReferenceCacheDisabled) {
  auto log = std::make_shared<TestCache::RequestLog>();
  auto cache = GetTestCache(nullptr, "cache", log);
  auto entry_a = GetCacheEntry(cache, "a");
  auto weak_ref = entry_a->AcquireWeakReference();
  entry_a = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre(Pair("", "a")));
  entry_a = GetCacheEntry(cache, "a");
  entry_a = {};
  EXPECT_THAT(log->entry_destroy_log,
              ElementsAre(Pair("", "a"), Pair("", "a")));
  weak_ref = {};
  EXPECT_THAT(log->entry_destroy_log,
              ElementsAre(Pair("", "a"), Pair("", "a")));  // Unchanged
}

TEST(CacheTest, PoolWithNonZeroBytesLimit) {
  auto log = std::make_shared<TestCache::RequestLog>();
  auto pool = CachePool::Make(kSmallCacheLimits);
  auto cache = GetTestCache(pool.get(), "cache", log);
  {
    auto entry_a = GetCacheEntry(cache, "a");
    auto weak_ref = entry_a->AcquireWeakReference();
  }
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  pool = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  cache = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre(Pair("cache", "a")));
}

TEST(CacheTest, WeakRefOwnedByEntry) {
  auto log = std::make_shared<TestCache::RequestLog>();
  auto pool = CachePool::Make(kSmallCacheLimits);
  auto cache1 = GetTestCache(pool.get(), "cache1", log);
  auto cache2 = GetTestCache(pool.get(), "cache2", log);
  {
    auto entry_a = GetCacheEntry(cache1, "a");
    auto entry_b = GetCacheEntry(cache1, "b");
    entry_a->weak_ref = entry_b->AcquireWeakReference();
  }
  { auto entry_c = GetCacheEntry(cache2, "c"); }

  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  pool = {};
  EXPECT_THAT(log->entry_destroy_log, ElementsAre());
  cache1 = {};
  EXPECT_THAT(log->entry_destroy_log,
              UnorderedElementsAre(Pair("cache1", "a"), Pair("cache1", "b")));
  cache2 = {};
  EXPECT_THAT(log->entry_destroy_log,
              UnorderedElementsAre(Pair("cache1", "a"), Pair("cache1", "b"),
                                   Pair("cache2", "c")));
}

}  // namespace
