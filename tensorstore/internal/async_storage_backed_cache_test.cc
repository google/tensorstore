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

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::Future;
using tensorstore::Status;
using tensorstore::internal::AsyncStorageBackedCache;
using tensorstore::internal::Cache;
using tensorstore::internal::CacheEntryQueueState;
using tensorstore::internal::CachePool;
using tensorstore::internal::PinnedCacheEntry;
using tensorstore::internal::static_pointer_cast;
using tensorstore::internal::UniqueNow;
using WriteFlags = tensorstore::internal::AsyncStorageBackedCache::WriteFlags;

constexpr CachePool::Limits kSmallCacheLimits{10000000, 5000000};

struct RequestLog {
  struct ReadRequest {
    PinnedCacheEntry<AsyncStorageBackedCache> entry;
    absl::Time staleness_bound;
  };
  struct WritebackRequest {
    PinnedCacheEntry<AsyncStorageBackedCache> entry;
  };
  std::vector<ReadRequest> reads;
  std::vector<WritebackRequest> writebacks;
};

class TestCache
    : public tensorstore::internal::CacheBase<TestCache,
                                              AsyncStorageBackedCache> {
 public:
  class Entry : public AsyncStorageBackedCache::Entry {};

  TestCache(RequestLog* log) : log_(log) {}

  void DoRead(PinnedEntry entry, absl::Time staleness_bound) override {
    log_->reads.push_back(RequestLog::ReadRequest{
        tensorstore::internal::static_pointer_cast<Entry>(std::move(entry)),
        staleness_bound});
  }
  void DoWriteback(PinnedEntry entry) override {
    log_->writebacks.push_back(RequestLog::WritebackRequest{
        tensorstore::internal::static_pointer_cast<Entry>(std::move(entry)),
    });
  }

 private:
  RequestLog* log_;
};

TEST(AsyncCacheTest, ReadBasic) {
  auto pool = CachePool::Make(CachePool::Limits{});
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  absl::Time read_time1, read_time2;
  {
    auto init_time = absl::Now();
    auto read_future = entry->Read(init_time);
    ASSERT_FALSE(read_future.ready());

    // Tests that calling Read again with a time prior to the first Read returns
    // the same Future.
    {
      auto read_future2 = entry->Read(init_time);
      EXPECT_TRUE(HaveSameSharedState(read_future, read_future2));
    }
    ASSERT_EQ(1u, log.reads.size());
    ASSERT_TRUE(log.writebacks.empty());
    read_time1 = absl::Now();
    {
      auto read_req = log.reads.front();
      EXPECT_EQ(absl::InfinitePast(), read_req.entry->last_read_time);
      log.reads.clear();
      auto lock = read_req.entry->AcquireReadStateWriterLock();
      read_req.entry->last_read_time = read_time1;
      cache->NotifyReadSuccess(read_req.entry.get(), std::move(lock));
    }
    ASSERT_TRUE(read_future.ready());
    EXPECT_TRUE(read_future.result());

    // Tests that calling Read again with an old time doesn't issue any more
    // read requests.
    {
      auto read_future3 = entry->Read(read_time1);
      ASSERT_TRUE(read_future3.ready());
      EXPECT_TRUE(read_future3.result());
      ASSERT_TRUE(log.reads.empty());
      ASSERT_TRUE(log.writebacks.empty());
    }
  }

  // Tests that calling Read with a newer time issues another read request.
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    ASSERT_EQ(1u, log.reads.size());
    ASSERT_TRUE(log.writebacks.empty());
    read_time2 = absl::Now();
    {
      auto read_req = log.reads.front();
      log.reads.clear();
      EXPECT_EQ(read_time1, read_req.entry->last_read_time);
      auto lock = read_req.entry->AcquireReadStateWriterLock();
      read_req.entry->last_read_time = read_time2;
      cache->NotifyReadSuccess(read_req.entry.get(), std::move(lock));
    }
    ASSERT_TRUE(read_future.ready());
    EXPECT_TRUE(read_future.result());
  }

  // Tests that calling Read before another Read completes queues the second
  // read.
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    auto read_time = UniqueNow();
    auto read_future1 = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future1.ready());
    EXPECT_FALSE(HaveSameSharedState(read_future, read_future1));
    {
      auto read_future2 = entry->Read(absl::InfiniteFuture());
      EXPECT_TRUE(HaveSameSharedState(read_future1, read_future2));
    }
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.front();
      log.reads.clear();
      EXPECT_EQ(read_time2, read_req.entry->last_read_time);
      read_req.entry->last_read_time = read_time;
      cache->NotifyReadSuccess(read_req.entry.get(),
                               read_req.entry->AcquireReadStateWriterLock());
    }
    ASSERT_TRUE(read_future.ready());
    ASSERT_FALSE(read_future1.ready());
    EXPECT_TRUE(read_future.result());

    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    auto read_time2 = absl::Now();
    {
      auto read_req = log.reads.front();
      log.reads.clear();
      EXPECT_EQ(read_time, read_req.entry->last_read_time);
      read_req.entry->last_read_time = read_time2;
      cache->NotifyReadSuccess(read_req.entry.get(),
                               read_req.entry->AcquireReadStateWriterLock());
    }
    ASSERT_TRUE(read_future1.ready());
    EXPECT_TRUE(read_future1.result());
  }

  // Tests that a queued read can be resolved by the completion of the issued
  // read if the time is newer.
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    auto read_future1 = entry->Read(absl::InfiniteFuture());
    auto read_time = absl::Now();
    ASSERT_FALSE(read_future.ready());
    ASSERT_FALSE(read_future1.ready());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.front();
      log.reads.clear();
      read_req.entry->last_read_time = read_time;
      cache->NotifyReadSuccess(read_req.entry.get(),
                               read_req.entry->AcquireReadStateWriterLock());
    }
    ASSERT_TRUE(read_future.ready());
    EXPECT_TRUE(read_future.result());
    ASSERT_TRUE(read_future1.ready());
    EXPECT_TRUE(read_future1.result());

    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
  }

  // Tests that a queued read that is cancelled doesn't result in another read
  // request.
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    auto read_time = absl::Now();
    {
      auto read_future1 = entry->Read(absl::InfiniteFuture());
      ASSERT_FALSE(read_future1.ready());
    }
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.front();
      log.reads.clear();
      read_req.entry->last_read_time = read_time;
      cache->NotifyReadSuccess(read_req.entry.get(),
                               read_req.entry->AcquireReadStateWriterLock());
    }
    ASSERT_TRUE(read_future.ready());
    EXPECT_TRUE(read_future.result());
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
  }

  // Tests that a queued read that is cancelled and then requested again leads
  // to a new Future.
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    {
      auto read_future1 = entry->Read(absl::InfiniteFuture());
      ASSERT_FALSE(read_future1.ready());
    }
    auto read_future1 = entry->Read(absl::InfiniteFuture());
    auto read_time = absl::Now();
    ASSERT_FALSE(read_future1.ready());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.front();
      log.reads.clear();
      read_req.entry->last_read_time = read_time;
      cache->NotifyReadSuccess(read_req.entry.get(),
                               read_req.entry->AcquireReadStateWriterLock());
    }
    ASSERT_TRUE(read_future.ready());
    EXPECT_TRUE(read_future.result());
    ASSERT_TRUE(read_future1.ready());
    EXPECT_TRUE(read_future1.result());
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
  }
}

TEST(AsyncCacheTest, ReadFailed) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  const Status read_status = absl::UnknownError("read failed");
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.front();
      log.reads.clear();
      cache->NotifyReadError(read_req.entry.get(), read_status);
    }
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    ASSERT_TRUE(read_future.ready());
    EXPECT_EQ(read_status, GetStatus(read_future.result()));
  }

  // Check that error result is cached.
  {
    auto read_future = entry->Read(absl::InfinitePast());
    ASSERT_TRUE(read_future.ready());
    EXPECT_EQ(read_status, GetStatus(read_future.result()));
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
  }

  // Check that a new read can be issued.
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.front();
      log.reads.clear();
      read_req.entry->last_read_time = absl::Now();
      cache->NotifyReadSuccess(read_req.entry.get(),
                               read_req.entry->AcquireReadStateWriterLock());
    }
    ASSERT_TRUE(read_future.ready());
    EXPECT_TRUE(read_future.result());
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
  }
}

TEST(AsyncCacheTest, ReadFailedAfterSuccessfulRead) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  // First initialize the entry with a successful read.
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.front();
      log.reads.clear();
      read_req.entry->last_read_time = absl::Now();
      cache->NotifyReadSuccess(read_req.entry.get(),
                               read_req.entry->AcquireReadStateWriterLock());
    }
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    ASSERT_TRUE(read_future.ready());
    EXPECT_TRUE(read_future.result());
  }

  const Status read_status = absl::UnknownError("read failed");
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.front();
      log.reads.clear();
      cache->NotifyReadError(read_req.entry.get(), read_status);
    }
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    ASSERT_TRUE(read_future.ready());
    EXPECT_EQ(read_status, GetStatus(read_future.result()));
  }

  // Check that error result is cached.
  {
    auto read_future = entry->Read(absl::InfinitePast());
    ASSERT_TRUE(read_future.ready());
    EXPECT_EQ(read_status, GetStatus(read_future.result()));
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
  }

  // Check that a new read can be issued.
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.front();
      log.reads.clear();
      read_req.entry->last_read_time = absl::Now();
      cache->NotifyReadSuccess(read_req.entry.get(),
                               read_req.entry->AcquireReadStateWriterLock());
    }
    ASSERT_TRUE(read_future.ready());
    EXPECT_TRUE(read_future.result());
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
  }
}

TEST(AsyncCacheTest, Write) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_req.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_TRUE(write_future.ready());
  ASSERT_TRUE(write_future.result());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

TEST(AsyncCacheTest, ReadAfterUnconditionalWriteback) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  // Read still requires a new read request.
  auto read_future = entry->Read(absl::InfiniteFuture());

  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  EXPECT_TRUE(read_future.ready());
  EXPECT_TRUE(read_future.result());
}

TEST(AsyncCacheTest, UnconditionalWriteback) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());
  auto write_time = absl::Now();
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_req.entry->last_read_time = write_time;
    cache->NotifyWritebackSuccess(
        write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_TRUE(write_future.ready());
  ASSERT_TRUE(write_future.result());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  // Reading after writeback with an old time doesn't require a new read
  // request.
  {
    auto read_future = entry->Read(write_time);
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    EXPECT_TRUE(read_future.ready());
    EXPECT_TRUE(read_future.result());
  }

  // Reading after writeback with a new time does require a new read request.
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    EXPECT_FALSE(read_future.ready());
  }
}

TEST(AsyncCacheTest, FullyOverwrittenAfterSuccessfulRead) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto read_future = entry->Read(absl::InfinitePast());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  ASSERT_FALSE(read_future.ready());
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  ASSERT_TRUE(read_future.ready());
  ASSERT_TRUE(read_future.result());

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  // Reading still results in a new read request.
  auto read_future2 = entry->Read(absl::InfiniteFuture());
  EXPECT_EQ(1, log.reads.size());
  EXPECT_EQ(0, log.writebacks.size());
  EXPECT_FALSE(read_future2.ready());
}

TEST(AsyncCacheTest, FullyOverwrittenAfterFailedRead) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto read_future = entry->Read(absl::InfinitePast());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  ASSERT_FALSE(read_future.ready());
  const Status read_error = absl::UnknownError("read failed");
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    cache->NotifyReadError(read_req.entry.get(), read_error);
  }
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  ASSERT_TRUE(read_future.ready());
  ASSERT_FALSE(read_future.result());
  EXPECT_EQ(read_error, read_future.result().status());

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  // Reading still results in a new read request.
  auto read_future2 = entry->Read(absl::InfiniteFuture());
  EXPECT_EQ(1, log.reads.size());
  EXPECT_EQ(0, log.writebacks.size());
  EXPECT_FALSE(read_future2.ready());
}

TEST(AsyncCacheTest, FullyOverwrittenWithIssuedRead) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto read_future = entry->Read(absl::InfiniteFuture());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);
  // Write doesn't affect read request.
  ASSERT_FALSE(write_future.ready());
  ASSERT_FALSE(read_future.ready());

  write_future.Force();

  // Writeback can't start until the read completes.

  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  ASSERT_TRUE(read_future.ready());
  TENSORSTORE_ASSERT_OK(read_future.result());

  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  auto write_req = log.writebacks.front();
  log.writebacks.clear();
}

TEST(AsyncCacheTest, FullyOverwrittenWithIssuedReadAndQueuedRead) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto read_future = entry->Read(absl::InfiniteFuture());
  auto read_future2 = entry->Read(absl::InfiniteFuture());
  ASSERT_FALSE(HaveSameSharedState(read_future, read_future2));
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  // Write doesn't affect reads.
  ASSERT_FALSE(read_future.ready());
  ASSERT_FALSE(read_future2.ready());

  {
    auto read_req = log.reads.front();
    log.reads.clear();
  }
}

TEST(AsyncCacheTest, ForcedQueuedWritebackIssuedAfterWriteback) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  auto write_req = log.writebacks.front();
  log.writebacks.clear();
  cache->NotifyWritebackStarted(write_req.entry.get(),
                                write_req.entry->AcquireWriteStateLock());
  auto write_future2 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                          WriteFlags::kConditionalWriteback);
  ASSERT_FALSE(HaveSameSharedState(write_future, write_future2));
  write_future2.Force();
  write_req.entry->last_read_time = absl::Now();
  cache->NotifyWritebackSuccess(
      write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  ASSERT_TRUE(write_future.ready());
  ASSERT_TRUE(write_future.result());
  ASSERT_FALSE(write_future2.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  {
    auto write_req2 = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req2.entry.get(),
                                  write_req2.entry->AcquireWriteStateLock());
    write_req2.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req2.entry.get(),
        write_req2.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_TRUE(write_future2.ready());
  ASSERT_TRUE(write_future2.result());
}

TEST(AsyncCacheTest, QueuedWritebackNotIssuedAfterWriteback) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  auto write_req = log.writebacks.front();
  log.writebacks.clear();
  cache->NotifyWritebackStarted(write_req.entry.get(),
                                write_req.entry->AcquireWriteStateLock());
  auto write_future2 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                          WriteFlags::kConditionalWriteback);
  ASSERT_FALSE(HaveSameSharedState(write_future, write_future2));
  write_req.entry->last_read_time = absl::Now();
  cache->NotifyWritebackSuccess(
      write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  ASSERT_TRUE(write_future.ready());
  ASSERT_TRUE(write_future.result());
  ASSERT_FALSE(write_future2.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  write_future2.Force();
  ASSERT_FALSE(write_future2.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  {
    auto write_req2 = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req2.entry.get(),
                                  write_req2.entry->AcquireWriteStateLock());
    write_req2.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req2.entry.get(),
        write_req2.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_TRUE(write_future2.ready());
  ASSERT_TRUE(write_future2.result());
}

TEST(AsyncCacheTest, QueuedWritebackNotNeededAfterWriteback) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  auto write_req = log.writebacks.front();
  log.writebacks.clear();
  cache->NotifyWritebackStarted(write_req.entry.get(),
                                write_req.entry->AcquireWriteStateLock());
  auto write_future2 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                          WriteFlags::kConditionalWriteback);
  ASSERT_FALSE(HaveSameSharedState(write_future, write_future2));
  cache->NotifyWritebackStarted(write_req.entry.get(),
                                write_req.entry->AcquireWriteStateLock());
  write_req.entry->last_read_time = absl::Now();
  cache->NotifyWritebackSuccess(
      write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  ASSERT_TRUE(write_future.ready());
  ASSERT_TRUE(write_future.result());
  ASSERT_TRUE(write_future2.ready());
  ASSERT_TRUE(write_future2.result());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

TEST(AsyncCacheTest, WritebackFailedWhenFullyOverwritten) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  auto write_req = log.writebacks.front();
  log.writebacks.clear();
  cache->NotifyWritebackStarted(write_req.entry.get(),
                                write_req.entry->AcquireWriteStateLock());
  const Status error_status = absl::UnknownError("write failed");
  cache->NotifyWritebackError(write_req.entry.get(),
                              write_req.entry->AcquireWriteStateLock(),
                              error_status);
  ASSERT_TRUE(write_future.ready());
  ASSERT_EQ(error_status, write_future.result().status());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

TEST(AsyncCacheTest, WritebackFailedWhenNotFullyOverwritten) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  auto write_req = log.writebacks.front();
  log.writebacks.clear();
  cache->NotifyWritebackStarted(write_req.entry.get(),
                                write_req.entry->AcquireWriteStateLock());
  const Status error_status = absl::UnknownError("write failed");
  cache->NotifyWritebackError(write_req.entry.get(),
                              write_req.entry->AcquireWriteStateLock(),
                              error_status);
  ASSERT_TRUE(write_future.ready());
  ASSERT_EQ(error_status, write_future.result().status());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

// Tests that a failed read after a writeback has been requested propagates the
// error to the writeback.
TEST(AsyncCacheTest, ReadFailedWithWritebackRequested) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  const Status read_status = absl::UnknownError("read failed");
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    cache->NotifyReadError(read_req.entry.get(), read_status);
  }
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  ASSERT_TRUE(write_future.ready());
  EXPECT_EQ(read_status, GetStatus(write_future.result()));
}

// Tests that a failed read does not propagate the error to a writeback that has
// not been requested.
TEST(AsyncCacheTest, ReadFailedWithWritebackQueuedButNotRequested) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);

  auto read_future = entry->Read(absl::InfinitePast());
  ASSERT_FALSE(read_future.ready());
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  const Status read_status = absl::UnknownError("read failed");
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    cache->NotifyReadError(read_req.entry.get(), read_status);
  }
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  ASSERT_TRUE(read_future.ready());
  EXPECT_EQ(read_status, GetStatus(read_future.result()));
  ASSERT_FALSE(write_future.ready());

  write_future.Force();
  ASSERT_FALSE(write_future.ready());

  // Another read is issued.
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  // Writeback request is issued.
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_req.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  ASSERT_TRUE(write_future.ready());
  ASSERT_TRUE(write_future.result());
}

// Tests that another read is issued when a writeback fails due to a generation
// mismatch.
TEST(AsyncCacheTest, NotifyGenerationMismatch) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);
  write_future.Force();
  ASSERT_FALSE(write_future.ready());

  // Writeback requires a read request.
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  // Complete the read request successfully.
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  // Writeback is issued.
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  // Start writeback, then abort the writeback request with generation mismatch.
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    cache->NotifyWritebackNeedsRead(write_req.entry.get(),
                                    write_req.entry->AcquireWriteStateLock(),
                                    absl::Now());
  }

  // Another read is required.
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.writebacks.size());
  ASSERT_EQ(1, log.reads.size());
}

// Tests that another read is not issued when a writeback fails due to a
// generation mismatch after the writeback has been cancelled.
TEST(AsyncCacheTest, NotifyGenerationMismatchAfterWritebackCancelled) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);
  write_future.Force();
  ASSERT_FALSE(write_future.ready());

  // Writeback requires a read request.
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  // Complete the read request successfully.
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  // Writeback is issued.
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  // Start writeback, cancel the writeback, then abort the writeback request due
  // to generation mismatch.
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_future = Future<void>();
    cache->NotifyWritebackNeedsRead(write_req.entry.get(),
                                    write_req.entry->AcquireWriteStateLock(),
                                    absl::Now());
  }

  ASSERT_EQ(0, log.writebacks.size());
  ASSERT_EQ(0, log.reads.size());
}

TEST(AsyncCacheTest, ReadFailedWithWritebackRequestedAndQueued) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);
  write_future.Force();
  ASSERT_FALSE(write_future.ready());

  // Writeback requires a read request.
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  // Complete the read request successfully.
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  // Writeback is issued.
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  // Start writeback, issue another write, and fail the writeback request due to
  // generation mismatch.
  Future<const void> write_future2;
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_future2 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                       WriteFlags::kConditionalWriteback);
    cache->NotifyWritebackNeedsRead(write_req.entry.get(),
                                    write_req.entry->AcquireWriteStateLock(),
                                    absl::Now());
  }

  // Another read is required.
  ASSERT_FALSE(write_future.ready());
  ASSERT_FALSE(write_future2.ready());
  ASSERT_EQ(0, log.writebacks.size());
  ASSERT_EQ(1, log.reads.size());

  // Fail the read request.
  const Status read_status = absl::UnknownError("read failed");
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    cache->NotifyReadError(read_req.entry.get(), read_status);
  }
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  // The read error propagates to both write futures.
  ASSERT_TRUE(write_future.ready());
  ASSERT_TRUE(write_future2.ready());
  EXPECT_EQ(read_status, GetStatus(write_future.result()));
  EXPECT_EQ(read_status, GetStatus(write_future2.result()));
}

// Tests that a writeback is not issued if there are no references to the
// writeback future.
TEST(AsyncCacheTest, WritebackCancelled) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  EXPECT_EQ(CacheEntryQueueState::dirty, entry->queue_state());
  write_future.Force();
  EXPECT_EQ(CacheEntryQueueState::writeback_requested, entry->queue_state());
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  write_future = Future<void>();
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }
  EXPECT_EQ(CacheEntryQueueState::dirty, entry->queue_state());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

TEST(AsyncCacheTest, WritebackAndQueuedWritebackCancelled) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);
  write_future.Force();
  ASSERT_FALSE(write_future.ready());

  // Writeback requires a read request.
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  // Complete the read request successfully.
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  // Writeback is issued.
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    // Make and immediately cancel another writeback request.
    entry->FinishWrite(entry->AcquireWriteStateLock(),
                       WriteFlags::kConditionalWriteback);

    // Cancel original writeback request.
    write_future = Future<void>();

    // Fail writeback.
    cache->NotifyWritebackNeedsRead(write_req.entry.get(),
                                    write_req.entry->AcquireWriteStateLock(),
                                    absl::Now());
  }

  // No more reads are issued.
  ASSERT_EQ(0, log.writebacks.size());
  ASSERT_EQ(0, log.reads.size());
}

TEST(AsyncCacheTest, WritebackCancelledWithNonCancelledQueuedWriteback) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);
  write_future.Force();
  ASSERT_FALSE(write_future.ready());

  // Writeback requires a read request.
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  // Complete the read request successfully.
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  // Writeback is issued.
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  Future<const void> write_future2;
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    // Make another writeback request.
    write_future2 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                       WriteFlags::kConditionalWriteback);
    write_future2.Force();

    // Cancel original writeback request.
    write_future = Future<void>();

    // Fail writeback.
    cache->NotifyWritebackNeedsRead(write_req.entry.get(),
                                    write_req.entry->AcquireWriteStateLock(),
                                    absl::Now());
  }

  ASSERT_FALSE(write_future2.ready());

  // Another read is issued.
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  ASSERT_FALSE(write_future2.ready());
  // Writeback request is issued.
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_req.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  ASSERT_TRUE(write_future2.ready());
  ASSERT_TRUE(write_future2.result());
}

// After marking an entry clean, any outstanding writeback requests are marked
// ready.  However, we have to release the entry mutex before marking the
// promises ready, so there is a possibility of a writeback request being forced
// after the entry mutex is released but before the writeback promise is marked
// ready.  This tests that such requests are correctly ignored.
TEST(AsyncCacheTest, ForceWritebackWhenClean) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);
  write_future.Force();
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());
  Future<const void> write_future2;
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    // Call WritebackStarted so that a subsequent write gets a new Future.
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_future2 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                       WriteFlags::kUnconditionalWriteback);
    ASSERT_FALSE(HaveSameSharedState(write_future, write_future2));
    write_future.ExecuteWhenReady(
        [write_future2](Future<const void>) { write_future2.Force(); });
    // Call WritebackStarted again so that this writeback request makes the
    // entry clean.
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_req.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_TRUE(write_future.ready());
  ASSERT_TRUE(write_future.result());
  ASSERT_TRUE(write_future2.ready());
  ASSERT_TRUE(write_future2.result());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

TEST(AsyncCacheTest, WritebackRequestedWithReadIssued) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto read_future = entry->Read(absl::InfiniteFuture());
  ASSERT_FALSE(read_future.ready());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_FALSE(read_future.ready());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  ASSERT_FALSE(write_future.ready());
  ASSERT_TRUE(read_future.ready());
  ASSERT_TRUE(read_future.result());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_req.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_TRUE(write_future.ready());
  ASSERT_TRUE(write_future.result());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

// Tests that two writes issued before writeback starts share the same future.
TEST(AsyncCacheTest, WriteFutureSharing) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);
  auto write_future2 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                          WriteFlags::kConditionalWriteback);
  EXPECT_TRUE(HaveSameSharedState(write_future, write_future2));
}

TEST(AsyncCacheTest, WriteFutureSharingAfterWritebackIssued) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  auto write_req = log.writebacks.front();
  log.writebacks.clear();
  cache->NotifyWritebackStarted(write_req.entry.get(),
                                write_req.entry->AcquireWriteStateLock());
  auto write_future2 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                          WriteFlags::kConditionalWriteback);
  EXPECT_FALSE(HaveSameSharedState(write_future, write_future2));
  auto write_future3 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                          WriteFlags::kConditionalWriteback);
  EXPECT_TRUE(HaveSameSharedState(write_future2, write_future3));
  ASSERT_FALSE(write_future2.ready());
}

TEST(AsyncCacheTest, FullyOverwrittenAfterWritebackStarted) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  Future<const void> write_future2;
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_future2 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                       WriteFlags::kUnconditionalWriteback);
    write_req.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_FALSE(HaveSameSharedState(write_future, write_future2));
  ASSERT_FALSE(write_future2.ready());

  auto read_future = entry->Read(absl::InfiniteFuture());
  // Read still requires additional read request.
  ASSERT_FALSE(read_future.ready());
}

TEST(AsyncCacheTest, WritebackRequestedByCache) {
  auto pool = CachePool::Make(CachePool::Limits{});
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_req.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_TRUE(write_future.ready());
  ASSERT_TRUE(write_future.result());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

// Tests that calling GetWritebackFuture after a writeback has been issued for
// the latest write generation, while there isn't an additional writeback
// queued, returns the same future as was previously returned.
TEST(AsyncCacheTest, GetWritebackFutureAfterWritebackIssued) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  write_future.Force();

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());

    {
      auto write_future2 = entry->GetWritebackFuture();
      EXPECT_TRUE(HaveSameSharedState(write_future, write_future2));
    }

    write_req.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_TRUE(write_future.ready());
  ASSERT_TRUE(write_future.result());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

// Tests that calling GetWritebackFuture while there is a writeback queued that
// has not been cancelled returns the same future.
TEST(AsyncCacheTest, GetWritebackFutureWithWritebackQueuedAndNotCancelled) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  write_future.Force();

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());
  Future<const void> write_future2;

  auto write_req = log.writebacks.front();
  log.writebacks.clear();
  cache->NotifyWritebackStarted(write_req.entry.get(),
                                write_req.entry->AcquireWriteStateLock());

  write_future2 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                     WriteFlags::kUnconditionalWriteback);
  EXPECT_FALSE(HaveSameSharedState(write_future2, write_future));
  {
    auto write_future3 = entry->GetWritebackFuture();
    EXPECT_TRUE(HaveSameSharedState(write_future2, write_future3));
  }
}

TEST(AsyncCacheTest, GetWritebackFutureWithWritebackQueuedAndCancelled) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kUnconditionalWriteback);

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  write_future.Force();

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  Future<const void> write_future3;
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());

    // Make another write, then cancel the returned future.
    {
      auto write_future2 = entry->FinishWrite(
          /*size_update=*/{}, WriteFlags::kConditionalWriteback);
      EXPECT_FALSE(HaveSameSharedState(write_future2, write_future));
    }

    write_future3 = entry->GetWritebackFuture();
    write_future3.Force();

    write_req.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_TRUE(write_future.ready());
  ASSERT_TRUE(write_future.result());
  ASSERT_FALSE(write_future3.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_req.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_TRUE(write_future3.ready());
  ASSERT_TRUE(write_future3.result());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

TEST(AsyncCacheTest, ReadIssuedDuetoWritebackWithReadQueued) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                         WriteFlags::kConditionalWriteback);
  write_future.Force();
  ASSERT_FALSE(write_future.ready());

  // Writeback requires a read request.
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  // Complete the read request successfully.
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  // Writeback is issued.
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  // Request another read, which won't be issued yet because a writeback is in
  // progress.
  auto read_future = entry->Read(absl::InfiniteFuture());
  ASSERT_FALSE(read_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    // Fail writeback due to generation mismatch.
    cache->NotifyWritebackNeedsRead(write_req.entry.get(),
                                    write_req.entry->AcquireWriteStateLock(),
                                    absl::Now());
  }
  ASSERT_FALSE(read_future.ready());
  ASSERT_FALSE(write_future.ready());

  // Another read is issued.
  ASSERT_EQ(0, log.writebacks.size());
  ASSERT_EQ(1, log.reads.size());

  // Complete the read successfully.
  {
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }
  ASSERT_TRUE(read_future.ready());
  ASSERT_TRUE(read_future.result());

  // Another writeback is issued.
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  // Complete the writeback request successfully.
  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_req.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  }
  ASSERT_TRUE(write_future.ready());
  ASSERT_TRUE(write_future.result());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

TEST(AsyncCacheTest, GetWritebackFutureWhileClean) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");
  {
    auto write_future = entry->GetWritebackFuture();
    EXPECT_FALSE(write_future.valid());
  }
}

TEST(AsyncCacheTest, WriteAfterNotifyStartedBeforeAbort) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future1 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                          WriteFlags::kUnconditionalWriteback);

  write_future1.Force();

  {
    ASSERT_EQ(1, log.writebacks.size());
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    cache->NotifyWritebackNeedsRead(write_req.entry.get(),
                                    write_req.entry->AcquireWriteStateLock(),
                                    absl::Now());
  }

  auto write_future2 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                          WriteFlags::kUnconditionalWriteback);
  write_future2.Force();

  {
    ASSERT_EQ(1, log.reads.size());
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  ASSERT_EQ(1, log.writebacks.size());
  auto write_req1 = log.writebacks.front();
  log.writebacks.clear();
  cache->NotifyWritebackStarted(write_req1.entry.get(),
                                write_req1.entry->AcquireWriteStateLock());

  auto write_future3 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                          WriteFlags::kUnconditionalWriteback);
  write_future3.Force();
  EXPECT_FALSE(HaveSameSharedState(write_future2, write_future3));

  write_req1.entry->last_read_time = absl::Now();
  cache->NotifyWritebackSuccess(
      write_req1.entry.get(), write_req1.entry->AcquireWriteAndReadStateLock());
  EXPECT_TRUE(write_future2.ready());
  EXPECT_FALSE(write_future3.ready());

  ASSERT_EQ(1, log.writebacks.size());

  {
    auto write_req = log.writebacks.front();
    log.writebacks.clear();
    cache->NotifyWritebackStarted(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock());
    write_req.entry->last_read_time = absl::Now();
    cache->NotifyWritebackSuccess(
        write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  }
  EXPECT_TRUE(write_future3.ready());
}

TEST(AsyncCacheTest, WriteAfterNotifyStartedAfterWriteAfterNotifyStarted) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return absl::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future1 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                          WriteFlags::kUnconditionalWriteback);

  write_future1.Force();

  ASSERT_EQ(1, log.writebacks.size());
  auto write_req = log.writebacks.front();
  log.writebacks.clear();
  cache->NotifyWritebackStarted(write_req.entry.get(),
                                write_req.entry->AcquireWriteStateLock());

  auto write_future2 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                          WriteFlags::kUnconditionalWriteback);
  EXPECT_FALSE(HaveSameSharedState(write_future1, write_future2));
  write_future2.Force();

  cache->NotifyWritebackNeedsRead(write_req.entry.get(),
                                  write_req.entry->AcquireWriteStateLock(),
                                  absl::Now());

  {
    ASSERT_EQ(1, log.reads.size());
    auto read_req = log.reads.front();
    log.reads.clear();
    read_req.entry->last_read_time = absl::Now();
    cache->NotifyReadSuccess(read_req.entry.get(),
                             read_req.entry->AcquireReadStateWriterLock());
  }

  ASSERT_EQ(1, log.writebacks.size());
  write_req = log.writebacks.front();
  log.writebacks.clear();
  cache->NotifyWritebackStarted(write_req.entry.get(),
                                write_req.entry->AcquireWriteStateLock());

  auto write_future3 = entry->FinishWrite(entry->AcquireWriteStateLock(),
                                          WriteFlags::kUnconditionalWriteback);
  write_future3.Force();
  EXPECT_FALSE(HaveSameSharedState(write_future2, write_future3));

  write_req.entry->last_read_time = absl::Now();
  cache->NotifyWritebackSuccess(
      write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  EXPECT_TRUE(write_future2.ready());
  EXPECT_FALSE(write_future3.ready());

  ASSERT_EQ(1, log.writebacks.size());
  write_req = log.writebacks.front();
  log.writebacks.clear();
  cache->NotifyWritebackStarted(write_req.entry.get(),
                                write_req.entry->AcquireWriteStateLock());
  write_req.entry->last_read_time = absl::Now();
  cache->NotifyWritebackSuccess(
      write_req.entry.get(), write_req.entry->AcquireWriteAndReadStateLock());
  EXPECT_TRUE(write_future3.ready());
}

}  // namespace
