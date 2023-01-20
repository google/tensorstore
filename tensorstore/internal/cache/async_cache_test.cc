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

#include "tensorstore/internal/cache/async_cache.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/concurrent_testutil.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/queue_testutil.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::Future;
using ::tensorstore::no_transaction;
using ::tensorstore::StorageGeneration;
using ::tensorstore::Transaction;
using ::tensorstore::UniqueWriterLock;
using ::tensorstore::internal::AsyncCache;
using ::tensorstore::internal::CacheEntryQueueState;
using ::tensorstore::internal::CachePool;
using ::tensorstore::internal::OpenTransactionPtr;
using ::tensorstore::internal::PinnedCacheEntry;
using ::tensorstore::internal::TransactionState;
using ::tensorstore::internal::UniqueNow;
using ::tensorstore::internal::WeakTransactionNodePtr;

constexpr CachePool::Limits kSmallCacheLimits{10000000, 5000000};

struct RequestLog {
  struct ReadRequest {
    AsyncCache::Entry* entry;
    void Success(absl::Time time = absl::Now(),
                 std::shared_ptr<const size_t> value = {}) {
      entry->ReadSuccess(
          {std::move(value),
           {tensorstore::StorageGeneration::FromString("g"), time}});
    }
    void Error(absl::Status error) { entry->ReadError(std::move(error)); }
  };

  struct TransactionReadRequest {
    AsyncCache::TransactionNode* node;
    void Success(absl::Time time = absl::Now(),
                 std::shared_ptr<const size_t> value = {}) {
      node->ReadSuccess(
          {std::move(value),
           {tensorstore::StorageGeneration::FromString("g"), time}});
    }
    void Error(absl::Status error) { node->ReadError(std::move(error)); }
  };

  struct WritebackRequest {
    AsyncCache::TransactionNode* node;
    void Success(absl::Time time = absl::Now(),
                 std::shared_ptr<const size_t> value = {}) {
      node->WritebackSuccess(
          {std::move(value),
           {tensorstore::StorageGeneration::FromString("g"), time}});
    }
    void Error(absl::Status error) {
      node->SetError(error);
      node->WritebackError();
    }
  };
  tensorstore::internal::ConcurrentQueue<ReadRequest> reads;
  tensorstore::internal::ConcurrentQueue<TransactionReadRequest>
      transaction_reads;
  tensorstore::internal::ConcurrentQueue<WritebackRequest> writebacks;
};

class TestCache : public tensorstore::internal::AsyncCache {
  using Base = tensorstore::internal::AsyncCache;

 public:
  using ReadData = size_t;
  class Entry : public AsyncCache::Entry {
   public:
    using OwningCache = TestCache;

    auto CreateWriteTransaction(OpenTransactionPtr transaction = {}) {
      return GetTransactionNode(*this, transaction).value();
    }

    Future<const void> CreateWriteTransactionFuture(
        OpenTransactionPtr transaction = {}) {
      return CreateWriteTransaction(std::move(transaction))
          ->transaction()
          ->future();
    }

    void DoRead(absl::Time staleness_bound) override {
      GetOwningCache(*this).log_->reads.push(RequestLog::ReadRequest{this});
    }

    bool ShareImplicitTransactionNodes() override {
      return share_implicit_transaction_nodes;
    }

    size_t ComputeReadDataSizeInBytes(const void* data) override {
      return *static_cast<const size_t*>(data);
    }

    absl::Status do_initialize_transaction_error;
    bool share_implicit_transaction_nodes = true;
  };

  class TransactionNode : public Base::TransactionNode {
   public:
    using OwningCache = TestCache;
    using Base::TransactionNode::TransactionNode;

    absl::Status DoInitialize(OpenTransactionPtr& transaction) override {
      TENSORSTORE_RETURN_IF_ERROR(
          this->Base::TransactionNode::DoInitialize(transaction));
      auto& entry = GetOwningEntry(*this);
      ++value;
      SetReadsCommitted();
      return entry.do_initialize_transaction_error;
    }
    void DoRead(absl::Time staleness_bound) override {
      GetOwningCache(*this).log_->transaction_reads.push(
          RequestLog::TransactionReadRequest{this});
    }

    void Commit() override {
      GetOwningCache(*this).log_->writebacks.push(
          RequestLog::WritebackRequest{this});
      Base::TransactionNode::Commit();
    }

    size_t ComputeWriteStateSizeInBytes() override { return size; }

    int value = 0;
    size_t size = 0;
  };

  TestCache(RequestLog* log) : log_(log) {}

  Entry* DoAllocateEntry() final { return new Entry; }
  std::size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final {
    return new TransactionNode(static_cast<Entry&>(entry));
  }

 private:
  RequestLog* log_;
};

TEST(AsyncCacheTest, ReadBasic) {
  auto pool = CachePool::Make(CachePool::Limits{});
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
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
      auto read_req = log.reads.pop();
      EXPECT_EQ(absl::InfinitePast(),
                AsyncCache::ReadLock<void>(*read_req.entry).stamp().time);
      read_req.Success(read_time1);
    }
    ASSERT_TRUE(read_future.ready());
    TENSORSTORE_EXPECT_OK(read_future);

    // Tests that calling Read again with an old time doesn't issue any more
    // read requests.
    {
      auto read_future3 = entry->Read(read_time1);
      ASSERT_TRUE(read_future3.ready());
      TENSORSTORE_EXPECT_OK(read_future3);
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
      auto read_req = log.reads.pop();
      EXPECT_EQ(read_time1,
                AsyncCache::ReadLock<void>(*read_req.entry).stamp().time);
      read_req.Success(read_time2);
    }
    ASSERT_TRUE(read_future.ready());
    TENSORSTORE_EXPECT_OK(read_future);
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
      auto read_req = log.reads.pop();
      EXPECT_EQ(read_time2,
                AsyncCache::ReadLock<void>(*read_req.entry).stamp().time);
      read_req.Success(read_time);
    }
    ASSERT_TRUE(read_future.ready());
    ASSERT_FALSE(read_future1.ready());
    TENSORSTORE_EXPECT_OK(read_future);

    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    auto read_time2 = absl::Now();
    {
      auto read_req = log.reads.pop();
      EXPECT_EQ(read_time,
                AsyncCache::ReadLock<void>(*read_req.entry).stamp().time);
      read_req.Success(read_time2);
    }
    ASSERT_TRUE(read_future1.ready());
    TENSORSTORE_EXPECT_OK(read_future1);
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
      auto read_req = log.reads.pop();
      read_req.Success(read_time);
    }
    ASSERT_TRUE(read_future.ready());
    TENSORSTORE_EXPECT_OK(read_future);
    ASSERT_TRUE(read_future1.ready());
    TENSORSTORE_EXPECT_OK(read_future1);

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
      auto read_req = log.reads.pop();
      read_req.Success(read_time);
    }
    ASSERT_TRUE(read_future.ready());
    TENSORSTORE_EXPECT_OK(read_future);
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
      auto read_req = log.reads.pop();
      read_req.Success(read_time);
    }
    ASSERT_TRUE(read_future.ready());
    TENSORSTORE_EXPECT_OK(read_future);
    ASSERT_TRUE(read_future1.ready());
    TENSORSTORE_EXPECT_OK(read_future1);
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
  }
}

TEST(AsyncCacheTest, ReadFailed) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  const auto read_status = absl::UnknownError("read failed");
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.pop();
      read_req.Error(read_status);
    }
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    ASSERT_TRUE(read_future.ready());
    EXPECT_EQ(read_status, read_future.status());
  }

  // Check that a new read can be issued.
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.pop();
      read_req.Success();
    }
    ASSERT_TRUE(read_future.ready());
    TENSORSTORE_EXPECT_OK(read_future);
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
  }
}

TEST(AsyncCacheTest, ReadFailedAfterSuccessfulRead) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  // First initialize the entry with a successful read.
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.pop();
      read_req.Success();
    }
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    ASSERT_TRUE(read_future.ready());
    TENSORSTORE_EXPECT_OK(read_future);
  }

  const auto read_status = absl::UnknownError("read failed");
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.pop();
      read_req.Error(read_status);
    }
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    ASSERT_TRUE(read_future.ready());
    EXPECT_EQ(read_status, read_future.status());
  }

  // Check that a new read can be issued.
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_FALSE(read_future.ready());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    {
      auto read_req = log.reads.pop();
      read_req.Success();
    }
    ASSERT_TRUE(read_future.ready());
    TENSORSTORE_EXPECT_OK(read_future);
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
  }
}

TEST(AsyncCacheTest, NonTransactionalWrite) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  WeakTransactionNodePtr<TestCache::TransactionNode> weak_node;

  Future<const void> write_future;
  {
    auto node = entry->CreateWriteTransaction();
    weak_node.reset(node.get());
    write_future = node->transaction()->future();
  }

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());
  {
    auto write_req = log.writebacks.pop();
    EXPECT_EQ(weak_node.get(), write_req.node);
    write_req.Success();
  }
  ASSERT_TRUE(write_future.ready());
  TENSORSTORE_ASSERT_OK(write_future);
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

TEST(AsyncCacheTest, NonTransactionalWriteback) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->CreateWriteTransactionFuture();

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());
  auto write_time = absl::Now();
  {
    auto write_req = log.writebacks.pop();
    write_req.Success(write_time);
  }
  ASSERT_TRUE(write_future.ready());
  TENSORSTORE_ASSERT_OK(write_future);
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  // Reading after writeback with an old time doesn't require a new read
  // request.
  {
    auto read_future = entry->Read(write_time);
    ASSERT_EQ(0, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    ASSERT_TRUE(read_future.ready());
    TENSORSTORE_EXPECT_OK(read_future);
  }

  // Reading after writeback with a new time does require a new read request.
  {
    auto read_future = entry->Read(absl::InfiniteFuture());
    ASSERT_EQ(1, log.reads.size());
    ASSERT_EQ(0, log.writebacks.size());
    EXPECT_FALSE(read_future.ready());
    auto read_req = log.reads.pop();
    read_req.Success();
    EXPECT_TRUE(read_future.ready());
  }
}

// Tests that an implicit transaction is destroyed if there are no references to
// the writeback future.
TEST(AsyncCacheTest, WritebackCancelled) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->CreateWriteTransactionFuture();

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  EXPECT_EQ(CacheEntryQueueState::dirty, entry->queue_state());

  write_future = Future<void>();

  EXPECT_EQ(CacheEntryQueueState::clean_and_in_use, entry->queue_state());

  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

TEST(AsyncCacheTest, WritebackRequestedWithReadIssued) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto read_future = entry->Read(absl::InfiniteFuture());
  ASSERT_FALSE(read_future.ready());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  auto write_future = entry->CreateWriteTransactionFuture();
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_FALSE(read_future.ready());
  ASSERT_EQ(1, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());

  {
    auto read_req = log.reads.pop();
    read_req.Success();
  }

  ASSERT_FALSE(write_future.ready());
  ASSERT_TRUE(read_future.ready());
  TENSORSTORE_ASSERT_OK(read_future);
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());
  {
    auto write_req = log.writebacks.pop();
    write_req.Success();
  }
  ASSERT_TRUE(write_future.ready());
  TENSORSTORE_ASSERT_OK(write_future);
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
}

// Tests that two writes issued before writeback starts share the same
// transaction.
TEST(AsyncCacheTest, WriteFutureSharing) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->CreateWriteTransactionFuture();
  auto write_future2 = entry->CreateWriteTransactionFuture();
  EXPECT_TRUE(HaveSameSharedState(write_future, write_future2));
}

TEST(AsyncCacheTest, WriteFutureSharingAfterWritebackIssued) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->CreateWriteTransactionFuture();

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  write_future.Force();
  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  auto write_req = log.writebacks.pop();
  write_req.Success();
  auto write_future2 = entry->CreateWriteTransactionFuture();
  EXPECT_FALSE(HaveSameSharedState(write_future, write_future2));
  auto write_future3 = entry->CreateWriteTransactionFuture();
  EXPECT_TRUE(HaveSameSharedState(write_future2, write_future3));
  ASSERT_FALSE(write_future2.ready());
}

TEST(AsyncCacheTest, WritebackRequestedByCache) {
  auto pool = CachePool::Make(CachePool::Limits{});
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto write_future = entry->CreateWriteTransactionFuture();
  EXPECT_EQ(CacheEntryQueueState::writeback_requested, entry->queue_state());

  ASSERT_FALSE(write_future.ready());
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(1, log.writebacks.size());

  {
    auto write_req = log.writebacks.pop();
    write_req.Success();
  }
  ASSERT_TRUE(write_future.ready());
  TENSORSTORE_ASSERT_OK(write_future);
  ASSERT_EQ(0, log.reads.size());
  ASSERT_EQ(0, log.writebacks.size());
  EXPECT_EQ(CacheEntryQueueState::clean_and_in_use, entry->queue_state());
}

TEST(AsyncCacheTest, TransactionalWritebackSuccess) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");
  auto transaction = Transaction(tensorstore::atomic_isolated);

  WeakTransactionNodePtr<TestCache::TransactionNode> weak_node;
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    auto node = entry->CreateWriteTransaction(open_transaction);
    EXPECT_EQ(node, GetTransactionNode(*entry, open_transaction));
    weak_node.reset(node.get());
  }
  auto future = transaction.CommitAsync();
  EXPECT_TRUE(transaction.commit_started());

  {
    auto write_req = log.writebacks.pop();
    EXPECT_EQ(weak_node.get(), write_req.node);
    write_req.Success();
  }

  ASSERT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(AsyncCacheTest, TransactionalWritebackError) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");
  auto transaction = Transaction(tensorstore::isolated);
  WeakTransactionNodePtr<TestCache::TransactionNode> weak_node;
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    weak_node.reset(entry->CreateWriteTransaction(open_transaction).get());
  }
  auto future = transaction.CommitAsync();
  auto error = absl::UnknownError("write error");
  {
    auto write_req = log.writebacks.pop();
    EXPECT_EQ(weak_node.get(), write_req.node);
    write_req.Error(error);
  }

  ASSERT_TRUE(future.ready());
  EXPECT_EQ(error, future.status());
}

// Tests that concurrently committing multiple transactions over the same set of
// entries does not lead to deadlock.
TEST(AsyncCacheTest, ConcurrentTransactionCommit) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  static constexpr size_t kNumEntries = 2;
  tensorstore::internal::PinnedCacheEntry<TestCache> entries[kNumEntries];
  for (size_t i = 0; i < kNumEntries; ++i) {
    entries[i] = GetCacheEntry(cache, tensorstore::StrCat(i));
  }
  static constexpr size_t kNumTransactions = 3;
  std::vector<Transaction> transactions(kNumTransactions, no_transaction);
  tensorstore::internal::TestConcurrent<kNumTransactions>(
      /*num_iterations=*/100,
      /*initialize=*/
      [&] {
        for (size_t i = 0; i < kNumTransactions; ++i) {
          auto& transaction = transactions[i];
          transaction = Transaction(tensorstore::atomic_isolated);
          TENSORSTORE_ASSERT_OK_AND_ASSIGN(
              auto open_transaction,
              tensorstore::internal::AcquireOpenTransactionPtrOrError(
                  transaction));
          // Add entries in order dependent on `i`.
          for (size_t j = 0; j < kNumEntries; ++j) {
            entries[(i + j) % kNumEntries]->CreateWriteTransaction(
                open_transaction);
          }
          ASSERT_FALSE(transaction.future().ready());
        }
      },
      /*finalize=*/
      [&] {
        TransactionState* expected_transactions[kNumTransactions];
        for (size_t i = 0; i < kNumTransactions; ++i) {
          auto& transaction = transactions[i];
          ASSERT_TRUE(transaction.commit_started());
          ASSERT_FALSE(transaction.future().ready());
          expected_transactions[i] = TransactionState::get(transaction);
        }
        TransactionState* transaction_order[kNumTransactions];
        for (size_t i = 0; i < kNumTransactions; ++i) {
          PinnedCacheEntry<TestCache> entry_order[kNumEntries];
          ASSERT_EQ(kNumEntries, log.writebacks.size());
          for (size_t j = 0; j < kNumEntries; ++j) {
            auto write_req = log.writebacks.pop();
            entry_order[j].reset(static_cast<TestCache::Entry*>(
                &GetOwningEntry(*write_req.node)));
            if (j == 0) {
              transaction_order[i] = write_req.node->transaction();
            } else {
              ASSERT_EQ(transaction_order[i], write_req.node->transaction());
            }
            write_req.Success();
          }
          EXPECT_THAT(entry_order,
                      ::testing::UnorderedElementsAreArray(entries));
        }
        EXPECT_THAT(transaction_order, ::testing::UnorderedElementsAreArray(
                                           expected_transactions));
        for (auto& transaction : transactions) {
          ASSERT_TRUE(transaction.future().ready());
          TENSORSTORE_ASSERT_OK(transaction.future());
          transaction = no_transaction;
        }
      },
      /*concurrent_op=*/
      [&](size_t i) { transactions[i].CommitAsync().IgnoreFuture(); });
}

// Tests that an error return from `DoInitialize` is handled properly.
TEST(AsyncCacheTest, DoInitializeTransactionError) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");
  auto error = absl::UnknownError("initialize");
  entry->do_initialize_transaction_error = error;

  // Test implicit transaction error.
  {
    OpenTransactionPtr transaction;
    EXPECT_EQ(error, GetTransactionNode(*entry, transaction).status());
  }

  // Test explicit transaction error.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(
            Transaction(tensorstore::isolated)));
    EXPECT_EQ(error, GetTransactionNode(*entry, transaction).status());
  }
}

// Tests that concurrently adding a node for the same transaction to the same
// entry from multiple threads is handled properly.
TEST(AsyncCacheTest, ConcurrentInitializeExplicitTransaction) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");
  OpenTransactionPtr open_transaction;
  tensorstore::internal::TestConcurrent<2>(
      /*num_iterations=*/100,
      /*initialize=*/
      [&] {
        TENSORSTORE_ASSERT_OK_AND_ASSIGN(
            open_transaction,
            tensorstore::internal::AcquireOpenTransactionPtrOrError(
                Transaction(tensorstore::isolated)));
      },
      /*finalize=*/
      [] {},
      /*concurrent_op=*/
      [&](size_t i) {
        TENSORSTORE_ASSERT_OK_AND_ASSIGN(
            auto node, GetTransactionNode(*entry, open_transaction));
        EXPECT_EQ(1, node->value);
      });
}

// Tests that concurrently adding an implicit transaction node to the same entry
// from multiple threads is handled properly.
TEST(AsyncCacheTest, ConcurrentInitializeImplicitTransaction) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");
  tensorstore::internal::TestConcurrent<2>(
      /*num_iterations=*/100,
      /*initialize=*/
      [] {},
      /*finalize=*/
      [] {},
      /*concurrent_op=*/
      [&](size_t i) {
        OpenTransactionPtr transaction;
        TENSORSTORE_ASSERT_OK_AND_ASSIGN(
            auto node, GetTransactionNode(*entry, transaction));
        EXPECT_EQ(1, node->value);
      });
}

// Tests that implicit transaction nodes are shared when
// `ShareImplicitTransactionNodes()` returns `true`.
TEST(AsyncCacheTest, ShareImplicitTransactionNodesTrue) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");
  auto node = entry->CreateWriteTransaction();
  EXPECT_EQ(node, entry->CreateWriteTransaction());
}

// Tests that implicit transaction nodes are not shared when
// `ShareImplicitTransactionNodes()` returns `false`.
TEST(AsyncCacheTest, ShareImplicitTransactionNodesFalse) {
  auto pool = CachePool::Make(kSmallCacheLimits);
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");
  entry->share_implicit_transaction_nodes = false;
  auto node = entry->CreateWriteTransaction();
  EXPECT_NE(node, entry->CreateWriteTransaction());
}

// Tests that a single entry cannot have more than two associated implicit
// transactions if `ShareImplicitTransactionNodes()` is `false`.
TEST(AsyncCacheTest, NotMoreThanTwoImplicitTransactions) {
  auto pool = CachePool::Make(CachePool::Limits{});
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");

  auto future0 = entry->CreateWriteTransactionFuture();
  future0.Force();
  auto future1 = entry->CreateWriteTransactionFuture();
  future1.Force();
  auto future2 = entry->CreateWriteTransactionFuture();
  future2.Force();

  EXPECT_FALSE(HaveSameSharedState(future0, future1));
  EXPECT_TRUE(HaveSameSharedState(future1, future2));

  log.writebacks.pop().Success();
  log.writebacks.pop().Success();
  EXPECT_EQ(0, log.writebacks.size());
}

TEST(AsyncCacheTest, ReadSizeInBytes) {
  auto pool = CachePool::Make(CachePool::Limits{20000, 10000});
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });

  {
    auto entry = GetCacheEntry(cache, "a");
    auto read_future = entry->Read(absl::Now());
    log.reads.pop().Success(absl::Now(), std::make_shared<size_t>(19000));
  }

  // Entry was not be evicted because its size fits within the cache pool
  // `total_bytes_limit`.
  {
    auto entry = GetCacheEntry(cache, "a");
    EXPECT_THAT(AsyncCache::ReadLock<size_t>(*entry).data(),
                ::testing::Pointee(19000));

    auto read_future = entry->Read(absl::InfiniteFuture());
    log.reads.pop().Success(absl::Now(), std::make_shared<size_t>(21000));
    ASSERT_TRUE(read_future.ready());
  }

  // Entry was evicted because its size now exceeds the cache pool
  // `total_bytes_limit`.
  {
    auto entry = GetCacheEntry(cache, "a");
    EXPECT_THAT(AsyncCache::ReadLock<size_t>(*entry).data(),
                ::testing::IsNull());

    // Increase entry size back to 1000.
    auto read_future = entry->Read(absl::InfiniteFuture());
    log.reads.pop().Success(absl::Now(), std::make_shared<size_t>(1000));
    ASSERT_TRUE(read_future.ready());
  }

  // Entry was not be evicted.
  {
    auto entry = GetCacheEntry(cache, "a");
    EXPECT_THAT(AsyncCache::ReadLock<size_t>(*entry).data(),
                ::testing::Pointee(1000));

    auto write_future = entry->CreateWriteTransactionFuture();
    write_future.Force();

    log.writebacks.pop().Success(absl::Now(), std::make_shared<size_t>(21000));
    ASSERT_TRUE(write_future.ready());
  }

  // Entry was evicted after writeback because its size exceeded the limit.
  {
    auto entry = GetCacheEntry(cache, "a");
    EXPECT_THAT(AsyncCache::ReadLock<size_t>(*entry).data(),
                ::testing::IsNull());
  }
}

TEST(AsyncCacheTest, ImplicitTransactionSize) {
  auto pool = CachePool::Make(CachePool::Limits{20000, 10000});
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });

  auto entry = GetCacheEntry(cache, "a");

  Future<const void> write_future;
  {
    auto node = entry->CreateWriteTransaction();
    UniqueWriterLock lock(*node);
    node->size = 9000;
    node->MarkSizeUpdated();
    write_future = node->transaction()->future();
  }

  // Writeback was not started because size is less than
  // `queued_for_writeback_bytes_limit`.
  EXPECT_TRUE(log.writebacks.empty());

  {
    auto node = entry->CreateWriteTransaction();
    UniqueWriterLock lock(*node);
    EXPECT_EQ(9000, node->size);
    node->size = 11000;
    node->MarkSizeUpdated();
  }

  log.writebacks.pop().Success();

  ASSERT_TRUE(write_future.ready());

  auto entry2 = GetCacheEntry(cache, "b");
  {
    auto node = entry2->CreateWriteTransaction();
    UniqueWriterLock lock(*node);
    node->size = 9000;
    node->MarkSizeUpdated();
  }

  // Writeback was not started because size is less than
  // `queued_for_writeback_bytes_limit`, now that the implicit transaction for
  // `entry` has been committed.
  EXPECT_TRUE(log.writebacks.empty());
}

TEST(AsyncCacheTest, ExplicitTransactionSize) {
  auto pool = CachePool::Make(CachePool::Limits{20000, 10000});
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });

  // Create canary entry to detect eviction.
  {
    auto entry_b = GetCacheEntry(cache, "b");
    auto read_future = entry_b->Read(absl::Now());
    log.reads.pop().Success(absl::Now(), std::make_shared<size_t>(1000));
  }

  auto transaction = Transaction(tensorstore::isolated);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto open_transaction,
      tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));

  {
    auto entry_a = GetCacheEntry(cache, "a");
    {
      auto node = entry_a->CreateWriteTransaction(open_transaction);
      UniqueWriterLock lock(*node);
      node->size = 100000;
      node->MarkSizeUpdated();
    }
    EXPECT_EQ(100000, transaction.total_bytes());

    auto entry_c = GetCacheEntry(cache, "c");
    {
      auto node = entry_c->CreateWriteTransaction(open_transaction);
      UniqueWriterLock lock(*node);
      node->size = 500;
      node->MarkSizeUpdated();
    }
    EXPECT_EQ(100500, transaction.total_bytes());

    {
      auto node = entry_a->CreateWriteTransaction(open_transaction);
      UniqueWriterLock lock(*node);
      node->size = 110000;
      node->MarkSizeUpdated();
    }
    EXPECT_EQ(110500, transaction.total_bytes());
  }

  // Verify that "b" was not evicted.
  {
    auto entry_b = GetCacheEntry(cache, "b");
    EXPECT_THAT(AsyncCache::ReadLock<size_t>(*entry_b).data(),
                ::testing::Pointee(1000));
  }
}

void TestRevokedTransactionNode(bool reverse_order) {
  auto pool = CachePool::Make(CachePool::Limits{});
  RequestLog log;
  auto cache = pool->GetCache<TestCache>(
      "", [&] { return std::make_unique<TestCache>(&log); });
  auto entry = GetCacheEntry(cache, "a");
  auto transaction = Transaction(tensorstore::atomic_isolated);

  WeakTransactionNodePtr<TestCache::TransactionNode> weak_node1;
  WeakTransactionNodePtr<TestCache::TransactionNode> weak_node2;
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    {
      auto node = entry->CreateWriteTransaction(open_transaction);
      EXPECT_EQ(node, GetTransactionNode(*entry, open_transaction));
      weak_node1.reset(node.get());
      node->Revoke();
    }
    {
      auto node = entry->CreateWriteTransaction(open_transaction);
      EXPECT_EQ(node, GetTransactionNode(*entry, open_transaction));
      weak_node2.reset(node.get());
    }
  }
  auto future = transaction.CommitAsync();
  EXPECT_TRUE(transaction.commit_started());

  {
    auto write_req1 = log.writebacks.pop();
    EXPECT_EQ(weak_node1.get(), write_req1.node);

    auto write_req2 = log.writebacks.pop();
    EXPECT_EQ(weak_node2.get(), write_req2.node);
    if (reverse_order) {
      write_req2.Success();
      write_req1.Success();
    } else {
      write_req1.Success();
      write_req2.Success();
    }
  }
  ASSERT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(AsyncCacheTest, RevokedTransactionNodeFifo) {
  TestRevokedTransactionNode(false);
}

TEST(AsyncCacheTest, RevokedTransactionNodeLifo) {
  TestRevokedTransactionNode(true);
}

}  // namespace
