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

/// \file
///
/// Tests for `kvs_backed_cache` and for `internal_kvs::MultiPhaseMutation`.

#include "tensorstore/internal/kvs_backed_cache.h"

#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/async_cache.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/kvs_backed_cache_testutil.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/kvstore/key_value_store_testutil.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::KeyRange;
using tensorstore::MatchesStatus;
using tensorstore::StorageGeneration;
using tensorstore::TimestampedStorageGeneration;
using tensorstore::Transaction;
using tensorstore::internal::CachePool;
using tensorstore::internal::KvsBackedTestCache;
using tensorstore::internal::MatchesKvsReadResult;
using tensorstore::internal::OpenTransactionPtr;

TENSORSTORE_GLOBAL_INITIALIZER {
  using tensorstore::internal::KvsBackedCacheBasicTransactionalTestOptions;
  using tensorstore::internal::RegisterKvsBackedCacheBasicTransactionalTest;
  {
    KvsBackedCacheBasicTransactionalTestOptions options;
    options.test_name = "MemoryNonAtomic";
    options.get_store = [] {
      return tensorstore::GetMemoryKeyValueStore(/*atomic=*/false);
    };
    options.multi_key_atomic_supported = false;
    RegisterKvsBackedCacheBasicTransactionalTest(options);
  }

  {
    KvsBackedCacheBasicTransactionalTestOptions options;
    options.test_name = "MemoryAtomic";
    options.get_store = [] {
      return tensorstore::GetMemoryKeyValueStore(/*atomic=*/true);
    };
    RegisterKvsBackedCacheBasicTransactionalTest(options);
  }
}

class MockStoreTest : public ::testing::Test {
 protected:
  CachePool::StrongPtr pool = CachePool::Make(CachePool::Limits{});
  tensorstore::KeyValueStore::PtrT<tensorstore::internal::MockKeyValueStore>
      mock_store{new tensorstore::internal::MockKeyValueStore};
  tensorstore::KeyValueStore::Ptr memory_store =
      tensorstore::GetMemoryKeyValueStore();

  tensorstore::internal::CachePtr<KvsBackedTestCache> GetCache(
      std::string cache_identifier = {},
      tensorstore::KeyValueStore::Ptr kvstore = {}) {
    if (!kvstore) kvstore = mock_store;
    return pool->GetCache<KvsBackedTestCache>(cache_identifier, [&] {
      return std::make_unique<KvsBackedTestCache>(kvstore);
    });
  }

  tensorstore::internal::CachePtr<KvsBackedTestCache> cache = GetCache();
};

TEST_F(MockStoreTest, ReadSuccess) {
  auto entry = GetCacheEntry(cache, "a");
  auto read_time = absl::Now();
  auto read_future = entry->Read(read_time);
  auto read_req = mock_store->read_requests.pop();
  EXPECT_EQ("a", read_req.key);
  EXPECT_EQ(StorageGeneration::Unknown(), read_req.options.if_equal);
  EXPECT_EQ(StorageGeneration::Unknown(), read_req.options.if_not_equal);
  EXPECT_EQ(tensorstore::OptionalByteRangeRequest{},
            read_req.options.byte_range);
  EXPECT_EQ(read_time, read_req.options.staleness_bound);
  read_req(memory_store);
}

TEST_F(MockStoreTest, ReadError) {
  auto entry = GetCacheEntry(cache, "a");
  auto read_future = entry->Read(absl::Now());
  auto read_req = mock_store->read_requests.pop();
  read_req.promise.SetResult(absl::FailedPreconditionError("read error"));
  EXPECT_THAT(read_future.result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error reading \"a\": read error"));
}

TEST_F(MockStoreTest, WriteError) {
  auto entry = GetCacheEntry(cache, "a");

  auto transaction = Transaction(tensorstore::atomic_isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(entry->Modify(open_transaction, true, "abc"));
  }
  transaction.CommitAsync().IgnoreFuture();
  auto write_req = mock_store->write_requests.pop();
  write_req.promise.SetResult(absl::FailedPreconditionError("write error"));
  EXPECT_THAT(transaction.future().result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error writing \"a\": write error"));
}

TEST_F(MockStoreTest, ReadErrorDuringWriteback) {
  auto entry = GetCacheEntry(cache, "a");

  auto transaction = Transaction(tensorstore::atomic_isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(entry->Modify(open_transaction, false, "abc"));
  }
  transaction.CommitAsync().IgnoreFuture();
  auto read_req = mock_store->read_requests.pop();
  read_req.promise.SetResult(absl::FailedPreconditionError("read error"));
  EXPECT_THAT(transaction.future().result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error reading \"a\": read error"));
}

TEST_F(MockStoreTest, ReadErrorDueToValidateDuringWriteback) {
  auto entry = GetCacheEntry(cache, "a");

  auto transaction = Transaction(tensorstore::atomic_isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(entry->Validate(
        open_transaction, [](absl::Cord data) { return absl::OkStatus(); }));
    // Read value to exercise writeback path where there is already a cached
    // value.
    auto read_future = entry->ReadValue(open_transaction);
    mock_store->read_requests.pop()(memory_store);
    EXPECT_THAT(read_future.result(), ::testing::Optional(absl::Cord()));
  }
  transaction.CommitAsync().IgnoreFuture();
  auto read_req = mock_store->read_requests.pop();
  read_req.promise.SetResult(absl::FailedPreconditionError("read error"));
  EXPECT_THAT(transaction.future().result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error reading \"a\": read error"));
}

TEST_F(MockStoreTest, MultiPhaseSeparateKeys) {
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    open_transaction->Barrier();
    TENSORSTORE_ASSERT_OK(GetCacheEntry(GetCache("x"), "a")
                              ->Modify(open_transaction, false, "abc"));
    open_transaction->Barrier();
    TENSORSTORE_ASSERT_OK(GetCacheEntry(GetCache("x"), "b")
                              ->Modify(open_transaction, false, "de"));
    TENSORSTORE_ASSERT_OK(GetCacheEntry(GetCache("y"), "b")
                              ->Modify(open_transaction, false, "f"));
  }
  transaction.CommitAsync().IgnoreFuture();
  {
    auto read_req = mock_store->read_requests.pop();
    EXPECT_TRUE(mock_store->read_requests.empty());
    EXPECT_TRUE(mock_store->write_requests.empty());
    EXPECT_EQ("a", read_req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), read_req.options.if_not_equal);
    read_req(memory_store);
  }
  {
    auto write_req = mock_store->write_requests.pop();
    EXPECT_TRUE(mock_store->read_requests.empty());
    EXPECT_TRUE(mock_store->write_requests.empty());
    EXPECT_EQ("a", write_req.key);
    EXPECT_EQ(StorageGeneration::NoValue(), write_req.options.if_equal);
    EXPECT_EQ("abc", write_req.value);
    write_req(memory_store);
  }
  EXPECT_THAT(memory_store->Read("a").result(),
              MatchesKvsReadResult(absl::Cord("abc")));
  {
    auto read_req = mock_store->read_requests.pop();
    EXPECT_TRUE(mock_store->read_requests.empty());
    EXPECT_TRUE(mock_store->write_requests.empty());
    EXPECT_EQ("b", read_req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), read_req.options.if_not_equal);
    read_req(memory_store);
  }
  {
    auto write_req = mock_store->write_requests.pop();
    EXPECT_TRUE(mock_store->read_requests.empty());
    EXPECT_TRUE(mock_store->write_requests.empty());
    EXPECT_EQ("b", write_req.key);
    EXPECT_EQ(StorageGeneration::NoValue(), write_req.options.if_equal);
    EXPECT_EQ("def", write_req.value);
    write_req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  TENSORSTORE_EXPECT_OK(transaction.future());
  EXPECT_THAT(memory_store->Read("b").result(),
              MatchesKvsReadResult(absl::Cord("def")));
}

TEST_F(MockStoreTest, MultiPhaseSameKey) {
  auto entry = GetCacheEntry(cache, "a");
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(entry->Modify(open_transaction, false, "abc"));
    open_transaction->Barrier();
    TENSORSTORE_ASSERT_OK(entry->Modify(open_transaction, false, "def"));
  }
  transaction.CommitAsync().IgnoreFuture();
  {
    auto read_req = mock_store->read_requests.pop();
    EXPECT_TRUE(mock_store->read_requests.empty());
    EXPECT_TRUE(mock_store->write_requests.empty());
    EXPECT_EQ("a", read_req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), read_req.options.if_not_equal);
    read_req(memory_store);
  }
  {
    auto write_req = mock_store->write_requests.pop();
    EXPECT_TRUE(mock_store->read_requests.empty());
    EXPECT_TRUE(mock_store->write_requests.empty());
    EXPECT_EQ("a", write_req.key);
    EXPECT_EQ(StorageGeneration::NoValue(), write_req.options.if_equal);
    EXPECT_EQ("abc", write_req.value);
    write_req(memory_store);
  }
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result,
                                   memory_store->Read("a").result());
  EXPECT_THAT(read_result, MatchesKvsReadResult(absl::Cord("abc")));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto write_stamp, memory_store->Write("a", absl::Cord("xyz")).result());
  {
    auto write_req = mock_store->write_requests.pop();
    EXPECT_TRUE(mock_store->read_requests.empty());
    EXPECT_TRUE(mock_store->write_requests.empty());
    EXPECT_EQ("a", write_req.key);
    EXPECT_EQ(read_result.stamp.generation, write_req.options.if_equal);
    EXPECT_EQ("abcdef", write_req.value);
    write_req(memory_store);
  }
  {
    auto read_req = mock_store->read_requests.pop();
    EXPECT_TRUE(mock_store->read_requests.empty());
    EXPECT_TRUE(mock_store->write_requests.empty());
    EXPECT_EQ("a", read_req.key);
    EXPECT_EQ(read_result.stamp.generation, read_req.options.if_not_equal);
    read_req(memory_store);
  }
  {
    auto write_req = mock_store->write_requests.pop();
    EXPECT_TRUE(mock_store->read_requests.empty());
    EXPECT_TRUE(mock_store->write_requests.empty());
    EXPECT_EQ("a", write_req.key);
    EXPECT_EQ(write_stamp.generation, write_req.options.if_equal);
    EXPECT_EQ("xyzdef", write_req.value);
    write_req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  TENSORSTORE_EXPECT_OK(transaction.future());
  EXPECT_THAT(memory_store->Read("a").result(),
              MatchesKvsReadResult(absl::Cord("xyzdef")));
}

TEST_F(MockStoreTest, MultiPhaseSameKeyAbort) {
  auto entry = GetCacheEntry(cache, "a");
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(entry->Modify(open_transaction, false, "abc"));
    open_transaction->Barrier();
    TENSORSTORE_ASSERT_OK(entry->Modify(open_transaction, false, "def"));
  }
  transaction.Abort();
}

TEST_F(MockStoreTest, DeleteRangeSingle) {
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", "c"}));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("a", "c"), req.range);
    req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  TENSORSTORE_EXPECT_OK(transaction.future());
}

TEST_F(MockStoreTest, DeleteRangeError) {
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", "c"}));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("a", "c"), req.range);
    req.promise.SetResult(absl::FailedPreconditionError("delete range error"));
  }
  ASSERT_TRUE(transaction.future().ready());
  EXPECT_THAT(transaction.future().result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "delete range error"));
}

TEST_F(MockStoreTest, DeleteRangeAtomicError) {
  auto transaction = Transaction(tensorstore::atomic_isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    EXPECT_THAT(mock_store->TransactionalDeleteRange(open_transaction,
                                                     KeyRange{"a", "c"}),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              "Cannot delete range starting at \"a\" as single "
                              "atomic transaction"));
  }
}

TEST_F(MockStoreTest, DeleteRangeMultipleDisjoint) {
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", "c"}));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"d", "f"}));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("a", "c"), req.range);
    req(memory_store);
  }
  ASSERT_FALSE(transaction.future().ready());
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("d", "f"), req.range);
    req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  TENSORSTORE_EXPECT_OK(transaction.future());
}

TEST_F(MockStoreTest, DeleteRangeMultipleOverlapping) {
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", "c"}));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"b", "f"}));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("a", "f"), req.range);
    req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  TENSORSTORE_EXPECT_OK(transaction.future());
}

TEST_F(MockStoreTest, DeleteRangeBeforeWrite) {
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", "c"}));
    TENSORSTORE_ASSERT_OK(
        GetCacheEntry(cache, "b")->Modify(open_transaction, false, "abc"));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("a", "b"), req.range);
    req(memory_store);
  }
  ASSERT_FALSE(transaction.future().ready());
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange(KeyRange::Successor("b"), "c"), req.range);
    req(memory_store);
  }
  ASSERT_FALSE(transaction.future().ready());
  {
    auto write_req = mock_store->write_requests.pop();
    EXPECT_TRUE(mock_store->read_requests.empty());
    EXPECT_TRUE(mock_store->write_requests.empty());
    EXPECT_EQ("b", write_req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), write_req.options.if_equal);
    EXPECT_THAT(write_req.value, ::testing::Optional(std::string("abc")));
    write_req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  TENSORSTORE_EXPECT_OK(transaction.future());
}

TEST_F(MockStoreTest, DeleteRangeBeforeWriteJustBeforeExclusiveMax) {
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", KeyRange::Successor("b")}));
    TENSORSTORE_ASSERT_OK(
        GetCacheEntry(cache, "b")->Modify(open_transaction, false, "abc"));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("a", "b"), req.range);
    req(memory_store);
  }
  ASSERT_FALSE(transaction.future().ready());
  {
    auto write_req = mock_store->write_requests.pop();
    EXPECT_TRUE(mock_store->read_requests.empty());
    EXPECT_TRUE(mock_store->write_requests.empty());
    EXPECT_EQ("b", write_req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), write_req.options.if_equal);
    EXPECT_EQ("abc", write_req.value);
    write_req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  TENSORSTORE_EXPECT_OK(transaction.future());
}

TEST_F(MockStoreTest, DeleteRangeAfterWrite) {
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(
        GetCacheEntry(cache, "b")->Modify(open_transaction, false, "abc"));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", "c"}));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("a", "c"), req.range);
    req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  TENSORSTORE_EXPECT_OK(transaction.future());
}

TEST_F(MockStoreTest, DeleteRangeAfterValidateError) {
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(
        GetCacheEntry(cache, "b")
            ->Validate(open_transaction, [](absl::Cord value) {
              return absl::FailedPreconditionError("validate error");
            }));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", "c"}));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  mock_store->read_requests.pop()(memory_store);
  EXPECT_TRUE(mock_store->read_requests.empty());
  EXPECT_TRUE(mock_store->write_requests.empty());
  EXPECT_TRUE(mock_store->delete_range_requests.empty());
  ASSERT_TRUE(transaction.future().ready());
  EXPECT_THAT(transaction.future().result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error writing \"b\": validate error"));
}

TEST_F(MockStoreTest, DeleteRangeAfterValidateAndModify) {
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(
        GetCacheEntry(cache, "b")
            ->Validate(open_transaction, [](const absl::Cord& input) {
              // This validator always succeeds, but by adding it we force a
              // read.
              return absl::OkStatus();
            }));
    TENSORSTORE_ASSERT_OK(
        GetCacheEntry(cache, "b")->Modify(open_transaction, false, "abc"));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", "c"}));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  {
    auto read_req = mock_store->read_requests.pop();
    EXPECT_TRUE(mock_store->read_requests.empty());
    EXPECT_TRUE(mock_store->write_requests.empty());
    EXPECT_TRUE(mock_store->delete_range_requests.empty());
    EXPECT_EQ("b", read_req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), read_req.options.if_not_equal);
    read_req(memory_store);
  }
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("a", "c"), req.range);
    req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  TENSORSTORE_EXPECT_OK(transaction.future());
}

TEST_F(MockStoreTest, MultiPhaseValidateError) {
  auto transaction = Transaction(tensorstore::isolated);
  auto entry = GetCacheEntry(cache, "a");
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(entry->Modify(open_transaction, true, "abc"));
    open_transaction->Barrier();
    auto validator = [](absl::Cord value) {
      if (value != "abc") {
        return absl::AbortedError("validation");
      }
      return absl::OkStatus();
    };
    TENSORSTORE_ASSERT_OK(entry->Validate(open_transaction, validator));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  {
    auto write_req = mock_store->write_requests.pop();
    EXPECT_EQ("a", write_req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), write_req.options.if_equal);
    write_req(memory_store);
  }
  TENSORSTORE_ASSERT_OK(memory_store->Write("a", absl::Cord("def")));
  ASSERT_FALSE(transaction.future().ready());
  // Read request as part of writeback to verify value hasn't changed (but it
  // has changed).
  {
    auto read_req = mock_store->read_requests.pop();
    EXPECT_EQ("a", read_req.key);
    EXPECT_EQ(tensorstore::OptionalByteRangeRequest(0, 0),
              read_req.options.byte_range);
    read_req(memory_store);
  }
  // Read request to obtain updated value.
  {
    auto read_req = mock_store->read_requests.pop();
    EXPECT_EQ("a", read_req.key);
    read_req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  EXPECT_THAT(transaction.future().result(),
              MatchesStatus(absl::StatusCode::kAborted));
}

TEST_F(MockStoreTest, MultiPhaseValidateErrorAfterReadValue) {
  auto transaction = Transaction(tensorstore::isolated);
  auto entry = GetCacheEntry(cache, "a");
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(entry->Modify(open_transaction, true, "abc"));
    open_transaction->Barrier();
    auto validator = [](absl::Cord value) {
      if (value != "abc") {
        return absl::AbortedError("validation: " + std::string(value));
      }
      return absl::OkStatus();
    };
    TENSORSTORE_ASSERT_OK(entry->Validate(open_transaction, validator));
    TENSORSTORE_ASSERT_OK(entry->Modify(open_transaction, true, "xyz"));
    TENSORSTORE_ASSERT_OK(entry->Validate(
        open_transaction, [](absl::Cord value) { return absl::OkStatus(); }));
    EXPECT_THAT(entry->ReadValue(open_transaction).result(),
                ::testing::Optional(absl::Cord("xyz")));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  {
    auto write_req = mock_store->write_requests.pop();
    EXPECT_EQ("a", write_req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), write_req.options.if_equal);
    write_req(memory_store);
  }
  TENSORSTORE_ASSERT_OK(memory_store->Write("a", absl::Cord("def")));
  ASSERT_FALSE(transaction.future().ready());
  // Writeback assuming "abc".
  {
    auto write_req = mock_store->write_requests.pop();
    EXPECT_EQ("a", write_req.key);
    write_req(memory_store);
  }
  // Read request to obtain updated value.
  {
    auto read_req = mock_store->read_requests.pop();
    EXPECT_EQ("a", read_req.key);
    read_req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  EXPECT_THAT(transaction.future().result(),
              MatchesStatus(absl::StatusCode::kAborted));
}

TEST_F(MockStoreTest, UnboundedDeleteRangeAfterWrite) {
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(
        GetCacheEntry(cache, "b")->Modify(open_transaction, false, "abc"));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", ""}));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("a", ""), req.range);
    req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  TENSORSTORE_EXPECT_OK(transaction.future());
}

TEST_F(MockStoreTest, DeleteRangeThenWriteThenDeleteRange) {
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", "c"}));
    TENSORSTORE_ASSERT_OK(
        GetCacheEntry(cache, "b")->Modify(open_transaction, false, "abc"));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", "d"}));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("a", "d"), req.range);
    req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  TENSORSTORE_EXPECT_OK(transaction.future());
}

TEST_F(MockStoreTest, MultiPhaseDeleteRangeOverlapEnd) {
  const std::vector<std::vector<KeyRange>> test_cases = {
      {
          KeyRange{"a", "c"},
          KeyRange{"a", "c"},
      },
      {
          KeyRange{"a", "c"},
          KeyRange{"a", "d"},
      },
      {
          KeyRange{"b", "c"},
          KeyRange{"a", "c"},
      },
      {
          KeyRange{"b", "c"},
          KeyRange{"a", "d"},
      },
      {
          KeyRange{"a", "d"},
          KeyRange{"b", "c"},
      },
  };
  for (const auto& test_case : test_cases) {
    SCOPED_TRACE("test_case=" + ::testing::PrintToString(test_case));
    auto transaction = Transaction(tensorstore::isolated);
    {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto open_transaction,
          tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
      for (const auto& range : test_case) {
        TENSORSTORE_ASSERT_OK(
            mock_store->TransactionalDeleteRange(open_transaction, range));
        open_transaction->Barrier();
      }
    }
    transaction.CommitAsync().IgnoreFuture();
    ASSERT_FALSE(transaction.future().ready());
    for (const auto& range : test_case) {
      auto req = mock_store->delete_range_requests.pop();
      EXPECT_TRUE(mock_store->delete_range_requests.empty());
      EXPECT_EQ(range, req.range);
      req(memory_store);
    }
    ASSERT_TRUE(transaction.future().ready());
    TENSORSTORE_EXPECT_OK(transaction.future());
  }
}

TEST_F(MockStoreTest, MultiPhaseDeleteRangeAndWrite) {
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", "c"}));
    open_transaction->Barrier();
    TENSORSTORE_ASSERT_OK(
        GetCacheEntry(cache, "b")->Modify(open_transaction, false, "abc"));
    TENSORSTORE_ASSERT_OK(mock_store->TransactionalDeleteRange(
        open_transaction, KeyRange{"a", "d"}));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_TRUE(mock_store->delete_range_requests.empty());
    EXPECT_EQ(KeyRange("a", "c"), req.range);
    req(memory_store);
  }
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("a", "d"), req.range);
    req(memory_store);
  }
  ASSERT_TRUE(transaction.future().ready());
  TENSORSTORE_EXPECT_OK(transaction.future());
}

TEST_F(MockStoreTest, MultipleKeyValueStoreAtomicError) {
  auto transaction = Transaction(tensorstore::atomic_isolated);
  tensorstore::KeyValueStore::PtrT<tensorstore::internal::MockKeyValueStore>
      mock_store2{new tensorstore::internal::MockKeyValueStore};
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(
        GetCacheEntry(cache, "x")->Modify(open_transaction, false, "abc"));
    EXPECT_THAT(GetCacheEntry(GetCache("", mock_store2), "y")
                    ->Modify(open_transaction, false, "abc"),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              "Cannot read/write \"x\" and read/write \"y\" as "
                              "single atomic transaction"));
  }
}

}  // namespace
