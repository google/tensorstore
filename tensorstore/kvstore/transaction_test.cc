// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/transaction.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = tensorstore::kvstore;

using ::tensorstore::MatchesStatus;
using ::tensorstore::OptionalByteRangeRequest;
using ::tensorstore::StorageGeneration;
using ::tensorstore::TimestampedStorageGeneration;
using ::tensorstore::Transaction;
using ::tensorstore::internal::MatchesKvsReadResult;
using ::tensorstore::internal::MockKeyValueStore;
using ::tensorstore::kvstore::KvStore;

TEST(KvStoreTest, WriteThenRead) {
  auto mock_driver = MockKeyValueStore::Make();

  Transaction txn(tensorstore::isolated);

  KvStore store(mock_driver, "", txn);

  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "a", absl::Cord("value")));

  EXPECT_THAT(kvstore::Read(store, "a").result(),
              ::testing::Optional(MatchesKvsReadResult(absl::Cord("value"))));

  auto future = txn.CommitAsync();

  {
    auto req = mock_driver->write_requests.pop();
    EXPECT_THAT(req.key, "a");
    EXPECT_THAT(req.value, ::testing::Optional(absl::Cord("value")));
    EXPECT_THAT(req.options.if_equal, StorageGeneration::Unknown());
    req.promise.SetResult(TimestampedStorageGeneration(
        StorageGeneration::FromString("abc"), absl::Now()));
  }

  TENSORSTORE_ASSERT_OK(future);
}

TEST(KvStoreTest, Read) {
  auto mock_driver = MockKeyValueStore::Make();

  Transaction txn(tensorstore::isolated);

  KvStore store(mock_driver, "", txn);

  {
    auto read_future = kvstore::Read(store, "a");

    {
      auto req = mock_driver->read_requests.pop();
      EXPECT_THAT(req.key, "a");
      req.promise.SetResult(
          std::in_place, kvstore::ReadResult::kValue, absl::Cord("value"),
          TimestampedStorageGeneration(StorageGeneration::FromString("abc"),
                                       absl::Now()));
    }

    EXPECT_THAT(read_future.result(),
                ::testing::Optional(MatchesKvsReadResult(absl::Cord("value"))));
  }

  auto future = txn.CommitAsync();

  {
    auto req = mock_driver->read_requests.pop();
    EXPECT_THAT(req.key, "a");
    EXPECT_THAT(req.options.byte_range, OptionalByteRangeRequest(0, 0));
    EXPECT_THAT(req.options.if_not_equal, StorageGeneration::FromString("abc"));
    req.promise.SetResult(
        std::in_place, TimestampedStorageGeneration(
                           StorageGeneration::FromString("abc"), absl::Now()));
  }

  TENSORSTORE_ASSERT_OK(future);
}

TEST(KvStoreTest, ReadInvalidOptionIfEqual) {
  auto mock_driver = MockKeyValueStore::Make();

  Transaction txn(tensorstore::isolated);

  KvStore store(mock_driver, "", txn);
  kvstore::ReadOptions options;
  options.if_equal = StorageGeneration::FromString("abc");

  EXPECT_THAT(kvstore::Read(store, "a", std::move(options)).result(),
              MatchesStatus(absl::StatusCode::kUnimplemented));
}

TEST(KvStoreTest, ReadInvalidOptionByteRange) {
  auto mock_driver = MockKeyValueStore::Make();

  Transaction txn(tensorstore::isolated);

  KvStore store(mock_driver, "", txn);
  kvstore::ReadOptions options;
  options.byte_range = {5, 10};

  EXPECT_THAT(kvstore::Read(store, "a", std::move(options)).result(),
              MatchesStatus(absl::StatusCode::kUnimplemented));
}

TEST(KvStoreTest, ReadMismatch) {
  auto mock_driver = MockKeyValueStore::Make();

  Transaction txn(tensorstore::isolated);

  KvStore store(mock_driver, "", txn);

  {
    auto read_future = kvstore::Read(store, "a");

    {
      auto req = mock_driver->read_requests.pop();
      EXPECT_THAT(req.key, "a");
      req.promise.SetResult(
          std::in_place, kvstore::ReadResult::kValue, absl::Cord("value"),
          TimestampedStorageGeneration(StorageGeneration::FromString("abc"),
                                       absl::Now()));
    }

    EXPECT_THAT(read_future.result(),
                ::testing::Optional(MatchesKvsReadResult(absl::Cord("value"))));
  }

  auto future = txn.CommitAsync();

  // Initial writeback
  {
    auto req = mock_driver->read_requests.pop();
    EXPECT_THAT(req.key, "a");
    EXPECT_THAT(req.options.byte_range, OptionalByteRangeRequest(0, 0));
    EXPECT_THAT(req.options.if_not_equal, StorageGeneration::FromString("abc"));
    req.promise.SetResult(
        std::in_place, kvstore::ReadResult::kMissing, absl::Cord(),
        TimestampedStorageGeneration(StorageGeneration::FromString("def"),
                                     absl::Now()));
  }

  // Re-read by `ReadViaExistingTransaction`.  This read is actually redundant,
  // but is required because we don't currently have a mechanism for providing
  // an updated read result to a `ReadModifyWriteSource` in response to a failed
  // writeback.
  {
    auto req = mock_driver->read_requests.pop();
    EXPECT_THAT(req.key, "a");
    req.promise.SetResult(
        std::in_place, kvstore::ReadResult::kMissing, absl::Cord(),
        TimestampedStorageGeneration(StorageGeneration::FromString("def"),
                                     absl::Now()));
  }

  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kAborted,
                            "Error writing \"a\": Generation mismatch"));
}

TEST(KvStoreTest, ListInvalid) {
  auto mock_driver = MockKeyValueStore::Make();

  Transaction txn(tensorstore::isolated);
  KvStore store(mock_driver, "", txn);

  EXPECT_THAT(kvstore::ListFuture(store).result(),
              MatchesStatus(absl::StatusCode::kUnimplemented));
}

}  // namespace
