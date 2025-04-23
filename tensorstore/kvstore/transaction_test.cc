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

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/clock.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = tensorstore::kvstore;

using ::tensorstore::JsonSubValuesMatch;
using ::tensorstore::KeyRange;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::OptionalByteRangeRequest;
using ::tensorstore::StorageGeneration;
using ::tensorstore::TimestampedStorageGeneration;
using ::tensorstore::Transaction;
using ::tensorstore::internal::MatchesKvsReadResult;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore::internal::MatchesListEntry;
using ::tensorstore::internal::MockKeyValueStore;
using ::tensorstore::kvstore::KvStore;
using ::tensorstore::kvstore::ReadResult;

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
    EXPECT_THAT(req.options.generation_conditions.if_equal,
                StorageGeneration::Unknown());
    req.promise.SetResult(TimestampedStorageGeneration(
        StorageGeneration::FromString("abc"), absl::Now()));
  }

  TENSORSTORE_ASSERT_OK(future);
}

TEST(KvStoreTest, ListWithUncommittedWrite) {
  auto mock_driver = MockKeyValueStore::Make();
  mock_driver->log_requests = true;
  mock_driver->forward_to = tensorstore::GetMemoryKeyValueStore();

  TENSORSTORE_ASSERT_OK(
      mock_driver->forward_to->Write("x", absl::Cord("value")));

  Transaction txn(tensorstore::isolated);

  KvStore store(mock_driver, "", txn);

  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "a", absl::Cord("value")));

  EXPECT_THAT(kvstore::ListFuture(store).result(),
              ::testing::Optional(::testing::UnorderedElementsAre(
                  MatchesListEntry("a", -1), MatchesListEntry("x", 5))));

  EXPECT_THAT(mock_driver->request_log.pop_all(),
              ::testing::ElementsAre(
                  MatchesJson({{"type", "list"}, {"range", {"", ""}}})));
}

TEST(KvStoreTest, ListWithUncommittedWriteOutsideRange) {
  auto mock_driver = MockKeyValueStore::Make();
  mock_driver->log_requests = true;
  mock_driver->forward_to = tensorstore::GetMemoryKeyValueStore();

  TENSORSTORE_ASSERT_OK(
      mock_driver->forward_to->Write("x", absl::Cord("value")));

  for (const auto &write_set :
       std::vector<std::vector<std::string>>{{}, {"a"}, {"z"}, {"a", "z"}}) {
    Transaction txn(tensorstore::isolated);

    KvStore store(mock_driver, "", txn);

    for (const auto &key : write_set) {
      TENSORSTORE_ASSERT_OK(kvstore::Write(store, key, absl::Cord("value")));
    }

    {
      kvstore::ListOptions options;
      options.range = KeyRange{"b", "y"};
      EXPECT_THAT(kvstore::ListFuture(store, std::move(options)).result(),
                  ::testing::Optional(::testing::UnorderedElementsAre(
                      MatchesListEntry("x", 5))));
    }

    EXPECT_THAT(mock_driver->request_log.pop_all(),
                ::testing::ElementsAre(
                    MatchesJson({{"type", "list"}, {"range", {"b", "y"}}})));
  }
}

TEST(KvStoreTest, ListWithUncommittedConditionalWrite) {
  auto mock_driver = MockKeyValueStore::Make();
  mock_driver->log_requests = true;
  mock_driver->forward_to = tensorstore::GetMemoryKeyValueStore();

  TENSORSTORE_ASSERT_OK(
      mock_driver->forward_to->Write("x", absl::Cord("value")));

  Transaction txn(tensorstore::isolated);

  KvStore store(mock_driver, "", txn);

  {
    kvstore::WriteOptions options;
    options.generation_conditions.if_equal = StorageGeneration::NoValue();
    auto write_future = kvstore::WriteCommitted(store, "a", absl::Cord("value"),
                                                std::move(options));
  }

  EXPECT_THAT(kvstore::ListFuture(store).result(),
              ::testing::Optional(::testing::UnorderedElementsAre(
                  MatchesListEntry("a", -1), MatchesListEntry("x", 5))));

  EXPECT_THAT(mock_driver->request_log.pop_all(),
              ::testing::ElementsAre(
                  JsonSubValuesMatch({{"/type", "list"}, {"/range", {"", ""}}}),
                  JsonSubValuesMatch({{"/type", "read"},
                                      {"/key", "a"},
                                      {"/byte_range_exclusive_max", 0}})));
}

TEST(KvStoreTest, ListWithCommittedAndUncommittedWrite) {
  auto mock_driver = MockKeyValueStore::Make();
  mock_driver->log_requests = true;
  mock_driver->forward_to = tensorstore::GetMemoryKeyValueStore();

  TENSORSTORE_ASSERT_OK(
      mock_driver->forward_to->Write("a", absl::Cord("value")));
  TENSORSTORE_ASSERT_OK(
      mock_driver->forward_to->Write("b", absl::Cord("value")));
  TENSORSTORE_ASSERT_OK(
      mock_driver->forward_to->Write("c", absl::Cord("value")));

  Transaction txn(tensorstore::isolated);

  KvStore store(mock_driver, "", txn);

  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "a", absl::Cord("value")));
  TENSORSTORE_ASSERT_OK(kvstore::Delete(store, "b"));
  EXPECT_THAT(kvstore::Read(store, "c").result(),
              ::testing::Optional(MatchesKvsReadResult(absl::Cord("value"))));

  EXPECT_THAT(
      mock_driver->request_log.pop_all(),
      ::testing::ElementsAre(MatchesJson({{"type", "read"}, {"key", "c"}})));

  EXPECT_THAT(kvstore::ListFuture(store).result(),
              ::testing::Optional(::testing::UnorderedElementsAre(
                  MatchesListEntry("a", -1), MatchesListEntry("c", 5))));

  EXPECT_THAT(mock_driver->request_log.pop_all(),
              ::testing::ElementsAre(
                  MatchesJson({{"type", "list"}, {"range", {"", ""}}})));
}

TEST(KvStoreTest, DeleteRangeThenList) {
  auto mock_driver = MockKeyValueStore::Make();
  mock_driver->log_requests = true;
  mock_driver->forward_to = tensorstore::GetMemoryKeyValueStore();

  TENSORSTORE_ASSERT_OK(
      mock_driver->forward_to->Write("1", absl::Cord("value")));
  TENSORSTORE_ASSERT_OK(
      mock_driver->forward_to->Write("a1", absl::Cord("value")));
  TENSORSTORE_ASSERT_OK(
      mock_driver->forward_to->Write("b1", absl::Cord("value")));
  TENSORSTORE_ASSERT_OK(
      mock_driver->forward_to->Write("c1", absl::Cord("value")));
  TENSORSTORE_ASSERT_OK(
      mock_driver->forward_to->Write("d1", absl::Cord("value")));

  Transaction txn(tensorstore::isolated);

  KvStore store(mock_driver, "", txn);

  TENSORSTORE_ASSERT_OK(kvstore::DeleteRange(store, KeyRange{"a", "b"}));
  TENSORSTORE_ASSERT_OK(kvstore::DeleteRange(store, KeyRange{"c", "d"}));

  EXPECT_THAT(kvstore::ListFuture(store).result(),
              ::testing::Optional(::testing::UnorderedElementsAre(
                  MatchesListEntry("1", 5), MatchesListEntry("b1", 5),
                  MatchesListEntry("d1", 5))));

  EXPECT_THAT(mock_driver->request_log.pop_all(),
              ::testing::ElementsAre(
                  MatchesJson({{"type", "list"}, {"range", {"", "a"}}}),
                  MatchesJson({{"type", "list"}, {"range", {"b", "c"}}}),
                  MatchesJson({{"type", "list"}, {"range", {"d", ""}}})));
}

TEST(KvStoreTest, ReadWithoutRepeatableReadIsolation) {
  auto mock_driver = MockKeyValueStore::Make();

  Transaction txn(tensorstore::isolated);

  KvStore store(mock_driver, "", txn);

  {
    auto read_future = kvstore::Read(store, "a");

    {
      auto req = mock_driver->read_requests.pop();
      EXPECT_THAT(req.key, "a");
      req.promise.SetResult(ReadResult::Value(
          absl::Cord("value"),
          TimestampedStorageGeneration(StorageGeneration::FromString("abc"),
                                       absl::Now())));
    }

    EXPECT_THAT(read_future.result(),
                ::testing::Optional(MatchesKvsReadResult(absl::Cord("value"))));
  }

  TENSORSTORE_ASSERT_OK(txn.CommitAsync().result());
}

TEST(KvStoreTest, ReadWithRepeatableReadIsolation) {
  auto mock_driver = MockKeyValueStore::Make();

  Transaction txn(tensorstore::isolated | tensorstore::repeatable_read);

  KvStore store(mock_driver, "", txn);

  {
    auto read_future = kvstore::Read(store, "a");

    {
      auto req = mock_driver->read_requests.pop();
      EXPECT_THAT(req.key, "a");
      req.promise.SetResult(ReadResult::Value(
          absl::Cord("value"),
          TimestampedStorageGeneration(StorageGeneration::FromString("abc"),
                                       absl::Now())));
    }

    EXPECT_THAT(read_future.result(),
                ::testing::Optional(MatchesKvsReadResult(absl::Cord("value"))));
  }

  auto future = txn.CommitAsync();

  {
    auto req = mock_driver->read_requests.pop();
    EXPECT_THAT(req.key, "a");
    EXPECT_THAT(req.options.byte_range, OptionalByteRangeRequest::Stat());
    EXPECT_THAT(req.options.generation_conditions.if_not_equal,
                StorageGeneration::FromString("abc"));
    req.promise.SetResult(ReadResult::Unspecified(TimestampedStorageGeneration(
        StorageGeneration::FromString("abc"), absl::Now())));
  }

  TENSORSTORE_ASSERT_OK(future);
}

TEST(KvStoreTest, ByteRangeRead) {
  auto mock_driver = MockKeyValueStore::Make();
  tensorstore::kvstore::DriverPtr memory_store =
      tensorstore::GetMemoryKeyValueStore();
  mock_driver->forward_to = memory_store;
  mock_driver->log_requests = true;

  Transaction txn(tensorstore::isolated);

  KvStore store(mock_driver, "", txn);

  {
    kvstore::ReadOptions options;
    options.byte_range = tensorstore::OptionalByteRangeRequest{1, 3};
    EXPECT_THAT(kvstore::Read(store, "a", options).result(),
                MatchesKvsReadResultNotFound());
    EXPECT_THAT(mock_driver->request_log.pop_all(),
                ::testing::ElementsAreArray({
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "a"},
                                        {"/byte_range_inclusive_min", 1},
                                        {"/byte_range_exclusive_max", 3}}),
                }));
  }

  TENSORSTORE_ASSERT_OK(
      memory_store->Write("a", absl::Cord("0123456789")).result());

  {
    kvstore::ReadOptions options;
    options.byte_range = tensorstore::OptionalByteRangeRequest{1, 3};
    EXPECT_THAT(kvstore::Read(store, "a", options).result(),
                MatchesKvsReadResult(absl::Cord("12")));
    EXPECT_THAT(mock_driver->request_log.pop_all(),
                ::testing::ElementsAreArray({
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "a"},
                                        {"/byte_range_inclusive_min", 1},
                                        {"/byte_range_exclusive_max", 3}}),
                }));
  }
}

TEST(KvStoreTest, ByteRangeRepeatableReadSuccess) {
  auto mock_driver = MockKeyValueStore::Make();
  tensorstore::kvstore::DriverPtr memory_store =
      tensorstore::GetMemoryKeyValueStore();
  mock_driver->forward_to = memory_store;
  mock_driver->log_requests = true;

  Transaction txn(tensorstore::isolated | tensorstore::repeatable_read);

  KvStore store(mock_driver, "", txn);

  TENSORSTORE_ASSERT_OK(
      memory_store->Write("a", absl::Cord("0123456789")).result());

  {
    kvstore::ReadOptions options;
    options.byte_range = tensorstore::OptionalByteRangeRequest{1, 3};
    EXPECT_THAT(kvstore::Read(store, "a", options).result(),
                MatchesKvsReadResult(absl::Cord("12")));
    EXPECT_THAT(mock_driver->request_log.pop_all(),
                ::testing::ElementsAreArray({
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "a"},
                                        {"/byte_range_inclusive_min", 1},
                                        {"/byte_range_exclusive_max", 3}}),
                }));
  }

  {
    kvstore::ReadOptions options;
    options.byte_range = tensorstore::OptionalByteRangeRequest{1, 3};
    EXPECT_THAT(kvstore::Read(store, "a", options).result(),
                MatchesKvsReadResult(absl::Cord("12")));
    EXPECT_THAT(mock_driver->request_log.pop_all(),
                ::testing::ElementsAreArray({
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "a"},
                                        {"/byte_range_inclusive_min", 1},
                                        {"/byte_range_exclusive_max", 3}}),
                }));
  }

  TENSORSTORE_EXPECT_OK(txn.CommitAsync().result());
  EXPECT_THAT(mock_driver->request_log.pop_all(),
              ::testing::ElementsAreArray({
                  JsonSubValuesMatch({{"/type", "read"},
                                      {"/key", "a"},
                                      {"/byte_range_exclusive_max", 0}}),
              }));
}

TEST(KvStoreTest, ZeroByteRangeRepeatableReadSuccess) {
  auto mock_driver = MockKeyValueStore::Make();
  tensorstore::kvstore::DriverPtr memory_store =
      tensorstore::GetMemoryKeyValueStore();
  mock_driver->forward_to = memory_store;
  mock_driver->log_requests = true;

  Transaction txn(tensorstore::isolated | tensorstore::repeatable_read);

  KvStore store(mock_driver, "", txn);

  TENSORSTORE_ASSERT_OK(
      memory_store->Write("a", absl::Cord("0123456789")).result());

  {
    kvstore::ReadOptions options;
    options.byte_range = tensorstore::OptionalByteRangeRequest::Stat();
    EXPECT_THAT(kvstore::Read(store, "a", options).result(),
                MatchesKvsReadResult(absl::Cord("")));
    EXPECT_THAT(mock_driver->request_log.pop_all(),
                ::testing::ElementsAreArray({
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "a"},
                                        {"/byte_range_exclusive_max", 0}}),
                }));
  }
}

TEST(KvStoreTest, ZeroByteRangeRepeatableReadNotFound) {
  auto mock_driver = MockKeyValueStore::Make();
  tensorstore::kvstore::DriverPtr memory_store =
      tensorstore::GetMemoryKeyValueStore();
  mock_driver->forward_to = memory_store;
  mock_driver->log_requests = true;

  Transaction txn(tensorstore::isolated | tensorstore::repeatable_read);

  KvStore store(mock_driver, "", txn);

  {
    kvstore::ReadOptions options;
    options.byte_range = tensorstore::OptionalByteRangeRequest::Stat();
    EXPECT_THAT(kvstore::Read(store, "a", options).result(),
                MatchesKvsReadResultNotFound());
    EXPECT_THAT(mock_driver->request_log.pop_all(),
                ::testing::ElementsAreArray({
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "a"},
                                        {"/byte_range_exclusive_max", 0}}),
                }));
  }
}

TEST(KvStoreTest, ByteRangeRepeatableReadMismatchBeforeCommit) {
  auto mock_driver = MockKeyValueStore::Make();
  tensorstore::kvstore::DriverPtr memory_store =
      tensorstore::GetMemoryKeyValueStore();
  mock_driver->forward_to = memory_store;
  mock_driver->log_requests = true;

  Transaction txn(tensorstore::isolated | tensorstore::repeatable_read);

  KvStore store(mock_driver, "", txn);

  TENSORSTORE_ASSERT_OK(
      memory_store->Write("a", absl::Cord("0123456789")).result());

  {
    kvstore::ReadOptions options;
    options.byte_range = tensorstore::OptionalByteRangeRequest{1, 3};
    EXPECT_THAT(kvstore::Read(store, "a", options).result(),
                MatchesKvsReadResult(absl::Cord("12")));
    EXPECT_THAT(mock_driver->request_log.pop_all(),
                ::testing::ElementsAreArray({
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "a"},
                                        {"/byte_range_inclusive_min", 1},
                                        {"/byte_range_exclusive_max", 3}}),
                }));
  }

  TENSORSTORE_ASSERT_OK(
      memory_store->Write("a", absl::Cord("0123456789x")).result());

  {
    kvstore::ReadOptions options;
    options.byte_range = tensorstore::OptionalByteRangeRequest{1, 3};
    EXPECT_THAT(
        kvstore::Read(store, "a", options).result(),
        MatchesStatus(absl::StatusCode::kAborted,
                      "Generation mismatch in repeatable_read transaction"));
    EXPECT_THAT(mock_driver->request_log.pop_all(),
                ::testing::ElementsAreArray({
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "a"},
                                        {"/byte_range_inclusive_min", 1},
                                        {"/byte_range_exclusive_max", 3}}),
                }));
  }

  EXPECT_THAT(txn.CommitAsync().result(),
              MatchesStatus(absl::StatusCode::kAborted,
                            "Error reading \"a\": Generation mismatch in "
                            "repeatable_read transaction"));
  EXPECT_THAT(mock_driver->request_log.pop_all(), ::testing::ElementsAre());
}

TEST(KvStoreTest, ByteRangeRepeatableReadMismatchDuringCommit) {
  auto mock_driver = MockKeyValueStore::Make();
  tensorstore::kvstore::DriverPtr memory_store =
      tensorstore::GetMemoryKeyValueStore();
  mock_driver->forward_to = memory_store;
  mock_driver->log_requests = true;

  Transaction txn(tensorstore::isolated | tensorstore::repeatable_read);

  KvStore store(mock_driver, "", txn);

  TENSORSTORE_ASSERT_OK(
      memory_store->Write("a", absl::Cord("0123456789")).result());

  {
    kvstore::ReadOptions options;
    options.byte_range = tensorstore::OptionalByteRangeRequest{1, 3};
    EXPECT_THAT(kvstore::Read(store, "a", options).result(),
                MatchesKvsReadResult(absl::Cord("12")));
    EXPECT_THAT(mock_driver->request_log.pop_all(),
                ::testing::ElementsAreArray({
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "a"},
                                        {"/byte_range_inclusive_min", 1},
                                        {"/byte_range_exclusive_max", 3}}),
                }));
  }

  TENSORSTORE_ASSERT_OK(
      memory_store->Write("a", absl::Cord("0123456789x")).result());

  EXPECT_THAT(txn.CommitAsync().result(),
              MatchesStatus(absl::StatusCode::kAborted,
                            "Error writing \"a\": Generation mismatch"));
  EXPECT_THAT(mock_driver->request_log.pop_all(),
              ::testing::ElementsAreArray({
                  JsonSubValuesMatch({{"/type", "read"},
                                      {"/key", "a"},
                                      {"/byte_range_exclusive_max", 0}}),
              }));
}

TEST(KvStoreTest, ReadMismatch) {
  auto mock_driver = MockKeyValueStore::Make();

  Transaction txn(tensorstore::isolated | tensorstore::repeatable_read);

  KvStore store(mock_driver, "", txn);

  {
    auto read_future = kvstore::Read(store, "a");

    {
      auto req = mock_driver->read_requests.pop();
      EXPECT_THAT(req.key, "a");
      req.promise.SetResult(ReadResult::Value(
          absl::Cord("value"),
          TimestampedStorageGeneration(StorageGeneration::FromString("abc"),
                                       absl::Now())));
    }

    EXPECT_THAT(read_future.result(),
                ::testing::Optional(MatchesKvsReadResult(absl::Cord("value"))));
  }

  auto future = txn.CommitAsync();

  // Initial writeback
  {
    auto req = mock_driver->read_requests.pop();
    EXPECT_THAT(req.key, "a");
    EXPECT_THAT(req.options.byte_range, OptionalByteRangeRequest::Stat());
    EXPECT_THAT(req.options.generation_conditions.if_not_equal,
                StorageGeneration::FromString("abc"));
    req.promise.SetResult(ReadResult::Missing(TimestampedStorageGeneration(
        StorageGeneration::FromString("def"), absl::Now())));
  }

  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kAborted,
                            "Error writing \"a\": Generation mismatch"));
}

TEST(KvStoreTest, ListInvalid) {
  auto mock_driver = MockKeyValueStore::Make();

  Transaction txn(tensorstore::isolated | tensorstore::repeatable_read);
  KvStore store(mock_driver, "", txn);

  EXPECT_THAT(kvstore::ListFuture(store).result(),
              MatchesStatus(absl::StatusCode::kUnimplemented));
}

}  // namespace
