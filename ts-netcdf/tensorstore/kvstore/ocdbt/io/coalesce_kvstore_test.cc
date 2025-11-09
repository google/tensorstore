// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/kvstore/ocdbt/io/coalesce_kvstore.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/thread/thread_pool.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;
using ::tensorstore::Context;
using ::tensorstore::OptionalByteRangeRequest;
using ::tensorstore::internal::MockKeyValueStore;
using ::tensorstore::internal_ocdbt::MakeCoalesceKvStoreDriver;
using ::tensorstore::kvstore::ReadOptions;

TEST(CoalesceKvstoreTest, SimpleRead) {
  // make sure a simple write then read can be done properly
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_store,
                                   kvstore::Open("memory://").result());

  auto mock_key_value_store = MockKeyValueStore::Make();

  auto coalesce_driver = MakeCoalesceKvStoreDriver(
      mock_key_value_store, /*threshold=*/100, /*merged_threshold=*/0,
      /*interval=*/absl::ZeroDuration(),
      tensorstore::internal::DetachedThreadPool(1));

  auto write_future = kvstore::Write(coalesce_driver, "a", absl::Cord("a"));
  write_future.Force();
  {
    auto req = mock_key_value_store->write_requests.pop();
    EXPECT_EQ("a", req.key);
    req(base_store.driver);
  }

  auto read_future = kvstore::Read(coalesce_driver, "a");
  read_future.Force();
  {
    auto req = mock_key_value_store->read_requests.pop();
    EXPECT_EQ("a", req.key);
    req(base_store.driver);
  }
  ASSERT_TRUE(read_future.result().has_value());
  ASSERT_TRUE(read_future.result().value().has_value());
  EXPECT_EQ(read_future.result().value().value, absl::Cord("a"));
}

TEST(CoalesceKvstoreTest, ReadWithThreshold) {
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_store,
                                   kvstore::Open("memory://").result());

  auto mock_key_value_store = MockKeyValueStore::Make();

  auto coalesce_driver = MakeCoalesceKvStoreDriver(
      mock_key_value_store, /*threshold=*/1, /*merged_threshold=*/0,
      /*interval=*/absl::ZeroDuration(),
      tensorstore::internal::DetachedThreadPool(1));

  auto write_future =
      kvstore::Write(coalesce_driver, "a", absl::Cord("0123456789"));
  write_future.Force();
  {
    auto req = mock_key_value_store->write_requests.pop();
    EXPECT_EQ("a", req.key);
    req(base_store.driver);
  }

  ReadOptions ro1, ro2, ro3, ro4;
  ro1.byte_range =
      OptionalByteRangeRequest(0, 1);  // 1st read will not be coalesced
  ro2.byte_range = OptionalByteRangeRequest(2, 3);  // coalesced
  ro3.byte_range = OptionalByteRangeRequest(4, 5);  /// coalesced
  ro4.byte_range =
      OptionalByteRangeRequest(7, 8);  // out of threshold to be coalesced

  auto read_future1 = kvstore::Read(coalesce_driver, "a", ro1);
  auto read_future2 = kvstore::Read(coalesce_driver, "a", ro2);
  auto read_future3 = kvstore::Read(coalesce_driver, "a", ro3);
  auto read_future4 = kvstore::Read(coalesce_driver, "a", ro4);

  {
    auto req = mock_key_value_store->read_requests.pop();
    EXPECT_EQ("a", req.key);
    EXPECT_EQ(req.options.byte_range, ro1.byte_range);
    req(base_store.driver);
  }
  TENSORSTORE_EXPECT_OK(read_future1.result());
  EXPECT_EQ(read_future1.result().value().value, absl::Cord("0"));

  {
    auto req = mock_key_value_store->read_requests.pop();
    EXPECT_EQ("a", req.key);
    // merged range
    EXPECT_EQ(req.options.byte_range, OptionalByteRangeRequest(2, 5));
    req(base_store.driver);
  }
  TENSORSTORE_EXPECT_OK(read_future2.result());
  EXPECT_EQ(read_future2.result().value().value, absl::Cord("2"));

  TENSORSTORE_EXPECT_OK(read_future3.result());
  EXPECT_EQ(read_future3.result().value().value, absl::Cord("4"));

  {
    auto req = mock_key_value_store->read_requests.pop();
    EXPECT_EQ("a", req.key);
    EXPECT_EQ(req.options.byte_range, OptionalByteRangeRequest(7, 8));
    req(base_store.driver);
  }
  TENSORSTORE_EXPECT_OK(read_future4.result());
  EXPECT_EQ(read_future4.result().value().value, absl::Cord("7"));
}

TEST(CoalesceKvstoreTest, ReadWithMergedThreshold) {
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_store,
                                   kvstore::Open("memory://").result());

  auto mock_key_value_store = MockKeyValueStore::Make();

  auto coalesce_driver = MakeCoalesceKvStoreDriver(
      mock_key_value_store, /*threshold=*/1, /*merged_threshold=*/2,
      /*interval=*/absl::ZeroDuration(),
      tensorstore::internal::DetachedThreadPool(1));

  auto write_future =
      kvstore::Write(coalesce_driver, "a", absl::Cord("0123456789"));
  write_future.Force();
  {
    auto req = mock_key_value_store->write_requests.pop();
    EXPECT_EQ("a", req.key);
    req(base_store.driver);
  }

  ReadOptions ro1, ro2, ro3, ro4, ro5;
  ro1.byte_range =
      OptionalByteRangeRequest(0, 1);  // 1st read will not be coalesced
  ro2.byte_range = OptionalByteRangeRequest(2, 3);  // coalesced in 2nd read
  ro3.byte_range = OptionalByteRangeRequest(4, 5);  // coalesced in 2nd read
  ro4.byte_range = OptionalByteRangeRequest(6, 7);  // coalesced in 3rd read
  ro5.byte_range = OptionalByteRangeRequest(8, 9);  // coalesced in 3rd read

  auto read_future1 = kvstore::Read(coalesce_driver, "a", ro1);
  auto read_future2 = kvstore::Read(coalesce_driver, "a", ro2);
  auto read_future3 = kvstore::Read(coalesce_driver, "a", ro3);
  auto read_future4 = kvstore::Read(coalesce_driver, "a", ro4);
  auto read_future5 = kvstore::Read(coalesce_driver, "a", ro5);

  {
    auto req = mock_key_value_store->read_requests.pop();
    EXPECT_EQ("a", req.key);
    EXPECT_EQ(req.options.byte_range, ro1.byte_range);
    req(base_store.driver);
  }
  TENSORSTORE_EXPECT_OK(read_future1.result());
  EXPECT_EQ(read_future1.result().value().value, absl::Cord("0"));

  {
    auto req = mock_key_value_store->read_requests.pop();
    EXPECT_EQ("a", req.key);
    // merged range
    EXPECT_EQ(req.options.byte_range, OptionalByteRangeRequest(2, 5));
    req(base_store.driver);
  }
  TENSORSTORE_EXPECT_OK(read_future2.result());
  EXPECT_EQ(read_future2.result().value().value, absl::Cord("2"));

  TENSORSTORE_EXPECT_OK(read_future3.result());
  EXPECT_EQ(read_future3.result().value().value, absl::Cord("4"));

  {
    auto req = mock_key_value_store->read_requests.pop();
    EXPECT_EQ("a", req.key);
    EXPECT_EQ(req.options.byte_range, OptionalByteRangeRequest(6, 9));
    req(base_store.driver);
  }
  TENSORSTORE_EXPECT_OK(read_future4.result());
  EXPECT_EQ(read_future4.result().value().value, absl::Cord("6"));
  TENSORSTORE_EXPECT_OK(read_future5.result());
  EXPECT_EQ(read_future5.result().value().value, absl::Cord("8"));
}

TEST(CoalesceKvstoreTest, ReadWithInterval) {
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_store,
                                   kvstore::Open("memory://").result());

  auto mock_key_value_store = MockKeyValueStore::Make();

  auto coalesce_driver = MakeCoalesceKvStoreDriver(
      mock_key_value_store, /*threshold=*/1, /*merged_threshold=*/0,
      /*interval=*/absl::Milliseconds(10),
      tensorstore::internal::DetachedThreadPool(1));

  auto write_future =
      kvstore::Write(coalesce_driver, "a", absl::Cord("0123456789"));
  write_future.Force();
  {
    auto req = mock_key_value_store->write_requests.pop();
    EXPECT_EQ("a", req.key);
    req(base_store.driver);
  }

  ReadOptions ro1, ro2, ro3, ro4;
  ro1.byte_range = OptionalByteRangeRequest(0, 1);  // will be coalesced as well
  ro2.byte_range = OptionalByteRangeRequest(2, 3);  // coalesced
  ro3.byte_range = OptionalByteRangeRequest(4, 5);  /// coalesced
  ro4.byte_range =
      OptionalByteRangeRequest(7, 8);  // out of threshold to be coalesced

  auto read_future1 = kvstore::Read(coalesce_driver, "a", ro1);
  auto read_future2 = kvstore::Read(coalesce_driver, "a", ro2);
  auto read_future3 = kvstore::Read(coalesce_driver, "a", ro3);
  auto read_future4 = kvstore::Read(coalesce_driver, "a", ro4);

  {
    auto req = mock_key_value_store->read_requests.pop();
    EXPECT_EQ("a", req.key);
    EXPECT_EQ(req.options.byte_range, OptionalByteRangeRequest(0, 5));
    req(base_store.driver);
  }
  TENSORSTORE_EXPECT_OK(read_future1.result());
  EXPECT_EQ(read_future1.result().value().value, absl::Cord("0"));

  TENSORSTORE_EXPECT_OK(read_future2.result());
  EXPECT_EQ(read_future2.result().value().value, absl::Cord("2"));

  TENSORSTORE_EXPECT_OK(read_future3.result());
  EXPECT_EQ(read_future3.result().value().value, absl::Cord("4"));

  {
    auto req = mock_key_value_store->read_requests.pop();
    EXPECT_EQ("a", req.key);
    EXPECT_EQ(req.options.byte_range, OptionalByteRangeRequest(7, 8));
    req(base_store.driver);
  }
  TENSORSTORE_EXPECT_OK(read_future4.result());
  EXPECT_EQ(read_future4.result().value().value, absl::Cord("7"));
}

}  // namespace
