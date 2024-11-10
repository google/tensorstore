// Copyright 2024 The TensorStore Authors
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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/kvs_backed_cache_testutil.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {
namespace kvstore = ::tensorstore::kvstore;

using ::tensorstore::Context;
using ::tensorstore::IsOk;
using ::tensorstore::KeyRange;
using ::tensorstore::Result;
using ::tensorstore::StatusIs;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore::internal::MatchesListEntry;
using ::tensorstore::internal::MockKeyValueStore;
using ::tensorstore::internal::MockKeyValueStoreResource;
using ::tensorstore::kvstore::KvStore;
using ::testing::_;

class KvStackTest : public ::testing::Test {
 public:
  KvStackTest() : context_(Context::Default()) {}

  Result<KvStore> KvStoreOpen() const {
    return kvstore::Open(
               {{"driver", "kvstack"},
                {"layers",
                 ::nlohmann::json::array_t{
                     {
                         {"base", {{"driver", "memory"}, {"path", "range/"}}},
                     },
                     {
                         {"base", {{"driver", "memory"}, {"path", "prefix/"}}},
                         {"prefix", "a"},
                         {"strip_prefix", 0},
                     },
                     /**/
                 }}},
               context_)
        .result();
  }

  Context context_;
};

TEST_F(KvStackTest, ReadNotFound) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, KvStoreOpen());

  auto read_future = kvstore::Read(store, "abc");
  EXPECT_THAT(read_future.result(), MatchesKvsReadResultNotFound());
}

TEST_F(KvStackTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, KvStoreOpen());
  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST_F(KvStackTest, DeletePrefix) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, KvStoreOpen());
  tensorstore::internal::TestKeyValueStoreDeletePrefix(store);
}

TEST_F(KvStackTest, DeleteRange) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, KvStoreOpen());
  tensorstore::internal::TestKeyValueStoreDeleteRange(store);
}

TEST_F(KvStackTest, DeleteRangeToEnd) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, KvStoreOpen());
  tensorstore::internal::TestKeyValueStoreDeleteRangeToEnd(store);
}

TEST_F(KvStackTest, DeleteRangeFromBeginning) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, KvStoreOpen());
  tensorstore::internal::TestKeyValueStoreDeleteRangeFromBeginning(store);
}

TEST_F(KvStackTest, List) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, KvStoreOpen());
  tensorstore::internal::TestKeyValueStoreList(store);
}

TEST_F(KvStackTest, PrefixCheck) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base, kvstore::Open({{"driver", "memory"}}, context_).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, KvStoreOpen());

  const absl::Cord value("xyz");

  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "/a", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/b", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/d", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "aa/b", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "b", value));

  EXPECT_THAT(
      tensorstore::kvstore::ListFuture(base).value(),
      ::testing::UnorderedElementsAre(
          MatchesListEntry("range//a", _), MatchesListEntry("prefix/a/b", _),
          MatchesListEntry("prefix/a/d", _), MatchesListEntry("prefix/aa/b", _),
          MatchesListEntry("range/b", _)));
}

TEST_F(KvStackTest, DescribeKey) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, KvStoreOpen());

  EXPECT_EQ("\"range/c/d\"", store.driver->DescribeKey("c/d"));
  EXPECT_EQ("\"prefix/a/b\"", store.driver->DescribeKey("a/b"));
}

TEST_F(KvStackTest, TransactionalDeleteRange) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, KvStoreOpen());

  const absl::Cord value("xyz");
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "b", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "c", value));

  auto transaction = tensorstore::Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(store.driver->TransactionalDeleteRange(
        open_transaction, tensorstore::KeyRange{"a", "z"}));
  }
  TENSORSTORE_EXPECT_OK(transaction.Commit());

  EXPECT_THAT(tensorstore::kvstore::ListFuture(store).value(),
              ::testing::IsEmpty());
}

TEST_F(KvStackTest, TransactionalDeleteRangeWithMock) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context_.GetResource<MockKeyValueStoreResource>());
  MockKeyValueStore *mock_store = mock_key_value_store_resource->get();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open(
          {{"driver", "kvstack"},
           {"layers",
            ::nlohmann::json::array_t{
                {
                    {"base",
                     {{"driver", "mock_key_value_store"}, {"path", "base/"}}},
                },
                {
                    {"base",
                     {{"driver", "mock_key_value_store"}, {"path", "prefix/"}}},
                    {"prefix", "a"},
                },
            }}},
          context_)
          .result());

  auto transaction = tensorstore::Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(store.driver->TransactionalDeleteRange(
        open_transaction, tensorstore::KeyRange("a", "z")));
  }
  transaction.CommitAsync().IgnoreFuture();
  ASSERT_FALSE(transaction.future().ready());

  ASSERT_THAT(mock_store->delete_range_requests.size(), ::testing::Eq(2));
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("base/b", "base/z"), req.range);
    req.promise.SetResult(absl::OkStatus());
  }
  {
    auto req = mock_store->delete_range_requests.pop();
    EXPECT_EQ(KeyRange("prefix/", "prefix0"), req.range);
    req.promise.SetResult(absl::OkStatus());
  }

  ASSERT_TRUE(transaction.future().ready());
  EXPECT_THAT(transaction.future().result(), IsOk());
}

TEST_F(KvStackTest, ExperimentalCopyRange) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base, kvstore::Open("memory://", context_).result());

  const absl::Cord value("xyz");
  TENSORSTORE_EXPECT_OK(kvstore::Write(base, "src/a", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(base, "src/b", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(base, "src/c", value));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, KvStoreOpen());
  auto transaction = tensorstore::Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    kvstore::CopyRangeOptions options;
    options.source_range = tensorstore::KeyRange::Prefix("src/");
    EXPECT_THAT(
        store.driver
            ->ExperimentalCopyRangeFrom(open_transaction, base, "a/", options)
            .status(),
        StatusIs(absl::StatusCode::kUnimplemented));
  }
  transaction.Commit().IgnoreError();

  EXPECT_THAT(tensorstore::kvstore::ListFuture(base).value(),
              ::testing::UnorderedElementsAre(MatchesListEntry("src/a", _),
                                              MatchesListEntry("src/b", _),
                                              MatchesListEntry("src/c", _)));
}

// TODO: Test ReadModifyWrite

TEST(KvStackSpecTest, SpecRoundtrip) {
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  std::string metadata_json = "metadata.json";
  metadata_json.push_back('\0');
  options.create_spec = {{"driver", "kvstack"},
                         {"layers", ::nlohmann::json::array_t{
                                        {
                                            {"base", "memory://range/"},
                                        },
                                        {
                                            {"base", "memory://prefix/"},
                                            {"prefix", "a"},
                                        },
                                        {
                                            {"base", "memory://exact/m.json"},
                                            {"exact", "metadata.json"},
                                        },
                                        /**/
                                    }}};

  options.full_spec = {
      {"driver", "kvstack"},
      {"layers",
       ::nlohmann::json::array_t{
           {
               {"base", {{"driver", "memory"}, {"path", "range/"}}},
               {"exclusive_max", "a"},
           },
           {
               {"base", {{"driver", "memory"}, {"path", "prefix/"}}},
               {"prefix", "a"},
           },
           {
               {"base", {{"driver", "memory"}, {"path", "range/"}}},
               {"inclusive_min", "b"},
               {"exclusive_max", "metadata.json"},
           },
           {
               {"base", {{"driver", "memory"}, {"path", "exact/m.json"}}},
               {"exact", "metadata.json"},
           },
           {
               {"base", {{"driver", "memory"}, {"path", "range/"}}},
               {"inclusive_min", metadata_json},
           },
           /**/
       }}};

  options.check_data_after_serialization = false;
  options.check_data_persists = false;
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TENSORSTORE_GLOBAL_INITIALIZER {
  using ::tensorstore::internal::KvsBackedCacheBasicTransactionalTestOptions;
  using ::tensorstore::internal::RegisterKvsBackedCacheBasicTransactionalTest;

  KvsBackedCacheBasicTransactionalTestOptions options;
  options.test_name = "KvStackTransactional";
  options.get_store = [] {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto store,
        kvstore::Open(
            {{"driver", "kvstack"},
             {"layers",
              ::nlohmann::json::array_t{
                  {
                      {"base", {{"driver", "memory"}, {"path", "range/"}}},
                  },
                  {
                      {"base", {{"driver", "memory"}, {"path", "prefix/"}}},
                      {"prefix", "a"},
                  },
                  /**/
              }}})
            .result());
    return store.driver;
  };
  options.delete_range_supported = true;
  options.multi_key_atomic_supported = true;
  RegisterKvsBackedCacheBasicTransactionalTest(options);
}

}  // namespace
