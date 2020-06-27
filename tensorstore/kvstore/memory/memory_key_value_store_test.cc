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

#include "tensorstore/kvstore/memory/memory_key_value_store.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/cache_key.h"
#include "tensorstore/kvstore/key_value_store_testutil.h"
#include "tensorstore/util/execution.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/sender.h"
#include "tensorstore/util/sender_testutil.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::Context;
using tensorstore::KeyRange;
using tensorstore::KeyValueStore;
using tensorstore::MatchesStatus;
using tensorstore::Status;
using tensorstore::internal::MatchesKvsReadResult;
using tensorstore::internal::MatchesKvsReadResultNotFound;

TEST(MemoryKeyValueStoreTest, Basic) {
  auto store = tensorstore::GetMemoryKeyValueStore();
  tensorstore::internal::TestKeyValueStoreBasicFunctionality(store);
}

TEST(MemoryKeyValueStoreTest, DeleteRange) {
  auto store = tensorstore::GetMemoryKeyValueStore();
  TENSORSTORE_EXPECT_OK(store->Write("a/b", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/d", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/x", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/y", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/z/e", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/z/f", absl::Cord("xyz")));

  TENSORSTORE_EXPECT_OK(store->DeleteRange(KeyRange::Prefix("a/c")));

  EXPECT_EQ("xyz", store->Read("a/b").value().value);
  EXPECT_EQ("xyz", store->Read("a/d").value().value);

  EXPECT_THAT(store->Read("a/c/x").result(), MatchesKvsReadResultNotFound());
  EXPECT_THAT(store->Read("a/c/y").result(), MatchesKvsReadResultNotFound());
  EXPECT_THAT(store->Read("a/c/z/e").result(), MatchesKvsReadResultNotFound());
  EXPECT_THAT(store->Read("a/c/z/f").result(), MatchesKvsReadResultNotFound());
}

TEST(MemoryKeyValueStoreTest, List) {
  auto store = tensorstore::GetMemoryKeyValueStore();

  {
    std::vector<std::string> log;
    tensorstore::execution::submit(store->List({}),
                                   tensorstore::LoggingReceiver{&log});
    EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_done",
                                            "set_stopping"));
  }

  TENSORSTORE_EXPECT_OK(store->Write("a/b", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/d", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/x", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/y", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/z/e", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/z/f", absl::Cord("xyz")));

  // Listing the entire stream works.
  {
    std::vector<std::string> log;
    tensorstore::execution::submit(store->List({}),
                                   tensorstore::LoggingReceiver{&log});

    EXPECT_THAT(
        log, ::testing::UnorderedElementsAre(
                 "set_starting", "set_value: a/d", "set_value: a/c/z/f",
                 "set_value: a/c/y", "set_value: a/c/z/e", "set_value: a/c/x",
                 "set_value: a/b", "set_done", "set_stopping"));
  }

  // Listing a subset of the stream works.
  {
    std::vector<std::string> log;
    tensorstore::execution::submit(store->List({KeyRange::Prefix("a/c/")}),
                                   tensorstore::LoggingReceiver{&log});

    EXPECT_THAT(log, ::testing::UnorderedElementsAre(
                         "set_starting", "set_value: a/c/z/f",
                         "set_value: a/c/y", "set_value: a/c/z/e",
                         "set_value: a/c/x", "set_done", "set_stopping"));
  }

  // Cancellation immediately after starting yields nothing..
  struct CancelOnStarting : public tensorstore::LoggingReceiver {
    void set_starting(tensorstore::AnyCancelReceiver do_cancel) {
      this->tensorstore::LoggingReceiver::set_starting({});
      do_cancel();
    }
  };

  {
    std::vector<std::string> log;
    tensorstore::execution::submit(store->List({}), CancelOnStarting{{&log}});

    EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_done",
                                            "set_stopping"));
  }

  // Cancellation in the middle of the stream stops the stream.
  struct CancelAfter2 : public tensorstore::LoggingReceiver {
    using Key = tensorstore::KeyValueStore::Key;
    tensorstore::AnyCancelReceiver cancel;

    void set_starting(tensorstore::AnyCancelReceiver do_cancel) {
      this->cancel = std::move(do_cancel);
      this->tensorstore::LoggingReceiver::set_starting({});
    }

    void set_value(Key k) {
      this->tensorstore::LoggingReceiver::set_value(std::move(k));
      if (this->log->size() == 2) {
        this->cancel();
      }
    }
  };

  {
    std::vector<std::string> log;
    tensorstore::execution::submit(store->List({}), CancelAfter2{{&log}});

    EXPECT_THAT(log,
                ::testing::ElementsAre(
                    "set_starting",
                    ::testing::AnyOf("set_value: a/d", "set_value: a/c/z/f",
                                     "set_value: a/c/y", "set_value: a/c/z/e",
                                     "set_value: a/c/x", "set_value: a/b"),
                    "set_done", "set_stopping"));
  }
}

TEST(MemoryKeyValueStoreTest, Open) {
  auto context = Context::Default();

  {
    auto store =
        KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value();
    TENSORSTORE_ASSERT_OK(store->Write("key", absl::Cord("value")));

    {
      auto store2 =
          KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value();
      // Verify that `store2` shares the same underlying storage as `store`.
      EXPECT_THAT(store2->Read("key").result(),
                  MatchesKvsReadResult(absl::Cord("value")));
    }

    auto other_context =
        Context(Context::Spec::FromJson(
                    {{"memory_key_value_store", ::nlohmann::json::object_t{}}})
                    .value(),
                context);
    auto store3 =
        KeyValueStore::Open(other_context, {{"driver", "memory"}}, {}).value();
    // Verify that `store3` does not share the same underlying storage as
    // `store`.
    EXPECT_THAT(store3->Read("key").result(), MatchesKvsReadResultNotFound());
  }

  // Test that the data persists even when there are no references to the store.
  {
    auto store =
        KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value();
    EXPECT_EQ("value", store->Read("key").value().value);
  }
}

TEST(MemoryKeyValueStoreTest, SpecRoundtrip) {
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip({{"driver", "memory"}});
}

TEST(MemoryKeyValueStoreTest, InvalidSpec) {
  auto context = tensorstore::Context::Default();

  // Test with extra key.
  EXPECT_THAT(
      KeyValueStore::Open(context, {{"driver", "memory"}, {"extra", "key"}}, {})
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(MemoryKeyValueStoreTest, BoundSpec) {
  auto context = tensorstore::Context::Default();
  ::nlohmann::json json_spec{{"driver", "memory"}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, KeyValueStore::Spec::Ptr::FromJson(json_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto bound_spec, spec->Bind(context));
  std::string bound_spec_cache_key;
  tensorstore::internal::EncodeCacheKey(&bound_spec_cache_key, bound_spec);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, bound_spec->Open().result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_bound_spec, store->GetBoundSpec());
  std::string store_cache_key;
  tensorstore::internal::EncodeCacheKey(&store_cache_key, store);
  EXPECT_EQ(bound_spec_cache_key, store_cache_key);
  auto new_spec = new_bound_spec->Unbind();
  EXPECT_THAT(new_spec.ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(json_spec));

  // Reopen the same KeyValueStore, using the same spec and context.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store2, KeyValueStore::Open(context, json_spec).result());
    std::string store2_cache_key;
    tensorstore::internal::EncodeCacheKey(&store2_cache_key, store2);
    EXPECT_EQ(store_cache_key, store2_cache_key);
  }

  // Reopen the same KeyValueStore, using an indirect context reference.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store2,
        KeyValueStore::Open(
            context,
            {{"driver", "memory"},
             {"context",
              {{"memory_key_value_store#a", "memory_key_value_store"}}},
             {"memory_key_value_store", "memory_key_value_store#a"}})
            .result());
    std::string store2_cache_key;
    tensorstore::internal::EncodeCacheKey(&store2_cache_key, store2);
    EXPECT_EQ(store_cache_key, store2_cache_key);
  }

  // Reopen a different KeyValueStore, using the same spec but different
  // context.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store3,
        KeyValueStore::Open(Context::Default(), json_spec).result());
    std::string store3_cache_key;
    tensorstore::internal::EncodeCacheKey(&store3_cache_key, store3);
    EXPECT_NE(store_cache_key, store3_cache_key);
  }
}

TEST(MemoryKeyValueStoreTest, OpenCache) {
  auto context = tensorstore::Context::Default();
  ::nlohmann::json json_spec{{"driver", "memory"}};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store1, KeyValueStore::Open(context, json_spec).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store2, KeyValueStore::Open(context, json_spec).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store3,
      KeyValueStore::Open(tensorstore::Context::Default(), json_spec).result());
  EXPECT_EQ(store1.get(), store2.get());
  EXPECT_NE(store1.get(), store3.get());

  std::string cache_key1, cache_key3;
  tensorstore::internal::EncodeCacheKey(&cache_key1, store1);
  tensorstore::internal::EncodeCacheKey(&cache_key3, store3);
  EXPECT_NE(cache_key1, cache_key3);
}

}  // namespace
