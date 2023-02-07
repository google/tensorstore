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
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = tensorstore::kvstore;
using ::tensorstore::Context;
using ::tensorstore::KeyRange;
using ::tensorstore::KvStore;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::MatchesKvsReadResult;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore::serialization::SerializationRoundTrip;

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
    using Key = tensorstore::kvstore::Key;
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
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, kvstore::Open({{"driver", "memory"}}, context).result());
    TENSORSTORE_ASSERT_OK(kvstore::Write(store, "key", absl::Cord("value")));

    {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto store2, kvstore::Open({{"driver", "memory"}}, context).result());
      // Verify that `store2` shares the same underlying storage as `store`.
      EXPECT_THAT(kvstore::Read(store2, "key").result(),
                  MatchesKvsReadResult(absl::Cord("value")));
    }

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto other_context, Context::FromJson({{"memory_key_value_store",
                                                ::nlohmann::json::object_t{}}},
                                              context));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store3,
        kvstore::Open({{"driver", "memory"}}, other_context).result());
    // Verify that `store3` does not share the same underlying storage as
    // `store`.
    EXPECT_THAT(kvstore::Read(store3, "key").result(),
                MatchesKvsReadResultNotFound());
  }

  // Test that the data persists even when there are no references to the store.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, kvstore::Open({{"driver", "memory"}}, context).result());
    EXPECT_EQ("value", kvstore::Read(store, "key").value().value);
  }
}

TEST(MemoryKeyValueStoreTest, SpecRoundtrip) {
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.full_spec = {
      {"driver", "memory"},
  };
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(MemoryKeyValueStoreTest, SpecRoundtripWithContextSpec) {
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.spec_request_options.Set(tensorstore::unbind_context);
  options.full_spec = {
      {"driver", "memory"},
      {"memory_key_value_store", "memory_key_value_store#a"},
      {"context",
       {
           {"memory_key_value_store#a", ::nlohmann::json::object_t()},
       }},
  };
  // Since spec includes context resources, if we re-open we get a different
  // context resource.
  options.check_data_persists = false;
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(MemoryKeyValueStoreTest, InvalidSpec) {
  auto context = tensorstore::Context::Default();

  // Test with extra key.
  EXPECT_THAT(
      kvstore::Open({{"driver", "memory"}, {"extra", "key"}}, context).result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(MemoryKeyValueStoreTest, BoundSpec) {
  auto context = tensorstore::Context::Default();
  ::nlohmann::json json_spec{{"driver", "memory"}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   kvstore::Spec::FromJson(json_spec));
  TENSORSTORE_ASSERT_OK(spec.BindContext(context));
  std::string bound_spec_cache_key;
  tensorstore::internal::EncodeCacheKey(&bound_spec_cache_key, spec.driver);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, kvstore::Open(spec).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_spec,
                                   store.spec(tensorstore::retain_context));
  std::string store_cache_key;
  tensorstore::internal::EncodeCacheKey(&store_cache_key, store.driver);
  EXPECT_EQ(bound_spec_cache_key, store_cache_key);
  new_spec.StripContext();
  EXPECT_THAT(new_spec.ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(json_spec));

  // Reopen the same KvStore, using the same spec and context.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store2, kvstore::Open(json_spec, context).result());
    std::string store2_cache_key;
    tensorstore::internal::EncodeCacheKey(&store2_cache_key, store2.driver);
    EXPECT_EQ(store_cache_key, store2_cache_key);
  }

  // Reopen the same KvStore, using an indirect context reference.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store2,
        kvstore::Open(
            {{"driver", "memory"},
             {"context",
              {{"memory_key_value_store#a", "memory_key_value_store"}}},
             {"memory_key_value_store", "memory_key_value_store#a"}},
            context)
            .result());
    std::string store2_cache_key;
    tensorstore::internal::EncodeCacheKey(&store2_cache_key, store2.driver);
    EXPECT_EQ(store_cache_key, store2_cache_key);
  }

  // Reopen a different KeyValueStore, using the same spec but different
  // context.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store3,
                                     kvstore::Open(json_spec).result());
    std::string store3_cache_key;
    tensorstore::internal::EncodeCacheKey(&store3_cache_key, store3.driver);
    EXPECT_NE(store_cache_key, store3_cache_key);
  }
}

TEST(MemoryKeyValueStoreTest, OpenCache) {
  auto context = tensorstore::Context::Default();
  ::nlohmann::json json_spec{{"driver", "memory"}};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store1,
                                   kvstore::Open(json_spec, context).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store2,
                                   kvstore::Open(json_spec, context).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store3,
                                   kvstore::Open(json_spec).result());
  EXPECT_EQ(store1.driver.get(), store2.driver.get());
  EXPECT_NE(store1.driver.get(), store3.driver.get());

  std::string cache_key1, cache_key3;
  tensorstore::internal::EncodeCacheKey(&cache_key1, store1.driver);
  tensorstore::internal::EncodeCacheKey(&cache_key3, store3.driver);
  EXPECT_NE(cache_key1, cache_key3);
}

TEST(MemoryKeyValueStoreTest, ContextBinding) {
  auto context1 = Context::Default();
  auto context2 = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec, kvstore::Spec::FromJson({{"driver", "memory"}}));
  auto base_spec1 = base_spec;
  TENSORSTORE_ASSERT_OK(base_spec1.Set(context1));

  // Check JSON conversion of bound spec.
  EXPECT_THAT(
      base_spec1.ToJson(),
      ::testing::Optional(MatchesJson(
          {{"driver", "memory"},
           {"context",
            {{"memory_key_value_store", ::nlohmann::json::object_t()}}}})));

  auto base_spec2 = base_spec;
  TENSORSTORE_ASSERT_OK(base_spec2.Set(context2));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store1,
                                   kvstore::Open(base_spec, context1).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store2,
                                   kvstore::Open(base_spec, context2).result());
  ASSERT_NE(store1.driver, store2.driver);
  EXPECT_THAT(kvstore::Open(base_spec1).result(), ::testing::Optional(store1));
  EXPECT_THAT(kvstore::Open(base_spec2).result(), ::testing::Optional(store2));

  auto base_spec3 = base_spec1;
  // All resources are already bound, setting `context2` has no effect.
  TENSORSTORE_ASSERT_OK(base_spec3.Set(context2));
  EXPECT_THAT(kvstore::Open(base_spec3).result(), ::testing::Optional(store1));

  // Rebind resources with `context2`
  TENSORSTORE_ASSERT_OK(base_spec3.Set(tensorstore::strip_context, context2));
  EXPECT_THAT(kvstore::Open(base_spec3).result(), ::testing::Optional(store2));
}

TEST(MemoryKeyValueStoreTest, SpecSerialization) {
  ::nlohmann::json json_spec{{"driver", "memory"}, {"path", "abc/"}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   kvstore::Spec::FromJson(json_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_roundtripped,
                                   SerializationRoundTrip(spec));
  EXPECT_THAT(spec_roundtripped.ToJson(),
              ::testing::Optional(MatchesJson(json_spec)));
}

TEST(MemoryKeyValueStoreTest, KvStoreSerialization) {
  ::nlohmann::json json_spec{{"driver", "memory"}, {"path", "abc/"}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   kvstore::Open(json_spec).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_roundtripped,
                                   SerializationRoundTrip(store));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_roundtripped,
                                   store_roundtripped.spec());
  EXPECT_THAT(spec_roundtripped.ToJson(),
              ::testing::Optional(MatchesJson(json_spec)));
}

TEST(MemoryKeyValueStoreTest, UrlRoundtrip) {
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip({{"driver", "memory"}},
                                                       "memory://");
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "memory"}, {"path", "abc/"}}, "memory://abc/");
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "memory"}, {"path", "abc def/"}}, "memory://abc%20def/");
}

TEST(MemoryKeyValueStoreTest, InvalidUri) {
  EXPECT_THAT(kvstore::Spec::FromUrl("memory://abc?query"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: Query string not supported"));
  EXPECT_THAT(kvstore::Spec::FromUrl("memory://abc#fragment"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: Fragment identifier not supported"));
}

}  // namespace
