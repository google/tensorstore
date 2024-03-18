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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = tensorstore::kvstore;
using ::tensorstore::Context;
using ::tensorstore::KvStore;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::MatchesKvsReadResult;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore::serialization::SerializationRoundTrip;

TEST(MemoryKeyValueStoreTest, Basic) {
  auto store = tensorstore::GetMemoryKeyValueStore();
  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST(MemoryKeyValueStoreTest, DeletePrefix) {
  auto store = tensorstore::GetMemoryKeyValueStore();
  tensorstore::internal::TestKeyValueStoreDeletePrefix(store);
}

TEST(MemoryKeyValueStoreTest, DeleteRange) {
  auto store = tensorstore::GetMemoryKeyValueStore();
  tensorstore::internal::TestKeyValueStoreDeleteRange(store);
}

TEST(MemoryKeyValueStoreTest, DeleteRangeToEnd) {
  auto store = tensorstore::GetMemoryKeyValueStore();
  tensorstore::internal::TestKeyValueStoreDeleteRangeToEnd(store);
}

TEST(MemoryKeyValueStoreTest, DeleteRangeFromBeginning) {
  auto store = tensorstore::GetMemoryKeyValueStore();
  tensorstore::internal::TestKeyValueStoreDeleteRangeFromBeginning(store);
}

#if 0
TEST(MemoryKeyValueStoreTest, CopyRange) {
  auto store = tensorstore::GetMemoryKeyValueStore();
  tensorstore::internal::TestKeyValueStoreCopyRange(store);
}
#endif

TEST(MemoryKeyValueStoreTest, List) {
  auto store = tensorstore::GetMemoryKeyValueStore();
  tensorstore::internal::TestKeyValueStoreList(store);
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

TEST(MemoryKeyValueStoreTest, ListWithPath) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "memory"}, {"path", "p/"}}, context).result());

  tensorstore::internal::TestKeyValueStoreList(store);
}

TEST(MemoryKeyValueStoreTest, SpecRoundtrip) {
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.full_spec = {
      {"driver", "memory"},
  };
  // Not possible with "memory" driver.
  options.check_data_after_serialization = false;
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
  // Not possible with "memory" driver.
  options.check_data_after_serialization = false;
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
