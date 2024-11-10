// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/kvstore/ocdbt/driver.h"

#include <stdint.h>

#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/kvs_backed_cache_testutil.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/testing/dynamic.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/test_util.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;
using ::tensorstore::Context;
using ::tensorstore::JsonSubValueMatches;
using ::tensorstore::KeyRange;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::GetMap;
using ::tensorstore::internal::MatchesKvsReadResult;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore::internal::MatchesListEntry;
using ::tensorstore::internal::MockKeyValueStore;
using ::tensorstore::internal_ocdbt::Config;
using ::tensorstore::internal_ocdbt::ConfigConstraints;
using ::tensorstore::internal_ocdbt::ManifestKind;
using ::tensorstore::internal_ocdbt::OcdbtDriver;
using ::tensorstore::internal_ocdbt::ReadManifest;
using ::tensorstore::internal_testing::RegisterGoogleTestCaseDynamically;
using ::tensorstore::kvstore::SupportedFeatures;

TEST(OcdbtTest, WriteSingleKey) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "ocdbt"}, {"base", "memory://"}}).result());
  auto& driver = static_cast<OcdbtDriver&>(*store.driver);
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "a", absl::Cord("value")));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto manifest, ReadManifest(driver));
  ASSERT_TRUE(manifest);
  auto& version = manifest->latest_version();
  EXPECT_EQ(2, version.generation_number);
  EXPECT_FALSE(version.root.location.IsMissing());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto map, GetMap(store));
  EXPECT_THAT(
      map, ::testing::ElementsAre(::testing::Pair("a", absl::Cord("value"))));
}

TEST(OcdbtTest, Base) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, kvstore::Spec::FromJson({
                                                  {"driver", "ocdbt"},
                                                  {"base", "memory://abc/"},
                                                  {"path", "def"},
                                              }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_spec,
                                   kvstore::Spec::FromJson("memory://abc/"));
  EXPECT_THAT(spec.base(), ::testing::Optional(base_spec));

  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   kvstore::Open(spec, context).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_store,
                                   kvstore::Open(base_spec, context).result());
  EXPECT_THAT(store.base(), ::testing::Optional(base_store));

  // Check that the transaction is *not* propagated to the base.
  auto transaction = tensorstore::Transaction(tensorstore::atomic_isolated);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_with_txn, store | transaction);
  EXPECT_THAT(store_with_txn.base(), ::testing::Optional(base_store));
}

TEST(OcdbtTest, SeparatePrefixes) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      kvstore::Spec::FromJson({
          {"driver", "ocdbt"},
          {"base", "memory://abc/"},
          {"value_data_prefix", "x/"},
          {"btree_node_data_prefix", "b/"},
          {"version_tree_node_data_prefix", "v/"},
          {"config",
           {{"max_inline_value_bytes", 0}, {"version_tree_arity_log2", 1}}},
          {"path", "def"},
      }));
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   kvstore::Open(spec, context).result());

  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "testa", absl::Cord("a")));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base, store.base());
  EXPECT_THAT(kvstore::ListFuture(base).result(),
              ::testing::Optional(::testing::UnorderedElementsAre(
                  MatchesListEntry("manifest.ocdbt"),
                  MatchesListEntry(::testing::StartsWith("x/")),
                  MatchesListEntry(::testing::StartsWith("b/")))));

  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "testa", absl::Cord("b")));
  EXPECT_THAT(kvstore::ListFuture(base).result(),
              ::testing::Optional(::testing::UnorderedElementsAre(
                  MatchesListEntry("manifest.ocdbt"),
                  MatchesListEntry(::testing::StartsWith("x/")),
                  MatchesListEntry(::testing::StartsWith("x/")),
                  MatchesListEntry(::testing::StartsWith("b/")),
                  MatchesListEntry(::testing::StartsWith("b/")),
                  MatchesListEntry(::testing::StartsWith("v")))));
}

TEST(OcdbtTest, SeparateManifestKvStore) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context,
      Context::FromJson(
          {{"memory_key_value_store#a", ::nlohmann::json::object_t()},
           {"memory_key_value_store#b", ::nlohmann::json::object_t()}}));

  ::nlohmann::json spec_json{
      {"driver", "ocdbt"},
      {"base",
       {{"driver", "memory"},
        {"memory_key_value_store", "memory_key_value_store#a"},
        {"path", "abc/"}}},
      {"manifest",
       {{"driver", "memory"},
        {"memory_key_value_store", "memory_key_value_store#b"},
        {"path", "def/"}}},
      {"config",
       {{"max_inline_value_bytes", 0}, {"version_tree_arity_log2", 1}}},
      {"path", "def"},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto memory_a,
      kvstore::Open({{"driver", "memory"},
                     {"memory_key_value_store", "memory_key_value_store#a"}},
                    context)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto memory_b,
      kvstore::Open({{"driver", "memory"},
                     {"memory_key_value_store", "memory_key_value_store#b"}},
                    context)
          .result());

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, kvstore::Open(spec_json, context).result());

    TENSORSTORE_ASSERT_OK(kvstore::Write(store, "testa", absl::Cord("a")));

    EXPECT_THAT(kvstore::ListFuture(memory_a).result(),
                ::testing::Optional(::testing::UnorderedElementsAre(
                    MatchesListEntry(::testing::StartsWith("abc/d/")))));
    EXPECT_THAT(kvstore::ListFuture(memory_b).result(),
                ::testing::Optional(::testing::UnorderedElementsAre(
                    MatchesListEntry("def/manifest.ocdbt"))));

    // Verify that the returned spec includes the manifest kvstore.
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto retrieved_spec,
                                     store.spec(tensorstore::MinimalSpec{}));
    EXPECT_THAT(retrieved_spec.ToJson(),
                ::testing::Optional(tensorstore::MatchesJson({
                    {"driver", "ocdbt"},
                    {"base", {{"driver", "memory"}, {"path", "abc/"}}},
                    {"manifest", {{"driver", "memory"}, {"path", "def/"}}},
                    {"path", "def"},
                })));
  }

  // Verify that reopening succeeds.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, kvstore::Open(spec_json, context).result());
    EXPECT_THAT(kvstore::Read(store, "testa").result(),
                MatchesKvsReadResult(absl::Cord("a")));
  }
}

TEST(OcdbtTest, SeparateManifestKvStoreNumbered) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context,
      Context::FromJson(
          {{"memory_key_value_store#a", ::nlohmann::json::object_t()},
           {"memory_key_value_store#b", ::nlohmann::json::object_t()}}));

  ::nlohmann::json spec_json{
      {"driver", "ocdbt"},
      {"base",
       {{"driver", "memory"},
        {"memory_key_value_store", "memory_key_value_store#a"},
        {"path", "abc/"}}},
      {"manifest",
       {{"driver", "memory"},
        {"memory_key_value_store", "memory_key_value_store#b"},
        {"path", "def/"}}},
      {"config",
       {{"max_inline_value_bytes", 0},
        {"version_tree_arity_log2", 1},
        {"manifest_kind", "numbered"}}},
      {"path", "def"},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto memory_a,
      kvstore::Open({{"driver", "memory"},
                     {"memory_key_value_store", "memory_key_value_store#a"}},
                    context)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto memory_b,
      kvstore::Open({{"driver", "memory"},
                     {"memory_key_value_store", "memory_key_value_store#b"}},
                    context)
          .result());

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, kvstore::Open(spec_json, context).result());

    TENSORSTORE_ASSERT_OK(kvstore::Write(store, "testa", absl::Cord("a")));

    EXPECT_THAT(kvstore::ListFuture(memory_a).result(),
                ::testing::Optional(::testing::UnorderedElementsAre(
                    MatchesListEntry(::testing::StartsWith("abc/d/")))));
    EXPECT_THAT(
        kvstore::ListFuture(memory_b).result(),
        ::testing::Optional(::testing::UnorderedElementsAre(
            MatchesListEntry("def/manifest.ocdbt"),
            MatchesListEntry(::testing::StartsWith("def/manifest.0")),
            MatchesListEntry(::testing::StartsWith("def/manifest.0")))));
  }

  // Verify that reopening succeeds.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, kvstore::Open(spec_json, context).result());
    EXPECT_THAT(kvstore::Read(store, "testa").result(),
                MatchesKvsReadResult(absl::Cord("a")));
  }
}

TEST(OcdbtTest, WriteTwoKeys) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "ocdbt"}, {"base", "memory://"}}).result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "testa", absl::Cord("a")));
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "testb", absl::Cord("b")));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto map, GetMap(store));
  EXPECT_THAT(
      map, ::testing::ElementsAre(::testing::Pair("testa", absl::Cord("a")),
                                  ::testing::Pair("testb", absl::Cord("b"))));
}

TEST(OcdbtTest, SimpleMinArity) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::kvstore::Open({{"driver", "ocdbt"},
                                  {"base", "memory://"},
                                  {"config", {{"max_decoded_node_bytes", 1}}},
                                  {"data_copy_concurrency", {{"limit", 1}}}})
          .result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "testa", absl::Cord("a")));
  EXPECT_THAT(GetMap(store), ::testing::Optional(::testing::ElementsAreArray({
                                 ::testing::Pair("testa", absl::Cord("a")),
                             })));
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "testb", absl::Cord("b")));
  EXPECT_THAT(GetMap(store), ::testing::Optional(::testing::ElementsAreArray({
                                 ::testing::Pair("testa", absl::Cord("a")),
                                 ::testing::Pair("testb", absl::Cord("b")),
                             })));
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "testc", absl::Cord("c")));
  EXPECT_THAT(GetMap(store), ::testing::Optional(::testing::ElementsAreArray({
                                 ::testing::Pair("testa", absl::Cord("a")),
                                 ::testing::Pair("testb", absl::Cord("b")),
                                 ::testing::Pair("testc", absl::Cord("c")),
                             })));
}

TEST(OcdbtTest, DeleteRangeMinArity) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::kvstore::Open({{"driver", "ocdbt"},
                                  {"base", "memory://"},
                                  {"config", {{"max_decoded_node_bytes", 1}}},
                                  {"data_copy_concurrency", {{"limit", 1}}}})
          .result());
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/b", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/d", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/x", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/y", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/e", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/f", absl::Cord("xyz")));

  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange::Prefix("a/c")));

  EXPECT_EQ("xyz", kvstore::Read(store, "a/b").value().value);
  EXPECT_EQ("xyz", kvstore::Read(store, "a/d").value().value);

  EXPECT_THAT(kvstore::Read(store, "a/c/x").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, "a/c/y").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, "a/c/z/e").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, "a/c/z/f").result(),
              MatchesKvsReadResultNotFound());
}

TEST(OcdbtTest, WithExperimentalSpec) {
  ::nlohmann::json json_spec{
      {"driver", "ocdbt"},
      {"base", {{"driver", "memory"}}},
      {"config", {{"max_decoded_node_bytes", 1}}},
      {"experimental_read_coalescing_threshold_bytes", 1024},
      {"experimental_read_coalescing_merged_bytes", 2048},
      {"experimental_read_coalescing_interval", "10ms"},
      {"target_data_file_size", 1024},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::kvstore::Open(json_spec).result());
  EXPECT_THAT(store.spec().value().ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(tensorstore::MatchesJson(json_spec)));

  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST(OcdbtTest, SpecRoundtrip) {
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.create_spec = {
      {"driver", "ocdbt"},
      {"base", {{"driver", "memory"}}},
      {"config",
       {
           {"uuid", "000102030405060708090a0b0c0d0e0f"},
           {"compression", {{"id", "zstd"}}},
       }},
  };
  options.full_spec = {
      {"driver", "ocdbt"},
      {"base", {{"driver", "memory"}}},
      {"config",
       {{"uuid", "000102030405060708090a0b0c0d0e0f"},
        {"compression", {{"id", "zstd"}}},
        {"max_decoded_node_bytes", 8388608},
        {"max_inline_value_bytes", 100},
        {"version_tree_arity_log2", 4}}},
  };
  options.full_base_spec = {{"driver", "memory"}};
  options.minimal_spec = {
      {"driver", "ocdbt"},
      {"base", {{"driver", "memory"}}},
  };
  options.check_data_after_serialization = false;
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(OcdbtTest, SpecRoundtripFile) {
  tensorstore::internal_testing::ScopedTemporaryDirectory tempdir;
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.full_base_spec = {{"driver", "file"}, {"path", tempdir.path() + "/"}};
  options.create_spec = {
      {"driver", "ocdbt"},
      {"base", options.full_base_spec},
      {"config",
       {
           {"uuid", "000102030405060708090a0b0c0d0e0f"},
           {"compression", {{"id", "zstd"}}},
       }},
  };
  options.full_spec = {
      {"driver", "ocdbt"},
      {"base", options.full_base_spec},
      {"config",
       {{"uuid", "000102030405060708090a0b0c0d0e0f"},
        {"compression", {{"id", "zstd"}}},
        {"max_decoded_node_bytes", 8388608},
        {"max_inline_value_bytes", 100},
        {"version_tree_arity_log2", 4}}},
  };
  options.minimal_spec = {
      {"driver", "ocdbt"},
      {"base", options.full_base_spec},
  };
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(OcdbtTest, CacheKey) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store1,
      kvstore::Open({{"driver", "ocdbt"}, {"base", "memory://"}}, context)
          .result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(store1, "abc", absl::Cord("value")));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store2,
      kvstore::Open({{"driver", "ocdbt"}, {"base", "memory://"}}, context)
          .result());
  EXPECT_EQ(store2.driver, store1.driver);
}

TEST(OcdbtTest, ConcurrentWrites) {
  auto context = Context::Default();
  tensorstore::internal::TestConcurrentWritesOptions options;
  options.get_store = [&] {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto store, kvstore::Open(
                        {
                            {"driver", "ocdbt"},
                            {"base", "memory://"},
                            {"cache_pool", {{"total_bytes_limit", 0}}},
                        },
                        context)
                        .result());
    return store;
  };
  options.num_threads = 16;
  tensorstore::internal::TestConcurrentWrites(options);
}

TEST(OcdbtTest, ConcurrentWritesNumbered) {
  auto context = Context::Default();
  tensorstore::internal::TestConcurrentWritesOptions options;
  options.get_store = [&] {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto store,
        kvstore::Open(
            {{"driver", "ocdbt"},
             {"base", "memory://"},
             // Use separate cache to ensure independent driver objects.
             {"cache_pool", {{"total_bytes_limit", 0}}},
             {"config", {{"manifest_kind", "numbered"}}}},
            context)
            .result());
    return store;
  };
  options.num_threads = 16;
  options.num_iterations = 100;
  tensorstore::internal::TestConcurrentWrites(options);
}

// TODO(jbms): Consider refactoring into TEST_P.
TENSORSTORE_GLOBAL_INITIALIZER {
  const auto register_test_suite = [](ConfigConstraints config) {
    const auto register_test_case = [&](std::string case_name, auto op) {
      RegisterGoogleTestCaseDynamically(
          "OcdbtBasicFunctionalityTest." + case_name,
          config.ToJson().value().dump(), [config, op] {
            TENSORSTORE_ASSERT_OK_AND_ASSIGN(
                auto store, tensorstore::kvstore::Open(
                                {{"driver", "ocdbt"},
                                 {"base", "memory://"},
                                 {"config", config.ToJson().value()}})
                                .result());
            op(store);
          });
    };

    register_test_case("ReadWriteOps", [](auto& store) {
      tensorstore::internal::TestKeyValueReadWriteOps(store);
    });
    register_test_case("DeletePrefix", [](auto& store) {
      tensorstore::internal::TestKeyValueStoreDeletePrefix(store);
    });
    register_test_case("DeleteRange", [](auto& store) {
      tensorstore::internal::TestKeyValueStoreDeleteRange(store);
    });
    register_test_case("DeleteRangeToEnd", [](auto& store) {
      tensorstore::internal::TestKeyValueStoreDeleteRangeToEnd(store);
    });
    register_test_case("DeleteRangeFromBeginning", [](auto& store) {
      tensorstore::internal::TestKeyValueStoreDeleteRangeFromBeginning(store);
    });
    register_test_case("CopyRange", [](auto& store) {
      tensorstore::internal::TestKeyValueStoreCopyRange(store);
    });
    register_test_case("List", [](auto& store) {
      tensorstore::internal::TestKeyValueStoreList(store);
    });
  };
  for (const auto max_decoded_node_bytes : {0, 1, 1048576}) {
    ConfigConstraints config;
    config.max_decoded_node_bytes = max_decoded_node_bytes;
    config.max_inline_value_bytes = 0;
    config.version_tree_arity_log2 = 1;
    config.compression = Config::NoCompression{};
    register_test_suite(config);
  }

  for (const auto max_inline_value_bytes : {0, 1, 1048576}) {
    ConfigConstraints config;
    config.max_decoded_node_bytes = 1048576;
    config.max_inline_value_bytes = max_inline_value_bytes;
    config.version_tree_arity_log2 = 1;
    config.compression = Config::NoCompression{};
    register_test_suite(config);
  }

  for (const auto version_tree_arity_log2 : {1, 16}) {
    ConfigConstraints config;
    config.max_decoded_node_bytes = 0;
    config.max_inline_value_bytes = 0;
    config.version_tree_arity_log2 = version_tree_arity_log2;
    config.compression = Config::NoCompression{};
    register_test_suite(config);
  }

  for (const auto compression : std::initializer_list<Config::Compression>{
           Config::NoCompression{}, Config::ZstdCompression{0}}) {
    ConfigConstraints config;
    config.max_decoded_node_bytes = 0;
    config.max_inline_value_bytes = 0;
    config.version_tree_arity_log2 = 16;
    config.compression = compression;
    register_test_suite(config);
  }

  {
    ConfigConstraints config;
    config.manifest_kind = ManifestKind::kNumbered;
    register_test_suite(config);
  }
}

// Tests that if a batch of writes leaves a node unmodified, it is not
// rewritten.
TEST(OcdbtTest, UnmodifiedNode) {
  tensorstore::internal_ocdbt::TestUnmodifiedNode();
}

// Disable this test until the corresponding error is re-enabled.
#if 0
TEST(OcdbtTest, NoSuitableManifestKind) {
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_store,
                                   kvstore::Open("memory://").result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context.GetResource<tensorstore::internal::MockKeyValueStoreResource>());
  MockKeyValueStore* mock_key_value_store =
      mock_key_value_store_resource->get();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store,
      kvstore::Open(
          {{"driver", "ocdbt"}, {"base", {{"driver", "mock_key_value_store"}}}},
          context)
          .result());

  auto write_future = kvstore::Write(ocdbt_store, "a", absl::Cord("b"));
  write_future.Force();

  {
    auto req = mock_key_value_store->read_requests.pop();
    EXPECT_EQ("manifest.ocdbt", req.key);
    req(base_store.driver);
  }

  EXPECT_THAT(
      write_future.result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*Cannot choose OCDBT manifest_kind automatically .*"));
}
#endif

TEST(OcdbtTest, ChooseNumberedManifestKind) {
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_store,
                                   kvstore::Open("memory://").result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context.GetResource<tensorstore::internal::MockKeyValueStoreResource>());
  MockKeyValueStore* mock_key_value_store =
      mock_key_value_store_resource->get();
  mock_key_value_store->supported_features =
      SupportedFeatures::kAtomicWriteWithoutOverwrite;

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store,
      kvstore::Open(
          {{"driver", "ocdbt"}, {"base", {{"driver", "mock_key_value_store"}}}},
          context)
          .result());

  auto write_future = kvstore::Write(ocdbt_store, "a", absl::Cord("b"));
  write_future.Force();

  {
    auto req = mock_key_value_store->read_requests.pop();
    EXPECT_EQ("manifest.ocdbt", req.key);
    req(base_store.driver);
  }

  {
    auto req = mock_key_value_store->write_requests.pop();
    EXPECT_EQ("manifest.ocdbt", req.key);
    req(base_store.driver);
  }

  {
    auto req = mock_key_value_store->write_requests.pop();
    EXPECT_EQ("manifest.0000000000000001", req.key);
    req(base_store.driver);
  }

  {
    auto req = mock_key_value_store->list_requests.pop();
    req(base_store.driver);
  }

  {
    auto req = mock_key_value_store->write_requests.pop();
    EXPECT_THAT(req.key, ::testing::StartsWith("d/"));
    req(base_store.driver);
  }

  {
    auto req = mock_key_value_store->write_requests.pop();
    EXPECT_EQ("manifest.0000000000000002", req.key);
    req(base_store.driver);
  }

  {
    auto req = mock_key_value_store->list_requests.pop();
    req(base_store.driver);
  }

  TENSORSTORE_EXPECT_OK(write_future);
}

TEST(OcdbtTest, NumberedManifest) {
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store, kvstore::Open("memory://", context).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store,
      kvstore::Open({{"driver", "ocdbt"},
                     {"config", {{"manifest_kind", "numbered"}}},
                     {"base", "memory://"}},
                    context)
          .result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(ocdbt_store, "a", absl::Cord("b")));
  EXPECT_THAT(kvstore::ListFuture(base_store).result(),
              ::testing::Optional(::testing::UnorderedElementsAre(
                  MatchesListEntry("manifest.ocdbt"),
                  MatchesListEntry("manifest.0000000000000001"),
                  MatchesListEntry("manifest.0000000000000002"),
                  MatchesListEntry(::testing::StartsWith("d/")))));
}

TEST(OcdbtTest, NumberedManifestNumNumberedManifestsToKeep) {
  using ::tensorstore::internal_ocdbt::kNumNumberedManifestsToKeep;
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store, kvstore::Open("memory://", context).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store,
      kvstore::Open({{"driver", "ocdbt"},
                     {"config", {{"manifest_kind", "numbered"}}},
                     {"base", "memory://"}},
                    context)
          .result());
  for (uint64_t i = 0; i < kNumNumberedManifestsToKeep + 4; ++i) {
    TENSORSTORE_ASSERT_OK(kvstore::Write(ocdbt_store, "a", absl::Cord("b")));
    std::vector<::testing::Matcher<kvstore::ListEntry>> matchers;
    uint64_t max_generation = i + 2;
    uint64_t min_generation = (max_generation > kNumNumberedManifestsToKeep)
                                  ? max_generation - kNumNumberedManifestsToKeep
                                  : 1;
    for (uint64_t j = min_generation; j <= max_generation; ++j) {
      matchers.push_back(
          MatchesListEntry(absl::StrFormat("manifest.%016x", j)));
    }
    matchers.push_back(MatchesListEntry("manifest.ocdbt"));
    EXPECT_THAT(kvstore::ListFuture(base_store, {KeyRange::Prefix("manifest.")})
                    .result(),
                ::testing::Optional(::testing::ElementsAreArray(matchers)));
  }
}

TEST(OcdbtTest, CopyRange) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "ocdbt"}, {"base", "memory://"}}).result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "x/a", absl::Cord("value_a")));
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "x/b", absl::Cord("value_b")));
  TENSORSTORE_ASSERT_OK(kvstore::ExperimentalCopyRange(
      store.WithPathSuffix("x/"), store.WithPathSuffix("y/")));
  EXPECT_THAT(GetMap(store), ::testing::Optional(::testing::ElementsAreArray({
                                 ::testing::Pair("x/a", absl::Cord("value_a")),
                                 ::testing::Pair("x/b", absl::Cord("value_b")),
                                 ::testing::Pair("y/a", absl::Cord("value_a")),
                                 ::testing::Pair("y/b", absl::Cord("value_b")),
                             })));
}

TEST(OcdbtTest, TransactionalCopyRange) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open({
                                    {"driver", "ocdbt"},
                                    {"base", "memory://"},
                                    {"config", {{"max_inline_value_bytes", 0}}},
                                })
                      .result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "x/a", absl::Cord("value_a")));
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "x/b", absl::Cord("value_b")));
  auto transaction = tensorstore::Transaction(tensorstore::atomic_isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto transactional_store,
                                     store | transaction);

    TENSORSTORE_ASSERT_OK(kvstore::ExperimentalCopyRange(
        store.WithPathSuffix("x/"), transactional_store.WithPathSuffix("y/")));
    TENSORSTORE_ASSERT_OK(kvstore::ExperimentalCopyRange(
        store.WithPathSuffix("x/"), transactional_store.WithPathSuffix("z/")));
    // Overwrite existing copy.
    TENSORSTORE_ASSERT_OK(kvstore::ExperimentalCopyRange(
        store.WithPathSuffix("x/"), transactional_store.WithPathSuffix("z/")));
    EXPECT_THAT(kvstore::Read(transactional_store, "y/a").result(),
                MatchesKvsReadResult(absl::Cord("value_a")));
    TENSORSTORE_ASSERT_OK(transaction.CommitAsync());
  }
  EXPECT_THAT(GetMap(store), ::testing::Optional(::testing::ElementsAreArray({
                                 ::testing::Pair("x/a", absl::Cord("value_a")),
                                 ::testing::Pair("x/b", absl::Cord("value_b")),
                                 ::testing::Pair("y/a", absl::Cord("value_a")),
                                 ::testing::Pair("y/b", absl::Cord("value_b")),
                                 ::testing::Pair("z/a", absl::Cord("value_a")),
                                 ::testing::Pair("z/b", absl::Cord("value_b")),
                             })));
}

TEST(OcdbtTest, AssumeConfig) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_store,
                                   kvstore::Open("memory://").result());
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context.GetResource<tensorstore::internal::MockKeyValueStoreResource>());
  MockKeyValueStore* mock_key_value_store =
      mock_key_value_store_resource->get();
  mock_key_value_store->supported_features =
      SupportedFeatures::kSingleKeyAtomicReadModifyWrite;
  mock_key_value_store->forward_to = base_store.driver;
  mock_key_value_store->log_requests = true;

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open(
                      {
                          {"driver", "ocdbt"},
                          {"base", {{"driver", "mock_key_value_store"}}},
                          {"config", {{"max_inline_value_bytes", 0}}},
                          {"assume_config", true},
                      },
                      context)
                      .result());

  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "x/a", absl::Cord("value_a")));

  EXPECT_THAT(
      mock_key_value_store->request_log.pop_all(),
      ::testing::ElementsAre(
          ::testing::AllOf(JsonSubValueMatches("/type", "supported_features")),
          ::testing::AllOf(JsonSubValueMatches("/type", "read"),
                           JsonSubValueMatches("/key", "manifest.ocdbt")),
          ::testing::AllOf(
              JsonSubValueMatches("/type", "write"),
              JsonSubValueMatches("/key", ::testing::StartsWith("d/"))),
          ::testing::AllOf(JsonSubValueMatches("/type", "write"),
                           JsonSubValueMatches("/key", "manifest.ocdbt"))));
}

TEST(OcdbtTest, AssumeConfigMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_store,
                                   kvstore::Open("memory://").result());
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context.GetResource<tensorstore::internal::MockKeyValueStoreResource>());
  MockKeyValueStore* mock_key_value_store =
      mock_key_value_store_resource->get();
  mock_key_value_store->supported_features =
      SupportedFeatures::kSingleKeyAtomicReadModifyWrite;

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open(
                      {
                          {"driver", "ocdbt"},
                          {"base", {{"driver", "mock_key_value_store"}}},
                          {"config", {{"version_tree_arity_log2", 2}}},
                          // Specify separate cache pool to ensure that separate
                          // driver instances are used.
                          {"cache_pool", {{"total_bytes_limit", 0}}},
                      },
                      context)
                      .result());

  mock_key_value_store->forward_to = base_store.driver;
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "a", absl::Cord("value")));
  mock_key_value_store->forward_to = {};

  for (bool assume_config : {false, true}) {
    SCOPED_TRACE(absl::StrFormat("assume_config=%d", assume_config));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store2, kvstore::Open(
                         {
                             {"driver", "ocdbt"},
                             {"base", {{"driver", "mock_key_value_store"}}},
                             // Specify separate cache pool to ensure that
                             // separate driver instances are used.
                             {"cache_pool", {{"total_bytes_limit", 0}}},
                             {"config", {{"max_inline_value_bytes", 1}}},
                             {"target_data_file_size", 1},
                             {"assume_config", assume_config},
                         },
                         context)
                         .result());
    auto write_future =
        kvstore::Write(store2, "b", absl::Cord(std::string(200, 0)));

    if (assume_config) {
      auto req = mock_key_value_store->write_requests.pop();
      EXPECT_THAT(req.key, ::testing::StartsWith("d/"));
      req(base_store.driver);
    }

    {
      auto req = mock_key_value_store->read_requests.pop();
      EXPECT_THAT(req.key, "manifest.ocdbt");
      req(base_store.driver);
    }

    EXPECT_THAT(write_future.result(),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              "Configuration mismatch .*"));
  }

  // Reading also fails.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store2, kvstore::Open(
                         {
                             {"driver", "ocdbt"},
                             {"base", {{"driver", "mock_key_value_store"}}},
                             // Note: Even though no configuration constraints
                             // are specified, the assumed config fixes all
                             // configuration options to their defaults, and
                             // therefore does not match the actual config with
                             // a non-default version_tree_arity_log2.
                             {"assume_config", true},
                         },
                         context)
                         .result());
    auto read_future = kvstore::Read(store2, "a");
    {
      auto req = mock_key_value_store->read_requests.pop();
      EXPECT_THAT(req.key, "manifest.ocdbt");
      req(base_store.driver);
    }
    EXPECT_THAT(read_future.result(),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              "Observed config does not match assumed config: "
                              "Configuration mismatch .*"));
  }
}

TENSORSTORE_GLOBAL_INITIALIZER {
  using ::tensorstore::internal::KvsBackedCacheBasicTransactionalTestOptions;
  using ::tensorstore::internal::RegisterKvsBackedCacheBasicTransactionalTest;

  KvsBackedCacheBasicTransactionalTestOptions options;
  options.test_name = "OcdbtDriverTransactional";
  options.get_store = [] {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto store,
        kvstore::Open({{"driver", "ocdbt"}, {"base", "memory://"}}).result());
    return store.driver;
  };
  options.delete_range_supported = true;
  options.multi_key_atomic_supported = true;
  RegisterKvsBackedCacheBasicTransactionalTest(options);
}

}  // namespace
