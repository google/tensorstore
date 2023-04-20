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

#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/test_util.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;
using ::tensorstore::Context;
using ::tensorstore::KeyRange;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::GetMap;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore::internal::MockKeyValueStore;
using ::tensorstore::internal_ocdbt::Config;
using ::tensorstore::internal_ocdbt::ConfigConstraints;
using ::tensorstore::internal_ocdbt::ManifestKind;
using ::tensorstore::internal_ocdbt::OcdbtDriver;
using ::tensorstore::internal_ocdbt::ReadManifest;
using ::tensorstore::kvstore::SupportedFeatures;

TEST(OcdbtTest, WriteSingleKey) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_store,
                                   kvstore::Open("memory://").result());
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
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::kvstore::Open(json_spec).result());
  EXPECT_THAT(store.spec().value().ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(tensorstore::MatchesJson(json_spec)));
}

TEST(OcdbtTest, Spec) {
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
  options.minimal_spec = {
      {"driver", "ocdbt"},
      {"base", {{"driver", "memory"}}},
  };
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

// TODO(jbms): Consider refactoring into TEST_P.
TENSORSTORE_GLOBAL_INITIALIZER {
  const auto register_test_case = [](ConfigConstraints config) {
    tensorstore::internal::RegisterGoogleTestCaseDynamically(
        "OcdbtBasicFunctionalityTest", config.ToJson().value().dump(),
        [config] {
          TENSORSTORE_ASSERT_OK_AND_ASSIGN(
              auto store,
              tensorstore::kvstore::Open({{"driver", "ocdbt"},
                                          {"base", "memory://"},
                                          {"config", config.ToJson().value()}})
                  .result());
          tensorstore::internal::TestKeyValueStoreBasicFunctionality(store);
        });
  };
  for (const auto max_decoded_node_bytes : {0, 1, 1048576}) {
    for (const auto max_inline_value_bytes : {0, 1, 1048576}) {
      for (const auto version_tree_arity_log2 : {1, 16}) {
        for (const auto compression :
             std::initializer_list<Config::Compression>{
                 Config::NoCompression{}, Config::ZstdCompression{0}}) {
          ConfigConstraints config;
          config.max_decoded_node_bytes = max_decoded_node_bytes;
          config.max_inline_value_bytes = max_inline_value_bytes;
          config.version_tree_arity_log2 = version_tree_arity_log2;
          config.compression = compression;
          register_test_case(config);
        }
      }
    }
  }

  {
    ConfigConstraints config;
    config.manifest_kind = ManifestKind::kNumbered;
    register_test_case(config);
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
                  "manifest.ocdbt", "manifest.0000000000000001",
                  "manifest.0000000000000002", ::testing::StartsWith("d/"))));
}

}  // namespace
