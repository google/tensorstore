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

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/testing/random_seed.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/distributed/coordinator_server.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/test_util.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;
using ::tensorstore::Context;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::GetMap;
using ::tensorstore::internal_ocdbt::OcdbtDriver;
using ::tensorstore::internal_ocdbt::ReadManifest;
using ::tensorstore::ocdbt::CoordinatorServer;

class DistributedTest : public ::testing::Test {
 protected:
  CoordinatorServer coordinator_server_;
  std::string coordinator_address_;
  Context::Spec context_spec;
  DistributedTest() {
    ::nlohmann::json security_json = ::nlohmann::json::value_t::discarded;
    {
      CoordinatorServer::Options options;
      options.spec = CoordinatorServer::Spec::FromJson(
                         {{"bind_addresses", {"localhost:0"}},
                          {"security", security_json}})
                         .value();
      TENSORSTORE_CHECK_OK_AND_ASSIGN(
          coordinator_server_, CoordinatorServer::Start(std::move(options)));
    }

    assert(coordinator_server_.port() != 0);
    coordinator_address_ =
        tensorstore::StrCat("localhost:", coordinator_server_.port());

    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        context_spec,
        Context::Spec::FromJson({{"ocdbt_coordinator",
                                  {{"address", coordinator_address_},
                                   {"security", security_json}}}}));
  }
};

TEST_F(DistributedTest, WriteSingleKey) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_store,
                                   kvstore::Open("memory://").result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open({{"driver", "ocdbt"}, {"base", "memory://"}},
                                Context(context_spec))
                      .result());
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

TEST_F(DistributedTest, WriteTwoKeys) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open({{"driver", "ocdbt"}, {"base", "memory://"}},
                                Context(context_spec))
                      .result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "testa", absl::Cord("a")));
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "testb", absl::Cord("b")));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto map, GetMap(store));
  EXPECT_THAT(
      map, ::testing::ElementsAre(::testing::Pair("testa", absl::Cord("a")),
                                  ::testing::Pair("testb", absl::Cord("b"))));
}

TEST_F(DistributedTest, BasicFunctionality) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open({{"driver", "ocdbt"}, {"base", "memory://"}},
                                Context(context_spec))
                      .result());
  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST_F(DistributedTest, BasicFunctionalityMinArity) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::kvstore::Open({{"driver", "ocdbt"},
                                  {"base", "memory://"},
                                  {"config", {{"max_decoded_node_bytes", 1}}}},
                                 Context(context_spec))
          .result());
  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST_F(DistributedTest, BasicFunctionalityMinArityNoInline) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::kvstore::Open({{"driver", "ocdbt"},
                                  {"base", "memory://"},
                                  {"config",
                                   {
                                       {"max_decoded_node_bytes", 1},
                                       {"max_inline_value_bytes", 0},
                                   }}},
                                 Context(context_spec))
          .result());

  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST_F(DistributedTest, TwoCooperators) {
  tensorstore::internal_testing::ScopedTemporaryDirectory tempdir;
  ::nlohmann::json base_kvs_store_spec{{"driver", "file"},
                                       {"path", tempdir.path() + "/"}};
  ::nlohmann::json kvs_spec{
      {"driver", "ocdbt"},
      {"base", base_kvs_store_spec},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store1, kvstore::Open(kvs_spec, Context(context_spec)).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store2, kvstore::Open(kvs_spec, Context(context_spec)).result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(store1, "testa", absl::Cord("a")));
  TENSORSTORE_ASSERT_OK(kvstore::Write(store2, "testb", absl::Cord("b")));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto map, GetMap(store1));
  EXPECT_THAT(
      map, ::testing::ElementsAre(::testing::Pair("testa", absl::Cord("a")),
                                  ::testing::Pair("testb", absl::Cord("b"))));
}

TEST_F(DistributedTest, MultipleCooperatorsManyWrites) {
  tensorstore::internal_testing::ScopedTemporaryDirectory tempdir;
  ::nlohmann::json base_kvs_store_spec{{"driver", "file"},
                                       {"path", tempdir.path() + "/"}};
  ::nlohmann::json kvs_spec{
      {"driver", "ocdbt"},
      {"base", base_kvs_store_spec},
      {"config", {{"max_decoded_node_bytes", 500}}},
  };
  constexpr size_t kNumCooperators = 3;
  constexpr size_t kNumWrites = 30;
  constexpr size_t kIterations = 5;
  std::vector<kvstore::KvStore> stores;
  for (size_t i = 0; i < kNumCooperators; ++i) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, kvstore::Open(kvs_spec, Context(context_spec)).result());
    stores.push_back(store);
  }
  std::minstd_rand gen{tensorstore::internal_testing::GetRandomSeedForTest(
      "TENSORSTORE_OCDBT_DRIVER_TEST_SEED")};
  for (size_t iter = 0; iter < kIterations; ++iter) {
    std::vector<tensorstore::AnyFuture> write_futures;
    for (size_t i = 0; i < kNumWrites; ++i) {
      auto k = absl::Uniform<uint16_t>(gen);
      write_futures.push_back(kvstore::Write(stores[i % kNumCooperators],
                                             absl::StrFormat("%04x", k),
                                             absl::Cord("a")));
    }
    for (auto& future : write_futures) {
      TENSORSTORE_ASSERT_OK(future.status());
    }
  }
}

TEST_F(DistributedTest, TwoCooperatorsManifestDeleted) {
  ::nlohmann::json base_kvs_store_spec = "memory://";
  ::nlohmann::json kvs_spec{
      {"driver", "ocdbt"},
      {"base", base_kvs_store_spec},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store1, kvstore::Open(kvs_spec, Context(context_spec)).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store2, kvstore::Open(kvs_spec, Context(context_spec)).result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(store1, "testa", absl::Cord("a")));
  EXPECT_THAT(kvstore::Write(store2, "testb", absl::Cord("b")).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition));
}

// Tests that if a batch of writes leaves a node unmodified, it is not
// rewritten.
TEST_F(DistributedTest, UnmodifiedNode) {
  tensorstore::internal_ocdbt::TestUnmodifiedNode(Context(context_spec));
}

TEST_F(DistributedTest, ManifestDeleted) {
  auto context = Context(context_spec);
  ::nlohmann::json base_kvs_store_spec = "memory://";
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "ocdbt"}, {"base", base_kvs_store_spec}},
                    context)
          .result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "testa", absl::Cord("a")));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store, kvstore::Open(base_kvs_store_spec, context).result());
  TENSORSTORE_ASSERT_OK(kvstore::Delete(base_store, "manifest.ocdbt"));
  EXPECT_THAT(kvstore::Write(store, "testb", absl::Cord("b")).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
