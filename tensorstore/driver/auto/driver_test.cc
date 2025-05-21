// Copyright 2025 The TensorStore Authors
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

#include <stdint.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/testing/json_gtest.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/schema.h"
#include "tensorstore/spec.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Context;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Spec;
using ::tensorstore::internal_testing::ScopedTemporaryDirectory;

auto MakeInitial(const Context &context) {
  return tensorstore::Open({{"driver", "zarr3"}, {"kvstore", "memory://"}},
                           context, tensorstore::dtype_v<int32_t>,
                           tensorstore::Schema::Shape({10, 20}),
                           tensorstore::OpenMode::create);
}

TEST(AutoTest, ExplicitDriver) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_initial,
                                   MakeInitial(context).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec_initial, store_initial.spec(tensorstore::retain_context));

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store_auto,
        tensorstore::Open({{"driver", "auto"}, {"kvstore", "memory://"}},
                          context)
            .result());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto spec_auto, store_auto.spec(tensorstore::retain_context));
    EXPECT_EQ(spec_initial, spec_auto);
  }
}

TEST(AutoTest, KvStoreDriver) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto explicit_auto_spec,
      Spec::FromJson(
          {{"driver", "auto"}, {"kvstore", {{"driver", "memory"}}}}));
  EXPECT_THAT(Spec::FromJson({{"driver", "memory"}}),
              ::testing::Optional(explicit_auto_spec));
}

TEST(AutoTest, ExplicitAutoUrl) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto explicit_auto_spec,
      Spec::FromJson(
          {{"driver", "auto"}, {"kvstore", {{"driver", "memory"}}}}));
  EXPECT_THAT(Spec::FromJson("memory://|auto:"),
              ::testing::Optional(explicit_auto_spec));
}

TEST(AutoTest, KvStoreUrl) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto explicit_auto_spec,
      Spec::FromJson(
          {{"driver", "auto"}, {"kvstore", {{"driver", "memory"}}}}));
  EXPECT_THAT(Spec::FromJson("memory://"),
              ::testing::Optional(explicit_auto_spec));
}

TEST(AutoTest, UnbindContextSimple) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      Spec::FromJson(
          {{"driver", "auto"},
           {"kvstore", {{"driver", "memory"}}},
           {"context", {{"cache_pool", {{"total_bytes_limit", 1000}}}}}}));

  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK(spec.BindContext(context));
  spec.UnbindContext();

  EXPECT_THAT(
      spec.ToJson(),
      ::testing::Optional(MatchesJson(
          {{"driver", "auto"},
           {"kvstore", {{"driver", "memory"}}},
           {"context",
            {
                {"memory_key_value_store", ::nlohmann::json::object_t()},
                {"data_copy_concurrency", ::nlohmann::json::object_t()},
                {"cache_pool", {{"total_bytes_limit", 1000}}},
                {"file_io_concurrency", ::nlohmann::json::object_t()},
                {"file_io_locking", ::nlohmann::json::object_t()},
                {"file_io_memmap", false},
                {"file_io_sync", true},
                {"ocdbt_coordinator", ::nlohmann::json::object_t()},
            }}})));
}

TEST(AutoTest, UnbindContext1) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      Spec::FromJson(
          {{"driver", "auto"},
           {"kvstore",
            {{"driver", "memory"},
             {"context",
              {{"memory_key_value_store", ::nlohmann::json::object_t()}}}}},
           {"context", {{"cache_pool", {{"total_bytes_limit", 1000}}}}}}));

  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK(spec.BindContext(context));
  spec.UnbindContext();

  EXPECT_THAT(
      spec.ToJson(),
      ::testing::Optional(MatchesJson(
          {{"driver", "auto"},
           {"kvstore",
            {{"driver", "memory"},
             {"memory_key_value_store", "memory_key_value_store#0"}}},
           {"memory_key_value_store", "memory_key_value_store#1"},
           {"context",
            {
                {"memory_key_value_store#0", ::nlohmann::json::object_t()},
                {"memory_key_value_store#1", ::nlohmann::json::object_t()},
                {"data_copy_concurrency", ::nlohmann::json::object_t()},
                {"cache_pool", {{"total_bytes_limit", 1000}}},
                {"file_io_concurrency", ::nlohmann::json::object_t()},
                {"file_io_locking", ::nlohmann::json::object_t()},
                {"file_io_memmap", false},
                {"file_io_sync", true},
                {"ocdbt_coordinator", ::nlohmann::json::object_t()},
            }}})));
}

TEST(AutoTest, MemoryZarr3) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK(tensorstore::Open("memory://tmp/dataset.zarr|zarr3",
                                          context,
                                          tensorstore::dtype_v<int32_t>,
                                          tensorstore::Schema::Shape({5}),
                                          tensorstore::OpenMode::create)
                            .result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::Open("memory://tmp/dataset.zarr", context).result());
}

TEST(AutoTest, MemoryOcdbtZarr3) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK(
      tensorstore::Open("memory://tmp/dataset.zarr.ocdbt|ocdbt|zarr3", context,
                        tensorstore::dtype_v<int32_t>,
                        tensorstore::Schema::Shape({5}),
                        tensorstore::OpenMode::create)
          .result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::Open("memory://tmp/dataset.zarr.ocdbt", context).result());
}

TEST(AutoTest, FileZarr3) {
  ScopedTemporaryDirectory tempdir;
  TENSORSTORE_ASSERT_OK(tensorstore::Open("file://" + tempdir.path() + "|zarr3",
                                          tensorstore::dtype_v<int32_t>,
                                          tensorstore::Schema::Shape({5}),
                                          tensorstore::OpenMode::create)
                            .result());
  TENSORSTORE_ASSERT_OK(tensorstore::Open("file://" + tempdir.path()).result());
}

TEST(AutoTest, MemoryZarr3Transaction) {
  auto context = Context::Default();
  auto transaction = tensorstore::Transaction(tensorstore::isolated);
  TENSORSTORE_ASSERT_OK(tensorstore::Open("memory://tmp/dataset.zarr|zarr3",
                                          context, transaction,
                                          tensorstore::dtype_v<int32_t>,
                                          tensorstore::Schema::Shape({5}),
                                          tensorstore::OpenMode::create)
                            .result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::Open("memory://tmp/dataset.zarr", context, transaction)
          .result());
}

TEST(AutoTest, MemoryOcdbtZarr3Transaction) {
  auto context = Context::Default();
  auto transaction = tensorstore::Transaction(tensorstore::isolated);
  TENSORSTORE_ASSERT_OK(
      tensorstore::Open("memory://tmp/dataset.zarr.ocdbt|ocdbt|zarr3", context,
                        transaction, tensorstore::dtype_v<int32_t>,
                        tensorstore::Schema::Shape({5}),
                        tensorstore::OpenMode::create)
          .result());
  // An uncommitted transactional write to OCDBT does not cause the
  // manifest.ocdbt file to be created.  Therefore, the uncommitted OCDBT
  // database cannot be detected.
  EXPECT_THAT(
      tensorstore::Open("memory://tmp/dataset.zarr.ocdbt", context, transaction)
          .result(),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Error opening \"auto\" driver: Failed to detect format for .*"));

  // Write an extra file to ensure manifest.ocdbt is written.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto ocdbt_kvs, tensorstore::kvstore::Open(
                            "memory://tmp/dataset.zarr.ocdbt|ocdbt", context)
                            .result());
    TENSORSTORE_ASSERT_OK(
        tensorstore::kvstore::Write(ocdbt_kvs, "junk", absl::Cord()).result());
  }

  TENSORSTORE_ASSERT_OK(
      tensorstore::Open("memory://tmp/dataset.zarr.ocdbt", context, transaction)
          .result());
}

}  // namespace
