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

/// End-to-end tests of the json driver.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/driver_testutil.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/parse_json_matches.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/kvstore/key_value_store_testutil.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/open.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::MakeArray;
using tensorstore::MakeScalarArray;
using tensorstore::MatchesStatus;
using tensorstore::internal::GetMap;
using tensorstore::internal::ParseJsonMatches;
using testing::Optional;
using testing::Pair;

::nlohmann::json GetKvstoreSpec() { return {{"driver", "memory"}}; }
std::string GetPath() { return "path.json"; }

::nlohmann::json GetSpec(std::string json_pointer) {
  return ::nlohmann::json{
      {"driver", "json"},
      {"kvstore", GetKvstoreSpec()},
      {"path", GetPath()},
      {"json_pointer", json_pointer},
  };
}

TEST(JsonDriverTest, Basic) {
  auto context = tensorstore::Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto kvstore,
      tensorstore::KeyValueStore::Open(context, GetKvstoreSpec()).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(context, GetSpec("")).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store_a, tensorstore::Open(context, GetSpec("/a")).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store_b, tensorstore::Open(context, GetSpec("/b")).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store_c_x, tensorstore::Open(context, GetSpec("/c/x")).result());

  EXPECT_THAT(tensorstore::Read(store_a).result(),
              MatchesStatus(absl::StatusCode::kNotFound,
                            "Error reading \"path\\.json\""));

  TENSORSTORE_EXPECT_OK(tensorstore::Write(MakeScalarArray(42), store_a));

  EXPECT_THAT(tensorstore::Read(store).result(),
              Optional(MakeScalarArray<::nlohmann::json>({{"a", 42}})));

  // Test with an index transform.
  {
    ::nlohmann::json v{{"a", 42}};
    EXPECT_THAT(tensorstore::Read(
                    store | tensorstore::Dims(0, 1).AddNew().SizedInterval(
                                {0, 0}, {2, 3}))
                    .result(),
                Optional(MakeArray({{v, v, v}, {v, v, v}})));
  }

  EXPECT_THAT(
      GetMap(kvstore).value(),
      testing::ElementsAre(
          Pair(GetPath(), ::testing::MatcherCast<absl::Cord>(
                              ParseJsonMatches(::nlohmann::json{{"a", 42}})))));

  // Test transactional read/write.
  {
    tensorstore::Transaction transaction(tensorstore::isolated);
    TENSORSTORE_EXPECT_OK(tensorstore::Write(
        MakeScalarArray<::nlohmann::json>(false), store_b | transaction));
    // Test transactional read.
    EXPECT_THAT(
        tensorstore::Read(store | transaction).result(),
        Optional(MakeScalarArray<::nlohmann::json>({{"a", 42}, {"b", false}})));
    // Overwrite previous writes, deleting "/a" and "/b".
    TENSORSTORE_EXPECT_OK(tensorstore::Write(
        MakeScalarArray<::nlohmann::json>({{"c", 50}}), store | transaction));
    EXPECT_THAT(tensorstore::Read(store_a | transaction).result(),
                MatchesStatus(absl::StatusCode::kNotFound));
    // Actual contents of kvstore has not yet changed since transaction has not
    // been committed.
    EXPECT_THAT(
        GetMap(kvstore).value(),
        testing::ElementsAre(Pair(
            GetPath(), ::testing::MatcherCast<absl::Cord>(
                           ParseJsonMatches(::nlohmann::json{{"a", 42}})))));
    TENSORSTORE_EXPECT_OK(transaction.CommitAsync());
  }

  EXPECT_THAT(
      GetMap(kvstore).value(),
      testing::ElementsAre(
          Pair(GetPath(), ::testing::MatcherCast<absl::Cord>(
                              ParseJsonMatches(::nlohmann::json{{"c", 50}})))));
  EXPECT_THAT(tensorstore::Read(store_c_x).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error reading \"path.json\": "
                            "JSON Pointer reference \"/c/x\" cannot be applied "
                            "to number value: 50"));

  TENSORSTORE_EXPECT_OK(kvstore->Write(GetPath(), absl::Cord("{\"x\":42}")));

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store_new, tensorstore::Open(context, GetSpec("")).result());
    EXPECT_THAT(tensorstore::Read(store_new).result(),
                Optional(MakeScalarArray<::nlohmann::json>({{"x", 42}})));
  }
}

TEST(JsonDriverTest, WriteIncompatibleWithExisting) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(context, GetSpec("")).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store_a, tensorstore::Open(context, GetSpec("/a")).result());

  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(MakeScalarArray<::nlohmann::json>(42), store));
  EXPECT_THAT(
      tensorstore::Write(MakeScalarArray<::nlohmann::json>(true), store_a)
          .result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error writing \"path\\.json\": JSON Pointer reference "
                    "\"/a\" cannot be applied to number value: 42"));
}

TEST(JsonDriverTest, WriteDiscarded) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto kvstore,
      tensorstore::KeyValueStore::Open(context, GetKvstoreSpec()).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(context, GetSpec("")).result());
  // Write initial value (42)
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(MakeScalarArray<::nlohmann::json>(42), store));
  EXPECT_THAT(GetMap(kvstore).value(),
              testing::ElementsAre(Pair(
                  GetPath(), ::testing::MatcherCast<absl::Cord>(
                                 ParseJsonMatches(::nlohmann::json(42))))));
  // Write `discarded` to delete.
  TENSORSTORE_EXPECT_OK(tensorstore::Write(
      MakeScalarArray<::nlohmann::json>(::nlohmann::json::value_t::discarded),
      store));
  EXPECT_THAT(GetMap(kvstore).value(), testing::ElementsAre());
}

TEST(JsonDriverTest, IncompatibleWrites) {
  auto context = tensorstore::Context::Default();
  tensorstore::Transaction transaction(tensorstore::isolated);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(context, GetSpec("")).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store_a, tensorstore::Open(context, GetSpec("/a")).result());
  // Write initial value (42) to root
  TENSORSTORE_EXPECT_OK(tensorstore::Write(
      MakeScalarArray<::nlohmann::json>(42), store | transaction));
  // Write value for "/a"
  EXPECT_THAT(
      tensorstore::Write(MakeScalarArray<::nlohmann::json>(true),
                         store_a | transaction)
          .result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error writing \"path\\.json\": JSON Pointer reference "
                    "\"/a\" cannot be applied to number value: 42"));
}

TEST(JsonDriverTest, InvalidJson) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto kvstore,
      tensorstore::KeyValueStore::Open(context, GetKvstoreSpec()).result());
  TENSORSTORE_EXPECT_OK(kvstore->Write(GetPath(), absl::Cord("invalid")));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(context, GetSpec("")).result());
  EXPECT_THAT(tensorstore::Read(store).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error reading \"path\\.json\": Invalid JSON"));
}

TEST(JsonDriverTest, InvalidSpec) {
  auto json_spec = GetSpec("foo");
  EXPECT_THAT(tensorstore::Spec::FromJson(json_spec),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"json_pointer\": "
                            "JSON Pointer does not start with '/': \"foo\""));
}

TEST(JsonDriverTest, ReadError) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context.GetResource(
          tensorstore::Context::ResourceSpec<
              tensorstore::internal::MockKeyValueStoreResource>::Default()));
  auto mock_key_value_store = *mock_key_value_store_resource;
  auto spec = GetSpec("/a");
  spec["kvstore"] = {{"driver", "mock_key_value_store"}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(context, spec).result());

  // Test error handling during read request
  {
    auto read_future = tensorstore::Read(store);
    mock_key_value_store->read_requests.pop().promise.SetResult(
        absl::UnknownError("read error"));
    EXPECT_THAT(read_future.result(),
                MatchesStatus(absl::StatusCode::kUnknown,
                              "Error reading \"path\\.json\": read error"));
  }

  // Test read error handling during writeback
  {
    auto write_future =
        tensorstore::Write(MakeScalarArray<::nlohmann::json>(42), store);
    TENSORSTORE_EXPECT_OK(write_future.copy_future);
    mock_key_value_store->read_requests.pop().promise.SetResult(
        absl::UnknownError("read error2"));
    EXPECT_THAT(write_future.commit_future.result(),
                MatchesStatus(absl::StatusCode::kUnknown,
                              "Error reading \"path\\.json\": read error2"));
  }
}

TEST(JsonDriverTest, ConditionalWriteback) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context.GetResource(
          tensorstore::Context::ResourceSpec<
              tensorstore::internal::MockKeyValueStoreResource>::Default()));
  auto mock_key_value_store = *mock_key_value_store_resource;
  auto memory_store = tensorstore::GetMemoryKeyValueStore();
  auto spec = GetSpec("/a");
  spec["kvstore"] = {{"driver", "mock_key_value_store"}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(context, spec).result());

  // Write initial value.  Write is conditional since only sub-value is written.
  {
    auto write_future =
        tensorstore::Write(MakeScalarArray<::nlohmann::json>(42), store);
    TENSORSTORE_EXPECT_OK(write_future.copy_future);
    mock_key_value_store->read_requests.pop()(memory_store);
    mock_key_value_store->write_requests.pop()(memory_store);
    TENSORSTORE_EXPECT_OK(write_future.commit_future);
  }

  // Re-write same value.
  {
    auto write_future =
        tensorstore::Write(MakeScalarArray<::nlohmann::json>(42), store);
    TENSORSTORE_EXPECT_OK(write_future.copy_future);
    mock_key_value_store->read_requests.pop()(memory_store);
    // No write request, since value is unchanged.
    TENSORSTORE_EXPECT_OK(write_future.commit_future);
  }
}

TEST(JsonDriverTest, UnconditionalWriteback) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context.GetResource(
          tensorstore::Context::ResourceSpec<
              tensorstore::internal::MockKeyValueStoreResource>::Default()));
  auto mock_key_value_store = *mock_key_value_store_resource;
  auto memory_store = tensorstore::GetMemoryKeyValueStore();
  auto spec = GetSpec("");
  spec["kvstore"] = {{"driver", "mock_key_value_store"}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(context, spec).result());
  auto write_future =
      tensorstore::Write(MakeScalarArray<::nlohmann::json>(42), store);
  {
    auto write_req = mock_key_value_store->write_requests.pop();
    EXPECT_EQ(tensorstore::StorageGeneration::Unknown(),
              write_req.options.if_equal);
    write_req(memory_store);
  }
  TENSORSTORE_EXPECT_OK(write_future);
}

TEST(JsonDriverTest, ZeroElementWrite) {
  auto json_spec = GetSpec("");
  json_spec["cache_pool"] = {{"total_bytes_limit", 10000000}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(tensorstore::Context::Default(), json_spec).result());
  // Confirm that a one-element write is not immediately committed due to cache.
  {
    auto write_future =
        tensorstore::Write(MakeScalarArray<::nlohmann::json>(42), store);
    TENSORSTORE_EXPECT_OK(write_future.copy_future);
    absl::SleepFor(absl::Milliseconds(10));
    EXPECT_FALSE(write_future.commit_future.ready());
    // When forced, future becomes ready.
    TENSORSTORE_EXPECT_OK(write_future.commit_future);
  }

  // Test that a write to zero elements is detected as a non-modification, and
  // leads to an immediately-ready future.
  {
    auto write_future = tensorstore::Write(
        tensorstore::AllocateArray<::nlohmann::json>({0}),
        store | tensorstore::Dims(0).AddNew().SizedInterval(0, 0));
    TENSORSTORE_EXPECT_OK(write_future.copy_future);
    absl::SleepFor(absl::Milliseconds(10));
    EXPECT_TRUE(write_future.commit_future.ready());
    TENSORSTORE_EXPECT_OK(write_future.commit_future);
  }
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TestTensorStoreDriverSpecRoundtripOptions options;
  options.test_name = "json";
  options.create_spec = {
      {"driver", "json"},
      {"kvstore", GetKvstoreSpec()},
      {"path", GetPath()},
  };
  options.full_spec = {
      {"dtype", "json"},
      {"driver", "json"},
      {"kvstore", {{"driver", "memory"}, {"atomic", true}}},
      {"recheck_cached_data", "open"},
      {"json_pointer", ""},
      {"path", GetPath()},
      {"transform",
       {{"input_exclusive_max", ::nlohmann::json::array_t{}},
        {"input_inclusive_min", ::nlohmann::json::array_t{}}}},
  };
  options.minimal_spec = options.full_spec;
  options.check_not_found_before_create = false;
  options.check_not_found_before_commit = false;
  options.to_json_options = tensorstore::IncludeContext{false};
  tensorstore::internal::RegisterTensorStoreDriverSpecRoundtripTest(
      std::move(options));
}

}  // namespace
