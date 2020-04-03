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

#include "tensorstore/kvstore/key_value_store_testutil.h"

#include <cassert>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_object_with_type.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/util/function_view.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace tensorstore {
namespace internal {
namespace {

using tensorstore::MatchesStatus;

void Cleanup(KeyValueStore::Ptr store, std::vector<std::string> objects) {
  // Delete everything that we're going to use before starting.
  // This is helpful if, for instance, we run against a persistent
  // service and the test crashed half-way through last time.
  TENSORSTORE_LOG("Cleanup");
  for (const auto& to_remove : objects) {
    store->Delete(to_remove).result();
  }
}

StorageGeneration GetStorageGeneration(KeyValueStore::Ptr store,
                                       std::string key) {
  auto get = store->Read(key).result();
  StorageGeneration gen;
  if (GetStatus(get).ok()) {
    gen = get->generation.generation;
  }
  return gen;
}

StorageGeneration GetMismatchStorageGeneration(KeyValueStore::Ptr store,
                                               std::string name) {
  std::string value = "8888";
  auto last = store->Write(name, value).result();
  EXPECT_THAT(last, MatchesRegularTimestampedStorageGeneration());
  EXPECT_THAT(store->Delete(name).result(),
              MatchesKnownTimestampedStorageGeneration());
  StorageGeneration gen;
  if (last) {
    gen = last->generation;
  }
  return gen;
}

}  // namespace

void TestKeyValueStoreUnconditionalOps(
    KeyValueStore::Ptr store,
    FunctionView<std::string(std::string key)> get_key) {
  const auto key = get_key("test");
  Cleanup(store, {key});

  // Test unconditional read of missing key.
  TENSORSTORE_LOG("Test unconditional read of missing key");
  EXPECT_THAT(store->Read(key).result(), MatchesKvsReadResultNotFound());

  // Test unconditional write.
  TENSORSTORE_LOG("Test unconditional write");
  std::string value = "1234";
  auto write_result = store->Write(key, value).result();
  ASSERT_THAT(write_result, MatchesRegularTimestampedStorageGeneration());

  // Test unconditional read.
  TENSORSTORE_LOG("Test unconditional read");
  EXPECT_THAT(store->Read(key).result(),
              MatchesKvsReadResult(value, write_result->generation));

  // Test unconditional byte range read.
  TENSORSTORE_LOG("Test unconditional byte range read");
  {
    KeyValueStore::ReadOptions options;
    options.byte_range.inclusive_min = 1;
    EXPECT_THAT(store->Read(key, options).result(),
                MatchesKvsReadResult("234", write_result->generation));
  }

  // Test unconditional byte range read.
  TENSORSTORE_LOG("Test unconditional byte range read with exclusive_max");
  {
    KeyValueStore::ReadOptions options;
    options.byte_range.inclusive_min = 1;
    options.byte_range.exclusive_max = 3;
    EXPECT_THAT(store->Read(key, options).result(),
                MatchesKvsReadResult("23", write_result->generation));
  }

  TENSORSTORE_LOG("Test unconditional byte range read with invalid range");
  {
    KeyValueStore::ReadOptions options;
    options.byte_range.inclusive_min = 1;
    options.byte_range.exclusive_max = 10;
    EXPECT_THAT(store->Read(key, options).result(),
                MatchesStatus(absl::StatusCode::kOutOfRange));
  }

  // Test unconditional delete.
  TENSORSTORE_LOG("Test unconditional delete");
  EXPECT_THAT(store->Delete(key).result(),
              MatchesKnownTimestampedStorageGeneration());

  // Verify that read reflects deletion.
  EXPECT_THAT(store->Read(key).result(), MatchesKvsReadResultNotFound());
}

void TestKeyValueStoreConditionalReadOps(
    KeyValueStore::Ptr store,
    FunctionView<std::string(std::string key)> get_key) {
  // Conditional read use an if-not-exists condition.
  const auto key = get_key("test1");
  Cleanup(store, {key});

  // Generate a mismatched storage generation by writing and deleting.
  const StorageGeneration mismatch = GetMismatchStorageGeneration(store, key);

  TENSORSTORE_LOG("Test conditional read, missing");
  EXPECT_THAT(store->Read(key, {mismatch}).result(),
              MatchesKvsReadResultNotFound());

  TENSORSTORE_LOG(
      "Test conditional read, matching if_equal=StorageGeneration::NoValue");
  {
    KeyValueStore::ReadOptions options;
    options.if_equal = StorageGeneration::NoValue();
    EXPECT_THAT(store->Read(key, options).result(),
                MatchesKvsReadResultNotFound());
  }

  // Test conditional read of a non-existent object using
  // `if_not_equal=StorageGeneration::NoValue()`, which should return
  // `StorageGeneration::NoValue()` even though the `if_not_equal` condition
  // does not hold.
  TENSORSTORE_LOG("Test conditional read, StorageGeneration::NoValue");
  EXPECT_THAT(store->Read(key, {StorageGeneration::NoValue()}).result(),
              MatchesKvsReadResultNotFound());

  // Write a value.
  std::string value = "five by five";
  auto write_result = store->Write(key, value).result();

  // Preconditions for the rest of the function.
  ASSERT_THAT(write_result,
              ::testing::AllOf(MatchesRegularTimestampedStorageGeneration(),
                               MatchesTimestampedStorageGeneration(
                                   ::testing::Not(mismatch))));

  TENSORSTORE_LOG("Test conditional read, matching `if_not_equal` generation");
  EXPECT_THAT(store->Read(key, {write_result->generation}).result(),
              MatchesKvsReadResultAborted());

  TENSORSTORE_LOG(
      "Test conditional read, mismatched `if_not_equal` generation");
  EXPECT_THAT(store->Read(key, {mismatch}).result(),
              MatchesKvsReadResult(value, write_result->generation));

  TENSORSTORE_LOG(
      "Test conditional read, if_not_equal=StorageGeneration::NoValue");
  EXPECT_THAT(store->Read(key, {StorageGeneration::NoValue()}).result(),
              MatchesKvsReadResult(value, write_result->generation));

  TENSORSTORE_LOG("Test conditional read, matching `if_equal` generation");
  {
    KeyValueStore::ReadOptions options;
    options.if_equal = write_result->generation;
    EXPECT_THAT(store->Read(key, options).result(),
                MatchesKvsReadResult(value, write_result->generation));
  }

  TENSORSTORE_LOG(
      "Test conditional read, mismatched `if_not_equal` generation");
  EXPECT_THAT(store->Read(key, {mismatch}).result(),
              MatchesKvsReadResult(value, write_result->generation));

  TENSORSTORE_LOG("Test conditional read, mismatched `if_equal` generation");
  {
    KeyValueStore::ReadOptions options;
    options.if_equal = mismatch;
    EXPECT_THAT(store->Read(key, options).result(),
                MatchesKvsReadResultAborted());
  }

  TENSORSTORE_LOG(
      "Test conditional read, mismatched if_equal=StorageGeneration::NoValue");
  {
    KeyValueStore::ReadOptions options;
    options.if_equal = StorageGeneration::NoValue();
    EXPECT_THAT(store->Read(key, options).result(),
                MatchesKvsReadResultAborted());
  }

  TENSORSTORE_LOG(
      "Test conditional read, if_not_equal=StorageGeneration::NoValue");
  EXPECT_THAT(store->Read(key, {StorageGeneration::NoValue()}).result(),
              MatchesKvsReadResult(value, write_result->generation));
}

void TestKeyValueStoreConditionalWriteOps(
    KeyValueStore::Ptr store,
    FunctionView<std::string(std::string key)> get_key) {
  const auto key1 = get_key("test2a");
  const auto key2 = get_key("test2b");
  const auto key3 = get_key("test2c");
  Cleanup(store, {key1, key2, key3});

  // Mismatch should not match any other generation.
  const StorageGeneration mismatch2a =
      GetMismatchStorageGeneration(store, key1);
  const StorageGeneration mismatch2b =
      GetMismatchStorageGeneration(store, key2);
  const std::string value("007");

  // Create an existing key.
  auto write_result = store->Write(key2, ".-=-.").result();
  ASSERT_THAT(write_result,
              ::testing::AllOf(MatchesRegularTimestampedStorageGeneration(),
                               MatchesTimestampedStorageGeneration(
                                   ::testing::Not(mismatch2b))));

  TENSORSTORE_LOG("Test conditional write, non-existent key");
  EXPECT_THAT(
      store->Write(key1, value, {mismatch2a}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  TENSORSTORE_LOG("Test conditional write, mismatched generation");
  EXPECT_THAT(
      store->Write(key2, value, {mismatch2b}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  TENSORSTORE_LOG("Test conditional write, matching generation");
  {
    auto write_conditional =
        store->Write(key2, value, {write_result->generation}).result();
    ASSERT_THAT(write_conditional,
                MatchesRegularTimestampedStorageGeneration());

    // Read has the correct data.
    EXPECT_THAT(store->Read(key2).result(),
                MatchesKvsReadResult(value, write_conditional->generation));
  }

  TENSORSTORE_LOG(
      "Test conditional write, existing key, StorageGeneration::NoValue");
  EXPECT_THAT(
      store->Write(key2, value, {StorageGeneration::NoValue()}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  TENSORSTORE_LOG(
      "Test conditional write, non-existent key StorageGeneration::NoValue");
  {
    auto write_conditional =
        store->Write(key3, value, {StorageGeneration::NoValue()}).result();

    ASSERT_THAT(write_conditional,
                MatchesRegularTimestampedStorageGeneration());

    // Read has the correct data.
    EXPECT_THAT(store->Read(key3).result(),
                MatchesKvsReadResult(value, write_conditional->generation));
  }

  Cleanup(store, {key1, key2, key3});
}

void TestKeyValueStoreConditionalDeleteOps(
    KeyValueStore::Ptr store,
    FunctionView<std::string(std::string key)> get_key) {
  const auto key1 = get_key("test3a");
  const auto key2 = get_key("test3b");
  const auto key3 = get_key("test3c");
  const auto key4 = get_key("test3d");
  Cleanup(store, {key1, key2, key3, key4});

  // Mismatch should not match any other generation.
  const StorageGeneration mismatch3a =
      GetMismatchStorageGeneration(store, key1);
  const StorageGeneration mismatch3b =
      GetMismatchStorageGeneration(store, key1);

  // Create an existing key.
  StorageGeneration last_generation;
  for (const auto& name : {key4, key2}) {
    auto write_result = store->Write(name, ".-=-.").result();
    ASSERT_THAT(write_result, MatchesRegularTimestampedStorageGeneration());
    last_generation = std::move(write_result->generation);
  }
  ASSERT_NE(last_generation, mismatch3b);

  TENSORSTORE_LOG("Test conditional delete, non-existent key");
  EXPECT_THAT(
      store->Delete(key1, {mismatch3a}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  TENSORSTORE_LOG("Test conditional delete, mismatched generation");
  EXPECT_THAT(
      store->Delete(key2, {mismatch3b}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  TENSORSTORE_LOG("Test conditional delete, matching generation");
  ASSERT_THAT(store->Delete(key2, {last_generation}).result(),
              MatchesKnownTimestampedStorageGeneration());

  // Verify that read reflects deletion.
  EXPECT_THAT(store->Read(key2).result(), MatchesKvsReadResultNotFound());

  TENSORSTORE_LOG(
      "Test conditional delete, non-existent key StorageGeneration::NoValue");
  EXPECT_THAT(store->Delete(key3, {StorageGeneration::NoValue()}).result(),
              MatchesKnownTimestampedStorageGeneration());

  TENSORSTORE_LOG(
      "Test conditional delete, existing key, StorageGeneration::NoValue");
  EXPECT_THAT(
      store->Delete(key4, {StorageGeneration::NoValue()}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  TENSORSTORE_LOG("Test conditional delete, matching generation");
  {
    auto gen = GetStorageGeneration(store, key4);
    EXPECT_THAT(store->Delete(key4, {gen}).result(),
                MatchesKnownTimestampedStorageGeneration());

    // Verify that read reflects deletion.
    EXPECT_THAT(store->Read(key4).result(), MatchesKvsReadResultNotFound());
  }
}

void TestKeyValueStoreBasicFunctionality(
    KeyValueStore::Ptr store,
    FunctionView<std::string(std::string key)> get_key) {
  TestKeyValueStoreUnconditionalOps(store, get_key);
  TestKeyValueStoreConditionalReadOps(store, get_key);
  TestKeyValueStoreConditionalWriteOps(store, get_key);
  TestKeyValueStoreConditionalDeleteOps(store, get_key);
}

void TestKeyValueStoreSpecRoundtrip(::nlohmann::json json_spec) {
  auto context = Context::Default();

  // Open and populate `"mykey"`.
  {
    auto store_result = KeyValueStore::Open(context, json_spec, {}).result();
    ASSERT_EQ(Status(), GetStatus(store_result));
    auto store = *store_result;
    auto spec_result = store->spec();
    ASSERT_EQ(Status(), GetStatus(spec_result));
    EXPECT_THAT(spec_result->ToJson(tensorstore::IncludeDefaults{false}),
                json_spec);
    ASSERT_THAT(store->Write("mykey", "myvalue").result(),
                MatchesRegularTimestampedStorageGeneration());
    EXPECT_THAT(store->Read("mykey").result(), MatchesKvsReadResult("myvalue"));
  }

  // Reopen and verify contents.
  {
    auto store_result = KeyValueStore::Open(context, json_spec, {}).result();
    ASSERT_EQ(Status(), GetStatus(store_result));
    auto store = *store_result;
    auto spec_result = store->spec();
    EXPECT_THAT(store->Read("mykey").result(), MatchesKvsReadResult("myvalue"));
  }
}

Result<std::map<std::string, std::string>> GetMap(KeyValueStore::Ptr kv_store) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto keys, ListFuture(kv_store.get()).result());
  std::map<std::string, std::string> result;
  for (const auto& key : keys) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto read_result,
                                 kv_store->Read(key).result());
    assert(!read_result.aborted());
    assert(!read_result.not_found());
    result.emplace(key, *read_result.value);
  }
  return result;
}

}  // namespace internal
}  // namespace tensorstore
