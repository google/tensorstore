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

#include "tensorstore/kvstore/test_util.h"

#include <cassert>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/function_ref.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {
namespace {

using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;

class Cleanup {
 public:
  Cleanup(KvStore store, std::vector<std::string> objects)
      : store_(std::move(store)), objects_(std::move(objects)) {
    DoCleanup();
  }
  void DoCleanup() {
    // Delete everything that we're going to use before starting.
    // This is helpful if, for instance, we run against a persistent
    // service and the test crashed half-way through last time.
    ABSL_LOG(INFO) << "Cleanup";
    for (const auto& to_remove : objects_) {
      TENSORSTORE_CHECK_OK(kvstore::Delete(store_, to_remove).result());
    }
  }

  ~Cleanup() { DoCleanup(); }

 private:
  KvStore store_;
  std::vector<std::string> objects_;
};

StorageGeneration GetStorageGeneration(const KvStore& store, std::string key) {
  auto get = kvstore::Read(store, key).result();
  StorageGeneration gen;
  if (get.ok()) {
    gen = get->stamp.generation;
  }
  return gen;
}

// Return a highly-improbable storage generation
StorageGeneration GetMismatchStorageGeneration(const KvStore& store) {
  // Use a single uint64_t storage generation here for GCS compatibility.
  // Also, the generation looks like a nanosecond timestamp.
  return StorageGeneration::FromValues(uint64_t{/*3.*/ 1415926535897932});
}

void TestKeyValueStoreUnconditionalOps(
    const KvStore& store,
    absl::FunctionRef<std::string(std::string key)> get_key) {
  const auto key = get_key("test");
  Cleanup cleanup(store, {key});
  const absl::Cord value("1234");

  // Test unconditional read of missing key.
  ABSL_LOG(INFO) << "Test unconditional read of missing key";
  EXPECT_THAT(kvstore::Read(store, key).result(),
              MatchesKvsReadResultNotFound());

  // Test unconditional write of empty value.
  {
    ABSL_LOG(INFO) << "Test unconditional write of empty value";
    auto write_result = kvstore::Write(store, key, absl::Cord()).result();
    ASSERT_THAT(write_result, MatchesRegularTimestampedStorageGeneration());

    // Test unconditional read.
    ABSL_LOG(INFO) << "Test unconditional read of empty value";
    EXPECT_THAT(kvstore::Read(store, key).result(),
                MatchesKvsReadResult(absl::Cord(), write_result->generation));
  }

  // Test unconditional write.
  ABSL_LOG(INFO) << "Test unconditional write of non-empty value";
  auto write_result = kvstore::Write(store, key, value).result();
  ASSERT_THAT(write_result, MatchesRegularTimestampedStorageGeneration());

  // Test unconditional read.
  ABSL_LOG(INFO) << "Test unconditional read of non-empty value";
  EXPECT_THAT(kvstore::Read(store, key).result(),
              MatchesKvsReadResult(value, write_result->generation));

  // Test unconditional byte range read.
  ABSL_LOG(INFO) << "Test unconditional byte range read";
  {
    kvstore::ReadOptions options;
    options.byte_range.inclusive_min = 1;
    EXPECT_THAT(
        kvstore::Read(store, key, options).result(),
        MatchesKvsReadResult(absl::Cord("234"), write_result->generation));
  }

  // Test unconditional byte range read.
  ABSL_LOG(INFO) << "Test unconditional byte range read with exclusive_max";
  {
    kvstore::ReadOptions options;
    options.byte_range.inclusive_min = 1;
    options.byte_range.exclusive_max = 3;
    EXPECT_THAT(
        kvstore::Read(store, key, options).result(),
        MatchesKvsReadResult(absl::Cord("23"), write_result->generation));
  }

  // Test unconditional byte range read.
  ABSL_LOG(INFO) << "Test unconditional byte range read with size 0";
  {
    kvstore::ReadOptions options;
    options.byte_range.inclusive_min = 1;
    options.byte_range.exclusive_max = 1;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResult(absl::Cord(""), write_result->generation));
  }

  ABSL_LOG(INFO) << "Test unconditional byte range read, min too large";
  {
    kvstore::ReadOptions options;
    options.byte_range.inclusive_min = 10;
    options.byte_range.exclusive_max = 11;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                testing::AnyOf(MatchesStatus(absl::StatusCode::kOutOfRange)));
  }

  ABSL_LOG(INFO)
      << "Test unconditional byte range read, max exceeds value size";
  {
    kvstore::ReadOptions options;
    options.byte_range.inclusive_min = 1;
    options.byte_range.exclusive_max = 10;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesStatus(absl::StatusCode::kOutOfRange));
  }

  // Test unconditional delete.
  ABSL_LOG(INFO) << "Test unconditional delete";
  EXPECT_THAT(kvstore::Delete(store, key).result(),
              MatchesKnownTimestampedStorageGeneration());

  // Verify that read reflects deletion.
  EXPECT_THAT(kvstore::Read(store, key).result(),
              MatchesKvsReadResultNotFound());
}

void TestKeyValueStoreConditionalReadOps(
    const KvStore& store,
    absl::FunctionRef<std::string(std::string key)> get_key) {
  const auto missing_key = get_key("test1a");
  const StorageGeneration mismatch = GetMismatchStorageGeneration(store);
  ABSL_LOG(INFO) << "... Conditional read of missing values.";

  ABSL_LOG(INFO)
      << "Test conditional read, matching if_equal=StorageGeneration::NoValue";
  {
    kvstore::ReadOptions options;
    options.if_equal = StorageGeneration::NoValue();
    EXPECT_THAT(kvstore::Read(store, missing_key, options).result(),
                MatchesKvsReadResultNotFound());
  }

  ABSL_LOG(INFO) << "Test conditional read, if_equal mismatch";
  {
    kvstore::ReadOptions options;
    options.if_equal = mismatch;
    EXPECT_THAT(
        kvstore::Read(store, missing_key, options).result(),
        testing::AnyOf(MatchesKvsReadResultNotFound(),   /// Common result
                       MatchesKvsReadResultAborted()));  /// GCS result
  }

  // Test conditional read of a non-existent object using
  // `if_not_equal=StorageGeneration::NoValue()`, which should return
  // `StorageGeneration::NoValue()` even though the `if_not_equal` condition
  // does not hold.
  ABSL_LOG(INFO) << "Test conditional read, matching "
                    "if_not_equal=StorageGeneration::NoValue";
  {
    kvstore::ReadOptions options;
    options.if_not_equal = StorageGeneration::NoValue();
    EXPECT_THAT(kvstore::Read(store, missing_key, options).result(),
                MatchesKvsReadResultNotFound());
  }

  ABSL_LOG(INFO) << "Test conditional read, if_not_equal mismatch";
  {
    kvstore::ReadOptions options;
    options.if_not_equal = mismatch;
    EXPECT_THAT(kvstore::Read(store, missing_key, options).result(),
                MatchesKvsReadResultNotFound());
  }

  ABSL_LOG(INFO) << "... Conditional read of existing values.";

  // Write a value.
  const absl::Cord value("five by five");
  const auto key = get_key("test1b");
  Cleanup cleanup(store, {key});

  // Preconditions for the rest of the function.
  auto write_result = kvstore::Write(store, key, value).result();
  ASSERT_THAT(write_result,
              ::testing::AllOf(MatchesRegularTimestampedStorageGeneration(),
                               MatchesTimestampedStorageGeneration(
                                   ::testing::Not(mismatch))));

  // if_not_equal tests
  ABSL_LOG(INFO) << "Test conditional read, if_not_equal matching "
                 << write_result->generation;
  {
    kvstore::ReadOptions options;
    options.if_not_equal = write_result->generation;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResultAborted());
  }

  ABSL_LOG(INFO) << "Test conditional read, if_not_equal mismatched";
  {
    kvstore::ReadOptions options;
    options.if_not_equal = mismatch;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResult(value, write_result->generation));
  }

  ABSL_LOG(INFO)
      << "Test conditional read, if_not_equal=StorageGeneration::NoValue";
  {
    kvstore::ReadOptions options;
    options.if_not_equal = StorageGeneration::NoValue();
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResult(value, write_result->generation));
  }

  /// if_equal tests
  ABSL_LOG(INFO) << "Test conditional read, if_equal matching "
                 << write_result->generation;
  {
    kvstore::ReadOptions options;
    options.if_equal = write_result->generation;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResult(value, write_result->generation));
  }

  ABSL_LOG(INFO) << "Test conditional read, if_equal mismatched";
  {
    kvstore::ReadOptions options;
    options.if_equal = mismatch;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResultAborted());
  }

  ABSL_LOG(INFO) << "Test conditional read, mismatched "
                    "if_equal=StorageGeneration::NoValue";
  {
    kvstore::ReadOptions options;
    options.if_equal = StorageGeneration::NoValue();
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResultAborted());
  }

  // NOTE: Add tests for both if_equal and if_not_equal set.
}

void TestKeyValueStoreConditionalWriteOps(
    const KvStore& store,
    absl::FunctionRef<std::string(std::string key)> get_key) {
  const auto key1 = get_key("test2a");
  const auto key2 = get_key("test2b");
  const auto key3 = get_key("test2c");
  Cleanup cleanup(store, {key1, key2, key3});

  // Mismatch should not match any other generation.
  const StorageGeneration mismatch = GetMismatchStorageGeneration(store);
  const absl::Cord value("007");

  // Create an existing key.
  auto write_result = kvstore::Write(store, key2, absl::Cord(".-=-.")).result();
  ASSERT_THAT(write_result,
              ::testing::AllOf(MatchesRegularTimestampedStorageGeneration(),
                               MatchesTimestampedStorageGeneration(
                                   ::testing::Not(mismatch))));

  ABSL_LOG(INFO) << "Test conditional write, non-existent key";
  EXPECT_THAT(
      kvstore::Write(store, key1, value, {mismatch}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  ABSL_LOG(INFO) << "Test conditional write, mismatched generation";
  EXPECT_THAT(
      kvstore::Write(store, key2, value, {mismatch}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  ABSL_LOG(INFO) << "Test conditional write, matching generation "
                 << write_result->generation;
  {
    auto write_conditional =
        kvstore::Write(store, key2, value, {write_result->generation}).result();
    ASSERT_THAT(write_conditional,
                MatchesRegularTimestampedStorageGeneration());

    // Read has the correct data.
    EXPECT_THAT(kvstore::Read(store, key2).result(),
                MatchesKvsReadResult(value, write_conditional->generation));
  }

  ABSL_LOG(INFO)
      << "Test conditional write, existing key, StorageGeneration::NoValue";
  EXPECT_THAT(
      kvstore::Write(store, key2, value, {StorageGeneration::NoValue()})
          .result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  ABSL_LOG(INFO)
      << "Test conditional write, non-existent key StorageGeneration::NoValue";
  {
    auto write_conditional =
        kvstore::Write(store, key3, value, {StorageGeneration::NoValue()})
            .result();

    ASSERT_THAT(write_conditional,
                MatchesRegularTimestampedStorageGeneration());

    // Read has the correct data.
    EXPECT_THAT(kvstore::Read(store, key3).result(),
                MatchesKvsReadResult(value, write_conditional->generation));
  }
}

void TestKeyValueStoreConditionalDeleteOps(
    const KvStore& store,
    absl::FunctionRef<std::string(std::string key)> get_key) {
  const auto key1 = get_key("test3a");
  const auto key2 = get_key("test3b");
  const auto key3 = get_key("test3c");
  const auto key4 = get_key("test3d");
  Cleanup cleanup(store, {key1, key2, key3, key4});

  // Mismatch should not match any other generation.
  const StorageGeneration mismatch = GetMismatchStorageGeneration(store);

  // Create an existing key.
  StorageGeneration last_generation;
  absl::Cord existing_value(".-=-.");
  for (const auto& name : {key4, key2}) {
    auto write_result = kvstore::Write(store, name, existing_value).result();
    ASSERT_THAT(write_result, MatchesRegularTimestampedStorageGeneration());
    last_generation = std::move(write_result->generation);
  }
  ASSERT_NE(last_generation, mismatch);
  EXPECT_THAT(kvstore::Read(store, key2).result(),
              MatchesKvsReadResult(existing_value));
  EXPECT_THAT(kvstore::Read(store, key4).result(),
              MatchesKvsReadResult(existing_value));

  ABSL_LOG(INFO) << "Test conditional delete, non-existent key";
  EXPECT_THAT(
      kvstore::Delete(store, key1, {mismatch}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));
  EXPECT_THAT(kvstore::Read(store, key2).result(),
              MatchesKvsReadResult(existing_value));
  EXPECT_THAT(kvstore::Read(store, key4).result(),
              MatchesKvsReadResult(existing_value));

  ABSL_LOG(INFO) << "Test conditional delete, mismatched generation";
  EXPECT_THAT(
      kvstore::Delete(store, key2, {mismatch}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));
  EXPECT_THAT(kvstore::Read(store, key2).result(),
              MatchesKvsReadResult(existing_value));
  EXPECT_THAT(kvstore::Read(store, key4).result(),
              MatchesKvsReadResult(existing_value));

  ABSL_LOG(INFO) << "Test conditional delete, matching generation";
  ASSERT_THAT(kvstore::Delete(store, key2, {last_generation}).result(),
              MatchesKnownTimestampedStorageGeneration());

  // Verify that read reflects deletion.
  EXPECT_THAT(kvstore::Read(store, key2).result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, key4).result(),
              MatchesKvsReadResult(existing_value));

  ABSL_LOG(INFO)
      << "Test conditional delete, non-existent key StorageGeneration::NoValue";
  EXPECT_THAT(
      kvstore::Delete(store, key3, {StorageGeneration::NoValue()}).result(),
      MatchesKnownTimestampedStorageGeneration());

  ABSL_LOG(INFO)
      << "Test conditional delete, existing key, StorageGeneration::NoValue";
  EXPECT_THAT(kvstore::Read(store, key2).result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, key4).result(),
              MatchesKvsReadResult(existing_value));
  EXPECT_THAT(
      kvstore::Delete(store, key4, {StorageGeneration::NoValue()}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  ABSL_LOG(INFO) << "Test conditional delete, matching generation";
  {
    auto gen = GetStorageGeneration(store, key4);
    EXPECT_THAT(kvstore::Delete(store, key4, {gen}).result(),
                MatchesKnownTimestampedStorageGeneration());

    // Verify that read reflects deletion.
    EXPECT_THAT(kvstore::Read(store, key4).result(),
                MatchesKvsReadResultNotFound());
  }
}

void TestKeyValueStoreStalenessBoundOps(
    const KvStore& store,
    absl::FunctionRef<std::string(std::string key)> get_key) {
  const auto key = get_key("stale");
  Cleanup cleanup(store, {key});
  const absl::Cord value1("1234");
  const absl::Cord value2("abcd");

  kvstore::ReadOptions read_options;
  read_options.staleness_bound = absl::Now();

  // Test read of missing key
  ABSL_LOG(INFO) << "Test staleness_bound read of missing key";
  EXPECT_THAT(kvstore::Read(store, key, read_options).result(),
              MatchesKvsReadResultNotFound());

  auto write_result1 = kvstore::Write(store, key, value1).result();
  ASSERT_THAT(write_result1, MatchesRegularTimestampedStorageGeneration());

  // kvstore currently are not expected to cache missing values, even with
  // staleness_bound, however neuroglancer_uint64_sharded_test does.
  ABSL_LOG(INFO) << "Test staleness_bound read: value1";
  EXPECT_THAT(
      kvstore::Read(store, key, read_options).result(),
      testing::AnyOf(MatchesKvsReadResultNotFound(),
                     MatchesKvsReadResult(value1, write_result1->generation)));

  // Updating staleness_bound should guarantee a read.
  read_options.staleness_bound = absl::Now();

  ABSL_LOG(INFO) << "Test unconditional read: value1";
  EXPECT_THAT(kvstore::Read(store, key).result(),
              MatchesKvsReadResult(value1, write_result1->generation));

  // Generally same-host writes should invalidate staleness_bound in a kvstore.
  auto write_result2 = kvstore::Write(store, key, value2).result();
  ASSERT_THAT(write_result2, MatchesRegularTimestampedStorageGeneration());

  // However allow either version to satisfy this test.
  ABSL_LOG(INFO) << "Test staleness_bound read: value2";
  EXPECT_THAT(kvstore::Read(store, key, read_options).result(),
              ::testing::AnyOf(
                  MatchesKvsReadResult(value1, write_result1->generation),
                  MatchesKvsReadResult(value2, write_result2->generation)));
}

}  // namespace

void TestKeyValueStoreBasicFunctionality(
    const KvStore& store,
    absl::FunctionRef<std::string(std::string key)> get_key) {
  TestKeyValueStoreUnconditionalOps(store, get_key);
  TestKeyValueStoreConditionalReadOps(store, get_key);
  TestKeyValueStoreConditionalWriteOps(store, get_key);
  TestKeyValueStoreConditionalDeleteOps(store, get_key);
  TestKeyValueStoreStalenessBoundOps(store, get_key);
}

/// Tests List on `store`, which should be empty.
void TestKeyValueStoreList(const KvStore& store) {
  ABSL_LOG(INFO) << "Test list, empty";
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        CompletionNotifyingReceiver{&notification,
                                    tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();
    EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_done",
                                            "set_stopping"));
  }

  ABSL_LOG(INFO) << "Test list: ... write elements ...";

  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/b", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/d", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/x", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/y", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/e", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/f", absl::Cord("xyz")));

  // Listing the entire stream works.
  ABSL_LOG(INFO) << "Test list, entire stream";
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        CompletionNotifyingReceiver{&notification,
                                    tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();

    EXPECT_THAT(
        log, ::testing::UnorderedElementsAre(
                 "set_starting", "set_value: a/d", "set_value: a/c/z/f",
                 "set_value: a/c/y", "set_value: a/c/z/e", "set_value: a/c/x",
                 "set_value: a/b", "set_done", "set_stopping"));
  }

  // Listing a subset of the stream works.
  ABSL_LOG(INFO) << "Test list, prefix range";
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {KeyRange::Prefix("a/c/")}),
        CompletionNotifyingReceiver{&notification,
                                    tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();

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

  ABSL_LOG(INFO) << "Test list, cancel on start";
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        CompletionNotifyingReceiver{&notification, CancelOnStarting{{&log}}});
    notification.WaitForNotification();

    EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_done",
                                            "set_stopping"));
  }

  // Cancellation in the middle of the stream stops the stream.
  struct CancelAfter2 : public tensorstore::LoggingReceiver {
    using Key = kvstore::Key;
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

  ABSL_LOG(INFO) << "Test list, cancel after 2";
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        CompletionNotifyingReceiver{&notification, CancelAfter2{{&log}}});
    notification.WaitForNotification();

    EXPECT_THAT(log,
                ::testing::ElementsAre(
                    "set_starting",
                    ::testing::AnyOf("set_value: a/d", "set_value: a/c/z/f",
                                     "set_value: a/c/y", "set_value: a/c/z/e",
                                     "set_value: a/c/x", "set_value: a/b"),
                    "set_done", "set_stopping"));
  }

  ABSL_LOG(INFO) << "Test list: ... delete elements ...";
  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "a/b"));
  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "a/d"));
  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "a/c/x"));
  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "a/c/y"));
  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "a/c/z/e"));
  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "a/c/z/f"));
}

void TestKeyValueStoreDeleteRange(const KvStore& store) {
  std::vector<AnyFuture> futures;
  for (auto key : {"a/a", "a/b", "a/c/a", "a/c/b", "b/a", "b/b"}) {
    futures.push_back(kvstore::Write(store, key, absl::Cord()));
  }
  for (auto& f : futures) {
    TENSORSTORE_EXPECT_OK(f.status());
  }
  futures.clear();

  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange("a/b", "b/aa")));
  EXPECT_THAT(
      kvstore::ListFuture(store).result(),
      ::testing::Optional(::testing::UnorderedElementsAre("a/a", "b/b")));

  // Construct a lot of nested values.
  for (auto a : {"m", "n", "o", "p"}) {
    for (auto b : {"p", "q", "r", "s"}) {
      for (auto c : {"s", "t", "u", "v"}) {
        futures.push_back(
            kvstore::Write(store, absl::StrFormat("%s/%s/%s/data", a, b, c),
                           absl::Cord("abc")));
      }
    }
  }
  for (auto& f : futures) {
    TENSORSTORE_EXPECT_OK(f.status());
  }
  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange("l", "z")));
  EXPECT_THAT(
      kvstore::ListFuture(store).result(),
      ::testing::Optional(::testing::UnorderedElementsAre("a/a", "b/b")));
}

void TestKeyValueStoreDeletePrefix(const KvStore& store) {
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/b", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/d", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/x", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/y", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/e", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/f", absl::Cord("xyz")));
  EXPECT_THAT(kvstore::Read(store, "a/b").result(),
              MatchesKvsReadResult(absl::Cord("xyz")));

  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange::Prefix("a/c/")));

  EXPECT_THAT(kvstore::Read(store, "a/b").result(),
              MatchesKvsReadResult(absl::Cord("xyz")));
  EXPECT_THAT(kvstore::Read(store, "a/d").result(),
              MatchesKvsReadResult(absl::Cord("xyz")));

  EXPECT_THAT(kvstore::Read(store, "a/c/x").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, "a/c/y").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, "a/c/z/e").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, "a/c/z/f").result(),
              MatchesKvsReadResultNotFound());
}

void TestKeyValueStoreDeleteRangeToEnd(const KvStore& store) {
  for (auto key : {"a/a", "a/b", "a/c/a", "a/c/b", "b/a", "b/b"}) {
    TENSORSTORE_EXPECT_OK(kvstore::Write(store, key, absl::Cord()).result());
  }
  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange("a/b", "")));
  EXPECT_THAT(ListFuture(store).result(),
              ::testing::Optional(::testing::UnorderedElementsAre("a/a")));
}

void TestKeyValueStoreDeleteRangeFromBeginning(const KvStore& store) {
  for (auto key : {"a/a", "a/b", "a/c/a", "a/c/b", "b/a", "b/b"}) {
    TENSORSTORE_EXPECT_OK(kvstore::Write(store, key, absl::Cord()).result());
  }
  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange("", "a/c/aa")));
  EXPECT_THAT(ListFuture(store).result(),
              ::testing::Optional(
                  ::testing::UnorderedElementsAre("a/c/b", "b/a", "b/b")));
}

void TestKeyValueStoreSpecRoundtrip(
    const KeyValueStoreSpecRoundtripOptions& options) {
  const auto& expected_minimal_spec = options.minimal_spec.is_discarded()
                                          ? options.full_spec
                                          : options.minimal_spec;
  const auto& create_spec = options.create_spec.is_discarded()
                                ? options.full_spec
                                : options.create_spec;
  SCOPED_TRACE(tensorstore::StrCat("full_spec=", options.full_spec.dump()));
  SCOPED_TRACE(tensorstore::StrCat("create_spec=", create_spec.dump()));
  SCOPED_TRACE(
      tensorstore::StrCat("minimal_spec=", expected_minimal_spec.dump()));
  auto context = Context::Default();

  // Open and populate roundtrip_key.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, kvstore::Open(create_spec, context).result());

    if (options.check_write_read) {
      ASSERT_THAT(
          kvstore::Write(store, options.roundtrip_key, options.roundtrip_value)
              .result(),
          MatchesRegularTimestampedStorageGeneration());
      EXPECT_THAT(kvstore::Read(store, options.roundtrip_key).result(),
                  MatchesKvsReadResult(options.roundtrip_value));
    }

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto spec,
        store.spec(kvstore::SpecRequestOptions{options.spec_request_options}));
    EXPECT_THAT(spec.ToJson(options.json_serialization_options),
                ::testing::Optional(MatchesJson(options.full_spec)));

    auto minimal_spec_obj = spec;
    TENSORSTORE_ASSERT_OK(minimal_spec_obj.Set(tensorstore::MinimalSpec{true}));
    EXPECT_THAT(minimal_spec_obj.ToJson(options.json_serialization_options),
                ::testing::Optional(MatchesJson(expected_minimal_spec)));
  }

  ASSERT_TRUE(options.check_write_read || !options.check_data_persists);

  // Reopen and verify contents.
  if (options.check_data_persists) {
    // Reopen with full_spec
    {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto store, kvstore::Open(options.full_spec, context).result());
      TENSORSTORE_ASSERT_OK(store.spec());
      EXPECT_THAT(kvstore::Read(store, options.roundtrip_key).result(),
                  MatchesKvsReadResult(options.roundtrip_value));
    }
    if (!options.minimal_spec.is_discarded()) {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto store, kvstore::Open(expected_minimal_spec, context).result());
      TENSORSTORE_ASSERT_OK(store.spec());
      EXPECT_THAT(kvstore::Read(store, options.roundtrip_key).result(),
                  MatchesKvsReadResult(options.roundtrip_value));
    }
  }
}

void TestKeyValueStoreSpecRoundtripNormalize(
    ::nlohmann::json json_spec, ::nlohmann::json normalized_json_spec) {
  SCOPED_TRACE(tensorstore::StrCat("json_spec=", json_spec.dump()));
  SCOPED_TRACE(tensorstore::StrCat("normalized_json_spec=",
                                   normalized_json_spec.dump()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   kvstore::Open(json_spec).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, store.spec());
  EXPECT_THAT(spec.ToJson(),
              ::testing::Optional(MatchesJson(normalized_json_spec)));
}

void TestKeyValueStoreUrlRoundtrip(::nlohmann::json json_spec,
                                   std::string_view url) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_from_json,
                                   kvstore::Spec::FromJson(json_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_from_url,
                                   kvstore::Spec::FromUrl(url));
  EXPECT_THAT(spec_from_json.ToUrl(), ::testing::Optional(url));
  EXPECT_THAT(spec_from_url.ToJson(),
              ::testing::Optional(MatchesJson(json_spec)));
}

Result<std::map<kvstore::Key, kvstore::Value>> GetMap(const KvStore& store) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto keys, ListFuture(store).result());
  std::map<kvstore::Key, kvstore::Value> result;
  for (const auto& key : keys) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto read_result,
                                 kvstore::Read(store, key).result());
    assert(!read_result.aborted());
    assert(!read_result.not_found());
    result.emplace(key, std::move(read_result.value));
  }
  return result;
}

}  // namespace internal
}  // namespace tensorstore
