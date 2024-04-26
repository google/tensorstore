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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include <nlohmann/json.hpp>
#include "tensorstore/batch.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/internal/testing/random_seed.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {
namespace {

using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;

static const char kSep[] = "----------------------------------------------\n";

class Cleanup {
 public:
  Cleanup(KvStore store, std::vector<std::string> keys)
      : store_(std::move(store)), keys_(std::move(keys)) {
    DoCleanup();
  }

  void DoCleanup() {
    // Delete everything that we're going to use before starting.
    // This is helpful if, for instance, we run against a persistent
    // service and the test crashed half-way through last time.
    ABSL_LOG(INFO) << "Cleanup";
    for (const auto& to_remove : keys_) {
      TENSORSTORE_CHECK_OK(kvstore::Delete(store_, to_remove).result());
    }
  }

  ~Cleanup() { DoCleanup(); }

 private:
  KvStore store_;
  std::vector<std::string> keys_;
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
  const auto& url_result = store.ToUrl();

  if (url_result.ok() && absl::StrContains(url_result.value(), "s3://")) {
    return StorageGeneration::FromString("\"abcdef1234567890\"");
  }

  // Use a single uint64_t storage generation here for GCS compatibility.
  // Also, the generation looks like a nanosecond timestamp.
  return StorageGeneration::FromValues(uint64_t{/*3.*/ 1415926535897932});
}

void TestKeyValueStoreWriteOps(const KvStore& store,
                               std::array<std::string, 3> key,
                               absl::Cord expected_value,
                               absl::Cord other_value) {
  SCOPED_TRACE("TestKeyValueStoreWriteOps");
  ABSL_CHECK(expected_value.size() > 3);

  Cleanup cleanup(store, {key.begin(), key.end()});

  const StorageGeneration mismatch = GetMismatchStorageGeneration(store);

  // The key should not be found.
  ASSERT_THAT(kvstore::Read(store, key[0]).result(),
              MatchesKvsReadResultNotFound());

  // Test unconditional write of empty value.
  {
    ABSL_LOG(INFO) << kSep << "Test unconditional write of empty value";
    auto write_result = kvstore::Write(store, key[0], absl::Cord()).result();
    ASSERT_THAT(write_result, MatchesRegularTimestampedStorageGeneration());

    // Test unconditional read.
    ABSL_LOG(INFO) << kSep << "Test unconditional read of empty value";
    EXPECT_THAT(kvstore::Read(store, key[0]).result(),
                MatchesKvsReadResult(absl::Cord(), write_result->generation));
  }

  // Test unconditional write.
  {
    ABSL_LOG(INFO) << kSep << "Test unconditional write";
    auto write_result = kvstore::Write(store, key[0], expected_value).result();
    ASSERT_THAT(write_result, MatchesRegularTimestampedStorageGeneration());

    // Verify unconditional read.
    ABSL_LOG(INFO) << kSep << "Test unconditional read";
    EXPECT_THAT(kvstore::Read(store, key[0]).result(),
                MatchesKvsReadResult(expected_value, write_result->generation));

    // Verify unconditional byte range read.
    kvstore::ReadOptions options;
    options.byte_range.inclusive_min = 1;
    options.byte_range.exclusive_max = 3;
    EXPECT_THAT(kvstore::Read(store, key[0], options).result(),
                MatchesKvsReadResult(expected_value.Subcord(1, 2),
                                     write_result->generation));
  }

  // Test unconditional delete.
  ABSL_LOG(INFO) << kSep << "Test unconditional delete";
  EXPECT_THAT(kvstore::Delete(store, key[0]).result(),
              MatchesKnownTimestampedStorageGeneration());

  // Verify that read reflects deletion.
  EXPECT_THAT(kvstore::Read(store, key[0]).result(),
              MatchesKvsReadResultNotFound());

  ABSL_LOG(INFO) << kSep << "Test conditional write, non-existent key";
  EXPECT_THAT(
      kvstore::Write(store, key[1], expected_value, {mismatch}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  ABSL_LOG(INFO) << kSep << "Test conditional write, mismatched generation";
  auto write2 = kvstore::Write(store, key[1], other_value).result();
  ASSERT_THAT(write2,
              ::testing::AllOf(MatchesRegularTimestampedStorageGeneration(),
                               MatchesTimestampedStorageGeneration(
                                   ::testing::Not(mismatch))));

  EXPECT_THAT(
      kvstore::Write(store, key[1], expected_value, {mismatch}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  ABSL_LOG(INFO) << kSep << "Test conditional write, matching generation "
                 << write2->generation;
  {
    auto write_conditional =
        kvstore::Write(store, key[1], expected_value, {write2->generation})
            .result();
    ASSERT_THAT(write_conditional,
                MatchesRegularTimestampedStorageGeneration());

    // Read has the correct data.
    EXPECT_THAT(
        kvstore::Read(store, key[1]).result(),
        MatchesKvsReadResult(expected_value, write_conditional->generation));
  }

  ABSL_LOG(INFO)
      << kSep
      << "Test conditional write, existing key, StorageGeneration::NoValue";
  EXPECT_THAT(
      kvstore::Write(store, key[1], expected_value,
                     {StorageGeneration::NoValue()})
          .result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  ABSL_LOG(INFO)
      << kSep
      << "Test conditional write, non-existent key StorageGeneration::NoValue";
  {
    auto write_conditional = kvstore::Write(store, key[2], expected_value,
                                            {StorageGeneration::NoValue()})
                                 .result();

    ASSERT_THAT(write_conditional,
                MatchesRegularTimestampedStorageGeneration());

    // Read has the correct data.
    EXPECT_THAT(
        kvstore::Read(store, key[2]).result(),
        MatchesKvsReadResult(expected_value, write_conditional->generation));
  }
}

void TestKeyValueStoreDeleteOps(const KvStore& store,
                                std::array<std::string, 4> key,
                                absl::Cord expected_value) {
  SCOPED_TRACE("TestKeyValueStoreDeleteOps");
  Cleanup cleanup(store, {key.begin(), key.end()});

  // Mismatch should not match any other generation.
  const StorageGeneration mismatch = GetMismatchStorageGeneration(store);

  // Create an existing key.
  StorageGeneration last_generation;
  for (const auto& name : {key[3], key[1]}) {
    auto write_result = kvstore::Write(store, name, expected_value).result();
    ASSERT_THAT(write_result, MatchesRegularTimestampedStorageGeneration());
    last_generation = std::move(write_result->generation);
  }

  ASSERT_NE(last_generation, mismatch);
  EXPECT_THAT(kvstore::Read(store, key[1]).result(),
              MatchesKvsReadResult(expected_value));
  EXPECT_THAT(kvstore::Read(store, key[3]).result(),
              MatchesKvsReadResult(expected_value));

  ABSL_LOG(INFO) << kSep << "Test conditional delete, non-existent key";
  EXPECT_THAT(
      kvstore::Delete(store, key[0], {mismatch}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));
  EXPECT_THAT(kvstore::Read(store, key[1]).result(),
              MatchesKvsReadResult(expected_value));
  EXPECT_THAT(kvstore::Read(store, key[3]).result(),
              MatchesKvsReadResult(expected_value));

  ABSL_LOG(INFO) << kSep << "Test conditional delete, mismatched generation";
  EXPECT_THAT(
      kvstore::Delete(store, key[1], {mismatch}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));
  EXPECT_THAT(kvstore::Read(store, key[1]).result(),
              MatchesKvsReadResult(expected_value));
  EXPECT_THAT(kvstore::Read(store, key[3]).result(),
              MatchesKvsReadResult(expected_value));

  ABSL_LOG(INFO) << kSep << "Test conditional delete, matching generation";
  ASSERT_THAT(kvstore::Delete(store, key[1], {last_generation}).result(),
              MatchesKnownTimestampedStorageGeneration());

  // Verify that read reflects deletion.
  EXPECT_THAT(kvstore::Read(store, key[1]).result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, key[3]).result(),
              MatchesKvsReadResult(expected_value));

  ABSL_LOG(INFO)
      << kSep
      << "Test conditional delete, non-existent key StorageGeneration::NoValue";
  EXPECT_THAT(
      kvstore::Delete(store, key[2], {StorageGeneration::NoValue()}).result(),
      MatchesKnownTimestampedStorageGeneration());

  ABSL_LOG(INFO)
      << kSep
      << "Test conditional delete, existing key, StorageGeneration::NoValue";
  EXPECT_THAT(kvstore::Read(store, key[1]).result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, key[3]).result(),
              MatchesKvsReadResult(expected_value));
  EXPECT_THAT(
      kvstore::Delete(store, key[3], {StorageGeneration::NoValue()}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  ABSL_LOG(INFO) << kSep << "Test conditional delete, matching generation";
  {
    auto gen = GetStorageGeneration(store, key[3]);
    EXPECT_THAT(kvstore::Delete(store, key[3], {gen}).result(),
                MatchesKnownTimestampedStorageGeneration());

    // Verify that read reflects deletion.
    EXPECT_THAT(kvstore::Read(store, key[3]).result(),
                MatchesKvsReadResultNotFound());
  }
}

void TestKeyValueStoreStalenessBoundOps(const KvStore& store, std::string key,
                                        absl::Cord value1, absl::Cord value2) {
  SCOPED_TRACE("TestKeyValueStoreStalenessBoundOps");
  Cleanup cleanup(store, {key});

  kvstore::ReadOptions read_options;
  read_options.staleness_bound = absl::Now();

  // Test read of missing key
  ABSL_LOG(INFO) << kSep << "Test staleness_bound read of missing key";
  EXPECT_THAT(kvstore::Read(store, key, read_options).result(),
              MatchesKvsReadResultNotFound());

  auto write_result1 = kvstore::Write(store, key, value1).result();
  ASSERT_THAT(write_result1, MatchesRegularTimestampedStorageGeneration());

  // kvstore currently are not expected to cache missing values, even with
  // staleness_bound, however neuroglancer_uint64_sharded_test does.
  ABSL_LOG(INFO) << kSep << "Test staleness_bound read: value1";
  EXPECT_THAT(
      kvstore::Read(store, key, read_options).result(),
      testing::AnyOf(MatchesKvsReadResultNotFound(),
                     MatchesKvsReadResult(value1, write_result1->generation)));

  // Updating staleness_bound should guarantee a read.
  read_options.staleness_bound = absl::Now();

  ABSL_LOG(INFO) << kSep << "Test unconditional read: value1";
  EXPECT_THAT(kvstore::Read(store, key).result(),
              MatchesKvsReadResult(value1, write_result1->generation));

  // Generally same-host writes should invalidate staleness_bound in a kvstore.
  auto write_result2 = kvstore::Write(store, key, value2).result();
  ASSERT_THAT(write_result2, MatchesRegularTimestampedStorageGeneration());

  // However allow either version to satisfy this test.
  ABSL_LOG(INFO) << kSep << "Test staleness_bound read: value2";
  EXPECT_THAT(kvstore::Read(store, key, read_options).result(),
              ::testing::AnyOf(
                  MatchesKvsReadResult(value1, write_result1->generation),
                  MatchesKvsReadResult(value2, write_result2->generation)));
}

}  // namespace

void TestKeyValueStoreReadOps(const KvStore& store, std::string key,
                              absl::Cord expected_value,
                              std::string missing_key) {
  SCOPED_TRACE("TestKeyValueStoreReadOps");
  ABSL_CHECK(expected_value.size() > 3);
  ABSL_CHECK(!key.empty());
  ABSL_CHECK(!missing_key.empty());
  ABSL_CHECK(key != missing_key);

  StorageGeneration mismatch_generation = GetMismatchStorageGeneration(store);

  ABSL_LOG(INFO) << kSep << "Test unconditional read of key";
  auto read_result = kvstore::Read(store, key).result();
  EXPECT_THAT(read_result, MatchesKvsReadResult(expected_value, testing::_));

  ABSL_LOG(INFO) << kSep << "Test unconditional suffix read [1 ..]";
  {
    kvstore::ReadOptions options;
    options.byte_range.inclusive_min = 1;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResult(
                    expected_value.Subcord(1, expected_value.size() - 1),
                    read_result->stamp.generation));
  }

  ABSL_LOG(INFO) << kSep << "Test unconditional suffix length read [.. -1]";
  {
    kvstore::ReadOptions options;
    options.byte_range.inclusive_min = -1;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResult(
                    expected_value.Subcord(expected_value.size() - 1, 1),
                    read_result->stamp.generation));
  }

  ABSL_LOG(INFO) << kSep << "Test unconditional suffix length read [.. -2]";
  {
    kvstore::ReadOptions options;
    options.byte_range.inclusive_min = -2;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResult(
                    expected_value.Subcord(expected_value.size() - 2, 2),
                    read_result->stamp.generation));
  }

  ABSL_LOG(INFO) << kSep << "Test unconditional range read [1 .. 3]";
  {
    kvstore::ReadOptions options;
    options.byte_range.inclusive_min = 1;
    options.byte_range.exclusive_max = 3;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResult(expected_value.Subcord(1, 2),
                                     read_result->stamp.generation));
  }

  ABSL_LOG(INFO) << kSep << "Test unconditional range read [1 .. 1], size 0";
  {
    kvstore::ReadOptions options;
    options.byte_range.inclusive_min = 1;
    options.byte_range.exclusive_max = 1;
    EXPECT_THAT(
        kvstore::Read(store, key, options).result(),
        MatchesKvsReadResult(absl::Cord(), read_result->stamp.generation));
  }

  ABSL_LOG(INFO) << kSep << "Test unconditional suffix read, min too large";
  {
    kvstore::ReadOptions options;
    options.byte_range.inclusive_min = expected_value.size() + 1;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                testing::AnyOf(MatchesStatus(absl::StatusCode::kOutOfRange)));
  }

  ABSL_LOG(INFO) << kSep
                 << "Test unconditional range read, max exceeds value size";
  if (testing::UnitTest::GetInstance()
          ->current_test_info()
          ->test_suite_name() == std::string_view("GcsTestbenchTest")) {
    ABSL_LOG(INFO)
        << "Skipping due to "
           "https://github.com/googleapis/storage-testbench/pull/622";
  } else {
    kvstore::ReadOptions options;
    options.byte_range.inclusive_min = 1;
    options.byte_range.exclusive_max = expected_value.size() + 1;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesStatus(absl::StatusCode::kOutOfRange));
  }

  // --------------------------------------------------------------------
  ABSL_LOG(INFO) << kSep << "... Conditional read of existing values.";

  // if_not_equal tests
  ABSL_LOG(INFO) << kSep << "Test conditional read, if_not_equal matching "
                 << read_result->stamp.generation;
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_not_equal = read_result->stamp.generation;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResultAborted());
  }

  ABSL_LOG(INFO) << kSep << "Test conditional read, if_not_equal mismatched";
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_not_equal = mismatch_generation;
    EXPECT_THAT(
        kvstore::Read(store, key, options).result(),
        MatchesKvsReadResult(expected_value, read_result->stamp.generation));
  }

  ABSL_LOG(INFO)
      << kSep
      << "Test conditional read, if_not_equal=StorageGeneration::NoValue";
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_not_equal = StorageGeneration::NoValue();
    EXPECT_THAT(
        kvstore::Read(store, key, options).result(),
        MatchesKvsReadResult(expected_value, read_result->stamp.generation));
  }

  /// if_equal tests
  ABSL_LOG(INFO) << kSep << "Test conditional read, if_equal matching "
                 << read_result->stamp.generation;
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_equal = read_result->stamp.generation;
    EXPECT_THAT(
        kvstore::Read(store, key, options).result(),
        MatchesKvsReadResult(expected_value, read_result->stamp.generation));
  }

  ABSL_LOG(INFO) << kSep << "Test conditional read, if_equal mismatched";
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_equal = mismatch_generation;
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResultAborted());
  }

  ABSL_LOG(INFO) << kSep
                 << "Test conditional read, mismatched "
                    "if_equal=StorageGeneration::NoValue";
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_equal = StorageGeneration::NoValue();
    EXPECT_THAT(kvstore::Read(store, key, options).result(),
                MatchesKvsReadResultAborted());
  }

  ABSL_LOG(INFO) << kSep << "Test staleness_bound read of key";
  {
    // This should force a re-read.
    kvstore::ReadOptions read_options;
    read_options.staleness_bound = absl::Now();

    auto result = kvstore::Read(store, key, read_options).result();
    EXPECT_THAT(result, MatchesKvsReadResult(expected_value,
                                             read_result->stamp.generation));
    EXPECT_THAT(result->stamp.time, testing::Gt(read_result->stamp.time));
  }

  // NOTE: Add tests for both if_equal and if_not_equal set.

  // --------------------------------------------------------------------
  // Now test similar ops for missing keys.
  ABSL_LOG(INFO) << kSep << "Test unconditional read of missing key";
  EXPECT_THAT(kvstore::Read(store, missing_key).result(),
              MatchesKvsReadResultNotFound());

  ABSL_LOG(INFO) << kSep << "Test staleness_bound read of missing key";
  {
    kvstore::ReadOptions read_options;
    read_options.staleness_bound = absl::Now();

    // Test read of missing key
    EXPECT_THAT(kvstore::Read(store, missing_key, read_options).result(),
                MatchesKvsReadResultNotFound());
  }

  if (/* DISABLE*/ (false)) {
    // neuroglancer_uint64_sharded_test caches missing results.
    ABSL_LOG(INFO) << kSep
                   << "Test conditional read, matching "
                      "if_equal=StorageGeneration::NoValue";
    kvstore::ReadOptions options;
    options.generation_conditions.if_equal = StorageGeneration::NoValue();
    options.staleness_bound = absl::Now();
    EXPECT_THAT(kvstore::Read(store, missing_key, options).result(),
                MatchesKvsReadResultNotFound());
  }

  ABSL_LOG(INFO) << kSep << "Test conditional read, if_equal mismatch";
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_equal = mismatch_generation;
    EXPECT_THAT(
        kvstore::Read(store, missing_key, options).result(),
        testing::AnyOf(MatchesKvsReadResultNotFound(),   /// Common result
                       MatchesKvsReadResultAborted()));  /// GCS result
  }

  // Test conditional read of a non-existent object using
  // `if_not_equal=StorageGeneration::NoValue()`, which should return
  // `StorageGeneration::NoValue()` even though the `if_not_equal` condition
  // does not hold.
  ABSL_LOG(INFO) << kSep
                 << "Test conditional read, matching "
                    "if_not_equal=StorageGeneration::NoValue";
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_not_equal = StorageGeneration::NoValue();
    EXPECT_THAT(kvstore::Read(store, missing_key, options).result(),
                MatchesKvsReadResultNotFound());
  }

  ABSL_LOG(INFO) << kSep << "Test conditional read, if_not_equal mismatch";
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_not_equal = mismatch_generation;
    EXPECT_THAT(kvstore::Read(store, missing_key, options).result(),
                MatchesKvsReadResultNotFound());
  }
}

void TestKeyValueStoreBatchReadOps(const KvStore& store, std::string key,
                                   absl::Cord expected_value) {
  auto correct_generation = GetStorageGeneration(store, key);
  auto mismatch_generation = GetMismatchStorageGeneration(store);

  constexpr size_t kNumIterations = 100;
  constexpr size_t kMaxReadsPerBatch = 10;

  std::minstd_rand gen{internal_testing::GetRandomSeedForTest(
      "TENSORSTORE_INTERNAL_KVSTORE_BATCH_READ")};
  for (size_t iter_i = 0; iter_i < kNumIterations; ++iter_i) {
    auto batch = tensorstore::Batch::New();

    auto reads_per_batch = absl::Uniform<size_t>(absl::IntervalClosedClosed,
                                                 gen, 1, kMaxReadsPerBatch);
    std::vector<::testing::Matcher<Result<kvstore::ReadResult>>> matchers;
    std::vector<Future<kvstore::ReadResult>> futures;
    for (size_t read_i = 0; read_i < reads_per_batch; ++read_i) {
      kvstore::ReadOptions options;
      options.batch = batch;
      options.byte_range.inclusive_min = absl::Uniform<int64_t>(
          absl::IntervalClosedClosed, gen, 0, expected_value.size());
      options.byte_range.exclusive_max = absl::Uniform<int64_t>(
          absl::IntervalClosedClosed, gen, options.byte_range.inclusive_min,
          expected_value.size());
      bool mismatch = false;
      if (absl::Bernoulli(gen, 0.5)) {
        options.generation_conditions.if_equal =
            absl::Bernoulli(gen, 0.5)
                ? correct_generation
                : ((mismatch = true), mismatch_generation);
      }
      if (absl::Bernoulli(gen, 0.5)) {
        options.generation_conditions.if_not_equal =
            absl::Bernoulli(gen, 0.5) ? ((mismatch = true), correct_generation)
                                      : mismatch_generation;
      }
      futures.push_back(kvstore::Read(store, key, options));
      if (mismatch) {
        matchers.push_back(MatchesKvsReadResultAborted());
      } else {
        matchers.push_back(MatchesKvsReadResult(
            expected_value.Subcord(options.byte_range.inclusive_min,
                                   options.byte_range.exclusive_max -
                                       options.byte_range.inclusive_min),
            correct_generation));
      }
    }

    batch.Release();

    for (size_t read_i = 0; read_i < reads_per_batch; ++read_i) {
      EXPECT_THAT(futures[read_i].result(), matchers[read_i]);
    }
  }
}

void TestKeyValueReadWriteOps(const KvStore& store) {
  return TestKeyValueReadWriteOps(
      store, [](std::string key) { return absl::StrCat("key_", key); });
}

void TestKeyValueReadWriteOps(
    const KvStore& store,
    absl::FunctionRef<std::string(std::string key)> get_key) {
  absl::Cord expected_value("_kvstore_value_");
  absl::Cord other_value("._-=+=-_.");

  // Test read operations.
  {
    std::string missing_key = get_key("missing");
    kvstore::Delete(store, missing_key).result().status().IgnoreError();

    std::string key = get_key("read");
    auto write_result = kvstore::Write(store, key, expected_value).result();
    ASSERT_THAT(write_result, MatchesRegularTimestampedStorageGeneration());

    tensorstore::internal::TestKeyValueStoreReadOps(store, key, expected_value,
                                                    missing_key);

    absl::Cord longer_expected_value;
    for (size_t i = 0; i < 4096; ++i) {
      char x = static_cast<char>(i);
      longer_expected_value.Append(std::string_view(&x, 1));
    }

    ASSERT_THAT(kvstore::Write(store, key, longer_expected_value).result(),
                MatchesRegularTimestampedStorageGeneration());

    tensorstore::internal::TestKeyValueStoreBatchReadOps(store, key,
                                                         longer_expected_value);

    kvstore::Delete(store, key).result().status().IgnoreError();
  }

  // Validate write/read of special characters.
  {
    ABSL_LOG(INFO) << kSep << "Test read/write of non-alphanumeric key.";

    std::string special_key = get_key("subdir/a!b@c$d");
    kvstore::Delete(store, special_key).result().status().IgnoreError();

    auto write_result =
        kvstore::Write(store, special_key, expected_value).result();
    ASSERT_THAT(write_result, MatchesRegularTimestampedStorageGeneration());

    auto read_result = kvstore::Read(store, special_key).result();
    EXPECT_THAT(read_result, MatchesKvsReadResult(expected_value, testing::_));

    kvstore::Delete(store, special_key).result().status().IgnoreError();
  }

  // Test write operations.
  TestKeyValueStoreWriteOps(
      store, {get_key("write1"), get_key("write2"), get_key("write3")},
      expected_value, other_value);

  TestKeyValueStoreDeleteOps(
      store,
      {get_key("del1"), get_key("del2"), get_key("del3"), get_key("del4")},
      expected_value);

  TestKeyValueStoreStalenessBoundOps(store, get_key("stale"), expected_value,
                                     other_value);
}

/// Tests List on `store`, which should be empty.
void TestKeyValueStoreList(const KvStore& store, bool match_size) {
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

  const absl::Cord value("xyz");

  ABSL_LOG(INFO) << "Test list: ... write elements ...";
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/b", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/d", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/x", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/y", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/e", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/f", value));

  ::testing::Matcher<int64_t> size_matcher = ::testing::_;
  if (match_size) {
    size_matcher = ::testing::Eq(value.size());
  }

  // Listing the entire stream works.
  ABSL_LOG(INFO) << "Test list, entire stream";
  {
    EXPECT_THAT(tensorstore::kvstore::ListFuture(store).value(),
                ::testing::UnorderedElementsAre(
                    MatchesListEntry("a/d", size_matcher),
                    MatchesListEntry("a/c/z/f", size_matcher),
                    MatchesListEntry("a/c/y", size_matcher),
                    MatchesListEntry("a/c/z/e", size_matcher),
                    MatchesListEntry("a/c/x", size_matcher),
                    MatchesListEntry("a/b", size_matcher)));
  }

  // Listing a subset of the stream works.
  ABSL_LOG(INFO) << "Test list, prefix range";
  {
    EXPECT_THAT(
        tensorstore::kvstore::ListFuture(store, {KeyRange::Prefix("a/c/")})
            .value(),
        ::testing::UnorderedElementsAre(
            MatchesListEntry("a/c/z/f", size_matcher),
            MatchesListEntry("a/c/y", size_matcher),
            MatchesListEntry("a/c/z/e", size_matcher),
            MatchesListEntry("a/c/x", size_matcher)));
  }

  // Cancellation immediately after starting yields nothing.
  ABSL_LOG(INFO) << "Test list, cancel on start";
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        CompletionNotifyingReceiver{&notification,
                                    CancelOnStartingReceiver{{&log}}});
    notification.WaitForNotification();

    ASSERT_THAT(log, ::testing::SizeIs(::testing::Ge(3)));

    EXPECT_EQ("set_starting", log[0]);
    EXPECT_EQ("set_done", log[log.size() - 2]);
    EXPECT_EQ("set_stopping", log[log.size() - 1]);
    EXPECT_THAT(span<const std::string>(&log[1], log.size() - 3),
                ::testing::IsSubsetOf({"set_value: a/d", "set_value: a/c/z/f",
                                       "set_value: a/c/y", "set_value: a/c/z/e",
                                       "set_value: a/c/x", "set_value: a/b"}));
  }

  // Cancellation in the middle of the stream may stop the stream.
  ABSL_LOG(INFO) << "Test list, cancel after 2";
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        CompletionNotifyingReceiver{&notification,
                                    CancelAfterNReceiver<2>{{&log}}});
    notification.WaitForNotification();

    ASSERT_THAT(log, ::testing::SizeIs(::testing::Gt(3)));

    EXPECT_EQ("set_starting", log[0]);
    EXPECT_EQ("set_done", log[log.size() - 2]);
    EXPECT_EQ("set_stopping", log[log.size() - 1]);
    EXPECT_THAT(span<const std::string>(&log[1], log.size() - 3),
                ::testing::IsSubsetOf({"set_value: a/d", "set_value: a/c/z/f",
                                       "set_value: a/c/y", "set_value: a/c/z/e",
                                       "set_value: a/c/x", "set_value: a/b"}));
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
  EXPECT_THAT(kvstore::ListFuture(store).result(),
              ::testing::Optional(::testing::UnorderedElementsAre(
                  MatchesListEntry("a/a"), MatchesListEntry("b/b"))));

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
  EXPECT_THAT(kvstore::ListFuture(store).result(),
              ::testing::Optional(::testing::UnorderedElementsAre(
                  MatchesListEntry("a/a"), MatchesListEntry("b/b"))));
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
              ::testing::Optional(
                  ::testing::UnorderedElementsAre(MatchesListEntry("a/a"))));
}

void TestKeyValueStoreDeleteRangeFromBeginning(const KvStore& store) {
  for (auto key : {"a/a", "a/b", "a/c/a", "a/c/b", "b/a", "b/b"}) {
    TENSORSTORE_EXPECT_OK(kvstore::Write(store, key, absl::Cord()).result());
  }
  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange("", "a/c/aa")));
  EXPECT_THAT(ListFuture(store).result(),
              ::testing::Optional(::testing::UnorderedElementsAre(
                  MatchesListEntry("a/c/b"), MatchesListEntry("b/a"),
                  MatchesListEntry("b/b"))));
}

void TestKeyValueStoreCopyRange(const KvStore& store) {
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "w/a", absl::Cord("w_a")));
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "x/a", absl::Cord("value_a")));
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "x/b", absl::Cord("value_b")));
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "z/a", absl::Cord("z_a")));
  TENSORSTORE_ASSERT_OK(kvstore::ExperimentalCopyRange(
      store.WithPathSuffix("x/"), store.WithPathSuffix("y/")));
  EXPECT_THAT(GetMap(store), ::testing::Optional(::testing::ElementsAreArray({
                                 ::testing::Pair("w/a", absl::Cord("w_a")),
                                 ::testing::Pair("x/a", absl::Cord("value_a")),
                                 ::testing::Pair("x/b", absl::Cord("value_b")),
                                 ::testing::Pair("y/a", absl::Cord("value_a")),
                                 ::testing::Pair("y/b", absl::Cord("value_b")),
                                 ::testing::Pair("z/a", absl::Cord("z_a")),
                             })));
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

  ASSERT_TRUE(options.check_write_read || !options.check_data_persists);
  ASSERT_TRUE(options.check_write_read ||
              !options.check_data_after_serialization);

  KvStore serialized_store;
  kvstore::Spec serialized_spec;

  ASSERT_TRUE(options.check_store_serialization ||
              !options.check_data_after_serialization);

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

    if (options.check_store_serialization) {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          serialized_store, serialization::SerializationRoundTrip(store));
      {
        TENSORSTORE_ASSERT_OK_AND_ASSIGN(
            auto serialized_store_spec,
            serialized_store.spec(
                kvstore::SpecRequestOptions{options.spec_request_options}));
        EXPECT_THAT(
            serialized_store_spec.ToJson(options.json_serialization_options),
            ::testing::Optional(MatchesJson(options.full_spec)));
      }
    }

    if (options.check_data_after_serialization) {
      EXPECT_THAT(
          kvstore::Read(serialized_store, options.roundtrip_key).result(),
          MatchesKvsReadResult(options.roundtrip_value));
      TENSORSTORE_ASSERT_OK(
          kvstore::Delete(serialized_store, options.roundtrip_key).result());
      EXPECT_THAT(kvstore::Read(store, options.roundtrip_key).result(),
                  MatchesKvsReadResultNotFound());
      EXPECT_THAT(
          kvstore::Read(serialized_store, options.roundtrip_key).result(),
          MatchesKvsReadResultNotFound());
      ASSERT_THAT(kvstore::Write(serialized_store, options.roundtrip_key,
                                 options.roundtrip_value)
                      .result(),
                  MatchesRegularTimestampedStorageGeneration());
      EXPECT_THAT(kvstore::Read(store, options.roundtrip_key).result(),
                  MatchesKvsReadResult(options.roundtrip_value));
      EXPECT_THAT(
          kvstore::Read(serialized_store, options.roundtrip_key).result(),
          MatchesKvsReadResult(options.roundtrip_value));
    }

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto spec,
        store.spec(kvstore::SpecRequestOptions{options.spec_request_options}));
    EXPECT_THAT(spec.ToJson(options.json_serialization_options),
                ::testing::Optional(MatchesJson(options.full_spec)));

    // Test serialization of spec.
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        serialized_spec, serialization::SerializationRoundTrip(spec));

    EXPECT_THAT(serialized_spec.ToJson(options.json_serialization_options),
                ::testing::Optional(MatchesJson(options.full_spec)));

    auto minimal_spec_obj = spec;
    TENSORSTORE_ASSERT_OK(minimal_spec_obj.Set(tensorstore::MinimalSpec{true}));
    EXPECT_THAT(minimal_spec_obj.ToJson(options.json_serialization_options),
                ::testing::Optional(MatchesJson(expected_minimal_spec)));

    // Check base
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_base, store.base());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_base, spec.base());
    EXPECT_EQ(store_base.valid(), spec_base.valid());
    if (store_base.valid()) {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_base_spec, store_base.spec());
      EXPECT_THAT(spec_base.ToJson(),
                  ::testing::Optional(MatchesJson(options.full_base_spec)));
      EXPECT_THAT(store_base_spec.ToJson(),
                  ::testing::Optional(MatchesJson(options.full_base_spec)));

      // Check that base spec can be opened.
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto store_base_reopened, kvstore::Open(spec_base, context).result());
      EXPECT_EQ(store_base_reopened, store_base);
    } else {
      EXPECT_THAT(options.full_base_spec,
                  MatchesJson(::nlohmann::json::value_t::discarded));
    }
  }

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

    // Reopen with serialized spec.
    {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto store, kvstore::Open(serialized_spec, context).result());
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
  TENSORSTORE_ASSIGN_OR_RETURN(auto entries, ListFuture(store).result());
  std::map<kvstore::Key, kvstore::Value> result;
  for (const auto& entry : entries) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto read_result,
                                 kvstore::Read(store, entry.key).result());
    assert(!read_result.aborted());
    assert(!read_result.not_found());
    result.emplace(entry.key, std::move(read_result.value));
  }
  return result;
}

namespace {
std::vector<::nlohmann::json> CollectMatchingMetricsAsJson(
    std::string_view metric_prefix, bool include_zero_metrics = true) {
  std::vector<::nlohmann::json> lines;

  auto collected_metrics =
      internal_metrics::GetMetricRegistry().CollectWithPrefix(metric_prefix);
  std::sort(collected_metrics.begin(), collected_metrics.end(),
            [](const auto& a, const auto& b) {
              return a.metric_name < b.metric_name;
            });

  for (const auto& collected_metric : collected_metrics) {
    if (include_zero_metrics ||
        internal_metrics::IsCollectedMetricNonZero(collected_metric)) {
      lines.push_back(
          internal_metrics::CollectedMetricToJson(collected_metric));
    }
  }

  return lines;
}

struct BatchReadExample {
  std::string key;
  OptionalByteRangeRequest byte_range;
  StorageGeneration if_equal;
  StorageGeneration if_not_equal;
};

absl::Status ExecuteReadBatch(const KvStore& kvs,
                              span<const BatchReadExample> requests,
                              bool use_batch) {
  Batch batch{no_batch};
  if (use_batch) {
    batch = Batch::New();
  }

  std::vector<Future<kvstore::ReadResult>> futures;
  for (const auto& request : requests) {
    kvstore::ReadOptions options;
    options.batch = batch;
    options.byte_range = request.byte_range;
    options.generation_conditions.if_equal = request.if_equal;
    options.generation_conditions.if_not_equal = request.if_not_equal;
    futures.push_back(kvstore::Read(kvs, request.key, std::move(options)));
  }

  batch.Release();

  for (const auto& future : futures) {
    TENSORSTORE_RETURN_IF_ERROR(future.status());
  }

  return absl::OkStatus();
}

::nlohmann::json ExpectedCounterMetric(std::string name, int64_t value) {
  return {{"name", name}, {"values", {{{"value", value}}}}};
}

std::vector<::nlohmann::json> ExpectedCounterMetrics(
    std::string_view metric_prefix,
    std::vector<std::pair<std::string, int64_t>> counters) {
  std::vector<::nlohmann::json> metrics;
  metrics.reserve(counters.size());
  for (const auto& [name, value] : counters) {
    metrics.push_back(
        ExpectedCounterMetric(absl::StrCat(metric_prefix, name), value));
  }
  return metrics;
}

void TestBatchRead(const KvStore& kvs,
                   const std::vector<BatchReadExample>& requests,
                   std::string_view metric_prefix,
                   const std::vector<std::pair<std::string, int64_t>>&
                       expected_counters_for_non_batch_read,
                   const std::vector<std::pair<std::string, int64_t>>&
                       expected_counters_for_batch_read) {
  internal_metrics::GetMetricRegistry().Reset();
  TENSORSTORE_ASSERT_OK(ExecuteReadBatch(kvs, requests, /*use_batch=*/false));
  EXPECT_THAT(CollectMatchingMetricsAsJson(metric_prefix),
              ::testing::IsSupersetOf(ExpectedCounterMetrics(
                  metric_prefix, expected_counters_for_non_batch_read)));

  internal_metrics::GetMetricRegistry().Reset();
  TENSORSTORE_ASSERT_OK(ExecuteReadBatch(kvs, requests, /*use_batch=*/true));
  EXPECT_THAT(CollectMatchingMetricsAsJson(metric_prefix),
              ::testing::IsSupersetOf(ExpectedCounterMetrics(
                  metric_prefix, expected_counters_for_batch_read)));
}

void TestBatchRead(
    const KvStore& kvs, const std::vector<BatchReadExample>& requests,
    std::string_view metric_prefix,
    const std::vector<std::pair<std::string, int64_t>>& expected_counters) {
  return TestBatchRead(kvs, requests, metric_prefix, expected_counters,
                       expected_counters);
}

}  // namespace

void TestBatchReadGenericCoalescing(
    const KvStore& store,
    const BatchReadGenericCoalescingTestOptions& options) {
  const auto& coalescing_options = options.coalescing_options;

  const bool has_target_coalesced_size =
      coalescing_options.target_coalesced_size !=
      std::numeric_limits<int64_t>::max();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto x_stamp,
      kvstore::Write(
          store, "x",
          absl::Cord(std::string(
              std::max(static_cast<int64_t>(8192),
                       (has_target_coalesced_size
                            ? coalescing_options.target_coalesced_size
                            : coalescing_options.max_extra_read_bytes) +
                           1),
              '\0')))
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_stamp,
      kvstore::Write(
          store, "y",
          absl::Cord(std::string(
              2 * (1 + coalescing_options.max_extra_read_bytes), '\0')))
          .result());

  const auto get_metrics =
      [&](std::vector<std::pair<std::string, int64_t>> common_metrics,
          std::vector<std::pair<std::string, int64_t>> open_file_metrics) {
        if (options.has_file_open_metric) {
          common_metrics.insert(common_metrics.end(), open_file_metrics.begin(),
                                open_file_metrics.end());
        }
        return common_metrics;
      };

  {
    SCOPED_TRACE("Single key, single read");
    TestBatchRead(store,
                  {
                      {"x", OptionalByteRangeRequest::Range(1, 100)},
                  },
                  options.metric_prefix,
                  get_metrics(
                      {
                          {"batch_read", 1},
                          {"read", 1},
                          {"bytes_read", 99},
                      },
                      {
                          {"open_read", 1},
                      }));
  }

  {
    SCOPED_TRACE("Two keys, single read each");
    TestBatchRead(store,
                  {
                      {"x", OptionalByteRangeRequest::Range(1, 100)},
                      {"y", OptionalByteRangeRequest::Range(100, 200)},
                  },
                  options.metric_prefix,
                  get_metrics(
                      {
                          {"batch_read", 2},
                          {"read", 2},
                          {"bytes_read", 199},
                      },
                      {
                          {"open_read", 2},
                      }));
  }

  {
    SCOPED_TRACE("Single key, two reads that are coalesced with no gap");
    TestBatchRead(store,
                  {
                      {"x", OptionalByteRangeRequest::Range(1, 100)},
                      {"x", OptionalByteRangeRequest::Range(100, 200)},
                  },
                  options.metric_prefix,
                  get_metrics(
                      {
                          {"batch_read", 2},

                          {"read", 2},
                          {"bytes_read", 199},
                      },
                      {
                          {"open_read", 2},
                      }),
                  get_metrics(
                      {
                          {"batch_read", 1},

                          {"read", 2},
                          {"bytes_read", 199},
                      },
                      {
                          {"open_read", 1},
                      }));
  }

  {
    SCOPED_TRACE(absl::StrFormat(
        "Single key, two reads that are coalesced with gap of %d bytes",
        coalescing_options.max_extra_read_bytes));
    TestBatchRead(
        store,
        {
            {"x", OptionalByteRangeRequest::Range(1, 100)},
            {"x", OptionalByteRangeRequest::Range(
                      100 + coalescing_options.max_extra_read_bytes,
                      100 + coalescing_options.max_extra_read_bytes + 100)},
        },
        options.metric_prefix,
        get_metrics(
            {
                {"batch_read", 2},

                {"read", 2},
                {"bytes_read", 99 + 100},
            },
            {
                {"open_read", 2},
            }),
        get_metrics(
            {
                {"batch_read", 1},

                {"read", 2},
                {"bytes_read",
                 100 + coalescing_options.max_extra_read_bytes + 100 - 1},
            },
            {
                {"open_read", 1},
            }));
  }

  {
    SCOPED_TRACE(absl::StrFormat(
        "Single key, two reads that are not coalesced with gap of %d "
        "bytes",
        coalescing_options.max_extra_read_bytes + 1));
    TestBatchRead(
        store,
        {
            {"x", OptionalByteRangeRequest::Range(1, 100)},
            {"x", OptionalByteRangeRequest::Range(
                      100 + coalescing_options.max_extra_read_bytes + 1,
                      100 + coalescing_options.max_extra_read_bytes + 1 + 100)},
        },
        options.metric_prefix,
        get_metrics(
            {
                {"batch_read", 2},
                {"read", 2},
                {"bytes_read", 99 + 100},
            },
            {
                {"open_read", 2},
            }),
        get_metrics(
            {
                {"batch_read", 2},
                {"read", 2},
                {"bytes_read", 99 + 100},
            },
            {
                {"open_read", 1},
            }));
  }

  if (coalescing_options.target_coalesced_size !=
      std::numeric_limits<int64_t>::max()) {
    SCOPED_TRACE(
        "Single key, two reads that are not coalesced due to size limit");
    TestBatchRead(
        store,
        {
            {"x", OptionalByteRangeRequest::Range(
                      0, coalescing_options.target_coalesced_size)},
            {"x", OptionalByteRangeRequest::Range(
                      coalescing_options.target_coalesced_size,
                      coalescing_options.target_coalesced_size + 1)},
        },
        options.metric_prefix,
        get_metrics(
            {
                {"batch_read", 2},
                {"read", 2},
                {"bytes_read", coalescing_options.target_coalesced_size + 1},
            },
            {
                {"open_read", 2},
            }),
        get_metrics(
            {
                {"batch_read", 2},
                {"read", 2},
                {"bytes_read", coalescing_options.target_coalesced_size + 1},
            },
            {
                {"open_read", 1},
            }));
  }
}

}  // namespace internal
}  // namespace tensorstore
