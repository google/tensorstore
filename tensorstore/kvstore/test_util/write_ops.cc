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

#include "tensorstore/kvstore/test_util/write_ops.h"

#include <stddef.h>

#include <array>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/thread/thread.h"
#include "tensorstore/internal/thread/thread_pool.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util/internal.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

using ::testing::HasSubstr;

namespace tensorstore {
namespace internal {
namespace {

static const char kSep[] = "----------------------------------------------\n";

}  // namespace

void TestKeyValueStoreWriteOps(const KvStore& store,
                               std::array<std::string, 3> key,
                               absl::Cord expected_value,
                               absl::Cord other_value) {
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

void TestKeyValueStoreTransactionalWriteOps(const KvStore& store,
                                            TransactionMode transaction_mode,
                                            std::string key,
                                            absl::Cord expected_value,
                                            std::string_view operation) {
  Cleanup cleanup(store, {key});
  Transaction txn(transaction_mode);
  auto txn_store = (store | txn).value();
  kvstore::WriteOptions options;
  bool success;
  if (operation == "Unconditional") {
    success = true;
  } else if (operation == "MatchingCondition") {
    options.generation_conditions.if_equal = StorageGeneration::NoValue();
    success = true;
  } else if (operation == "MatchingConditionAfterWrite") {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto stamp, kvstore::Write(txn_store, key, absl::Cord()).result());
    options.generation_conditions.if_equal = stamp.generation;
    success = true;
  } else if (operation == "NonMatchingCondition") {
    options.generation_conditions.if_equal =
        GetMismatchStorageGeneration(store);
    success = false;
  } else if (operation == "NonMatchingConditionAfterWrite") {
    TENSORSTORE_ASSERT_OK(
        kvstore::Write(txn_store, key, absl::Cord()).result());
    options.generation_conditions.if_equal =
        GetMismatchStorageGeneration(store);
    success = false;
  } else {
    ABSL_LOG(FATAL) << "Unepxected operation: " << operation;
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto stamp_within_txn,
      kvstore::Write(txn_store, key, expected_value, std::move(options))
          .result());
  EXPECT_THAT(
      kvstore::Read(txn_store, key).result(),
      MatchesKvsReadResult(expected_value, stamp_within_txn.generation));
  if (success) {
    TENSORSTORE_ASSERT_OK(txn.Commit());
    EXPECT_THAT(
        kvstore::Read(store, key).result(),
        MatchesKvsReadResult(
            expected_value,
            ::testing::Not(::testing::Eq(stamp_within_txn.generation))));
  } else {
    EXPECT_THAT(txn.Commit(), StatusIs(absl::StatusCode::kAborted,
                                       HasSubstr("Generation mismatch")));
    EXPECT_THAT(kvstore::Read(store, key).result(),
                MatchesKvsReadResultNotFound());
  }
}

void TestConcurrentWrites(const TestConcurrentWritesOptions& options) {
  std::vector<tensorstore::internal::Thread> threads;
  threads.reserve(options.num_threads);
  StorageGeneration initial_generation;
  std::string initial_value;
  initial_value.resize(sizeof(size_t) * options.num_threads);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto initial_stamp, kvstore::Write(options.get_store(), options.key,
                                           absl::Cord(initial_value))
                                .result());
    initial_generation = initial_stamp.generation;
  }
  for (size_t thread_i = 0; thread_i < options.num_threads; ++thread_i) {
    threads.emplace_back(
        tensorstore::internal::Thread({"concurrent_write"}, [&, thread_i] {
          auto store = options.get_store();
          StorageGeneration generation = initial_generation;
          std::string value = initial_value;
          for (size_t i = 0; i < options.num_iterations; ++i) {
            const size_t value_offset = sizeof(size_t) * thread_i;
            while (true) {
              size_t x;
              std::memcpy(&x, &value[value_offset], sizeof(size_t));
              ABSL_CHECK_EQ(i, x);
              std::string new_value = value;
              x = i + 1;
              std::memcpy(&new_value[value_offset], &x, sizeof(size_t));
              TENSORSTORE_CHECK_OK_AND_ASSIGN(
                  auto write_result,
                  kvstore::Write(store, options.key, absl::Cord(new_value),
                                 {generation})
                      .result());
              if (!StorageGeneration::IsUnknown(write_result.generation)) {
                generation = write_result.generation;
                value = new_value;
                break;
              }
              TENSORSTORE_CHECK_OK_AND_ASSIGN(
                  auto read_result, kvstore::Read(store, options.key).result());
              ABSL_CHECK(!read_result.aborted());
              ABSL_CHECK(!read_result.not_found());
              value = std::string(read_result.value);
              ABSL_CHECK_EQ(sizeof(size_t) * options.num_threads, value.size());
              generation = read_result.stamp.generation;
            }
          }
        }));
  }
  for (auto& t : threads) t.Join();
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto read_result,
        kvstore::Read(options.get_store(), options.key).result());
    ASSERT_FALSE(read_result.aborted() || read_result.not_found());
    std::string expected_value;
    expected_value.resize(sizeof(size_t) * options.num_threads);
    {
      std::vector<size_t> expected_nums(options.num_threads,
                                        options.num_iterations);
      std::memcpy(const_cast<char*>(expected_value.data()),
                  expected_nums.data(), expected_value.size());
    }
    EXPECT_EQ(expected_value, read_result.value);
  }
}

}  // namespace internal
}  // namespace tensorstore
