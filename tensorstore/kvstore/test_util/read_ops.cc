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

#include "tensorstore/kvstore/test_util/read_ops.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "riegeli/base/byte_fill.h"
#include "tensorstore/batch.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/internal/testing/random_seed.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util/internal.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace tensorstore {
namespace internal {
namespace {

static const char kSep[] = "----------------------------------------------\n";

}  // namespace

void TestKeyValueStoreReadOps(const KvStore& store, std::string key,
                              absl::Cord expected_value,
                              std::string missing_key) {
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
                testing::AnyOf(StatusIs(absl::StatusCode::kOutOfRange)));
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
                StatusIs(absl::StatusCode::kOutOfRange));
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

  // if_equal tests
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
    // FIXME(jbms): Potentially unconditional writes should not produce
    // `absl::InfiniteFuture()`, because a subsequent write could result in a
    // different value. While this is not a problem for users of
    // `ReadModifyWrite` it does result in behavior inconsistent with
    // non-transactional reads and writes.
    if (read_result->stamp.time != absl::InfiniteFuture()) {
      EXPECT_THAT(result->stamp.time, testing::Gt(read_result->stamp.time));
    }
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
    EXPECT_THAT(kvstore::Read(store, missing_key, options).result(),
                testing::AnyOf(MatchesKvsReadResultNotFound(),  // Common result
                               MatchesKvsReadResultAborted()));  // GCS result
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

void TestKeyValueStoreStalenessBoundOps(const KvStore& store, std::string key,
                                        absl::Cord value1, absl::Cord value2) {
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

  // Generally same-host writes should invalidate staleness_bound in a
  // kvstore.
  auto write_result2 = kvstore::Write(store, key, value2).result();
  ASSERT_THAT(write_result2, MatchesRegularTimestampedStorageGeneration());

  // However allow either version to satisfy this test.
  ABSL_LOG(INFO) << kSep << "Test staleness_bound read: value2";
  EXPECT_THAT(kvstore::Read(store, key, read_options).result(),
              ::testing::AnyOf(
                  MatchesKvsReadResult(value1, write_result1->generation),
                  MatchesKvsReadResult(value2, write_result2->generation)));
}

namespace {

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

template <typename Matcher1, typename Matcher2>
void TestBatchRead(const KvStore& kvs,
                   const std::vector<BatchReadExample>& requests,
                   std::string_view metric_prefix,
                   Matcher1 non_batch_read_matcher,
                   Matcher2 batch_read_matcher) {
  auto before_metrics =
      internal_metrics::GetMetricRegistry().CollectWithPrefix(metric_prefix);

  TENSORSTORE_ASSERT_OK(ExecuteReadBatch(kvs, requests, /*use_batch=*/false));

  auto non_batch_read_metrics =
      internal_metrics::GetMetricRegistry().CollectWithPrefix(metric_prefix);
  auto non_batch_read_delta = internal_metrics::CollectedMetricsDelta(
      before_metrics, non_batch_read_metrics);

  TENSORSTORE_ASSERT_OK(ExecuteReadBatch(kvs, requests, /*use_batch=*/true));

  auto batch_read_metrics =
      internal_metrics::GetMetricRegistry().CollectWithPrefix(metric_prefix);
  auto batch_read_delta = internal_metrics::CollectedMetricsDelta(
      non_batch_read_metrics, batch_read_metrics);

  EXPECT_THAT(CollectedMetricsToJson(non_batch_read_delta),
              non_batch_read_matcher);

  EXPECT_THAT(CollectedMetricsToJson(batch_read_delta), batch_read_matcher);
}

template <typename Matcher>
void TestBatchRead(const KvStore& kvs,
                   const std::vector<BatchReadExample>& requests,
                   std::string_view metric_prefix, Matcher expected_counters) {
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
          absl::Cord(riegeli::ByteFill(std::max(
              int64_t{8192}, (has_target_coalesced_size
                                  ? coalescing_options.target_coalesced_size
                                  : coalescing_options.max_extra_read_bytes) +
                                 1))))
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_stamp,
      kvstore::Write(store, "y",
                     absl::Cord(riegeli::ByteFill(
                         2 * (1 + coalescing_options.max_extra_read_bytes))))
          .result());

  const auto format_counter_metric = [&](std::string name,
                                         int64_t value) -> ::nlohmann::json {
    return {{"name", absl::StrCat(options.metric_prefix, name)},
            {"values", {{{"value", value}}}}};
  };

  const auto get_metrics =
      [&](std::vector<std::pair<std::string, int64_t>> common_metrics,
          std::vector<std::pair<std::string, int64_t>> open_file_metrics) {
        std::vector<::nlohmann::json> metrics;
        metrics.reserve(common_metrics.size() + open_file_metrics.size());
        for (const auto& [name, value] : common_metrics) {
          metrics.push_back(format_counter_metric(name, value));
        }
        if (options.has_file_open_metric) {
          for (const auto& [name, value] : open_file_metrics) {
            metrics.push_back(format_counter_metric(name, value));
          }
        }
        return ::testing::IsSupersetOf(metrics);
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

void TestKeyValueStoreTransactionalReadOps(
    const TransactionalReadOpsParameters& p) {
  Cleanup cleanup(p.store, {p.key});
  StorageGeneration expected_generation;
  std::optional<absl::Cord> expected_value;
  auto mismatch_generation = GetMismatchStorageGeneration(p.store);
  if (p.write_outside_transaction) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto stamp, kvstore::Write(p.store, p.key, p.value1).result());
    expected_generation = stamp.generation;
    expected_value = p.value1;
  } else {
    expected_generation = StorageGeneration::NoValue();
  }

  if (p.write_to_other_node) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto stamp,
                                     p.write_to_other_node(p.key, p.value3));
    expected_value = p.value3;
    expected_generation = stamp.generation;
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto txn_store, p.store | tensorstore::Transaction(p.transaction_mode));

  if (p.write_operation_within_transaction == "Unmodified") {
    // Do nothing
  } else if (p.write_operation_within_transaction == "DeleteRange") {
    TENSORSTORE_ASSERT_OK(
        kvstore::DeleteRange(txn_store, KeyRange::Singleton(p.key)));
    expected_value = std::nullopt;
    expected_generation = StorageGeneration::Dirty(
        StorageGeneration::Unknown(), StorageGeneration::kDeletionMutationId);
  } else if (p.write_operation_within_transaction == "Delete") {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto stamp, kvstore::Delete(txn_store, p.key).result());
    expected_value = std::nullopt;
    expected_generation = stamp.generation;
  } else if (p.write_operation_within_transaction == "WriteUnconditionally") {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto stamp, kvstore::Write(txn_store, p.key, p.value2).result());
    expected_value = p.value2;
    expected_generation = stamp.generation;
  } else if (p.write_operation_within_transaction ==
             "WriteWithFalseCondition") {
    kvstore::WriteOptions options;
    options.generation_conditions.if_equal = mismatch_generation;
    auto future =
        kvstore::WriteCommitted(txn_store, p.key, p.value2, std::move(options));
    static_cast<void>(future);
  } else if (p.write_operation_within_transaction == "WriteWithTrueCondition") {
    kvstore::WriteOptions options;
    options.generation_conditions.if_equal = expected_generation;
    auto future =
        kvstore::WriteCommitted(txn_store, p.key, p.value2, std::move(options));
    static_cast<void>(future);
    expected_value = p.value2;
    expected_generation = StorageGeneration::Unknown();
  } else {
    ABSL_LOG(FATAL) << "Invalid write_operation_within_transaction: "
                    << p.write_operation_within_transaction;
  }

  // Read full value unconditionally
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result,
                                     kvstore::Read(txn_store, p.key).result());
    EXPECT_EQ(expected_value ? kvstore::ReadResult::kValue
                             : kvstore::ReadResult::kMissing,
              read_result.state);
    EXPECT_EQ(expected_value, read_result.optional_value());
    if (!StorageGeneration::IsUnknown(expected_generation)) {
      EXPECT_EQ(expected_generation, read_result.stamp.generation);
    }
    expected_generation = read_result.stamp.generation;
  }

  // Read with if_equal (true)
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_equal = expected_generation;
    EXPECT_THAT(kvstore::Read(txn_store, p.key, std::move(options)).result(),
                MatchesKvsReadResult(expected_value, expected_generation));
  }

  // Read with if_equal (false)
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_equal = mismatch_generation;
    if (expected_value.has_value()) {
      EXPECT_THAT(kvstore::Read(txn_store, p.key, std::move(options)).result(),
                  MatchesKvsReadResult(kvstore::ReadResult::kUnspecified));
    } else {
      EXPECT_THAT(kvstore::Read(txn_store, p.key, std::move(options)).result(),
                  ::testing::AnyOf(
                      MatchesKvsReadResult(kvstore::ReadResult::kUnspecified),
                      MatchesKvsReadResultNotFound()));
    }
  }

  // Read with if_not_equal (true)
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_not_equal = mismatch_generation;
    EXPECT_THAT(kvstore::Read(txn_store, p.key, std::move(options)).result(),
                MatchesKvsReadResult(expected_value, expected_generation));
  }

  // Read with if_not_equal (false)
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_not_equal = expected_generation;
    if (expected_value.has_value()) {
      EXPECT_THAT(kvstore::Read(txn_store, p.key, std::move(options)).result(),
                  MatchesKvsReadResult(kvstore::ReadResult::kUnspecified,
                                       expected_generation));
    } else {
      EXPECT_THAT(kvstore::Read(txn_store, p.key, std::move(options)).result(),
                  ::testing::AnyOf(
                      MatchesKvsReadResult(kvstore::ReadResult::kUnspecified,
                                           expected_generation),
                      MatchesKvsReadResultNotFound()));
    }
  }

  // Read partial byte range
  {
    kvstore::ReadOptions options;
    options.byte_range = OptionalByteRangeRequest::Range(1, 3);
    if (expected_value.has_value()) {
      EXPECT_THAT(kvstore::Read(txn_store, p.key, std::move(options)).result(),
                  MatchesKvsReadResult(expected_value->Subcord(1, 2),
                                       expected_generation));
    } else {
      EXPECT_THAT(kvstore::Read(txn_store, p.key, std::move(options)).result(),
                  MatchesKvsReadResultNotFound());
    }
  }

  // Read stat byte range
  {
    kvstore::ReadOptions options;
    options.byte_range = OptionalByteRangeRequest::Stat();
    if (expected_value.has_value()) {
      EXPECT_THAT(kvstore::Read(txn_store, p.key, std::move(options)).result(),
                  MatchesKvsReadResult(absl::Cord(), expected_generation));
    } else {
      EXPECT_THAT(kvstore::Read(txn_store, p.key, std::move(options)).result(),
                  MatchesKvsReadResultNotFound());
    }
  }
}

}  // namespace internal
}  // namespace tensorstore
