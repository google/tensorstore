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

#include <stddef.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorstore/batch.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

/// WARNING: This can modify live data!
///
/// This is a test-only binary which runs a set of io tests against a
/// kvstore driver with the intent of validating batch read requests.
///
/// WARNING: This can modify live data!

/* Examples

bazel run //tensorstore/kvstore:live_kvstore_test -- \
    --kvstore_spec='"file:///tmp/tensorstore_kvstore_test"'

bazel run //tensorstore/kvstore:live_kvstore_test -- \
    --kvstore_spec='{"driver":"ocdbt","base":"file:///tmp/tensorstore_kvstore_test"}'

*/

tensorstore::kvstore::Spec DefaultKvStoreSpec() {
  return tensorstore::kvstore::Spec::FromJson({{"driver", "memory"}}).value();
}

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec>, kvstore_spec,
          DefaultKvStoreSpec(),
          "KvStore spec for reading data.  See examples at the start of the "
          "source file.");

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Context::Spec>, context_spec,
          {},
          "Context spec for writing data.  This can be used to control the "
          "number of concurrent write operations of the underlying key-value "
          "store.");

ABSL_FLAG(std::string, metric_prefix, "/tensorstore",
          "Prefix for metrics to collect.");

namespace kvstore = ::tensorstore::kvstore;

using ::tensorstore::Context;
using ::tensorstore::Future;
using ::tensorstore::IsOk;
using ::tensorstore::OptionalByteRangeRequest;
using ::tensorstore::StatusIs;
using ::tensorstore::StorageGeneration;

namespace {

Context GetContext() {
  static Context* context =
      new Context(absl::GetFlag(FLAGS_context_spec).value);
  return *context;
}

tensorstore::kvstore::Spec GetSpec() {
  auto kvstore_spec = absl::GetFlag(FLAGS_kvstore_spec).value;
  if (!kvstore_spec.path.empty() && kvstore_spec.path.back() != '/') {
    kvstore_spec.AppendSuffix("/");
  }
  return kvstore_spec;
}

std::string GetTestData() {
  std::string data_1k(1024, 0);
  for (size_t i = 0; i < 1024; ++i) {
    data_1k[i] = '0' + (i % 10);
  }
  return data_1k;
}

tensorstore::kvstore::KvStore OpenStore() {
  static auto kvstore = []() {
    auto store = kvstore::Open(GetSpec(), GetContext()).value();
    TENSORSTORE_CHECK_OK(
        tensorstore::kvstore::Write(store, "abc", absl::Cord(GetTestData())));
    return store;
  }();
  return kvstore;
}

TEST(LiveReadTest, Read) {
  auto store = OpenStore();
  auto expected_value = absl::Cord(GetTestData());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result,
                                   kvstore::Read(store, "abc").result());

  // Individual result field verification.
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value, expected_value);
  EXPECT_TRUE(result.stamp.generation.IsValid());
}

TEST(LiveReadTest, PartialRead) {
  auto store = OpenStore();

  kvstore::ReadOptions options;
  options.generation_conditions.if_not_equal = StorageGeneration::FromUint64(3);
  options.staleness_bound = absl::InfiniteFuture();
  options.byte_range = OptionalByteRangeRequest{1, 10};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Read(store, "abc", options).result());

  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value, "123456789");
  EXPECT_TRUE(result.stamp.generation.IsValid());
}

TEST(LiveReadTest, BatchRead_NonAdjacent) {
  auto store = OpenStore();

  Future<kvstore::ReadResult> read1;
  Future<kvstore::ReadResult> read2;
  {
    auto batch = tensorstore::Batch::New();

    kvstore::ReadOptions options;
    options.batch = batch;

    options.byte_range.inclusive_min = 0;
    options.byte_range.exclusive_max = 10;
    read1 = kvstore::Read(store, "abc", options);

    options.byte_range.inclusive_min = 1010;
    options.byte_range.exclusive_max = 1020;
    read2 = kvstore::Read(store, "abc", options);
    batch.Release();
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result1, read1.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result2, read2.result());

  // Individual result field verification.
  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value, "0123456789");
  EXPECT_TRUE(result1.stamp.generation.IsValid());

  EXPECT_TRUE(result2.has_value());
  EXPECT_EQ(result2.value, "0123456789");
  EXPECT_TRUE(result2.stamp.generation.IsValid());
}

TEST(LiveReadTest, BatchRead_Adjacent) {
  auto store = OpenStore();

  Future<kvstore::ReadResult> read1;
  Future<kvstore::ReadResult> read2;
  {
    auto batch = tensorstore::Batch::New();

    kvstore::ReadOptions options;
    options.batch = batch;
    options.byte_range.inclusive_min = 0;
    options.byte_range.exclusive_max = 10;
    read1 = kvstore::Read(store, "abc", options);

    options.byte_range.inclusive_min = 11;
    options.byte_range.exclusive_max = 20;
    read2 = kvstore::Read(store, "abc", options);

    batch.Release();
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result1, read1.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result2, read2.result());

  // Individual result field verification.
  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value, "0123456789");
  EXPECT_TRUE(result1.stamp.generation.IsValid());

  EXPECT_TRUE(result2.has_value());
  EXPECT_EQ(result2.value, "123456789");
  EXPECT_TRUE(result2.stamp.generation.IsValid());
}

TEST(LiveReadTest, BatchRead_Overlapping) {
  auto store = OpenStore();

  Future<kvstore::ReadResult> read1;
  Future<kvstore::ReadResult> read2;
  {
    auto batch = tensorstore::Batch::New();

    kvstore::ReadOptions options;
    options.batch = batch;
    options.byte_range.inclusive_min = 0;
    options.byte_range.exclusive_max = 10;
    read1 = kvstore::Read(store, "abc", options);

    options.byte_range.inclusive_min = 5;
    options.byte_range.exclusive_max = 15;
    read2 = kvstore::Read(store, "abc", options);

    batch.Release();
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result1, read1.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result2, read2.result());

  // Individual result field verification.
  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value, "0123456789");
  EXPECT_TRUE(result1.stamp.generation.IsValid());

  EXPECT_TRUE(result2.has_value());
  EXPECT_EQ(result2.value, "5678901234");
  EXPECT_TRUE(result2.stamp.generation.IsValid());
}

TEST(LiveReadTest, BatchRead_UnboundedOverlapping) {
  auto store = OpenStore();

  Future<kvstore::ReadResult> read1;
  Future<kvstore::ReadResult> read2;
  {
    auto batch = tensorstore::Batch::New();

    kvstore::ReadOptions options;
    options.batch = batch;
    options.byte_range.inclusive_min = 5;
    read1 = kvstore::Read(store, "abc", options);

    options.byte_range.inclusive_min = 0;
    options.byte_range.exclusive_max = 10;
    read2 = kvstore::Read(store, "abc", options);
    batch.Release();
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result1, read1.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result2, read2.result());

  // Individual result field verification.
  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value, GetTestData().substr(5));
  EXPECT_TRUE(result1.stamp.generation.IsValid());

  EXPECT_TRUE(result2.has_value());
  EXPECT_EQ(result2.value, "0123456789");
  EXPECT_TRUE(result2.stamp.generation.IsValid());
}

TEST(LiveReadTest, BatchRead_Suffix) {
  auto store = OpenStore();

  Future<kvstore::ReadResult> read1;
  Future<kvstore::ReadResult> read2;
  {
    auto batch = tensorstore::Batch::New();

    kvstore::ReadOptions options;
    options.batch = batch;

    options.byte_range.inclusive_min = -10;
    read1 = kvstore::Read(store, "abc", options);

    options.byte_range.inclusive_min = -5;
    read2 = kvstore::Read(store, "abc", options);
    batch.Release();
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result1, read1.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result2, read2.result());

  // Individual result field verification.
  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value, "4567890123");
  EXPECT_TRUE(result1.stamp.generation.IsValid());

  EXPECT_TRUE(result2.has_value());
  EXPECT_EQ(result2.value, "90123");
  EXPECT_TRUE(result2.stamp.generation.IsValid());
}

TEST(LiveReadTest, BatchRead_Overlapping_TooLarge) {
  auto store = OpenStore();

  Future<kvstore::ReadResult> read1;
  Future<kvstore::ReadResult> read2;
  {
    auto batch = tensorstore::Batch::New();

    kvstore::ReadOptions options;
    options.batch = batch;
    options.byte_range.inclusive_min = 0;
    options.byte_range.exclusive_max = 10;
    read1 = kvstore::Read(store, "abc", options);

    options.byte_range.inclusive_min = 1;
    options.byte_range.exclusive_max = 2000;
    read2 = kvstore::Read(store, "abc", options);

    batch.Release();
  }

  // This test reveals a bug in the current gcs_http driver. It issues a request
  // with the header 'range: bytes=0-1999'. The gcs response is a code=206 with
  // the entire response, however read1 still fails.

  // Individual result field verification.
  EXPECT_THAT(read1, IsOk());
  if (read1.result().ok()) {
    EXPECT_TRUE(read1.result()->has_value());
    EXPECT_EQ(read1.result()->value, "0123456789");
    EXPECT_TRUE(read1.result()->stamp.generation.IsValid());
  }

  EXPECT_THAT(read2, StatusIs(absl::StatusCode::kOutOfRange));
}

TEST(LiveReadTest, BatchRead_Suffix_TooLarge) {
  auto store = OpenStore();

  Future<kvstore::ReadResult> read1;
  Future<kvstore::ReadResult> read2;
  {
    auto batch = tensorstore::Batch::New();

    kvstore::ReadOptions options;
    options.batch = batch;

    options.byte_range.inclusive_min = -10;
    read1 = kvstore::Read(store, "abc", options);

    options.byte_range.inclusive_min = -2000;
    read2 = kvstore::Read(store, "abc", options);
    batch.Release();
  }

  // Individual result field verification.
  EXPECT_THAT(read1.result(), IsOk());
  if (read1.result().ok()) {
    EXPECT_TRUE(read1.result()->has_value());
    EXPECT_EQ(read1.result()->value, "4567890123");
    EXPECT_TRUE(read1.result()->stamp.generation.IsValid());
  }

  EXPECT_THAT(read2.result(), StatusIs(absl::StatusCode::kOutOfRange));
}

void DumpAllMetrics() {
  std::vector<std::string> lines;
  for (const auto& metric :
       tensorstore::internal_metrics::GetMetricRegistry().CollectWithPrefix(
           absl::GetFlag(FLAGS_metric_prefix))) {
    tensorstore::internal_metrics::FormatCollectedMetric(
        metric, [&lines](bool has_value, std::string line) {
          if (has_value) lines.emplace_back(std::move(line));
        });
  }

  // `lines` is unordered, which isn't great for benchmark comparison.
  std::sort(std::begin(lines), std::end(lines));
  std::cout << std::endl;
  for (const auto& l : lines) {
    std::cout << l << std::endl;
  }
  std::cout << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  int test_result = RUN_ALL_TESTS();
  DumpAllMetrics();
  return test_result;
}
