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

/// \file kvstore_benchmark is a benchmark of kvstore operations with varied
/// chunk sizes and other parameters.
///
/* Examples

# 1 GB file

bazel run -c opt //tensorstore/internal/benchmark:kvstore_benchmark -- \
  --kvstore_spec='"file:///tmp/tensorstore_kvstore_benchmark"' \
  --clean_before_write \
  --repeat_writes=10 \
  --repeat_reads=10

# 4 GB memory, 4MB chunks

bazel run -c opt //tensorstore/internal/benchmark:kvstore_benchmark -- \
  --kvstore_spec='"memory://abc/"' \
  --chunk_size=4194304 \
  --total_bytes=4294967296 \
  --read_blowup=500 \
  --repeat_writes=10 \
  --repeat_reads=100

# Quick size reference:

16KB   --chunk_size=16384
512KB  --chunk_size=524288
1MB    --chunk_size=1048576
2MB    --chunk_size=2097152  (default)
4MB    --chunk_size=4194304

256MB  --total_bytes=268435456
1GB    --total_bytes=1073741824  (default)
4GB    --total_bytes=4294967296
*/

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/internal/init_tensorstore.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/internal/metrics/value.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec>, kvstore_spec,
          {},
          "KvStore spec for reading data.  See examples at the start of the "
          "source file.");

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Context::Spec>, context_spec,
          {},
          "Context spec for writing data.  This can be used to control the "
          "number of concurrent write operations of the underlying key-value "
          "store.");

ABSL_FLAG(int64_t, repeat_reads, 1,
          "Number of times to repeat read benchmark.");
ABSL_FLAG(int64_t, repeat_writes, 0,
          "Number of times to repeat write benchmark.");

ABSL_FLAG(size_t, chunk_size, 2 * 1024 * 1024,
          "When writing, use this chunk size (default 2MB).");

ABSL_FLAG(size_t, total_bytes, 1024 * 1024 * 1024,
          "When writing, write this many total bytes (default 1GB).");

ABSL_FLAG(bool, clean_before_write, false,
          "Delete before writing benchmark files.");

ABSL_FLAG(size_t, read_blowup, 1, "Number of chunk reads for each read loop.");

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec>,
          metric_kvstore_spec, {},
          "KvStore spec for writing the metric data in json.  See examples at "
          "the start of the source file.");

ABSL_FLAG(std::string, metric_kvstore_key, "",
          "key used to store the metric data");

ABSL_FLAG(bool, per_operation_metrics, false,
          "Whether to collect per-operation metrics.");

namespace tensorstore {
namespace {

auto& write_throughput = internal_metrics::Value<double>::New(
    "/tensorstore/kvstore_benchmark/write_throughput",
    "the write throughput in this test");

auto& read_throughput = internal_metrics::Value<double>::New(
    "/tensorstore/kvstore_benchmark/read_throughput",
    "the read throughput in this test");

void DumpMetrics(std::string_view prefix) {
  std::vector<std::string> lines;
  for (const auto& metric :
       internal_metrics::GetMetricRegistry().CollectWithPrefix(prefix)) {
    internal_metrics::FormatCollectedMetric(
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

::nlohmann::json StartMetrics() {
  ::nlohmann::json json_metrics = ::nlohmann::json::array();

  // collect flags into a structure that is similar to the metric one.
  json_metrics.emplace_back(::nlohmann::json{
      {"name", "/chunk_size"}, {"values", {absl::GetFlag(FLAGS_chunk_size)}}});
  json_metrics.emplace_back(
      ::nlohmann::json{{"name", "/total_bytes"},
                       {"values", {absl::GetFlag(FLAGS_total_bytes)}}});
  json_metrics.emplace_back(
      ::nlohmann::json{{"name", "/clean_before_write"},
                       {"values", {absl::GetFlag(FLAGS_clean_before_write)}}});
  json_metrics.emplace_back(
      ::nlohmann::json{{"name", "/kvstore_spec"},
                       {"values", {absl::GetFlag(FLAGS_kvstore_spec).value}}});
  json_metrics.emplace_back(
      ::nlohmann::json{{"name", "/context_spec"},
                       {"values", {absl::GetFlag(FLAGS_context_spec).value}}});
  json_metrics.emplace_back(::nlohmann::json{
      {"name", "/per_operation_metrics"},
      {"values", {absl::GetFlag(FLAGS_per_operation_metrics)}}});

  return json_metrics;
}

::nlohmann::json CollectMetricsToJson(std::string id, std::string_view prefix) {
  auto json_metrics = ::nlohmann::json::array();

  // Add an identifier for each collection.
  if (!id.empty()) {
    json_metrics.emplace_back(
        ::nlohmann::json{{"name", "/identifier"}, {"values", {id}}});
  }

  // collect metrics
  for (auto& metric :
       internal_metrics::GetMetricRegistry().CollectWithPrefix(prefix)) {
    if (internal_metrics::IsCollectedMetricNonZero(metric)) {
      json_metrics.emplace_back(
          internal_metrics::CollectedMetricToJson(metric));
    }
  }

  return json_metrics;
}

// When --per_operation_metrics is true, collect (or dump) metrics.
void PerOperationMetricCollection(::nlohmann::json* all_metrics,
                                  std::string id) {
  if (!absl::GetFlag(FLAGS_per_operation_metrics)) {
    return;
  }
  if (absl::GetFlag(FLAGS_metric_kvstore_spec).value.valid()) {
    all_metrics->emplace_back(
        CollectMetricsToJson(std::move(id), "/tensorstore/"));
  } else {
    DumpMetrics("/tensorstore/");
  }
}

// Finalize metrics. When --per_operation_metrics is false, collect (or dump),
// and output them to the --metric_kvstore_spec if set.
void FinishMetricCollection(::nlohmann::json all_metrics) {
  auto kvstore_spec = absl::GetFlag(FLAGS_metric_kvstore_spec).value;

  if (!kvstore_spec.valid()) {
    if (!absl::GetFlag(FLAGS_per_operation_metrics)) {
      DumpMetrics("");
    }
    return;
  }

  all_metrics.emplace_back(CollectMetricsToJson("final", ""));

  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto kvstore,
                                  kvstore::Open(kvstore_spec).result());

  // the key should be unique in each kvstore_benchmark run
  auto key = absl::GetFlag(FLAGS_metric_kvstore_key);
  if (key.empty()) {
    key = absl::FormatTime(absl::Now(), absl::UTCTimeZone());
  }
  TENSORSTORE_CHECK_OK(
      kvstore::Write(kvstore, key, absl::Cord(all_metrics.dump())).result());

  std::cout << "Dumped metrics to: " << kvstore_spec.path << " with key=" << key
            << std::endl;
}

void MaybeCleanExisting(Context context, kvstore::Spec kvstore_spec) {
  // When set, delete the kvstore. For ocdbt, delete everything at "base".
  if (!absl::GetFlag(FLAGS_clean_before_write)) {
    return;
  }
  std::cout << "Cleaning existing tensorstore." << std::endl;
  bool is_ocdbt = false;
  auto json = kvstore_spec.ToJson();
  if (json->is_object() && json->find("driver") != json->end() &&
      json->at("driver") == "ocdbt") {
    is_ocdbt = true;
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto base, kvstore::Open(json->at("base"), context).result());
    TENSORSTORE_CHECK_OK(kvstore::DeleteRange(base, {}));
  }
  if (!is_ocdbt) {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto kvstore, kvstore::Open(kvstore_spec, context).result());
    TENSORSTORE_CHECK_OK(kvstore::DeleteRange(kvstore, {}));
  }
}

struct Prepared {
  std::vector<std::string> keys;
  size_t write_size;
};

Prepared DoWriteBenchmark(Context context, kvstore::Spec kvstore_spec,
                          ::nlohmann::json* all_metrics) {
  Prepared result;
  if (absl::GetFlag(FLAGS_repeat_writes) == 0) {
    return result;
  }
  if (absl::GetFlag(FLAGS_chunk_size) < 8 ||
      absl::GetFlag(FLAGS_total_bytes) < 8) {
    return result;
  }
  std::cout << "Starting write benchmark." << std::endl;
  const bool clear_metrics_before_run =
      absl::GetFlag(FLAGS_per_operation_metrics);

  absl::InsecureBitGen gen;
  absl::Cord data = [&] {
    size_t size = absl::GetFlag(FLAGS_chunk_size);
    auto data = new char[size];
    uint64_t x = absl::Uniform<uint64_t>(gen);
    memcpy(data + size - sizeof(x), &x, sizeof(x));
    for (size_t i = 0; i < size - sizeof(uint64_t); i += sizeof(uint64_t)) {
      uint64_t x = absl::Uniform<uint64_t>(gen);
      memcpy(data + i, &x, sizeof(x));
    }
    return absl::MakeCordFromExternal(
        std::string_view(data, size),
        [](std::string_view s) { delete[] (const_cast<char*>(s.data())); });
  }();

  ABSL_CHECK(!data.empty());
  const size_t num_chunks = tensorstore::CeilOfRatio(
      absl::GetFlag(FLAGS_total_bytes), absl::GetFlag(FLAGS_chunk_size));

  for (size_t i = 0; i < num_chunks; ++i) {
    result.keys.push_back(absl::StrFormat("bm/%03d/%09d", i / 256, i));
  }
  result.write_size = data.size() * result.keys.size();

  // Repeat the benchmark `--repeat_writes` time using the same source arrays.
  for (int64_t i = 0, num_repeats = absl::GetFlag(FLAGS_repeat_writes);
       i < num_repeats; ++i) {
    if (clear_metrics_before_run) {
      internal_metrics::GetMetricRegistry().Reset();
    }

    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto kvstore, kvstore::Open(kvstore_spec, context).result());
    std::shuffle(result.keys.begin(), result.keys.end(), gen);

    // Perform the actual read.
    auto start_time = absl::Now();
    std::atomic<size_t> files_written = 0;
    std::atomic<size_t> bytes_written = 0;

    auto value_lambda =
        [&, sz = data.size()](
            Promise<void> a_promise,
            ReadyFuture<TimestampedStorageGeneration> a_future) {
          files_written.fetch_add(1);
          bytes_written.fetch_add(sz);
        };

    // Promise/Future pair used to track completion of all reads.
    auto [promise, future] = PromiseFuturePair<void>::Make(absl::OkStatus());
    for (const auto& key : result.keys) {
      LinkValue(value_lambda, promise, kvstore::Write(kvstore, key, data, {}));
    }

    // Wait until all writes are complete.
    promise = {};
    TENSORSTORE_CHECK_OK(future.result());

    auto elapsed_s =
        absl::FDivDuration(absl::Now() - start_time, absl::Seconds(1));
    double write_mb = static_cast<double>(bytes_written.load()) / 1e6;
    double throughput = write_mb / elapsed_s;
    write_throughput.Set(throughput);

    std::cout << "Write summary: "
              << absl::StrFormat("%d bytes in %.0f ms:  %.3f MB/second",
                                 bytes_written.load(), elapsed_s * 1e3,
                                 throughput)
              << std::endl;

    PerOperationMetricCollection(all_metrics, absl::StrFormat("write_%03d", i));
  }

  return result;
}

void DoReadBenchmark(Context context, kvstore::Spec kvstore_spec,
                     Prepared input, ::nlohmann::json* all_metrics) {
  if (absl::GetFlag(FLAGS_repeat_reads) == 0) {
    return;
  }
  std::cout << "Starting read benchmark." << std::endl;
  const bool clear_metrics_before_run =
      absl::GetFlag(FLAGS_per_operation_metrics);

  // If data was written, then re-read that data. Otherwise list the available
  // keys and read that dataset.
  if (input.keys.empty()) {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto kvstore, kvstore::Open(kvstore_spec, context).result());
    input.keys = kvstore::ListFuture(kvstore).result().value();
    ABSL_LOG(INFO) << "Read " << input.keys.size() << " keys from kvstore";
  }
  absl::InsecureBitGen gen;

  // Repeat the benchmark `--repeat_reads` times using the same source arrays.
  for (int64_t i = 0, num_repeats = absl::GetFlag(FLAGS_repeat_reads);
       i < num_repeats; ++i) {
    if (clear_metrics_before_run) {
      internal_metrics::GetMetricRegistry().Reset();
    }

    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto kvstore, kvstore::Open(kvstore_spec, context).result());

    std::shuffle(input.keys.begin(), input.keys.end(), gen);

    // Perform the actual read.
    auto start_time = absl::Now();
    std::atomic<size_t> bytes_read = 0;
    std::atomic<size_t> files_read = 0;

    auto value_lambda = [&](Promise<void> a_promise,
                            ReadyFuture<kvstore::ReadResult> a_future) {
      files_read.fetch_add(1);
      if (a_future.result().ok()) {
        bytes_read.fetch_add(a_future.result()->value.size());
      }
    };

    // Promise/Future pair used to track completion of all reads.
    auto [promise, future] = PromiseFuturePair<void>::Make(absl::OkStatus());
    for (size_t j = 0;
         j < std::max(std::size_t{1}, absl::GetFlag(FLAGS_read_blowup)); j++) {
      for (const auto& key : input.keys) {
        LinkValue(value_lambda, promise, kvstore::Read(kvstore, key));
      }
    }

    // Wait until all reads are complete.
    promise = {};
    TENSORSTORE_CHECK_OK(future.result());

    auto elapsed_s =
        absl::FDivDuration(absl::Now() - start_time, absl::Seconds(1));
    double read_mb = static_cast<double>(bytes_read.load()) / 1e6;

    double throughput = read_mb / elapsed_s;
    std::cout << "Read Summary: "
              << absl::StrFormat("%d bytes in %.0f ms:  %.3f MB/second",
                                 bytes_read.load(), elapsed_s * 1e3, throughput)
              << std::endl;

    read_throughput.Set(throughput);

    PerOperationMetricCollection(all_metrics, absl::StrFormat("read_%03d", i));
  }
}

void DoKvstoreBenchmark() {
  auto kvstore_spec = absl::GetFlag(FLAGS_kvstore_spec).value;
  if (!kvstore_spec.path.empty() && kvstore_spec.path.back() != '/') {
    kvstore_spec.AppendSuffix("/");
  }

  Context context(absl::GetFlag(FLAGS_context_spec).value);

  MaybeCleanExisting(context, kvstore_spec);

  // this array contains metrics for each run
  auto all_metrics = StartMetrics();

  auto prepared = DoWriteBenchmark(context, kvstore_spec, &all_metrics);
  DoReadBenchmark(context, kvstore_spec, std::move(prepared), &all_metrics);

  FinishMetricCollection(std::move(all_metrics));
}

}  // namespace
}  // namespace tensorstore

int main(int argc, char** argv) {
  tensorstore::InitTensorstore(&argc, &argv);
  tensorstore::DoKvstoreBenchmark();
  return 0;
}
