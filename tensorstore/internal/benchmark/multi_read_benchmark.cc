// Copyright 2024 The TensorStore Authors
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

// Runs a benchmark of reading a set of tensorstores.
//
// The benchmark is configured by a json file or a json object specifying a
// list of ShardVariable.  Each ShardVariable specifies a path to a tensorstore
// constructed by appending the path to the base tensorstore kvstore path.
// If no "array_boxes" are specified, then then entire tensorstore is read.
//
// The benchmark limits in-flight reads to --max_in_flight bytes.

/* Examples

bazel run -c opt \
  //tensorstore/internal/benchmark:multi_read_benchmark -- \
  --read_config=config.json
*/

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json_fwd.hpp>
#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "absl/flags/parse.h"
#include "tensorstore/internal/benchmark/metric_utils.h"
#include "tensorstore/internal/benchmark/multi_spec.h"
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/value.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

ABSL_FLAG(
    tensorstore::JsonAbslFlag<tensorstore::Context::Spec>, context_spec,
    []() {
      return tensorstore::Context::Spec::FromJson(
                 {
                     {"file_io_concurrency", {{"limit", 128}}},
                 })
          .value();
    }(),
    "Context spec for reading data.  This can be used to control the "
    "number of concurrent read operations, for example.");

ABSL_FLAG(std::string, read_config, {},
          "Paths to all the tensorstore specs to benchmark.");

ABSL_FLAG(int64_t, repeat_reads, 1,
          "Number of times to repeat read benchmark.");

ABSL_FLAG(int64_t, max_in_flight, 64ll * 1024 * 1024 * 1024,  // 64GB
          "Maximum number of in_flight bytes.");

ABSL_FLAG(std::string, context_mode, "",
          "Whether to reuse the tensorstore::Context.");

ABSL_FLAG(std::optional<std::string>, metrics_prefix, std::nullopt,
          "Prefix for metrics.");

ABSL_FLAG(int64_t, ith_spec, -1, "Start at the ith spec in the config file.");

namespace tensorstore {
namespace {

using ::tensorstore::internal_benchmark::ReadSpecsFromFile;

auto& read_throughput = internal_metrics::Value<double>::New(
    "/tensorstore/ts_read_benchmark/read_throughput",
    internal_metrics::MetricMetadata("the read throughput in this test"));

std::vector<tensorstore::TensorStore<>> MultiOpen(
    std::vector<tensorstore::Context> context,
    std::vector<tensorstore::Spec> specs) {
  ABSL_LOG(INFO) << "Opening " << specs.size() << " tensorstores";
  std::vector<Future<tensorstore::TensorStore<>>> pending_open;
  pending_open.reserve(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) {
    Context ctx = i < context.size() ? context[i] : context[0];
    pending_open.push_back(
        tensorstore::Open(specs[i], ctx, tensorstore::ReadWriteMode::read));
  }
  auto wait_all = tensorstore::WaitAllFuture(tensorstore::span(pending_open));
  std::vector<tensorstore::TensorStore<>> stores;
  stores.reserve(specs.size());
  for (size_t i = 0; i < pending_open.size(); ++i) {
    if (pending_open[i].status().ok()) {
      stores.push_back(std::move(pending_open[i]).value());
    } else {
      ABSL_LOG(ERROR) << "Failed to open: " << specs[i];
    }
  }
  return stores;
}

int64_t GetBytesEstimate(const tensorstore::TensorStore<>& ts) {
  return ts.domain().num_elements() * ts.dtype().size();
}

// This class manages the state of a single pass through all the tensorstores.
// It starts new reads when the count of in-flight bytes is below the limit,
// and tracks the total number of bytes read.
struct ReadContinuation {
  const std::vector<tensorstore::TensorStore<>>& stores;

  int64_t max_in_flight = 0;
  std::atomic<int64_t> bytes_read = 0;

  absl::Mutex mutex;
  size_t i = 0;
  int64_t in_flight = 0;

  ReadContinuation(const std::vector<tensorstore::TensorStore<>>& stores,
                   int64_t max_in_flight)
      : stores(stores), max_in_flight(max_in_flight) {}
};

static bool StartNextRead(tensorstore::Promise<void> promise,
                          std::shared_ptr<ReadContinuation> self,
                          size_t finish) {
  int64_t estimate = 0;
  Future<SharedOffsetArray<void>> read_future;

  {
    absl::MutexLock lock(&self->mutex);
    if (finish) {
      self->in_flight -= finish;
    }
    if (self->in_flight >= self->max_in_flight) {
      return false;
    }
    if (self->i >= self->stores.size()) {
      return false;
    }
    const size_t i = self->i++;
    const auto& ts = self->stores[i];

    int64_t estimate = GetBytesEstimate(ts);
    self->in_flight += estimate;
    read_future = tensorstore::Read(ts);
  }

  // Release the mutex before calling Link; the callback may be immediately
  // invoked and deadlock otherwise.
  tensorstore::Link(
      [self = std::move(self), estimate](
          Promise<void> a_promise, Future<SharedOffsetArray<void>> a_future) {
        if (a_future.status().ok()) {
          self->bytes_read.fetch_add(a_future.value().num_elements() *
                                     a_future.value().dtype().size());
        }
        size_t e = estimate;
        while (StartNextRead(a_promise, self, e)) {
          e = 0;
        }
      },
      promise, read_future);
  return true;
}

struct Stats {
  absl::Time start_time;
  absl::Time read_time;
  int64_t bytes = 0;
};

std::string FormatStats(const Stats& s) {
  double read_mb = static_cast<double>(s.bytes) / 1e6;
  auto elapsed_s =
      absl::FDivDuration(s.read_time - s.start_time, absl::Seconds(1));

  double throughput = read_mb / elapsed_s;
  read_throughput.Set(throughput);

  return absl::StrFormat("%.2f MB in %.0f ms, throughput: %.3f MB/second",
                         read_mb, elapsed_s * 1e3, throughput);
}

Stats DoSinglePass(const std::vector<tensorstore::TensorStore<>>& stores,
                   size_t bytes_semaphore) {
  auto [promise, future] = PromiseFuturePair<void>::Make(absl::OkStatus());

  auto cont = std::make_shared<ReadContinuation>(stores, bytes_semaphore);
  Stats stats;
  stats.start_time = absl::Now();

  while (StartNextRead(promise, cont, 0)) {
    /* */
  }
  promise = {};
  future.Wait();

  stats.read_time = absl::Now();
  stats.bytes = cont->bytes_read;
  return stats;
}

Stats DoSinglePassPerPass(tensorstore::Context::Spec context_spec,
                          std::vector<tensorstore::Spec> specs,
                          size_t bytes_semaphore) {
  Context context(context_spec);
  std::vector<tensorstore::TensorStore<>> stores = MultiOpen({context}, specs);
  return DoSinglePass(stores, bytes_semaphore);
}

Stats DoSinglePassDistinct(tensorstore::Context::Spec context_spec,
                           std::vector<tensorstore::Spec> specs,
                           size_t bytes_semaphore) {
  std::vector<tensorstore::Context> contexts;
  contexts.reserve(specs.size());
  while (contexts.size() < specs.size()) {
    contexts.push_back(Context(context_spec));
  }
  std::vector<tensorstore::TensorStore<>> stores = MultiOpen(contexts, specs);
  return DoSinglePass(stores, bytes_semaphore);
}

void RunBenchmark(tensorstore::Context::Spec context_spec,
                  std::vector<tensorstore::Spec> specs) {
  Context context(context_spec);
  std::vector<tensorstore::TensorStore<>> stores = MultiOpen({context}, specs);
  ABSL_QCHECK(!stores.empty()) << "No stores to benchmark";

  int64_t total_bytes = 0;
  int64_t max_ts_bytes = 0;
  for (size_t i = 0; i < stores.size(); ++i) {
    const int64_t estimate = GetBytesEstimate(stores[i]);
    ABSL_CHECK(estimate < 1ull * 1024 * 1024 * 1024 * 1024) << specs[i];
    total_bytes += estimate;
    max_ts_bytes = std::max(max_ts_bytes, estimate);
  }

  std::cout << absl::StrFormat(
                   "Reading %d mb (max %d mb) from %d stores\n"
                   "Read factor %.2f",
                   total_bytes / (1024 * 1024), max_ts_bytes / (1024 * 1024),
                   stores.size(),
                   static_cast<double>(total_bytes) /
                       static_cast<double>(max_ts_bytes))
            << std::endl;

  ABSL_QCHECK(absl::GetFlag(FLAGS_max_in_flight) >= max_ts_bytes)
      << "--max_in_flight=" << max_ts_bytes
      << " is the minimum allowed by this configuration.";

  if (absl::GetFlag(FLAGS_context_mode) == "distinct") {
    context = {};
    stores = {};
    for (size_t i = 0; i < absl::GetFlag(FLAGS_repeat_reads); ++i) {
      auto s = DoSinglePassDistinct(context_spec, specs,
                                    absl::GetFlag(FLAGS_max_in_flight));
      std::cout << FormatStats(s) << std::endl;
    }
  } else if (absl::GetFlag(FLAGS_context_mode) == "pass") {
    context = {};
    stores = {};
    for (size_t i = 0; i < absl::GetFlag(FLAGS_repeat_reads); ++i) {
      auto s = DoSinglePassPerPass(context_spec, specs,
                                   absl::GetFlag(FLAGS_max_in_flight));
      std::cout << FormatStats(s) << std::endl;
    }
  } else {
    for (size_t i = 0; i < absl::GetFlag(FLAGS_repeat_reads); ++i) {
      auto s = DoSinglePass(stores, absl::GetFlag(FLAGS_max_in_flight));
      std::cout << FormatStats(s) << std::endl;
    }
  }
}

void Run(int argc, char** argv) {
  ABSL_CHECK(absl::GetFlag(FLAGS_repeat_reads) > 0);

  std::vector<tensorstore::Spec> specs =
      ReadSpecsFromFile(absl::GetFlag(FLAGS_read_config));
  for (int i = 1; i < argc; ++i) {
    ::nlohmann::json j =
        ::nlohmann::json::parse(std::string(argv[i]), nullptr, false);
    auto spec = tensorstore::Spec::FromJson(j);
    if (spec.ok()) {
      specs.push_back(std::move(spec).value());
    }
  }

  ABSL_QCHECK(!specs.empty()) << "Empty config; supply non-empty --read_config "
                                 "or pass specs on the command line.";

  if (absl::GetFlag(FLAGS_ith_spec) >= 0) {
    ABSL_LOG(INFO) << "Reading only spec #" << absl::GetFlag(FLAGS_ith_spec);
    specs = {specs[absl::GetFlag(FLAGS_ith_spec)]};
  }

  RunBenchmark(absl::GetFlag(FLAGS_context_spec).value, std::move(specs));

  if (absl::GetFlag(FLAGS_metrics_prefix).has_value()) {
    internal::DumpMetrics(absl::GetFlag(FLAGS_metrics_prefix).value());
  }
}

}  // namespace
}  // namespace tensorstore

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);  // InitTensorstore
  tensorstore::Run(argc, argv);
  return 0;
}
