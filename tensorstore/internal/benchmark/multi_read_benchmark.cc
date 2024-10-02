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
  --base_spec='{
      "driver": "zarr3",
      "kvstore": {
          "driver": "ocdbt",
          "base": "gs://bucket/path/ocdbt.root/"
      }}'
  --read_config=config.json
*/

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
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
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/context.h"
#include "absl/flags/parse.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/benchmark/metric_utils.h"
#include "tensorstore/internal/benchmark/multi_spec.h"
#include "tensorstore/internal/benchmark/vector_flag.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/value.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

using ::tensorstore::internal_benchmark::ReadFromFileOrFlag;
using ::tensorstore::internal_benchmark::ShardVariable;

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Context::Spec>, context_spec,
          {},
          "Context spec for writing data.  This can be used to control the "
          "number of concurrent write operations of the underlying key-value "
          "store.  See examples at the start of the source file.");

ABSL_FLAG(
    tensorstore::JsonAbslFlag<tensorstore::Spec>, base_spec,
    []() {
      return tensorstore::Spec::FromJson({
                                             {"driver", "zarr"},
                                             {"kvstore", "memory://"},
                                         })
          .value();
    }(),
    "Base TensorStore Spec to use for reading; the kvstore path will be "
    "augmented with each specified --path.");

ABSL_FLAG(std::string, read_config, {},
          "Paths to all the tensorstore specs to benchmark. The actual spec "
          "is constructed by merging the base_spec with the path.");

ABSL_FLAG(tensorstore::VectorFlag<std::string>, paths, {},
          "Paths to all the tensorstore specs to benchmark. The actual spec "
          "is constructed by merging the base_spec with the path.");

ABSL_FLAG(int64_t, repeat_reads, 1,
          "Number of times to repeat read benchmark.");

ABSL_FLAG(int64_t, max_in_flight, 64ll * 1024 * 1024 * 1024,  // 64GB
          "Maximum number of in_flight bytes.");

ABSL_FLAG(bool, reuse_context, true,
          "Whether to reuse the tensorstore::Context.");

ABSL_FLAG(std::string, metrics_prefix, "/tensorstore", "Prefix for metrics.");

namespace tensorstore {
namespace {

auto& read_throughput = internal_metrics::Value<double>::New(
    "/tensorstore/ts_read_benchmark/read_throughput",
    internal_metrics::MetricMetadata("the read throughput in this test"));

std::vector<tensorstore::TensorStore<>> MultiOpen(
    Context context, std::vector<tensorstore::Spec> specs) {
  ABSL_LOG(INFO) << "Opening " << specs.size() << " tensorstores";
  std::vector<Future<tensorstore::TensorStore<>>> pending_open;
  pending_open.reserve(specs.size());
  for (const auto& spec : specs) {
    pending_open.push_back(
        tensorstore::Open(spec, context, tensorstore::ReadWriteMode::read));
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

// This class manages the state of a single pass through all the tensorstores.
// It starts new reads when the count of in-flight bytes is below the limit,
// and tracks the total number of bytes read.
struct ReadContinuation {
  const std::vector<ShardVariable>& config;
  const std::vector<tensorstore::TensorStore<>>& stores;

  int64_t max_in_flight = 0;
  std::atomic<int64_t> bytes_read = 0;

  absl::Mutex mutex;
  size_t i = 0;
  size_t b = 0;
  int64_t in_flight = 0;

  ReadContinuation(const std::vector<ShardVariable>& config,
                   const std::vector<tensorstore::TensorStore<>>& stores,
                   int64_t max_in_flight)
      : config(config), stores(stores), max_in_flight(max_in_flight) {}
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
    while (true) {
      if (self->i >= self->stores.size()) {
        return false;
      }
      if (self->b == self->config[self->i].array_boxes.size()) {
        self->b = 0;
        self->i++;
        continue;
      }
      break;
    }

    const size_t i = self->i;
    const size_t b = self->b++;

    const auto& ts = self->stores[i];
    const auto& var = self->config[i];
    const auto& box = var.array_boxes[b];

    int64_t estimate = box.num_elements() * ts.dtype().size();
    self->in_flight += estimate;

    read_future = tensorstore::Read(ts | tensorstore::AllDims().BoxSlice(box));
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

Stats DoSinglePass(tensorstore::Context context_spec,
                   const std::vector<ShardVariable>& config,
                   const std::vector<tensorstore::TensorStore<>>& stores,
                   size_t bytes_semaphore) {
  auto [promise, future] = PromiseFuturePair<void>::Make(absl::OkStatus());

  auto cont =
      std::make_shared<ReadContinuation>(config, stores, bytes_semaphore);
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

Stats DoSinglePass(tensorstore::Context::Spec context_spec,
                   const std::vector<ShardVariable>& config,
                   std::vector<tensorstore::Spec> specs,
                   size_t bytes_semaphore) {
  Context context(context_spec);
  std::vector<tensorstore::TensorStore<>> stores = MultiOpen(context, specs);
  return DoSinglePass(context, config, stores, bytes_semaphore);
}

void RunBenchmark(tensorstore::Context::Spec context_spec,
                  std::vector<ShardVariable> config,
                  std::vector<tensorstore::Spec> specs) {
  Context context(context_spec);
  std::vector<tensorstore::TensorStore<>> stores = MultiOpen(context, specs);
  ABSL_QCHECK(!stores.empty()) << "No stores to benchmark";
  ABSL_QCHECK(stores.size() == config.size())
      << "Number of stores and config size mismatch";

  int64_t total_bytes = 0;
  int64_t max_ts_bytes = 0;
  for (size_t i = 0; i < stores.size(); ++i) {
    if (config[i].shape.empty()) {
      config[i].shape.assign(stores[i].domain().shape().cbegin(),
                             stores[i].domain().shape().cend());
    }
    if (config[i].array_boxes.empty() && !config[i].shape.empty()) {
      config[i].array_boxes.push_back(Box<>(config[i].shape));
    }

    for (const auto& box : config[i].array_boxes) {
      ABSL_CHECK_EQ(box.rank(), stores[i].rank());
      if (box.rank() > 0) {
        const int64_t estimate = box.num_elements() * stores[i].dtype().size();
        ABSL_CHECK(estimate < 1ull * 1024 * 1024 * 1024 * 1024 * 1024) << box;
        total_bytes += estimate;
        max_ts_bytes = std::max(max_ts_bytes, estimate);
      }
    }

    // Log the config variable.
    [&](const ShardVariable& var) {
      auto j = internal_json_binding::ToJson(config[i]);
      if (j.ok() && !j->is_discarded()) {
        ABSL_LOG(INFO) << j->dump();
      }
    }(config[i]);
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

  if (absl::GetFlag(FLAGS_reuse_context)) {
    // Reuse the context on each pass. This should be generally faster.
    for (size_t i = 0; i < absl::GetFlag(FLAGS_repeat_reads); ++i) {
      auto s = DoSinglePass(context, config, stores,
                            absl::GetFlag(FLAGS_max_in_flight));
      std::cout << FormatStats(s) << std::endl;
    }
  } else {
    // Re
    context = {};
    stores = {};
    for (size_t i = 0; i < absl::GetFlag(FLAGS_repeat_reads); ++i) {
      auto s = DoSinglePass(context_spec, config, specs,
                            absl::GetFlag(FLAGS_max_in_flight));
      std::cout << FormatStats(s) << std::endl;
    }
  }
}

void Run() {
  ABSL_CHECK(absl::GetFlag(FLAGS_repeat_reads) > 0);

  // --read_config specifies a file or a json object.
  std::vector<ShardVariable> config = [&]() {
    if (absl::GetFlag(FLAGS_read_config).empty()) {
      std::vector<ShardVariable> config;
      for (const auto& name : absl::GetFlag(FLAGS_paths).elements) {
        config.push_back(ShardVariable{name});
      }
      return config;
    }
    return ReadFromFileOrFlag(absl::GetFlag(FLAGS_read_config));
  }();

  ABSL_QCHECK(!config.empty())
      << "Empty config; supply non-empty --read_config or --paths.";

  auto ensure_trailing_slash = [](auto& spec) {
    if (!spec.path.empty() && spec.path.back() != '/') {
      spec.AppendSuffix("/");
    }
  };

  auto kvstore_spec = absl::GetFlag(FLAGS_base_spec).value.kvstore();
  ensure_trailing_slash(kvstore_spec);
  auto base_spec = [&]() {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto ts_json, absl::GetFlag(FLAGS_base_spec).value.ToJson());
    ts_json.erase("kvstore");
    return tensorstore::Spec::FromJson(ts_json).value();
  }();
  TENSORSTORE_CHECK_OK(base_spec.Set(OpenMode::open));

  // Construct the specs to benchmark by combining the base spec with each
  // specified path. Orbax, for example, constructs multiple tensorstores
  // within the same ocdbt kvstore, so this can be used to benchmark the read
  // of an orbax-style checkpoint.
  std::vector<tensorstore::Spec> specs;
  for (const auto& variable : config) {
    ABSL_QCHECK(!variable.name.empty()) << "Variable name must not be empty.";

    auto read_spec = kvstore_spec;
    read_spec.AppendPathComponent(variable.name);
    ensure_trailing_slash(read_spec);

    auto spec = base_spec;
    TENSORSTORE_CHECK_OK(spec.Set(std::move(read_spec)));
    specs.push_back(std::move(spec));
  }

  RunBenchmark(absl::GetFlag(FLAGS_context_spec).value, std::move(config),
               std::move(specs));

  internal::DumpMetrics(absl::GetFlag(FLAGS_metrics_prefix));
}

}  // namespace
}  // namespace tensorstore

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);  // InitTensorstore
  tensorstore::Run();
  return 0;
}
