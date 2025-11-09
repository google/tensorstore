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

// Runs a benchmark of writing a set of tensorstores.
//
// The benchmark is configured by a json file or a json object specifying a
// list of ShardVariable.  Each ShardVariable specifies a path to a tensorstore
// constructed by appending the path to the base tensorstore kvstore path.
// If no "array_boxes" are specified, then then entire tensorstore is read.
//
// The benchmark limits in-flight writes to --max_in_flight bytes.

/* Examples

bazel run -c opt \
  //tensorstore/internal/benchmark:multi_write_benchmark -- \
  --base_spec='{
      "driver": "zarr3",
      "kvstore": {
          "driver": "ocdbt",
          "base": "file:///tmp/benchmark"
      }}'
  --write_config=config.json
*/

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/hash/hash.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/context.h"
#include "absl/flags/parse.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/benchmark/metric_utils.h"
#include "tensorstore/internal/benchmark/multi_spec.h"
#include "tensorstore/internal/data_type_random_generator.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/value.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/schema.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

ABSL_FLAG(
    tensorstore::JsonAbslFlag<tensorstore::Context::Spec>, context_spec,
    []() {
      return tensorstore::Context::Spec::FromJson(
                 {
                     {"file_io_concurrency", {{"limit", 128}}},
                 })
          .value();
    }(),
    "Context spec for writing data.  This can be used to control the "
    "number of concurrent write operations, for example.");

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec>, clean_spec, {},
          "kvstore::Spec to use for cleaning the kvstore before writing. "
          "When unset, no cleanup is done, which may cause ocdbt volumes to "
          "grow excessively large.");

ABSL_FLAG(std::string, write_config, {},
          "Paths to all the tensorstore specs to benchmark. The actual spec "
          "is constructed by merging the base_spec with the path.");

ABSL_FLAG(int64_t, repeat_writes, 1,
          "Number of times to repeat read benchmark.");

ABSL_FLAG(bool, clean_before_write, true, "Clean kvstore before writing.");

ABSL_FLAG(bool, reuse_context, false,
          "Whether to reuse the tensorstore::Context.");

ABSL_FLAG(std::optional<std::string>, metrics_prefix, std::nullopt,
          "Prefix for metrics.");

namespace tensorstore {
namespace {

using ::tensorstore::internal_benchmark::ReadSpecsFromFile;

auto& write_throughput = internal_metrics::Value<double>::New(
    "/tensorstore/ts_write_benchmark/write_throughput",
    internal_metrics::MetricMetadata("the write throughput in this test"));

using KeyType = std::pair<std::string_view, std::vector<Index>>;
struct ArrayMapEq {
  bool operator()(const KeyType& x, const KeyType& y) const {
    return x.first == y.first && x.second.size() == y.second.size() &&
           std::equal(x.second.begin(), x.second.end(), y.second.begin());
  }
};
using ArrayMap = absl::flat_hash_map<KeyType, SharedOffsetArray<const void>,
                                     absl::Hash<KeyType>, ArrayMapEq>;

int64_t GetBytesEstimate(const tensorstore::TensorStore<>& ts) {
  return ts.domain().num_elements() * ts.dtype().size();
}

struct Stats {
  absl::Time start_time;
  absl::Time open_time;
  absl::Time copy_time;
  absl::Time commit_time;
  int64_t bytes = 0;
  int64_t chunks = 0;
};

std::string FormatStats(const Stats& s) {
  double read_mb = static_cast<double>(s.bytes) / 1e6;
  auto open_elapsed_s =
      absl::FDivDuration(s.open_time - s.start_time, absl::Seconds(1));
  auto copy_elapsed_s =
      absl::FDivDuration(s.copy_time - s.start_time, absl::Seconds(1));
  auto commit_elapsed_s =
      absl::FDivDuration(s.commit_time - s.start_time, absl::Seconds(1));

  double throughput = read_mb / commit_elapsed_s;
  write_throughput.Set(throughput);

  return absl::StrFormat(
      "%.2f MB (%d chunks) in %.0f ms (open), %.0f ms "
      "(copy), %.0f "
      "ms (commit), throughput: %.3f MB/second",
      read_mb, s.chunks, open_elapsed_s * 1e3, copy_elapsed_s * 1e3,
      commit_elapsed_s * 1e3, throughput);
}

Stats DoSinglePass(tensorstore::Context context,
                   const std::vector<tensorstore::Spec>& specs,
                   const ArrayMap& chunk_arrays) {
  std::atomic<int64_t> bytes_completed = 0;
  std::atomic<int64_t> chunks_completed = 0;

  Stats stats;
  stats.start_time = absl::Now();

  // Promise/Future pair used to track completion of all writes.
  auto [open_promise, open_future] =
      PromiseFuturePair<void>::Make(absl::OkStatus());
  auto [copy_promise, copy_future] =
      PromiseFuturePair<void>::Make(absl::OkStatus());
  auto [commit_promise, commit_future] =
      PromiseFuturePair<void>::Make(absl::OkStatus());

  for (size_t i = 0; i < specs.size(); ++i) {
    auto cur_open_future = tensorstore::Open(
        specs[i], context, tensorstore::ReadWriteMode::read_write,
        tensorstore::OpenMode::open_or_create);
    // Open all tensorstores in parallel.  Once a given tensorstore is open,
    // write each of its partitions.
    Link(
        [&, copy_promise = copy_promise, commit_promise = commit_promise](
            Promise<void> open_promise, ReadyFuture<TensorStore<>> future) {
          if (!future.status().ok()) {
            // If one tensorstore fails to open, keep going with the rest.
            ABSL_LOG(ERROR) << future.status();
            return;
          }
          auto& ts = future.value();
          KeyType key(ts.dtype().name(),
                      {ts.chunk_layout()->write_chunk_shape().begin(),
                       ts.chunk_layout()->write_chunk_shape().end()});
          auto it = chunk_arrays.find(key);
          if (it == chunk_arrays.end()) {
            // No data to write.
            return;
          }

          const auto& array = it->second;
          auto chunk_bytes = array.num_elements() * array.dtype().size();

          const auto& domain = future.value().domain();

          const auto rank = array.rank();
          std::vector<int64_t> grid_shape(rank);
          std::vector<int64_t> grid_pos(rank);
          for (size_t i = 0; i < rank; ++i) {
            grid_shape[i] =
                tensorstore::CeilOfRatio(domain.shape()[i], array.shape()[i]);
          }
          std::vector<int64_t> slice_start(rank), slice_shape(rank);
          do {
            for (size_t i = 0; i < rank; ++i) {
              slice_start[i] = grid_pos[i] * array.shape()[i];
              slice_shape[i] = array.shape()[i];
              if (slice_start[i] + slice_shape[i] > domain.shape()[i]) {
                slice_shape[i] = domain.shape()[i] - slice_start[i];
              }
            }
            Box target(slice_start, slice_shape);

            auto write_futures = tensorstore::Write(
                array,
                future.value() |
                    tensorstore::AllDims().BoxSlice(target).TranslateTo(0));

            LinkError(copy_promise, std::move(write_futures.copy_future));
            Link(
                [&, chunk_bytes = chunk_bytes](Promise<void> commit_promise,
                                               ReadyFuture<void> f) {
                  chunks_completed.fetch_add(1);
                  if (f.status().ok()) {
                    bytes_completed.fetch_add(chunk_bytes);
                  } else {
                    commit_promise.SetResult(f.status());
                  }
                },
                commit_promise, std::move(write_futures.commit_future));
          } while (tensorstore::internal::AdvanceIndices(rank, grid_pos.data(),
                                                         grid_shape.data()));
        },
        open_promise, std::move(cur_open_future));
  }

  open_promise = {};
  copy_promise = {};
  commit_promise = {};

  // Wait until all opens complete.
  open_future.Wait();
  stats.open_time = absl::Now();

  // Wait until all copies complete.
  copy_future.Wait();
  if (!copy_future.result().ok()) {
    // write test failed, return to write_before
    ABSL_LOG(WARNING) << "Failed to copy:" << copy_future.status();
  }
  stats.copy_time = absl::Now();

  // Wait until all commits complete.
  commit_future.Wait();
  if (!commit_future.result().ok()) {
    // write test failed, return to write_before
    ABSL_LOG(FATAL) << "Failed to commit:" << commit_future.status();
  }

  stats.commit_time = absl::Now();
  stats.bytes = bytes_completed.load();
  stats.chunks = chunks_completed.load();
  return stats;
}

absl::Status RunBenchmark(tensorstore::Context::Spec context_spec,
                          tensorstore::kvstore::Spec clean_spec,
                          std::vector<tensorstore::Spec> specs) {
  ABSL_QCHECK(!specs.empty()) << "No stores to benchmark";

  int64_t total_bytes = 0;
  int64_t max_ts_bytes = 0;

  // Generate data arrays in the shape as the write chunks.
  ArrayMap chunk_arrays;
  {
    Context context(context_spec);
    absl::BitGen gen;
    for (const auto& spec : specs) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto ts, tensorstore::Open(spec, context,
                                     tensorstore::ReadWriteMode::read_write,
                                     tensorstore::OpenMode::open_or_create)
                       .result());
      KeyType key(ts.dtype().name(),
                  {ts.chunk_layout()->write_chunk_shape().begin(),
                   ts.chunk_layout()->write_chunk_shape().end()});
      if (key.second.empty()) continue;
      if (chunk_arrays.find(key) != chunk_arrays.end()) continue;
      Box<> box(key.second);
      ABSL_LOG(INFO) << "Generating random array: " << ts.dtype().name() << " "
                     << box;
      chunk_arrays[key] = internal::MakeRandomArray(gen, box, ts.dtype());
      total_bytes += GetBytesEstimate(ts);
      max_ts_bytes =
          std::max(max_ts_bytes, box.num_elements() * ts.dtype().size());
    }
  }

  std::cout << absl::StrFormat("Writing %d mb (max %d mb) to %d stores\n",
                               total_bytes / (1024 * 1024),
                               max_ts_bytes / (1024 * 1024), specs.size())
            << std::endl;

  auto maybe_clean = [&](Context context) {
    if (absl::GetFlag(FLAGS_clean_before_write)) {
      ABSL_LOG(INFO) << "Cleaning kvstore at: " << clean_spec.ToJson()->dump();
      TENSORSTORE_CHECK_OK_AND_ASSIGN(
          auto kvstore, kvstore::Open(clean_spec, context).result());
      TENSORSTORE_CHECK_OK(kvstore::DeleteRange(kvstore, {}));
    }
  };

  if (absl::GetFlag(FLAGS_reuse_context)) {
    if (absl::GetFlag(FLAGS_clean_before_write)) {
      // NOTE: The current behavior, when coupled with a persistent cache,
      // can cause a crash.
      ABSL_LOG(WARNING) << "This may cause tensorstore to crash.";
    }
    Context context(context_spec);
    for (size_t i = 0; i < absl::GetFlag(FLAGS_repeat_writes); ++i) {
      maybe_clean(context);
      auto s = DoSinglePass(context, specs, chunk_arrays);
      std::cout << "Write Summary: " << FormatStats(s) << std::endl;
    }
  } else {
    for (size_t i = 0; i < absl::GetFlag(FLAGS_repeat_writes); ++i) {
      Context context(context_spec);
      maybe_clean(context);
      auto s = DoSinglePass(context, specs, chunk_arrays);
      std::cout << "Write Summary: " << FormatStats(s) << std::endl;
    }
  }
  return absl::OkStatus();
}

void Run(int argc, char** argv) {
  ABSL_CHECK(absl::GetFlag(FLAGS_repeat_writes) > 0);

  std::vector<tensorstore::Spec> specs =
      ReadSpecsFromFile(absl::GetFlag(FLAGS_write_config));
  for (int i = 1; i < argc; ++i) {
    ::nlohmann::json j =
        ::nlohmann::json::parse(std::string(argv[i]), nullptr, false);
    auto spec = tensorstore::Spec::FromJson(j);
    if (spec.ok()) {
      specs.push_back(std::move(spec).value());
    }
  }

  ABSL_QCHECK(!specs.empty())
      << "Empty config; supply non-empty --write_config "
         "or pass specs on the command line.";

  // For OCDBT, deleting from the kvstore itself would not delete data
  // from the underlying storage system.  Instead, we must directly delete
  // from the underlying storage system.

#if 0
  tensorstore::kvstore::Spec clean_spec = [&]() {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto kvstore_json_spec,
                                    kvstore_spec.ToJson());
    if (kvstore_json_spec.is_object() &&
        kvstore_json_spec.find("driver") != kvstore_json_spec.end() &&
        kvstore_json_spec.at("driver") == "ocdbt") {
      TENSORSTORE_CHECK_OK_AND_ASSIGN(
          auto base, kvstore::Spec::FromJson(kvstore_json_spec.at("base")));
      return base;
    }
    return kvstore_spec;
  }();
#endif

  auto status =
      RunBenchmark(absl::GetFlag(FLAGS_context_spec).value,
                   absl::GetFlag(FLAGS_clean_spec).value, std::move(specs));
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run benchmark: " << status;
  } else if (absl::GetFlag(FLAGS_metrics_prefix).has_value()) {
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
