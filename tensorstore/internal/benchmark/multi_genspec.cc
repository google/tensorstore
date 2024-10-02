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

// Generates a config.json spec for multi_read_benchmark and/or
// multi_write_benchmark.

/* Examples

bazel run -c opt \
  //tensorstore/internal/benchmark:multi_genspec -- \
  --base_spec='{
      "driver": "zarr3",
      "kvstore": {
          "driver": "ocdbt",
          "base": "gs://bucket/path/ocdbt.root/"
      }}'
  --paths=opt_state.0.mu.params.decoder.decoder_norm.scale/,opt_state.0.mu.params.decoder.layers.mlp.wi_0.kernel/
  > config.json

*/

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "absl/flags/parse.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/benchmark/multi_spec.h"
#include "tensorstore/internal/benchmark/vector_flag.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/box.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
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

using ::tensorstore::internal_benchmark::ReadFromFileOrFlag;
using ::tensorstore::internal_benchmark::ShardVariable;

struct HostSpec {
  /// Number of partitions into which each variable should be divided.
  /// This is, for example, the number of accelerator devices across all hosts.
  int64_t total_partitions = 0;

  /// Number of partitions of each variable written by a single host.
  /// This is, for example, the number of accelerator devices on a single host.
  int64_t partitions_per_host = 1;

  /// Index of the host in the benchmark run.
  int32_t worker_index = 0;

  /// Total number of hosts in the benchmark run.
  int32_t total_workers = 0;

  constexpr static auto default_json_binder =
      [](auto is_loading, const auto& options, auto* obj, auto* j) {
        namespace jb = tensorstore::internal_json_binding;
        using Self = HostSpec;
        return jb::Object(
            jb::Member("total_partitions",
                       jb::Projection<&Self::total_partitions>(
                           jb::DefaultValue([](auto* x) { *x = 0; }))),
            jb::Member("partitions_per_host",
                       jb::Projection<&Self::partitions_per_host>(
                           jb::DefaultValue([](auto* x) { *x = 1; }))),
            jb::Member("worker_index", jb::Projection<&Self::worker_index>()),
            jb::Member("total_workers", jb::Projection<&Self::total_workers>()),
            jb::DiscardExtraMembers /**/
            )(is_loading, options, obj, j);
      };
};

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

ABSL_FLAG(std::string, config, {},
          "Paths to all the tensorstore specs to benchmark. The actual spec "
          "is constructed by merging the base_spec with the path.");

ABSL_FLAG(tensorstore::VectorFlag<std::string>, paths, {},
          "Paths to all the tensorstore specs to benchmark. The actual spec "
          "is constructed by merging the base_spec with the path.");

ABSL_FLAG(
    tensorstore::JsonAbslFlag<HostSpec>, host_spec, {},
    "HostShard json object to use for reading.  This can be used to control "
    "the number of partitions of each variable to read.");

namespace tensorstore {
namespace {

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

void FillShardVariableFields(const TensorStore<>& ts, ShardVariable& var) {
  // Fill in the shape/chunks if not specified.
  var.shape.assign(ts.domain().shape().cbegin(), ts.domain().shape().cend());
  auto layout = ts.chunk_layout();
  var.chunks.assign(layout->read_chunk_shape().cbegin(),
                    layout->read_chunk_shape().cend());
  var.dtype = ts.dtype().name();
  ABSL_CHECK_EQ(var.shape.size(), var.chunks.size()) << var.name;
}

void SetHostSpecArrayBoxes(ShardVariable& var, const HostSpec& host) {
  ABSL_CHECK(host.total_workers > 0);

  var.array_boxes.clear();
  if (var.shape.empty()) return;

  ABSL_CHECK_EQ(var.shape.size(), var.chunks.size()) << var.name;

  size_t rank = var.shape.size();
  int64_t total_chunks = 1;
  std::vector<int64_t> grid_shape(rank);
  std::vector<int64_t> grid_pos(rank);
  for (size_t i = 0; i < rank; ++i) {
    grid_shape[i] = tensorstore::CeilOfRatio(var.shape[i], var.chunks[i]);
    total_chunks *= grid_shape[i];
  }

  int64_t local_chunks_to_write = tensorstore::CeilOfRatio(
      total_chunks * host.partitions_per_host, host.total_partitions);
  int64_t local_start_chunk_i =
      std::min(total_chunks,
               static_cast<int64_t>(local_chunks_to_write * host.worker_index));
  int64_t local_end_chunk_i =
      std::min(total_chunks, local_start_chunk_i + local_chunks_to_write);
  int64_t global_end_chunk_i = std::min(
      total_chunks,
      static_cast<int64_t>(host.total_workers * local_chunks_to_write));
  std::vector<int64_t> slice_start(rank), slice_shape(rank);
  for (int64_t chunk_i = 0; chunk_i < global_end_chunk_i; ++chunk_i) {
    for (size_t i = 0; i < rank; ++i) {
      slice_start[i] = var.chunks[i] * grid_pos[i];
      slice_shape[i] = std::min(var.chunks[i], var.shape[i] - slice_start[i]);
    }
    const auto box = Box(slice_start, slice_shape);
    if (chunk_i >= local_start_chunk_i && chunk_i < local_end_chunk_i) {
      var.array_boxes.push_back(box);
    }
    tensorstore::internal::AdvanceIndices(rank, grid_pos.data(),
                                          grid_shape.data());
  }
}

bool FillUsingHostSpec(std::vector<ShardVariable>& config,
                       const std::vector<tensorstore::TensorStore<>>& stores,
                       HostSpec host) {
  if (host.total_workers == 0) return false;
  ABSL_LOG(INFO) << "Attempting to generate config from --host_spec.";

  auto total_chunks = [&](const ShardVariable& var) -> int64_t {
    int64_t total_chunks = 1;
    auto rank = var.shape.size();
    std::vector<int64_t> grid_shape(rank);
    for (size_t i = 0; i < rank; ++i) {
      grid_shape[i] = tensorstore::CeilOfRatio(var.shape[i], var.chunks[i]);
      total_chunks *= grid_shape[i];
    }
    return total_chunks;
  };

  if (host.total_partitions == 0) {
    int64_t total_partitions = 0;
    for (size_t i = 0; i < stores.size(); ++i) {
      total_partitions = std::max(total_partitions, total_chunks(config[i]));
    }
    ABSL_LOG(INFO) << "total_partitions=" << total_partitions;
    host.total_partitions = total_partitions;
  }
  if (host.partitions_per_host == 0) {
    host.partitions_per_host = 1;
  }
  for (size_t i = 0; i < stores.size(); ++i) {
    SetHostSpecArrayBoxes(config[i], host);
    if (config[i].array_boxes.empty() && !config[i].shape.empty()) {
      config[i].array_boxes.push_back(Box<>(config[i].shape));
    }
  }
  return true;
}

// Queries the storage statistics for each possible var.chunk shape contained
// within the tensorstore. All present ranges are added to the array_boxes.
void SetExistingChunkArrayBoxes(ShardVariable& var,
                                tensorstore::TensorStore<> ts) {
  size_t rank = var.shape.size();
  std::vector<int64_t> grid_shape(rank);
  std::vector<int64_t> grid_pos(rank);
  for (size_t i = 0; i < rank; ++i) {
    grid_shape[i] = tensorstore::CeilOfRatio(var.shape[i], var.chunks[i]);
  }

  auto [stat_promise, stat_future] =
      PromiseFuturePair<void>::Make(absl::OkStatus());

  absl::Mutex mutex;
  std::vector<Box<>> array_boxes;

  std::vector<int64_t> slice_start(rank), slice_shape(rank);
  do {
    for (size_t i = 0; i < rank; ++i) {
      slice_start[i] = var.chunks[i] * grid_pos[i];
      slice_shape[i] = std::min(var.chunks[i], var.shape[i] - slice_start[i]);
    }
    const auto box = Box(slice_start, slice_shape);
    LinkValue(
        [b = box, &var, &mutex, &array_boxes](
            auto promise, ReadyFuture<ArrayStorageStatistics> f) {
          ABSL_LOG(INFO) << var.name << " " << b
                         << (f.value().not_stored ? " <missing>" : "");
          absl::MutexLock lock(&mutex);
          if (!f.value().not_stored) {
            array_boxes.push_back(std::move(b));
          }
        },
        stat_promise,
        GetStorageStatistics(
            ts | tensorstore::AllDims().BoxSlice(box),
            tensorstore::ArrayStorageStatistics::query_not_stored));
  } while (tensorstore::internal::AdvanceIndices(rank, grid_pos.data(),
                                                 grid_shape.data()));
  stat_promise = {};
  stat_future.Wait();
  var.array_boxes = std::move(array_boxes);
}

bool FillUsingExistingData(
    std::vector<ShardVariable>& config,
    const std::vector<tensorstore::TensorStore<>>& stores) {
  ABSL_LOG(INFO)
      << "Attempting to generate config from existing data in --base_spec.";
  for (size_t i = 0; i < stores.size(); ++i) {
    SetExistingChunkArrayBoxes(config[i], stores[i]);
  }
  return true;
}

// Greedily merges adjacent boxes in the array.
// The basic greedy algorithm is to sort the boxes by each position, and then
// attempt to merge any boxes which are adjacent along a single dimension,
// repeating until no more merges are possible.
std::vector<Box<>> MergeAdjacentBoxes(std::vector<Box<>> boxes) {
  if (boxes.empty()) return boxes;
  bool merged = false;
  auto rank = boxes[0].rank();
  for (auto& box : boxes) {
    rank = std::max(rank, box.rank());
  }
  if (rank == 0) return boxes;

  auto merge_if_adjacent = [&](const Box<>& left,
                               const Box<>& right) -> std::optional<Box<>> {
    if (left.rank() != right.rank()) return std::nullopt;
    std::optional<size_t> dim_to_merge;
    std::vector<int64_t> shape(left.rank());
    for (size_t i = 0; i < left.rank(); ++i) {
      shape[i] = left.shape()[i];
      if (left.origin()[i] == right.origin()[i]) continue;
      if (left.origin()[i] + left.shape()[i] == right.origin()[i]) {
        if (!dim_to_merge.has_value()) {
          dim_to_merge = i;
          shape[i] = left.shape()[i] + right.shape()[i];
          continue;
        }
      }
      return std::nullopt;
    }
    return Box<>(left.origin(), shape);
  };

  std::vector<Box<>> result = std::move(boxes);

  do {
    merged = false;
    // Sort boxes by each rank.
    for (size_t i = 0; i < rank; ++i) {
      boxes = std::move(result);
      std::sort(boxes.begin(), boxes.end(),
                [i](const Box<>& a, const Box<>& b) {
                  if (a.rank() != b.rank()) {
                    return a.rank() < b.rank();
                  }
                  if (std::equal(a.origin().begin(), a.origin().end(),
                                 b.origin().begin())) {
                    return std::lexicographical_compare(
                        a.shape().begin(), a.shape().end(), b.shape().begin(),
                        b.shape().end());
                  }
                  if (i < a.rank() && a.origin()[i] != b.origin()[i]) {
                    return a.origin()[i] < b.origin()[i];
                  }
                  return std::lexicographical_compare(
                      a.origin().begin(), a.origin().end(), b.origin().begin(),
                      b.origin().end());
                });
      // Merge adjacent boxes.
      result = {};
      auto it = boxes.begin();
      result.push_back(std::move(*it));
      for (++it; it != boxes.end(); ++it) {
        auto box_maybe = merge_if_adjacent(result.back(), *it);
        if (box_maybe.has_value()) {
          result.back() = *std::move(box_maybe);
          merged = true;
        } else {
          result.push_back(std::move(*it));
        }
      }
    }
  } while (merged);

  // Sort the final set of merged boxes to ensure consistent ordering.
  std::sort(result.begin(), result.end(), [](const Box<>& a, const Box<>& b) {
    if (a.rank() != b.rank()) return a.rank() < b.rank();
    if (std::equal(a.origin().begin(), a.origin().end(), b.origin().begin())) {
      return std::lexicographical_compare(a.shape().begin(), a.shape().end(),
                                          b.shape().begin(), b.shape().end());
    } else {
      return std::lexicographical_compare(a.origin().begin(), a.origin().end(),
                                          b.origin().begin(), b.origin().end());
    }
  });
  return result;
}

void Dump(std::vector<ShardVariable>& config) {
  std::cout << "[";
  bool add_comma = false;
  for (auto& var : config) {
    if (add_comma) std::cout << ",";
    add_comma = true;
    auto j = internal_json_binding::ToJson(var);
    if (j.ok() && !j->is_discarded()) {
      std::cout << "\n  " << j->dump();
    }
  }
  std::cout << "\n]\n" << std::endl;
}

void Run() {
  // --read_config specifies a file or a json object.
  std::vector<ShardVariable> config = [&]() {
    if (absl::GetFlag(FLAGS_config).empty()) {
      std::vector<ShardVariable> config;
      for (const auto& name : absl::GetFlag(FLAGS_paths).elements) {
        config.push_back(ShardVariable{name});
      }
      return config;
    }
    return ReadFromFileOrFlag(absl::GetFlag(FLAGS_config));
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

    if (!variable.dtype.empty()) {
      auto dtype = GetDataType(variable.dtype);
      ABSL_QCHECK(dtype.valid()) << "Invalid dtype: " << variable.dtype;
      TENSORSTORE_CHECK_OK(spec.Set(dtype));
    }
    if (!variable.shape.empty()) {
      TENSORSTORE_CHECK_OK(spec.Set(Schema::Shape(variable.shape)));
    }
    if (!variable.chunks.empty()) {
      TENSORSTORE_CHECK_OK(
          spec.Set(ChunkLayout::ReadChunkShape(variable.chunks)));
    }
    specs.push_back(std::move(spec));
  }
  Context context(absl::GetFlag(FLAGS_context_spec).value);

  std::vector<tensorstore::TensorStore<>> stores = MultiOpen(context, specs);
  ABSL_QCHECK(stores.size() == config.size())
      << "Number of stores and config size mismatch";

  for (size_t i = 0; i < stores.size(); ++i) {
    FillShardVariableFields(stores[i], config[i]);
  }

  if (!FillUsingHostSpec(config, stores,
                         absl::GetFlag(FLAGS_host_spec).value)) {
    if (!FillUsingExistingData(config, stores)) {
      ABSL_LOG(FATAL) << "Failed to fill config.";
    }
  }

  // Merges adjacent boxes.
  for (auto& var : config) {
    var.array_boxes = MergeAdjacentBoxes(std::move(var.array_boxes));
  }

  Dump(config);
}

}  // namespace
}  // namespace tensorstore

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);  // InitTensorstore
  tensorstore::Run();
  return 0;
}
