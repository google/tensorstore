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

#include <cstddef>
#include <memory>
#include <new>
#include <ostream>
#include <utility>
#include <vector>

#include "absl/base/macros.h"
#include "absl/container/inlined_vector.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <benchmark/benchmark.h>
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/async_storage_backed_cache.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/chunk_cache.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/thread_pool.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/progress.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/result_util.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/to_string.h"

namespace {

using tensorstore::ArrayView;
using tensorstore::DimensionIndex;
using tensorstore::Executor;
using tensorstore::Future;
using tensorstore::Index;
using tensorstore::IndexTransform;
using tensorstore::MakeArray;
using tensorstore::MakeCopy;
using tensorstore::SharedArray;
using tensorstore::span;
using tensorstore::Status;
using tensorstore::StorageGeneration;
using tensorstore::TimestampedStorageGeneration;
using tensorstore::internal::CachePool;
using tensorstore::internal::CachePtr;
using tensorstore::internal::ChunkCache;
using tensorstore::internal::ChunkCacheDriver;
using tensorstore::internal::ChunkGridSpecification;
using tensorstore::internal::Driver;
using tensorstore::internal::ElementCopyFunction;

/// Benchmark configuration for read/write benchmark.
struct BenchmarkConfig {
  /// Source and target data type.
  tensorstore::DataType data_type;

  /// Shape of the contiguous source/target array to be copied each iteration.
  std::vector<Index> copy_shape;

  /// Stride (in units of index positions, not bytes) along each dimension
  /// within the chunk cache.  Must have length equal to `copy_shape.size()`.
  std::vector<Index> stride;

  /// Specifies for each dimension whether to use an index array to perform the
  /// indexing within the chunk cache.  Must have length equal to
  /// `copy_shape.size()`.
  std::vector<bool> indexed;

  /// Specifies the cell shape to use for the chunk cache.  Must have length
  /// equal to `copy_shape.size()`.
  std::vector<Index> cell_shape;

  /// Specifies for each cell dimension whether it is chunked.  Must have length
  /// equal to `copy_shape.size()`.
  std::vector<bool> chunked;

  /// Specifies whether to test reading/writing the same chunks repeatedly,
  /// rather than new chunks for each iteration.
  bool cached;

  /// Specifies the number of threads to use for copying.
  int threads;

  /// Specifies whether the benchmark is a read benchmark (`read == true`) or
  /// write benchmark (`read == false`).
  bool read;
};

std::ostream& operator<<(std::ostream& os, const BenchmarkConfig& config) {
  return os
         << (config.read ? "Read:  " : "Write: ")
         << "copy_shape=" << span(config.copy_shape)
         << ", stride=" << span(config.stride) << ", i="
         << span(std::vector<int>(config.indexed.begin(), config.indexed.end()))
         << ", cell_shape=" << span(config.cell_shape) << ", chunked="
         << span(std::vector<int>(config.chunked.begin(), config.chunked.end()))
         << ", cached=" << config.cached << ", threads=" << config.threads;
}

class BenchmarkCache : public ChunkCache {
 public:
  using ChunkCache::ChunkCache;
  Executor executor;

  void DoRead(ReadOptions options, ReadReceiver receiver) override {
    auto req_time = absl::Now();
    executor([receiver, req_time, this] {
      absl::InlinedVector<SharedArray<void>, 1> data;
      absl::InlinedVector<ArrayView<const void>, 1> data_ref;
      data.reserve(grid().components.size());
      data_ref.reserve(grid().components.size());
      for (const auto& component_spec : grid().components) {
        data.push_back(MakeCopy(component_spec.fill_value));
        data_ref.push_back(data.back());
      }
      receiver.NotifyDone(ReadReceiver::ComponentsWithGeneration{
          data_ref, {StorageGeneration{"gen"}, req_time}});
    });
  }
  void DoWriteback(TimestampedStorageGeneration existing_generation,
                   WritebackReceiver receiver) override {
    executor([receiver, this] {
      {
        ChunkCache::WritebackSnapshot snapshot(receiver);
        for (const auto& array : snapshot.component_arrays()) {
          ::benchmark::DoNotOptimize(MakeCopy(array));
        }
      }
      auto req_time = absl::Now();
      executor([receiver, req_time] {
        receiver.NotifyDone(
            {absl::in_place, StorageGeneration{"gen"}, req_time});
      });
    });
  }
};

class TestDriver : public ChunkCacheDriver {
 public:
  using ChunkCacheDriver::ChunkCacheDriver;

  /// Not actually used.
  Executor data_copy_executor() override {
    return static_cast<BenchmarkCache*>(cache())->executor;
  }
};

class CopyBenchmarkRunner {
 public:
  CopyBenchmarkRunner(const BenchmarkConfig& config) : config(config) {
    tensorstore::Executor executor;
    if (config.threads == 0) {
      executor = tensorstore::InlineExecutor{};
    } else {
      executor = tensorstore::internal::DetachedThreadPool(config.threads);
    }

    pool = CachePool::Make(CachePool::Limits{});
    const DimensionIndex rank = config.copy_shape.size();
    assert(rank == static_cast<DimensionIndex>(config.stride.size()));
    assert(rank == static_cast<DimensionIndex>(config.indexed.size()));
    assert(rank == static_cast<DimensionIndex>(config.cell_shape.size()));
    assert(rank == static_cast<DimensionIndex>(config.chunked.size()));
    std::vector<DimensionIndex> chunked_dims;
    for (DimensionIndex i = 0; i < rank; ++i) {
      if (config.chunked[i]) chunked_dims.push_back(i);
    }
    ChunkGridSpecification grid({ChunkGridSpecification::Component{
        AllocateArray(config.cell_shape, tensorstore::c_order,
                      tensorstore::value_init, config.data_type),
        chunked_dims}});
    cache = pool->GetCache<BenchmarkCache>(
        "", [&] { return absl::make_unique<BenchmarkCache>(grid); });
    cache->executor = executor;
    driver.reset(new TestDriver(cache, 0));
    array = AllocateArray(config.copy_shape, tensorstore::c_order,
                          tensorstore::value_init, config.data_type);
    transform = ChainResult(tensorstore::IdentityTransform(rank),
                            tensorstore::AllDims().SizedInterval(
                                0, config.copy_shape, config.stride))
                    .value();
    for (DimensionIndex i = 0; i < rank; ++i) {
      if (config.indexed[i]) {
        auto index_array =
            tensorstore::AllocateArray<Index>({config.copy_shape[i]});
        for (DimensionIndex j = 0; j < config.copy_shape[i]; ++j)
          index_array(j) = j;
        transform =
            ChainResult(transform,
                        tensorstore::Dims(i).OuterIndexArraySlice(index_array))
                .value();
      }
    }
    std::vector<Index> offset_amount(rank);
    if (!config.cached) {
      for (DimensionIndex i = 0; i < rank; ++i) {
        if (config.chunked[i]) {
          offset_amount[i] = config.stride[i] * config.copy_shape[i];
        }
      }
    }
    offset_transform =
        ChainResult(tensorstore::IdentityTransform(rank),
                    tensorstore::AllDims().TranslateBy(offset_amount))
            .value();
  }

  void RunOnce() {
    if (config.read) {
      tensorstore::internal::DriverRead(cache->executor, {driver, transform},
                                        array, {/*.progress_function=*/{}})
          .result();
    } else {
      tensorstore::internal::DriverWrite(cache->executor, array,
                                         /*target=*/{driver, transform},
                                         {/*.progress_function=*/{}})
          .commit_future.result();
    }
    transform =
        tensorstore::ComposeTransforms(offset_transform, transform).value();
  }

  BenchmarkConfig config;
  tensorstore::internal::CachePool::StrongPtr pool;
  tensorstore::internal::CachePtr<BenchmarkCache> cache;
  tensorstore::internal::Driver::Ptr driver;
  tensorstore::IndexTransform<> transform;
  tensorstore::SharedArray<void> array;
  tensorstore::IndexTransform<> offset_transform;
};

void BenchmarkCopy(const BenchmarkConfig& config, ::benchmark::State& state) {
  CopyBenchmarkRunner runner(config);

  const Index num_bytes =
      runner.array.num_elements() * runner.config.data_type->size;
  Index total_bytes = 0;
  while (state.KeepRunningBatch(num_bytes)) {
    runner.RunOnce();
    total_bytes += num_bytes;
  }
  state.SetBytesProcessed(total_bytes);
}

struct RegisterBenchmarks {
  static void Register(const BenchmarkConfig& config) {
    ::benchmark::RegisterBenchmark(
        tensorstore::StrCat(config).c_str(),
        [config](auto& state) { BenchmarkCopy(config, state); });
  }

  RegisterBenchmarks() {
    for (const bool read : {true, false}) {
      for (const bool cached : {true, false}) {
        for (const int threads : {0, 1, 2, 4}) {
          for (const Index copy_size : {16, 32, 64, 128, 256}) {
            Register({
                /*data_type=*/tensorstore::DataTypeOf<int>(),
                /*copy_shape=*/{copy_size, copy_size, copy_size},
                /*stride=*/{1, 1, 1},
                /*indexed=*/{false, false, false},
                /*cell_shape=*/{64, 64, 64},
                /*chunked=*/{true, true, true},
                /*cached=*/cached,
                /*threads=*/threads,
                /*read=*/read,
            });
            Register({
                /*data_type=*/tensorstore::DataTypeOf<int>(),
                /*copy_shape=*/{copy_size, copy_size, copy_size},
                /*stride=*/{2, 1, 1},
                /*indexed=*/{false, false, false},
                /*cell_shape=*/{64, 64, 64},
                /*chunked=*/{true, true, true},
                /*cached=*/cached,
                /*threads=*/threads,
                /*read=*/read,
            });
            Register({
                /*data_type=*/tensorstore::DataTypeOf<int>(),
                /*copy_shape=*/{copy_size, copy_size, copy_size},
                /*stride=*/{2, 2, 2},
                /*indexed=*/{false, false, false},
                /*cell_shape=*/{64, 64, 64},
                /*chunked=*/{true, true, true},
                /*cached=*/cached,
                /*threads=*/threads,
                /*read=*/read,
            });
            Register({
                /*data_type=*/tensorstore::DataTypeOf<int>(),
                /*copy_shape=*/{copy_size, copy_size, copy_size},
                /*stride=*/{1, 1, 1},
                /*indexed=*/{false, true, false},
                /*cell_shape=*/{64, 64, 64},
                /*chunked=*/{true, true, true},
                /*cached=*/cached,
                /*threads=*/threads,
                /*read=*/read,
            });
          }
        }
      }
    }
  }
} register_benchmarks_;

}  // namespace
