// Copyright 2023 The TensorStore Authors
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

// This benchmarks reading and writing 3-d volumes with the zarr3 driver.
//
// Both read and write benchmarks have 4 parameters:
//
// BM_Read/<total_size>/<write_chunk_size>/<read_chunk_size>/<parallelism>
// BM_Write/<total_size>/<write_chunk_size>/<read_chunk_size>/<parallelism>
//
// total_size:
//
//   Indicates a volume of shape `total_size^3`.
//
// write_chunk_size:
//
//   Indicates a top-level chunk of shape `write_chunk_size^3`.
//
// read_chunk_size:
//
//   Indicates a sub-chunk shape of `read_chunk_size^3`.  If this is equal to
//   `write_chunk_size`, then sharding is not used.
//
// parallelism:
//
//   Indicates that the first dimension of the array should be evenly split into
//   `parallelism` partitions, and parallel read or write operations are issued
//   separately for each partition.

#include <stdint.h>

#include <vector>

#include <benchmark/benchmark.h>
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/progress.h"
#include "tensorstore/schema.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace {

using ::tensorstore::ChunkLayout;
using ::tensorstore::Dims;
using ::tensorstore::dtype_v;
using ::tensorstore::Future;
using ::tensorstore::Index;
using ::tensorstore::Schema;
using ::tensorstore::Spec;
using ::tensorstore::WriteFutures;

struct BenchmarkHelper {
  explicit BenchmarkHelper(Index total_size, Index write_chunk_size,
                           Index read_chunk_size)
      : shape(3, total_size) {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(spec, Spec::FromJson({
                                              {"driver", "zarr3"},
                                              {"kvstore", "memory://"},
                                          }));
    TENSORSTORE_CHECK_OK(
        spec.Set(dtype_v<uint8_t>,
                 Schema::FillValue(tensorstore::MakeScalarArray<uint8_t>(42)),
                 Schema::Shape(shape),
                 ChunkLayout::WriteChunkShape(
                     {write_chunk_size, write_chunk_size, write_chunk_size}),
                 ChunkLayout::ReadChunkShape(
                     {read_chunk_size, read_chunk_size, read_chunk_size}),
                 tensorstore::OpenMode::create));
    source_data = tensorstore::AllocateArray(
        shape, tensorstore::c_order, tensorstore::value_init, spec.dtype());
    total_bytes = source_data.num_elements() * source_data.dtype().size();
  }
  tensorstore::Spec spec;
  const std::vector<Index> shape;
  tensorstore::SharedArray<void> source_data;
  int64_t total_bytes;
};

void BM_Write(benchmark::State& state) {
  BenchmarkHelper helper{state.range(0), state.range(1), state.range(2)};
  const int top_level_parallelism = state.range(3);
  const auto size0 = helper.source_data.shape()[0];
  const auto block_size = size0 / top_level_parallelism;
  for (auto s : state) {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store,
                                    tensorstore::Open(helper.spec).result());
    std::vector<WriteFutures> write_futures(top_level_parallelism);
    for (int i = 0; i < top_level_parallelism; ++i) {
      write_futures[i] = tensorstore::Write(
          helper.source_data |
              Dims(0).HalfOpenInterval(block_size * i, block_size * (i + 1)),
          store |
              Dims(0).HalfOpenInterval(block_size * i, block_size * (i + 1)));
      write_futures[i].Force();
    }
    for (int i = 0; i < top_level_parallelism; ++i) {
      TENSORSTORE_CHECK_OK(write_futures[i].result());
    }
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          helper.total_bytes);
}

void BM_Read(benchmark::State& state) {
  BenchmarkHelper helper{state.range(0), state.range(1), state.range(2)};
  const int top_level_parallelism = state.range(3);
  const auto size0 = helper.source_data.shape()[0];
  const auto block_size = size0 / top_level_parallelism;
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store,
                                  tensorstore::Open(helper.spec).result());
  TENSORSTORE_CHECK_OK(tensorstore::Write(helper.source_data, store).result());
  for (auto s : state) {
    std::vector<Future<const void>> read_futures(top_level_parallelism);
    for (int i = 0; i < top_level_parallelism; ++i) {
      read_futures[i] = tensorstore::Read(
          store |
              Dims(0).HalfOpenInterval(block_size * i, block_size * (i + 1)),
          helper.source_data |
              Dims(0).HalfOpenInterval(block_size * i, block_size * (i + 1)));
    }
    for (int i = 0; i < top_level_parallelism; ++i) {
      TENSORSTORE_CHECK_OK(read_futures[i].result());
    }
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          helper.total_bytes);
}

template <typename Bench>
void DefineArgs(Bench* bench) {
  for (auto chunk_size : {16, 32, 64, 128}) {
    for (auto parallelism : {1, 8}) {
      bench->Args({1024, chunk_size, chunk_size, parallelism});
      bench->Args({1024, 1024, chunk_size, parallelism});
    }
  }
  bench->UseRealTime();
}

BENCHMARK(BM_Write)->Apply(DefineArgs);
BENCHMARK(BM_Read)->Apply(DefineArgs);

}  // namespace
