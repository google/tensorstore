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

#include <stdint.h>

#include <vector>

#include <benchmark/benchmark.h>
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/schema.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace {

using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::Schema;
using ::tensorstore::Spec;

struct BenchmarkHelper {
  explicit BenchmarkHelper(Index sub_chunk_size)
      : shape(3, 1024), sub_chunk_shape(3, sub_chunk_size) {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        spec,
        Spec::FromJson({
            {"driver", "zarr3"},
            {"kvstore", "memory://"},
            {"metadata",
             {
                 {"chunk_grid",
                  {{"name", "regular"},
                   {"configuration", {{"chunk_shape", shape}}}}},
                 {"data_type", "uint8"},
                 {"fill_value", 42},
                 {"codecs",
                  {{{"name", "sharding_indexed"},
                    {"configuration", {{"chunk_shape", sub_chunk_shape}}}}}},
             }},
        }));
    TENSORSTORE_CHECK_OK(
        spec.Set(Schema::Shape(shape), tensorstore::OpenMode::create));
    source_data = tensorstore::AllocateArray(
        shape, tensorstore::c_order, tensorstore::value_init, spec.dtype());
    total_bytes = source_data.num_elements() * source_data.dtype().size();
  }
  tensorstore::Spec spec;
  const std::vector<Index> shape;
  const std::vector<Index> sub_chunk_shape;
  tensorstore::SharedArray<const void> source_data;
  int64_t total_bytes;
};

void BM_Write(benchmark::State& state) {
  BenchmarkHelper helper{state.range(0)};
  for (auto s : state) {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store,
                                    tensorstore::Open(helper.spec).result());
    TENSORSTORE_CHECK_OK(
        tensorstore::Write(helper.source_data, store).result());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          helper.total_bytes);
}

void BM_Read(benchmark::State& state) {
  BenchmarkHelper helper{state.range(0)};
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store,
                                  tensorstore::Open(helper.spec).result());
  TENSORSTORE_CHECK_OK(tensorstore::Write(helper.source_data, store).result());
  for (auto s : state) {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto value,
                                    tensorstore::Read(store).result());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          helper.total_bytes);
}

BENCHMARK(BM_Write)->Arg(32)->Arg(64)->Arg(128)->UseRealTime();
BENCHMARK(BM_Read)->Arg(32)->Arg(64)->Arg(128)->UseRealTime();

}  // namespace
