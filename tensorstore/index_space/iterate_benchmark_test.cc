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

#include <ostream>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/generic_stringify.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::GenericStringify;
using ::tensorstore::Index;
using ::tensorstore::IterationConstraints;
using ::tensorstore::span;
using ::tensorstore::TransformedArray;

struct BenchmarkArrayConfig {
  std::vector<Index> shape;
  std::vector<DimensionIndex> order;
  std::vector<bool> indexed;
};

template <typename Sink>
void AbslStringify(Sink& sink, const BenchmarkArrayConfig& config) {
  absl::Format(&sink, "s=%v, o=%v, i=%v", GenericStringify(config.shape),
               GenericStringify(config.order),
               GenericStringify(std::vector<int>(config.indexed.begin(),
                                                 config.indexed.end())));
}

tensorstore::TransformedSharedArray<char> Allocate(
    const BenchmarkArrayConfig& config, span<const Index> copy_shape) {
  const DimensionIndex rank = config.shape.size();
  ABSL_CHECK(rank == static_cast<DimensionIndex>(config.order.size()) &&
             rank == static_cast<DimensionIndex>(config.indexed.size()));
  std::vector<Index> alloc_shape(rank);
  for (DimensionIndex i = 0; i < rank; ++i) {
    alloc_shape[config.order[i]] = config.shape[i];
  }
  auto array =
      tensorstore::AllocateArray<char>(alloc_shape, tensorstore::c_order);
  auto transform =
      tensorstore::Dims(config.order)
          .Transpose()
          .SizedInterval(0, copy_shape)(IdentityTransformLike(array))
          .value();
  for (DimensionIndex i = 0; i < rank; ++i) {
    if (config.indexed[i]) {
      auto index_array = tensorstore::AllocateArray<Index>({copy_shape[i]});
      for (DimensionIndex j = 0; j < copy_shape[i]; ++j) index_array(j) = j;
      transform = tensorstore::Dims(i)
                      .OuterIndexArraySlice(index_array)(std::move(transform))
                      .value();
    }
  }
  return MakeTransformedArray(std::move(array), std::move(transform)).value();
}

struct BenchmarkConfig {
  std::vector<Index> copy_shape;
  IterationConstraints constraints;
  BenchmarkArrayConfig source, dest;
};

template <typename Sink>
void AbslStringify(Sink& sink, const BenchmarkConfig& config) {
  absl::Format(&sink, "%v", GenericStringify(config.copy_shape));
  if (config.constraints.order_constraint()) {
    absl::Format(&sink, "[%v]", config.constraints.order_constraint().order());
  }
  absl::Format(&sink, ": src={%v}, dst={%v}", config.source, config.dest);
}

void BenchmarkCopy(const BenchmarkConfig& config, ::benchmark::State& state) {
  const DimensionIndex rank = config.copy_shape.size();
  ABSL_CHECK(rank == static_cast<DimensionIndex>(config.source.shape.size()) &&
             rank == static_cast<DimensionIndex>(config.source.order.size()));
  auto source = Allocate(config.source, config.copy_shape);
  auto dest = Allocate(config.dest, config.copy_shape);

  const Index num_elements =
      tensorstore::ProductOfExtents(span(config.copy_shape));

  while (state.KeepRunningBatch(num_elements)) {
    ABSL_CHECK(IterateOverTransformedArrays(
        [&](const char* source_ptr, char* dest_ptr) {
          *dest_ptr = *source_ptr;
        },
        config.constraints, source, dest));
  }
}

struct RegisterIterateBenchmarks {
  static void Register(const BenchmarkConfig& config) {
    ::benchmark::RegisterBenchmark(
        absl::StrCat("IterateOverTransformedArrays: ", config).c_str(),
        [config](auto& state) { BenchmarkCopy(config, state); });
  }

  RegisterIterateBenchmarks() {
    for (const Index size : {16, 32, 64, 128}) {
      Register({
          /*copy_shape=*/{size, size, size},
          /*constraints=*/{},
          /*source=*/
          {/*shape=*/{size, size, size},
           /*order=*/{0, 1, 2},
           /*indexed=*/{false, false, false}},
          /*dest=*/
          {/*shape=*/{size, size, size},
           /*order=*/{0, 1, 2},
           /*indexed=*/{false, false, false}},
      });

      Register({
          /*copy_shape=*/{size, size, size},
          /*constraints=*/{},
          /*source=*/
          {/*shape=*/{size, size, size},
           /*order=*/{0, 1, 2},
           /*indexed=*/{false, false, false}},
          /*dest=*/
          {/*shape=*/{size, size, size},
           /*order=*/{2, 1, 0},
           /*indexed=*/{false, false, false}},
      });

      Register({
          /*copy_shape=*/{size, size, size},
          /*constraints=*/{},
          /*source=*/
          {/*shape=*/{size, size, size},
           /*order=*/{0, 1, 2},
           /*indexed=*/{false, false, true}},
          /*dest=*/
          {/*shape=*/{size, size, size},
           /*order=*/{0, 1, 2},
           /*indexed=*/{false, false, true}},
      });

      Register({
          /*copy_shape=*/{size, size, size},
          /*constraints=*/tensorstore::c_order,
          /*source=*/
          {/*shape=*/{size, size, size},
           /*order=*/{0, 1, 2},
           /*indexed=*/{false, false, true}},
          /*dest=*/
          {/*shape=*/{size, size, size},
           /*order=*/{0, 1, 2},
           /*indexed=*/{false, false, true}},
      });

      Register({
          /*copy_shape=*/{size, size, size},
          /*constraints=*/{},
          /*source=*/
          {/*shape=*/{size, size, size},
           /*order=*/{0, 1, 2},
           /*indexed=*/{false, true, false}},
          /*dest=*/
          {/*shape=*/{size, size, size},
           /*order=*/{0, 1, 2},
           /*indexed=*/{false, true, false}},
      });

      Register({
          /*copy_shape=*/{size, size, size},
          /*constraints=*/{},
          /*source=*/
          {/*shape=*/{size, size, size},
           /*order=*/{0, 1, 2},
           /*indexed=*/{true, false, false}},
          /*dest=*/
          {/*shape=*/{size, size, size},
           /*order=*/{0, 1, 2},
           /*indexed=*/{true, false, false}},
      });
    }
  }
} register_iterate_benchmarks_;

}  // namespace
