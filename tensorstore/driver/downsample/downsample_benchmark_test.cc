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

#include <vector>

#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include <benchmark/benchmark.h>
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/downsample/downsample_array.h"
#include "tensorstore/driver/downsample/downsample_nditerable.h"
#include "tensorstore/driver/downsample/downsample_util.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/data_type_random_generator.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::Box;
using ::tensorstore::BoxView;
using ::tensorstore::DataType;
using ::tensorstore::DimensionIndex;
using ::tensorstore::DownsampleMethod;
using ::tensorstore::Index;
using ::tensorstore::internal_downsample::DownsampleArray;
using ::tensorstore::internal_downsample::DownsampleBounds;

void BenchmarkDownsample(::benchmark::State& state, DataType dtype,
                         DownsampleMethod downsample_method,
                         DimensionIndex rank, Index downsample_factor,
                         Index block_size) {
  std::vector<Index> downsample_factors(rank, downsample_factor);
  std::vector<Index> block_shape(rank, block_size);
  absl::BitGen gen;
  BoxView<> base_domain(block_shape);
  auto base_array =
      tensorstore::internal::MakeRandomArray(gen, base_domain, dtype);
  Box<> downsampled_domain(rank);
  DownsampleBounds(base_domain, downsampled_domain, downsample_factors,
                   downsample_method);
  auto downsampled_array =
      tensorstore::AllocateArray(downsampled_domain, tensorstore::c_order,
                                 tensorstore::default_init, dtype);
  const Index num_elements = base_domain.num_elements();
  Index total_elements = 0;
  while (state.KeepRunningBatch(num_elements)) {
    ABSL_CHECK(DownsampleArray(base_array, downsampled_array,
                               downsample_factors, downsample_method)
                   .ok());
    total_elements += num_elements;
  }
  state.SetItemsProcessed(total_elements);
}

TENSORSTORE_GLOBAL_INITIALIZER {
  for (const DataType dtype : tensorstore::kDataTypes) {
    for (const DownsampleMethod downsample_method :
         {DownsampleMethod::kStride, DownsampleMethod::kMean,
          DownsampleMethod::kMedian, DownsampleMethod::kMode,
          DownsampleMethod::kMin,
          // Skip `kMax` because it is redundant with `kMin`.
          /*DownsampleMethod::kMax*/}) {
      if (!tensorstore::internal_downsample::IsDownsampleMethodSupported(
              dtype, downsample_method)) {
        continue;
      }
      for (const DimensionIndex rank : {1, 2, 3}) {
        for (const Index downsample_factor : {2, 3}) {
          for (const Index block_size : {16, 32, 64, 128, 256}) {
            ::benchmark::RegisterBenchmark(
                tensorstore::StrCat("DownsampleArray_", dtype, "_",
                                    downsample_method, "_Rank", rank, "_Factor",
                                    downsample_factor, "_BlockSize", block_size)
                    .c_str(),
                [=](auto& state) {
                  BenchmarkDownsample(state, dtype, downsample_method, rank,
                                      downsample_factor, block_size);
                });
          }
        }
      }
    }
  }
}

}  // namespace
