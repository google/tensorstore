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

#ifndef THIRD_PARTY_TENSORSTORE_INTERNAL_THREAD_THREAD_POOL_BENCHMARK_INC_
#define THIRD_PARTY_TENSORSTORE_INTERNAL_THREAD_THREAD_POOL_BENCHMARK_INC_

#include <stddef.h>

#include <algorithm>
#include <vector>

#include "testing/base/public/benchmark.h"
#include "absl/random/random.h"
#include "absl/synchronization/blocking_counter.h"
#include "tensorstore/util/executor.h"

namespace {

using ::tensorstore::Executor;
using ::tensorstore::internal::DetachedThreadPool;

// This is a benchmark which enqueues a lot of items into a thread pool.
// it doesn't have a goal of implementing an efficient matrix multiply; the
// multiply is incidental.
static void BM_ThreadPool_MatrixMultiply(benchmark::State& state) {
  const size_t sz = state.range(0);
  std::vector<float> A(sz * sz);
  std::vector<float> B(sz * sz);
  std::vector<float> C(sz * sz);

  absl::BitGen rng;
  std::generate(A.begin(), A.end(),
                [&] { return absl::Gaussian<float>(rng, 0, 1); });
  std::generate(B.begin(), B.end(),
                [&] { return absl::Gaussian<float>(rng, 0, 1); });

  for (auto s : state) {
    absl::BlockingCounter done(sz * sz);
    auto executor = DetachedThreadPool(state.range(1));
    for (size_t i = 0; i < sz; i++) {
      executor([&, i] {
        for (size_t j = 0; j < sz; ++j) {
          executor([&, i, j] {
            float sum = 0;
            for (size_t l = 0; l < sz; ++l) {
              sum += A[i + l * sz] * B[l + j * sz];
            }
            C[i + j * sz] = sum;
            done.DecrementCount();
          });
        }
      });
    }
    done.Wait();
  }
  state.SetItemsProcessed(state.iterations() * sz * sz);
}

BENCHMARK(BM_ThreadPool_MatrixMultiply)  //
    ->Args({256, 12})
    ->Args({1024, 12})
    ->Args({1024, 24})
    ->Args({2048, 24})
    ->Args({2048, 36})
    ->UseRealTime();

}  // namespace

#endif  // THIRD_PARTY_TENSORSTORE_INTERNAL_THREAD_THREAD_POOL_BENCHMARK_INC_
