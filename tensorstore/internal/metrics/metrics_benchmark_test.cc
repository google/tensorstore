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

#ifndef TENSORSTORE_METRICS_DISABLED

#include <stddef.h>
#include <stdint.h>

#include <benchmark/benchmark.h>
#include "absl/synchronization/blocking_counter.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/internal/thread/thread_pool.h"
#include "tensorstore/util/executor.h"

namespace {

using ::tensorstore::Executor;
using ::tensorstore::internal_metrics::Counter;
using ::tensorstore::internal_metrics::GetMetricRegistry;

Executor SetupThreadPoolTestEnv(size_t num_threads) {
  GetMetricRegistry().Reset();
  if (num_threads == 0) {
    return ::tensorstore::InlineExecutor{};
  }
  return ::tensorstore::internal::DetachedThreadPool(num_threads);
}

static auto& benchmark_counter_int64 =
    Counter<int64_t>::New("/tensorstore/benchmark/counter_int64", "A metric");

static auto& benchmark_counter_double =
    Counter<double>::New("/tensorstore/benchmark/counter_double", "A metric");

// This is a thread pool benchmark designed to create a lot of tasks with
// large fanout and some memory locality.  The task itself Xors data into a
// buffer by splitting the buffer into N x M x O chunks.
static void BM_Metric_Counter(benchmark::State& state) {
  const size_t ops = 16 * 1024 * 1024;
  const size_t num_threads = state.range(0) ? state.range(0) : 1;
  const size_t iters = ops / num_threads;

  auto executor = SetupThreadPoolTestEnv(state.range(0));

  for (auto s : state) {
    absl::BlockingCounter done(num_threads);
    for (size_t i = 0; i < num_threads; i++) {
      executor([&done, iters] {
        for (size_t j = 0; j < iters; j++) {
          benchmark_counter_int64.Increment();
          benchmark_counter_double.IncrementBy(1.1);
        }
        done.DecrementCount();
      });
    }
    done.Wait();
  }

  state.SetItemsProcessed(state.iterations() * iters * num_threads);
}

BENCHMARK(BM_Metric_Counter)  //
    ->Args({0})               // InlineExecutor
    ->Args({8})               //
    ->Args({32})              //
    ->Args({256})             //
    ->UseRealTime();

}  // namespace

#endif  // !defined(TENSORSTORE_METRICS_DISABLED)
