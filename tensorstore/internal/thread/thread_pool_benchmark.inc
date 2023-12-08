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
#include <stdint.h>

#include <algorithm>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/synchronization/blocking_counter.h"
#include "tensorstore/internal/digest/sha256.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/util/executor.h"

namespace {

// static constexpr const char kMetric[] =
// "/tensorstore/thread_pool/total_queue_time_ns";
static constexpr const char kMetric[] = "/tensorstore/thread_pool/steal_count";

using ::tensorstore::Executor;
using ::tensorstore::internal::SHA256Digester;
using ::tensorstore::internal_metrics::GetMetricRegistry;

Executor GetExecutor(size_t num_threads) {
  if (num_threads == 0) {
    return tensorstore::InlineExecutor{};
  }
  return ::tensorstore::internal::DetachedThreadPool(num_threads);
}

void SetLabels(benchmark::State& state, size_t num_threads) {
  if (num_threads == 0) {
    state.SetLabel("InlineExecutor");
  } else {
    auto metric = GetMetricRegistry().Collect(kMetric);
    if (metric) {
      tensorstore::internal_metrics::FormatCollectedMetric(
          *metric, [&state](bool has_value, std::string formatted_line) {
            if (has_value) {
              state.SetLabel(formatted_line);
            }
          });
    }
  }
}

// This is a thread pool benchmark designed to be fully compute bound.
// The task itself computes the sha hash of a random buffer concurrently.
static void BM_ThreadPool_Sha(benchmark::State& state) {
  SetupThreadPoolTestEnv();
  GetMetricRegistry().Reset();

  const size_t size = state.range(0);
  const size_t repetitions = state.range(1);
  const size_t n = state.range(2);

  std::string source(size, 0);
  absl::BitGen rng;
  std::generate(source.begin(), source.end(),
                [&] { return absl::Uniform<uint8_t>(rng); });

  // Pad to avoid sharing cache lines on write.
  struct DigestType {
    union {
      SHA256Digester::DigestType digest;
      char padding[64];
    };
  };

  std::vector<DigestType> digesters(n);
  for (auto s : state) {
    absl::BlockingCounter done(n);
    auto executor = GetExecutor(state.range(3));
    for (size_t i = 0; i < n; i++) {
      executor([&, i] {
        SHA256Digester d;
        for (size_t j = 0; j < repetitions; j++) {
          d.Write(source);
        }
        digesters[i].digest = d.Digest();
        done.DecrementCount();
      });
    }
    done.Wait();
  }

  state.SetItemsProcessed(state.iterations() * n);  // tasks
  state.SetBytesProcessed(state.iterations() * n * repetitions * source.size());
  SetLabels(state, state.range(3));
}

BENCHMARK(BM_ThreadPool_Sha)                //
    ->Args({1024, 4, 1024 * 1024, 0})       // InlineExecutor
    ->Args({1024, 4, 1024 * 1024, 32})      // 4kB, 1M tasks
    ->Args({1024 * 1024, 1, 4 * 1024, 32})  // 1MB, 4k tasks
    ->Args({1024, 4, 1024 * 1024, 1024})    // 4kB, 1M tasks
    ->Args({1024 * 1024, 4, 1024, 1024})    // 4MB, 1k tasks
    ->UseRealTime();

// This is a thread pool benchmark designed to create a lot of tasks with
// large fanout and some memory locality.  The task itself Xors data into a
// buffer by splitting the buffer into N x M x O chunks.
static void BM_ThreadPool_XorData(benchmark::State& state) {
  SetupThreadPoolTestEnv();
  GetMetricRegistry().Reset();

  const size_t total_size = state.range(0) / sizeof(uint64_t);
  const size_t buffer_size = state.range(1) / sizeof(uint64_t);
  ABSL_CHECK_GT(total_size, 0);
  ABSL_CHECK_GT(buffer_size, 0);
  ABSL_CHECK_GT(total_size, buffer_size);

  const size_t n = state.range(2);
  const size_t chunk0 = total_size / n;
  const size_t m = chunk0 / buffer_size;
  const size_t chunk1 = buffer_size;

  std::vector<uint64_t> destination(total_size);
  std::vector<uint64_t> source(buffer_size);

  absl::BitGen rng;
  std::generate(source.begin(), source.end(),
                [&] { return absl::Uniform<uint64_t>(rng); });

  for (auto s : state) {
    auto executor = GetExecutor(state.range(3));
    absl::BlockingCounter done(n * m);
    for (size_t i = 0; i < n; i++) {
      executor([&, i] {
        for (size_t j = 0; j < m; j++) {
          executor([&, i, j] {
            auto* d = destination.data() + (i * chunk0) + (j * chunk1);
            for (size_t k = 0; k < buffer_size; ++k) d[k] ^= source[k];
            done.DecrementCount();
          });
        }
      });
    }
    done.Wait();
  }

  state.SetItemsProcessed(state.iterations() * n * m);  // tasks
  state.SetBytesProcessed(state.iterations() * total_size *
                          sizeof(uint64_t));  // bytes
  SetLabels(state, state.range(3));
}

BENCHMARK(BM_ThreadPool_XorData)                  //
    ->Args({1024 * 1024 * 1024, 256, 1024, 0})    // InlineExecutor
    ->Args({1024 * 1024 * 1024, 256, 1024, 32})   // 1GB x 256-byte writes
    ->Args({1024 * 1024 * 1024, 64, 2048, 32})    // 1GB x 64-byte writes
    ->Args({1024 * 1024 * 1024, 64, 2048, 1024})  // 1GB x 64-byte writes
    ->UseRealTime();

// This is a benchmark which represents a fully memory-bound task. The
// benchmark decomposes a matrix multiply onto a lot of work units on a thread
// pool; the matrix multiply is incidental to the benchmark.
static void BM_ThreadPool_MatrixMultiply_Naive(benchmark::State& state) {
  SetupThreadPoolTestEnv();
  GetMetricRegistry().Reset();
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
    auto executor = GetExecutor(state.range(1));
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
  SetLabels(state, state.range(1));
}

BENCHMARK(BM_ThreadPool_MatrixMultiply_Naive)  //
    ->Args({512, 0})                           // Inline Executor
    ->Args({512, 32})
    ->Args({1024, 32})
    ->Args({2048, 32})
    ->UseRealTime();

}  // namespace

#endif  // THIRD_PARTY_TENSORSTORE_INTERNAL_THREAD_THREAD_POOL_BENCHMARK_INC_
