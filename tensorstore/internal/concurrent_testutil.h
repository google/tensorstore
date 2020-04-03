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

#ifndef TENSORSTORE_INTERNAL_CONCURRENT_TESTUTIL_H_
#define TENSORSTORE_INTERNAL_CONCURRENT_TESTUTIL_H_

#include <atomic>
#include <cstddef>
#include <ctime>
#include <thread>  // NOLINT

namespace tensorstore {
namespace internal {

/// Repeatedly calls `initialize()`, then calls `concurrent_ops()...`
/// concurrently from separate threads, then calls `finalize()` after the
/// `concurrent_ops` finish.
///
/// This is repeated `num_iterations` times.
template <typename Initialize, typename Finalize, typename... ConcurrentOps>
void TestConcurrent(std::size_t num_iterations, Initialize initialize,
                    Finalize finalize, ConcurrentOps... concurrent_ops) {
  std::atomic<std::size_t> counter(0);
  // One count per concurrent op, plus one for initialize/finalize.
  constexpr std::size_t counts_per_iteration = sizeof...(ConcurrentOps) + 1;
  // Start one thread for each concurrent operation.
  std::thread threads[]{std::thread([&] {
    for (std::size_t iteration = 0; iteration < num_iterations; ++iteration) {
      // Spin until `initialize` has run for this iteration.
      while (counter.load() < iteration * counts_per_iteration + 1) continue;
      concurrent_ops();
      ++counter;
    }
  })...};
  for (std::size_t iteration = 0; iteration < num_iterations; ++iteration) {
    initialize();
    ++counter;
    // Spin until all concurrent operations have run for this iteration.
    while (counter.load() < (iteration + 1) * counts_per_iteration) continue;
    finalize();
  }
  for (auto& t : threads) t.join();
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CONCURRENT_TESTUTIL_H_
