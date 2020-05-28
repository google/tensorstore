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

#ifdef _WIN32
/// On Windows, prevents concurrent invocations of `TestConcurrent` (including
/// across multiple processes) to avoid test timeouts.
class TestConcurrentLock {
 public:
  TestConcurrentLock();
  ~TestConcurrentLock();

 private:
  void* mutex_;
};
#endif

/// Repeatedly calls `initialize()`, then calls `concurrent_ops()...`
/// concurrently from separate threads, then calls `finalize()` after the
/// `concurrent_ops` finish.
///
/// This is repeated `num_iterations` times.
template <typename Initialize, typename Finalize, typename... ConcurrentOps>
void TestConcurrent(std::size_t num_iterations, Initialize initialize,
                    Finalize finalize, ConcurrentOps... concurrent_ops) {
#ifdef _WIN32
  TestConcurrentLock lock;
#endif
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

/// Same as above, but invokes `concurrent_op(Is)` concurrently for each index
/// in `Is` rather than invoking separate independently-specified
/// `concurrent_ops` functions.
template <typename Initialize, typename Finalize, typename ConcurrentOp,
          size_t... Is>
void TestConcurrent(std::index_sequence<Is...>, std::size_t num_iterations,
                    Initialize initialize, Finalize finalize,
                    ConcurrentOp concurrent_op) {
  TestConcurrent(
      num_iterations, std::move(initialize), std::move(finalize),
      [&] { concurrent_op(std::integral_constant<size_t, Is>{}); }...);
}

/// Same as above, but invokes
/// `concurrent_op(0), ..., concurrent_op(NumConcurrentOps-1)` concurrently.
template <size_t NumConcurrentOps, typename Initialize, typename Finalize,
          typename ConcurrentOp>
void TestConcurrent(std::size_t num_iterations, Initialize initialize,
                    Finalize finalize, ConcurrentOp concurrent_op) {
  return TestConcurrent(std::make_index_sequence<NumConcurrentOps>{},
                        num_iterations, std::move(initialize),
                        std::move(finalize), std::move(concurrent_op));
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CONCURRENT_TESTUTIL_H_
