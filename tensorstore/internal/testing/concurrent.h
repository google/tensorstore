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

#ifndef TENSORSTORE_INTERNAL_TESTING_CONCURRENT_H_
#define TENSORSTORE_INTERNAL_TESTING_CONCURRENT_H_

#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <ctime>
#include <thread>  // NOLINT
#include <type_traits>
#include <utility>

#include "tensorstore/internal/multi_barrier.h"
#include "tensorstore/internal/thread/thread.h"

namespace tensorstore {
namespace internal_testing {

#ifdef _WIN32
/// On Windows, prevents concurrent invocations of `TestConcurrent` (including
/// across multiple processes) to avoid test timeouts.
class TestConcurrentLock {
 public:
  TestConcurrentLock();
  ~TestConcurrentLock();

  TestConcurrentLock(TestConcurrentLock&& other) noexcept = delete;
  TestConcurrentLock& operator=(TestConcurrentLock&& other) = delete;
  TestConcurrentLock(const TestConcurrentLock& other) = delete;
  TestConcurrentLock& operator=(const TestConcurrentLock& other) = delete;

 private:
  void* handle_;
};
#endif

/// Repeatedly calls `initialize()`, then calls `concurrent_ops()...`
/// concurrently from separate threads, then calls `finalize()` after the
/// `concurrent_ops` finish.
///
/// This is repeated `num_iterations` times.
///
/// Example:
///
///   std::atomic<int> sum{0};
///   TestConcurrent(
///      /*num_iterations=*/100,
///      /*initialize=*/[&] {},
///      /*finalize=*/[&] {},
///      /* concurrent_ops = ... */
///      [&]() { sum += 1; },
///      [&]() { sum += 2; },
///      [&]() { sum += 3; });
///
template <typename Initialize, typename Finalize, typename... ConcurrentOps>
void TestConcurrent(size_t num_iterations, Initialize initialize,
                    Finalize finalize, ConcurrentOps... concurrent_ops) {
#ifdef _WIN32
  TestConcurrentLock lock;
#endif
  std::atomic<size_t> counter(0);
  constexpr size_t concurrent_op_size = sizeof...(ConcurrentOps);

  // The synchronization strategy used here is to introduce sync points
  // between the {initialize, op, finalize} functions using a barrier that
  // blocks until all threads + caller have arrived each point.
  internal::MultiBarrier sync_point(1 + concurrent_op_size);

  // In addition, try to increase the contention while running concurrent_ops.
  // This is a best-effort heueristic; to do it properly we'd need a mechanism
  // to tell the OS scheduler to start the threads at the same time after waking
  // from a mutex, but alas. Instead, busy-wait in an attempt to sync up to 4
  // threads at once. Empirically, on my linux workstation, yada yada,
  // concurrent_op contends a mutex in fewer than 1 in 200 iterations without
  // this busy-wait. With this busy-wait contention increases to approximately 1
  // in 10 iterations.
  size_t sync_mask = std::min(4u, std::thread::hardware_concurrency()) - 1;
  if (sync_mask == 2) sync_mask--;

  // Start one thread for each concurrent operation.
  internal::Thread threads[]{internal::Thread({"concurrent"}, [&] {
    for (size_t iteration = 0; iteration < num_iterations; ++iteration) {
      // Wait until `initialize` has run for this iteration.
      sync_point.Block();

      // See above: busy-wait to increase contention.
      size_t current = counter.fetch_add(1, std::memory_order_acq_rel) + 1;
      size_t target = std::min(current | sync_mask, concurrent_op_size);
      while (counter.load() < target) {
        // Allow another thread a chance to run.
        std::this_thread::yield();
      }
      concurrent_ops();

      // Signal that the op() has completed.
      sync_point.Block();
    }
  })...};

  for (size_t iteration = 0; iteration < num_iterations; ++iteration) {
    initialize();
    counter = 0;

    sync_point.Block();
    // Wait until the op() has run.
    sync_point.Block();

    finalize();
  }

  for (auto& t : threads) {
    t.Join();
  }
}

/// Repeatedly calls `initialize()`, then calls `concurrent_op(Is)...` for
/// each index in Is concurrently from separate threads, then calls
/// `finalize()` after the `concurrent_ops` finish.
///
/// This is repeated `num_iterations` times.
template <typename Initialize, typename Finalize, typename ConcurrentOp,
          size_t... Is>
void TestConcurrent(std::index_sequence<Is...>, size_t num_iterations,
                    Initialize initialize, Finalize finalize,
                    ConcurrentOp concurrent_op) {
  TestConcurrent(
      num_iterations, std::move(initialize), std::move(finalize),
      [&] { concurrent_op(std::integral_constant<size_t, Is>{}); }...);
}

/// Repeatedly calls `initialize()`, then calls `concurrent_op(0), ...,
/// concurrent_op(NumConcurrentOps-1)` concurrently from separate threads,
/// then calls `finalize()` after the `concurrent_ops` finish.
///
/// This is repeated `num_iterations` times.
///
/// Example:
///
///   std::atomic<int> sum{0};
///   TestConcurrent<3>(
///      /*num_iterations=*/100,
///      /*initialize=*/[&] {},
///      /*finalize=*/[&] {},
///      [&](auto i) { sum += i; });
///
template <size_t NumConcurrentOps, typename Initialize, typename Finalize,
          typename ConcurrentOp>
void TestConcurrent(size_t num_iterations, Initialize initialize,
                    Finalize finalize, ConcurrentOp concurrent_op) {
  return TestConcurrent(std::make_index_sequence<NumConcurrentOps>{},
                        num_iterations, std::move(initialize),
                        std::move(finalize), std::move(concurrent_op));
}

}  // namespace internal_testing
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_TESTING_CONCURRENT_H_
