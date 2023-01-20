// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/concurrent_testutil.h"

#include <atomic>
#include <type_traits>

#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"

namespace {

using ::tensorstore::internal::TestConcurrent;

TEST(TestConcurrent, EnsureContentionHappens) {
  // This test attempts to measure contention. Note, however, contention is
  // stochastic, and so running this multiple times is necessary to get a
  // handle on behavior. Empirically, on my linux workstation:
  // * The lock is contended about 1 in 10 iterations, on average.
  // * The lock is uncontended about 1 in 500 test runs, on average. (flaky).

  static constexpr int kIterations = 100;
  static constexpr int kN = 20;

  absl::Mutex lock;
  int uncontended{0};

  TestConcurrent<kN>(
      /*num_iterations=*/kIterations,
      /*initialize=*/[&] {},
      /*finalize=*/[&] {},  //
      [&](auto) {
        if (lock.TryLock()) {
          uncontended++;
          lock.Unlock();
        }
      });
  int contended = (kIterations * kN) - uncontended;
  ABSL_LOG(INFO) << "Contended in " << contended << " of 2000 iterations.";

  // This would be flaky; see above.
  // EXPECT_NE(0, contended);
}

TEST(TestConcurrent, Example1) {
  static constexpr int kIterations = 100;

  std::atomic<int> sum{0};
  TestConcurrent(
      /*num_iterations=*/kIterations,
      /*initialize=*/[&] {},
      /*finalize=*/[&] {},
      /* concurrent_ops = ... */
      [&]() { sum += 1; }, [&]() { sum += 2; }, [&]() { sum += 3; });

  EXPECT_EQ(100 + 200 + 300, sum);
}

/// Fixture to run the example for 1, 4, and 16 concurrent operations.
template <typename T>
struct TestConcurrentFixture : public ::testing::Test {};

using ConcurrentOpSizes = ::testing::Types<std::integral_constant<int, 1>,
                                           std::integral_constant<int, 4>,
                                           std::integral_constant<int, 16>>;

TYPED_TEST_SUITE(TestConcurrentFixture, ConcurrentOpSizes);

TYPED_TEST(TestConcurrentFixture, Example2) {
  static constexpr int kN = TypeParam{}();
  static constexpr int kIterations = 100;

  std::atomic<int> sum{0};
  TestConcurrent<kN>(
      /*num_iterations=*/kIterations,
      /*initialize=*/[&] {},
      /*finalize=*/[&] {}, [&](auto i) { sum += (i + 1); });

  EXPECT_EQ((kIterations / 2) * kN * (kN + 1), sum);
}

}  // namespace
