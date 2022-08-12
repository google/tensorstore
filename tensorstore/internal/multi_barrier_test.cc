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

#include "tensorstore/internal/multi_barrier.h"

#include <type_traits>

#include <gtest/gtest.h>
#include "tensorstore/internal/thread.h"

namespace internal = tensorstore::internal;

namespace {

template <typename T>
struct MultiBarrierFixture : public ::testing::Test {};

using NumThreadTypes = ::testing::Types<std::integral_constant<int, 1>,
                                        std::integral_constant<int, 2>,
                                        std::integral_constant<int, 16>>;

TYPED_TEST_SUITE(MultiBarrierFixture, NumThreadTypes);

TYPED_TEST(MultiBarrierFixture, Example) {
  constexpr int kIterations = 1000;
  constexpr int kNumThreads = TypeParam{}();

  internal::MultiBarrier barrier(kNumThreads);
  std::atomic<int> winner[kNumThreads] = {};
  std::atomic<int> loser[kNumThreads] = {};

  internal::Thread threads[kNumThreads];
  for (int i = 0; i < kNumThreads; i++) {
    threads[i] = internal::Thread({"sanity"}, [&, id = i]() {
      for (int j = 0; j < kIterations; j++) {
        if (barrier.Block()) {
          winner[id]++;
        } else {
          loser[id]++;
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.Join();
  }

  int sum = 0;
  for (auto& x : winner) {
    sum += x;
  }
  EXPECT_EQ(kIterations, sum);

  sum = 0;
  for (auto& x : loser) {
    sum += x;
  }
  EXPECT_EQ(kIterations * (kNumThreads - 1), sum);
}

}  // namespace
