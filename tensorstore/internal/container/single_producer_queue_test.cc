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

#include "tensorstore/internal/container/single_producer_queue.h"

#include <stddef.h>

#include <atomic>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "tensorstore/internal/thread/thread.h"

using ::tensorstore::internal_container::SingleProducerQueue;
using ::testing::Eq;
using ::testing::Optional;

namespace {

TEST(SingleProducerQueueTest, Basic) {
  SingleProducerQueue<int> q(32);

  EXPECT_THAT(q.capacity(), Eq(32));
  EXPECT_THAT(q.empty(), true);
  EXPECT_THAT(q.size(), Eq(0));

  q.push(1);

  EXPECT_THAT(q.capacity(), Eq(32));
  EXPECT_THAT(q.empty(), false);
  EXPECT_THAT(q.size(), Eq(1));

  EXPECT_THAT(q.try_pop(), Optional(1));
  EXPECT_THAT(q.try_pop(), Eq(std::nullopt));

  q.push(2);
  EXPECT_THAT(q.try_steal(), Optional(2));
  EXPECT_THAT(q.try_pop(), Eq(std::nullopt));
}

TEST(SingleProducerQueueTest, BasicPtr) {
  SingleProducerQueue<int*> q(32);
  int a[2];

  EXPECT_THAT(q.capacity(), Eq(32));
  EXPECT_THAT(q.empty(), true);
  EXPECT_THAT(q.size(), Eq(0));

  q.push(&a[0]);

  EXPECT_THAT(q.capacity(), Eq(32));
  EXPECT_THAT(q.empty(), false);
  EXPECT_THAT(q.size(), Eq(1));

  EXPECT_THAT(q.try_pop(), Eq(a));
  EXPECT_THAT(q.try_pop(), Eq(nullptr));

  q.push(&a[1]);
  EXPECT_THAT(q.try_steal(), Eq(a + 1));
  EXPECT_THAT(q.try_pop(), Eq(nullptr));
}

TEST(SimpleQueue, PushPop) {
  SingleProducerQueue<int, false> q(1);

  for (int i = 0; i < 4096; i++) {
    if (!q.push(i)) {
      q.try_pop();
      q.push(i);
      if (i & 0x2) q.try_pop();
    }
  }
}

TEST(SingleProducerQueueTest, ConcurrentSteal) {
  static constexpr size_t kNumThreads = 4;
  static constexpr int kSize = 10000;

  SingleProducerQueue<int> q(32);
  std::atomic<int> remaining(kSize);

  std::vector<tensorstore::internal::Thread> threads;
  threads.reserve(kNumThreads + 1);

  for (size_t thread_i = 0; thread_i < kNumThreads; ++thread_i) {
    bool c = thread_i & 1;
    threads.emplace_back(
        tensorstore::internal::Thread({"steal"}, [c, &remaining, &q]() {
          while (remaining.load(std::memory_order_seq_cst) > 0) {
            if (auto v = q.try_steal(); v.has_value()) {
              ABSL_CHECK_EQ(*v, 1);
              remaining.fetch_sub(1);
            }
            if (c) {
              q.capacity();
            } else {
              q.size();
            }
          }
        }));
  }

  threads.emplace_back(tensorstore::internal::Thread({"create"}, [&q]() {
    for (int i = 0; i < kSize; ++i) q.push(1);
  }));

  for (auto& t : threads) t.Join();

  // Q is empty.
  EXPECT_THAT(remaining.load(std::memory_order_seq_cst), 0);
  EXPECT_THAT(q.try_steal(), Eq(std::nullopt));
}

}  // namespace
