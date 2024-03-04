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

#include "tensorstore/internal/rate_limiter/scaling_rate_limiter.h"

#include <stddef.h>

#include <atomic>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/rate_limiter/rate_limiter.h"
#include "tensorstore/util/executor.h"

namespace {

using ::tensorstore::Executor;
using ::tensorstore::ExecutorTask;
using ::tensorstore::internal::adopt_object_ref;
using ::tensorstore::internal::AtomicReferenceCount;
using ::tensorstore::internal::ConstantRateLimiter;
using ::tensorstore::internal::DoublingRateLimiter;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::MakeIntrusivePtr;
using ::tensorstore::internal::RateLimiter;
using ::tensorstore::internal::RateLimiterNode;

struct Node : public RateLimiterNode, public AtomicReferenceCount<Node> {
  RateLimiter* queue_;
  ExecutorTask task_;

  Node(RateLimiter* queue, ExecutorTask task)
      : queue_(queue), task_(std::move(task)) {}

  ~Node() { queue_->Finish(this); }

  static void Start(void* task) {
    IntrusivePtr<Node> self(reinterpret_cast<Node*>(task), adopt_object_ref);
    std::move(self->task_)();
  }
};

TEST(ConstantRateLimiter, Basic) {
  absl::Time now = absl::Now();
  ConstantRateLimiter queue(0.2, [&now]() { return now; });

  EXPECT_EQ(0.2, queue.initial_rate());
  EXPECT_EQ(now, queue.start_time());
  EXPECT_EQ(now, queue.last_update());

  EXPECT_EQ(0, queue.available());

  EXPECT_EQ(0, queue.TokensToAdd(now, now));
  EXPECT_EQ(2, queue.TokensToAdd(now + absl::Seconds(10), now));
  EXPECT_EQ(60, queue.TokensToAdd(now + absl::Seconds(300), now));

  std::atomic<size_t> done{0};
  {
    for (int i = 0; i < 100; i++) {
      auto node = MakeIntrusivePtr<Node>(&queue, [&done] {  //
        done++;
      });

      intrusive_ptr_increment(node.get());  // adopted by Node::Start.
      queue.Admit(node.get(), &Node::Start);
    }
  }

  now += absl::Seconds(10);
  queue.PeriodicCallForTesting();
  EXPECT_EQ(2, done);

  now += absl::Seconds(100);
  queue.PeriodicCallForTesting();
  EXPECT_EQ(22, done);

  now += absl::Seconds(400);
  queue.PeriodicCallForTesting();
  EXPECT_EQ(100, done);
}

TEST(DoublingRateLimiter, Basic) {
  absl::Time now = absl::Now();

  DoublingRateLimiter queue(2, absl::Seconds(10), [&now]() { return now; });

  EXPECT_EQ(2, queue.initial_rate());
  EXPECT_EQ(absl::Seconds(10), queue.doubling_time());
  EXPECT_EQ(0, queue.available());

  EXPECT_EQ(0, queue.TokensToAdd(now, now));

  // Second 10 gets ~4 tokens
  EXPECT_THAT(
      queue.TokensToAdd(now + absl::Seconds(11), now + absl::Seconds(10)),
      ::testing::Gt(4));

  // Second 20 gets ~8 tokens
  EXPECT_THAT(
      queue.TokensToAdd(now + absl::Seconds(21), now + absl::Seconds(20)),
      ::testing::Gt(8));

  std::atomic<size_t> done{0};
  {
    for (int i = 0; i < 100; i++) {
      auto node = MakeIntrusivePtr<Node>(&queue, [&done] {  //
        done++;
      });

      intrusive_ptr_increment(node.get());  // adopted by Node::Start.
      queue.Admit(node.get(), &Node::Start);
    }
  }
  EXPECT_EQ(0, done);

  now += absl::Seconds(1);
  queue.PeriodicCallForTesting();
  EXPECT_EQ(2, done);

  now += absl::Seconds(10);
  queue.PeriodicCallForTesting();
  EXPECT_EQ(32, done);

  now += absl::Seconds(20);
  queue.PeriodicCallForTesting();
  EXPECT_EQ(100, done);
}

}  // namespace
