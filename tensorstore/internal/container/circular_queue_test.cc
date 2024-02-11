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

#include "tensorstore/internal/container/circular_queue.h"

#include <stdint.h>

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal_container::CircularQueue;

TEST(CircularQueue, Basic) {
  CircularQueue<int64_t> q(2);
  EXPECT_TRUE(q.empty());
  EXPECT_THAT(q.size(), 0);

  q.push_back(10);
  EXPECT_FALSE(q.empty());
  EXPECT_EQ(q.front(), 10);
  EXPECT_EQ(q.back(), 10);

  q.pop_front();
  EXPECT_TRUE(q.empty());

  for (int i = 0; i < 10; ++i) {
    q.push_back(i);
  }
  EXPECT_FALSE(q.empty());

  q.clear();
  EXPECT_TRUE(q.empty());
}

TEST(CircularQueue, BasicWithSharedPtr) {
  CircularQueue<std::shared_ptr<int64_t>> q(2);
  EXPECT_TRUE(q.empty());
  EXPECT_THAT(q.size(), 0);

  q.push_back(std::make_shared<int64_t>(10));
  EXPECT_FALSE(q.empty());
  EXPECT_EQ(*q.front(), 10);
  EXPECT_EQ(*q.back(), 10);

  q.pop_front();
  EXPECT_TRUE(q.empty());

  for (int i = 0; i < 10; ++i) {
    q.push_back(std::make_shared<int64_t>(i));
  }
  EXPECT_FALSE(q.empty());

  q.clear();
  EXPECT_TRUE(q.empty());
}

TEST(CircularQueue, Resize) {
  CircularQueue<int64_t> q(2);

  for (int64_t i = 0; i < 1234; ++i) {
    q.push_back(i);
  }

  EXPECT_FALSE(q.empty());
  EXPECT_THAT(q.size(), 1234);
  EXPECT_THAT(q.capacity(), ::testing::Gt(1234));

  for (int64_t i = 0; i < 1234; ++i) {
    EXPECT_THAT(q.front(), i);
    q.pop_front();
  }

  EXPECT_THAT(q.size(), ::testing::Eq(0));
}

}  // namespace
