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

#include "tensorstore/internal/container/block_queue.h"

#include <stdint.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal_container::BlockQueue;

TEST(BlockQueue, Basic) {
  BlockQueue<int64_t> q;
  EXPECT_TRUE(q.empty());
  EXPECT_THAT(q.size(), 0);

  q.push_back(10);
  EXPECT_FALSE(q.empty());
  EXPECT_EQ(q.front(), 10);
  EXPECT_EQ(q.back(), 10);

  q.pop_front();
  EXPECT_TRUE(q.empty());

  q.clear();
  EXPECT_TRUE(q.empty());
}

TEST(BlockQueue, PushPop) {
  BlockQueue<int64_t> q;

  for (int i = 0; i < 4096; i++) {
    q.push_back(i);
    if (i & 0x08) {
      q.pop_front();
    }
  }
  EXPECT_FALSE(q.empty());

  q.clear();
  EXPECT_TRUE(q.empty());
}

}  // namespace
