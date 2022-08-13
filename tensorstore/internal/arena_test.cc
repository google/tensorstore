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

#include "tensorstore/internal/arena.h"

#include <algorithm>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::internal::Arena;
using ::tensorstore::internal::ArenaAllocator;

bool Contains(tensorstore::span<const unsigned char> buffer, void* ptr) {
  return ptr >= buffer.data() && ptr < buffer.data() + buffer.size();
}

TEST(ArenaTest, Small) {
  unsigned char buffer[1024];
  Arena arena(buffer);
  std::vector<int, ArenaAllocator<int>> vec(100, &arena);
  EXPECT_EQ(&arena, vec.get_allocator().arena());
  std::fill(vec.begin(), vec.end(), 5);
  EXPECT_TRUE(Contains(buffer, vec.data()));
}

TEST(ArenaTest, Alignment) {
  alignas(16) unsigned char buffer[1024];
  // Test alignment for allocations that fit in fixed-size buffer.
  for (int x = 1; x <= 16; x *= 2) {
    Arena arena(buffer);
    // always results in an odd alignment
    unsigned char* ptr1 = arena.allocate(1, 1);
    EXPECT_EQ(&buffer[0], ptr1);
    // Always aligned to `x`.
    unsigned char* ptr2 = arena.allocate(1, x);
    EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(ptr2) % x);
    EXPECT_EQ(&buffer[x], ptr2);
    arena.deallocate(ptr1, 1, 1);
    arena.deallocate(ptr2, 1, x);
  }
  // Test alignment for allocations that don't fit in fixed-size buffer.
  {
    Arena arena(buffer);
    unsigned char* ptr = arena.allocate(2000, 16);
    EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(ptr) % 16);
    arena.deallocate(ptr, 2000, 16);
  }
}

TEST(ArenaTest, Large) {
  unsigned char buffer[1024];
  Arena arena(buffer);
  std::vector<int, ArenaAllocator<int>> vec(&arena);
  vec.resize(2000);
  std::fill(vec.begin(), vec.end(), 7);
  EXPECT_FALSE(Contains(buffer, vec.data()));
}

TEST(ArenaTest, MultipleSmall) {
  unsigned char buffer[1024];
  Arena arena(buffer);
  std::vector<std::int32_t, ArenaAllocator<int>> vec(100, &arena);
  EXPECT_EQ(&arena, vec.get_allocator().arena());
  std::fill(vec.begin(), vec.end(), 5);
  EXPECT_TRUE(Contains(buffer, vec.data()));

  std::vector<std::int32_t, ArenaAllocator<int>> vec2(100, &arena);
  std::fill(vec2.begin(), vec2.end(), 6);
  EXPECT_TRUE(Contains(buffer, vec2.data()));

  std::vector<std::int32_t, ArenaAllocator<int>> vec3(100, &arena);
  std::fill(vec3.begin(), vec3.end(), 7);
  EXPECT_FALSE(Contains(buffer, vec3.data()));

  std::vector<std::int32_t, ArenaAllocator<int>> vec4(5, &arena);
  std::fill(vec4.begin(), vec4.end(), 8);
  EXPECT_TRUE(Contains(buffer, vec4.data()));

  EXPECT_THAT(vec,
              ::testing::ElementsAreArray(std::vector<std::int32_t>(100, 5)));
  EXPECT_THAT(vec2,
              ::testing::ElementsAreArray(std::vector<std::int32_t>(100, 6)));
  EXPECT_THAT(vec3,
              ::testing::ElementsAreArray(std::vector<std::int32_t>(100, 7)));
  EXPECT_THAT(vec4,
              ::testing::ElementsAreArray(std::vector<std::int32_t>(5, 8)));
}

}  // namespace
