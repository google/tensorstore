// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/internal/os/memory_region.h"

#include <stddef.h>

#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/cord.h"

using ::tensorstore::internal_os::AllocateHeapRegion;

namespace {

TEST(MemoryRegionTest, EmptyRegion) {
  auto region = AllocateHeapRegion(0);
  EXPECT_EQ(region.as_string_view().size(), 0);
  absl::Cord a = std::move(region).as_cord();
}

TEST(MemoryRegionTest, Assignment) {
  auto region = AllocateHeapRegion(0);
  for (int i = 0; i < 10; ++i) region = AllocateHeapRegion(i);
}

TEST(MemoryRegionTest, AllocateHeapRegion) {
  auto region = AllocateHeapRegion(16 * 1024 * 1024);
  EXPECT_EQ(region.as_string_view().size(), 16 * 1024 * 1024);

  // Verify that assignment doesn't leak.
  region = AllocateHeapRegion(16);
  EXPECT_EQ(region.as_string_view().size(), 16);

  absl::Cord a = std::move(region).as_cord();
}

}  // namespace
