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

#include "tensorstore/internal/os/aligned_alloc.h"

#include <stddef.h>

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "tensorstore/internal/os/memory_region.h"

using ::tensorstore::internal_os::AllocatePageAlignedRegion;

namespace {

TEST(AlignedAllocTest, AllocatePageAlignedRegion) {
  auto region = AllocatePageAlignedRegion(1024, 16 * 1024 * 1024);
  EXPECT_THAT(region.as_string_view().size(), testing::Ge(16 * 1024 * 1024));

  // Verify that assignment doesn't leak.
  region = AllocatePageAlignedRegion(1024, 16);
  EXPECT_THAT(region.as_string_view().size(), testing::Ge(16));

  absl::Cord a = std::move(region).as_cord();
}

}  // namespace
