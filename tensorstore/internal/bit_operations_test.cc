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

#include "tensorstore/internal/bit_operations.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using tensorstore::internal::bit_width;
using tensorstore::internal_bits::CountLeadingZeros64Slow;

static_assert(CountLeadingZeros64Slow(0) == 64);
static_assert(CountLeadingZeros64Slow(~std::uint64_t(0)) == 0);
static_assert(CountLeadingZeros64Slow(0xffffffff) == 32);
static_assert(CountLeadingZeros64Slow(7) == 61);
static_assert(CountLeadingZeros64Slow(8) == 60);

TEST(BitWidthTest, Basic) {
  EXPECT_EQ(0, bit_width(0));
  EXPECT_EQ(1, bit_width(1));
  EXPECT_EQ(2, bit_width(2));
  EXPECT_EQ(2, bit_width(3));
  EXPECT_EQ(3, bit_width(4));
  EXPECT_EQ(3, bit_width(5));
  EXPECT_EQ(3, bit_width(6));
  EXPECT_EQ(3, bit_width(7));
  EXPECT_EQ(4, bit_width(8));
  EXPECT_EQ(32, bit_width(0xffffffff));
  EXPECT_EQ(64, bit_width(0xffffffffffffffff));
  EXPECT_EQ(63, bit_width(0x7fffffffffffffff));
  EXPECT_EQ(62, bit_width(0x3fffffffffffffff));
}

}  // namespace
