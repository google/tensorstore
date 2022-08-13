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

#include "tensorstore/util/constant_bit_vector.h"

#include <cstdint>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/bit_span.h"

namespace {
using ::tensorstore::BitSpan;
using ::tensorstore::GetConstantBitVector;
TEST(GetConstantBitVectorTest, StaticExtentFalse) {
  constexpr auto v = GetConstantBitVector<std::uint64_t, false, 113>();
  static_assert(
      std::is_same_v<decltype(v), const BitSpan<const std::uint64_t, 113>>);
  EXPECT_THAT(v, ::testing::ElementsAreArray(std::vector<bool>(113, false)));
}

TEST(GetConstantBitVectorTest, StaticExtentTrue) {
  constexpr auto v = GetConstantBitVector<std::uint64_t, true, 113>();
  static_assert(
      std::is_same_v<decltype(v), const BitSpan<const std::uint64_t, 113>>);
  EXPECT_THAT(v, ::testing::ElementsAreArray(std::vector<bool>(113, true)));
}

TEST(GetConstantBitVectorTest, DynamicExtentFalse) {
  auto v = GetConstantBitVector<std::uint64_t, false>(113);
  static_assert(std::is_same_v<decltype(v), BitSpan<const std::uint64_t>>);
  EXPECT_THAT(v, ::testing::ElementsAreArray(std::vector<bool>(113, false)));
}

TEST(GetConstantBitVectorTest, DynamicExtentTrue) {
  auto v = GetConstantBitVector<std::uint64_t, true>(113);
  static_assert(std::is_same_v<decltype(v), BitSpan<const std::uint64_t>>);
  EXPECT_THAT(v, ::testing::ElementsAreArray(std::vector<bool>(113, true)));
}

}  // namespace
