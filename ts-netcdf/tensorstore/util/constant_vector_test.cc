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

#include "tensorstore/util/constant_vector.h"

#include <string>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::GetConstantVector;
using ::tensorstore::span;

// Tests with a run time-specified length.  The request is satisfied with the
// initial static buffer of length 32.
TEST(GetConstantVectorTest, RunTimeLengthInt) {
  auto x = GetConstantVector<int, 3>(5);
  static_assert(std::is_same_v<decltype(x), span<const int>>);
  EXPECT_THAT(x, ::testing::ElementsAreArray(std::vector<int>(5, 3)));
}

// Tests with a length of 0 specified at run time.
TEST(GetConstantVectorTest, ZeroRunTimeLengthInt) {
  auto x = GetConstantVector<int, 3>(0);
  static_assert(std::is_same_v<decltype(x), span<const int>>);
  EXPECT_EQ(0, x.size());
}

// Tests with a static length.
TEST(GetConstantVectorTest, StaticLengthInt) {
  constexpr auto x = GetConstantVector<int, 3, 5>();
  static_assert(std::is_same_v<decltype(x), const span<const int, 5>>);
  EXPECT_THAT(x, ::testing::ElementsAreArray(std::vector<int>(5, 3)));
}

// Tests with a static length specified using a StaticRank value.
TEST(GetConstantVectorTest, StaticLengthIntUsingStaticRankValue) {
  constexpr auto x = GetConstantVector<int, 3>(tensorstore::StaticRank<5>{});
  static_assert(std::is_same_v<decltype(x), const span<const int, 5>>);
  EXPECT_THAT(x, ::testing::ElementsAreArray(std::vector<int>(5, 3)));
}

// Tests with a static length of 0.
TEST(GetConstantVectorTest, StaticZeroLengthInt) {
  constexpr auto x = GetConstantVector<int, 3, 0>();
  static_assert(std::is_same_v<decltype(x), const span<const int, 0>>);
}

TEST(GetDefaultStringVectorTest, StaticLength) {
  auto x = tensorstore::GetDefaultStringVector<2>();
  static_assert(std::is_same_v<decltype(x), span<const std::string, 2>>);
  EXPECT_THAT(x, ::testing::ElementsAre("", ""));
}

TEST(GetDefaultStringVectorTest, DynamicLength) {
  auto x = tensorstore::GetDefaultStringVector(2);
  static_assert(std::is_same_v<decltype(x), span<const std::string>>);
  EXPECT_THAT(x, ::testing::ElementsAre("", ""));
}

}  // namespace
