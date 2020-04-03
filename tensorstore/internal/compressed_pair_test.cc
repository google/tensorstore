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

#include "tensorstore/internal/compressed_pair.h"

#include <type_traits>

#include <gtest/gtest.h>

namespace {

using tensorstore::internal::CompressedFirstEmptyPair;
using tensorstore::internal::CompressedFirstSecondPair;
using tensorstore::internal::CompressedPair;
using tensorstore::internal::CompressedSecondEmptyPair;

struct Empty {
  Empty() = default;
  Empty(int) {}
};

TEST(CompressedPair, FirstEmpty) {
  static_assert(std::is_same<CompressedPair<Empty, int>,
                             CompressedFirstEmptyPair<Empty, int>>::value,
                "");
  CompressedPair<Empty, int> x;
  CompressedPair<Empty, int> x2(1, 2);
  static_assert(std::is_same<decltype(x.first()), Empty>::value, "");
  static_assert(
      std::is_same<
          decltype(std::declval<const CompressedPair<Empty, int>&>().second()),
          const int&>::value,
      "");
  static_assert(
      std::is_same<decltype(
                       std::declval<CompressedPair<Empty, int>>().second()),
                   int&&>::value,
      "");
  static_assert(std::is_same<decltype(x.second()), int&>::value, "");
  EXPECT_EQ(2, x2.second());
}

TEST(CompressedPair, SecondEmpty) {
  static_assert(std::is_same<CompressedPair<int, Empty>,
                             CompressedSecondEmptyPair<int, Empty>>::value,
                "");
  CompressedPair<int, Empty> x;
  CompressedPair<int, Empty> x2(1, 2);
  static_assert(std::is_same<decltype(x.second()), Empty>::value, "");
  static_assert(
      std::is_same<
          decltype(std::declval<const CompressedPair<int, Empty>&>().first()),
          const int&>::value,
      "");
  static_assert(
      std::is_same<decltype(std::declval<CompressedPair<int, Empty>>().first()),
                   int&&>::value,
      "");
  static_assert(std::is_same<decltype(x.first()), int&>::value, "");
  EXPECT_EQ(1, x2.first());
}

TEST(CompressedPair, BothNonEmpty) {
  static_assert(std::is_same<CompressedPair<int, float>,
                             CompressedFirstSecondPair<int, float>>::value,
                "");
  CompressedPair<int, float> x;
  CompressedPair<int, float> x2(1, 2);
  static_assert(
      std::is_same<
          decltype(std::declval<const CompressedPair<int, float>&>().first()),
          const int&>::value,
      "");
  static_assert(
      std::is_same<decltype(std::declval<CompressedPair<int, float>>().first()),
                   int&&>::value,
      "");
  static_assert(std::is_same<decltype(x.first()), int&>::value, "");

  static_assert(
      std::is_same<
          decltype(std::declval<const CompressedPair<int, float>&>().second()),
          const float&>::value,
      "");
  static_assert(
      std::is_same<decltype(
                       std::declval<CompressedPair<int, float>>().second()),
                   float&&>::value,
      "");
  static_assert(std::is_same<decltype(x.second()), float&>::value, "");

  EXPECT_EQ(1, x2.first());
  EXPECT_EQ(2, x2.second());
}

}  // namespace
