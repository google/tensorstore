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

#include "tensorstore/internal/multi_vector_view.h"

#include <cstddef>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::dynamic_rank;
using ::tensorstore::span;
using ::tensorstore::internal::MultiVectorAccess;
using ::tensorstore::internal::MultiVectorViewStorage;
using ::testing::ElementsAre;

static_assert(
    MultiVectorAccess<MultiVectorViewStorage<3, int, float>>::static_extent ==
    3);
static_assert(
    MultiVectorAccess<MultiVectorViewStorage<3, int, float>>::num_vectors == 2);

TEST(MultiVectorViewStorageTest, StaticExtent2) {
  using Container = MultiVectorViewStorage<2, float, int>;
  using Access = MultiVectorAccess<Container>;
  static_assert(
      std::is_same_v<float, typename Access::template ElementType<0>>);
  static_assert(std::is_same_v<int, typename Access::template ElementType<1>>);
  static_assert(
      std::is_same_v<float, typename Access::template ConstElementType<0>>);
  static_assert(
      std::is_same_v<int, typename Access::template ConstElementType<1>>);

  // Test default construction.
  Container vec;
  static_assert(std::is_same_v<std::integral_constant<std::ptrdiff_t, 2>,
                               decltype(Access::GetExtent(vec))>);
  EXPECT_EQ(2, Access::GetExtent(vec));
  EXPECT_EQ(nullptr, Access::template get<0>(&vec).data());
  EXPECT_EQ(nullptr, Access::template get<1>(&vec).data());

  // Test (rank, pointer...) assignment.
  float float_arr[] = {1, 2};
  int int_arr[] = {3, 4};
  Access::Assign(&vec, std::integral_constant<std::ptrdiff_t, 2>(), float_arr,
                 int_arr);
  EXPECT_THAT(Access::template get<0>(&vec), ElementsAre(1, 2));
  EXPECT_THAT(Access::template get<1>(&vec), ElementsAre(3, 4));
  EXPECT_EQ(&float_arr[0], Access::template get<0>(&vec).data());
  EXPECT_EQ(&int_arr[0], Access::template get<1>(&vec).data());

  // Test (span...) assignment.
  float float_arr2[] = {5, 6};
  int int_arr2[] = {7, 8};
  Access::Assign(&vec, span(float_arr2), span(int_arr2));
  EXPECT_EQ(&float_arr2[0], Access::template get<0>(&vec).data());
  EXPECT_EQ(&int_arr2[0], Access::template get<1>(&vec).data());
}

TEST(MultiVectorViewStorageTest, StaticExtent0) {
  using Container = MultiVectorViewStorage<0, float, int>;
  using Access = MultiVectorAccess<Container>;
  static_assert(
      std::is_same_v<float, typename Access::template ElementType<0>>);
  static_assert(std::is_same_v<int, typename Access::template ElementType<1>>);
  static_assert(
      std::is_same_v<float, typename Access::template ConstElementType<0>>);
  static_assert(
      std::is_same_v<int, typename Access::template ConstElementType<1>>);

  // Test default construction.
  Container vec;
  static_assert(std::is_same_v<std::integral_constant<std::ptrdiff_t, 0>,
                               decltype(Access::GetExtent(vec))>);
  EXPECT_EQ(0, Access::GetExtent(vec));
  EXPECT_EQ(nullptr, Access::template get<0>(&vec).data());
  EXPECT_EQ(nullptr, Access::template get<1>(&vec).data());

  // Test (rank, pointer...) assignment.
  Access::Assign(&vec, std::integral_constant<std::ptrdiff_t, 0>(), nullptr,
                 nullptr);
  EXPECT_EQ(nullptr, Access::template get<0>(&vec).data());
  EXPECT_EQ(nullptr, Access::template get<1>(&vec).data());

  // Test (span...) assignment.
  Access::Assign(&vec, span<float, 0>{}, span<int, 0>{});
}

TEST(MultiVectorViewStorageTest, DynamicExtent) {
  using Container = MultiVectorViewStorage<dynamic_rank, float, int>;
  using Access = MultiVectorAccess<Container>;
  static_assert(
      std::is_same_v<float, typename Access::template ElementType<0>>);
  static_assert(std::is_same_v<int, typename Access::template ElementType<1>>);
  static_assert(
      std::is_same_v<float, typename Access::template ConstElementType<0>>);
  static_assert(
      std::is_same_v<int, typename Access::template ConstElementType<1>>);

  // Test default construction.
  Container vec;
  static_assert(
      std::is_same_v<std::ptrdiff_t, decltype(Access::GetExtent(vec))>);
  EXPECT_EQ(0, Access::GetExtent(vec));
  EXPECT_EQ(nullptr, Access::template get<0>(&vec).data());
  EXPECT_EQ(nullptr, Access::template get<1>(&vec).data());

  // Test (rank, pointer...) assignment.
  float float_arr[] = {1, 2};
  int int_arr[] = {3, 4};
  Access::Assign(&vec, std::integral_constant<std::ptrdiff_t, 2>(), float_arr,
                 int_arr);
  EXPECT_EQ(2, Access::GetExtent(vec));
  EXPECT_THAT(Access::template get<0>(&vec), ElementsAre(1, 2));
  EXPECT_THAT(Access::template get<1>(&vec), ElementsAre(3, 4));
  EXPECT_EQ(&float_arr[0], Access::template get<0>(&vec).data());
  EXPECT_EQ(&int_arr[0], Access::template get<1>(&vec).data());

  // Test (span...) assignment.
  float float_arr2[] = {5, 6, 7};
  int int_arr2[] = {7, 8, 9};
  Access::Assign(&vec, span<float>(float_arr2), span<int>(int_arr2));
  EXPECT_EQ(3, Access::GetExtent(vec));
  EXPECT_EQ(&float_arr2[0], Access::template get<0>(&vec).data());
  EXPECT_EQ(&int_arr2[0], Access::template get<1>(&vec).data());
}

}  // namespace
