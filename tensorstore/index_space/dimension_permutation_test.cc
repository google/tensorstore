// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/index_space/dimension_permutation.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::Dims;
using ::tensorstore::IsValidPermutation;
using ::tensorstore::PermutationMatchesOrder;
using ::tensorstore::span;

TEST(SetPermutationTest, Rank0) {
  std::vector<DimensionIndex> permutation(0);
  // No effects, but verify it does not crash.
  tensorstore::SetPermutation(tensorstore::c_order, permutation);
  tensorstore::SetPermutation(tensorstore::fortran_order, permutation);
}

TEST(SetPermutationTest, Rank1COrder) {
  std::vector<DimensionIndex> permutation(1, 42);
  tensorstore::SetPermutation(tensorstore::c_order, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0));
}

TEST(SetPermutationTest, Rank1FortranOrder) {
  std::vector<DimensionIndex> permutation(1, 42);
  tensorstore::SetPermutation(tensorstore::fortran_order, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0));
}

TEST(SetPermutationTest, Rank2COrder) {
  std::vector<DimensionIndex> permutation(2, 42);
  tensorstore::SetPermutation(tensorstore::c_order, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0, 1));
}

TEST(SetPermutationTest, Rank2FortranOrder) {
  std::vector<DimensionIndex> permutation(2, 42);
  tensorstore::SetPermutation(tensorstore::fortran_order, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(1, 0));
}

TEST(SetPermutationTest, Rank3COrder) {
  std::vector<DimensionIndex> permutation(3, 42);
  tensorstore::SetPermutation(tensorstore::c_order, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0, 1, 2));
}

TEST(SetPermutationTest, Rank3FortranOrder) {
  std::vector<DimensionIndex> permutation(3, 42);
  tensorstore::SetPermutation(tensorstore::fortran_order, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(2, 1, 0));
}

TEST(IsValidPermutationTest, Basic) {
  EXPECT_TRUE(IsValidPermutation(span<const DimensionIndex>()));
  EXPECT_TRUE(IsValidPermutation(span<const DimensionIndex>({0})));
  EXPECT_FALSE(IsValidPermutation(span<const DimensionIndex>({1})));
  EXPECT_FALSE(IsValidPermutation(span<const DimensionIndex>({-1})));
  EXPECT_TRUE(IsValidPermutation(span<const DimensionIndex>({0, 1})));
  EXPECT_TRUE(IsValidPermutation(span<const DimensionIndex>({1, 0})));
  EXPECT_FALSE(IsValidPermutation(span<const DimensionIndex>({1, 1})));
  EXPECT_FALSE(IsValidPermutation(span<const DimensionIndex>({0, 0})));
  EXPECT_TRUE(IsValidPermutation(span<const DimensionIndex>({1, 2, 0})));
  EXPECT_FALSE(IsValidPermutation(span<const DimensionIndex>({1, 2, 1})));
}

TEST(PermutationMatchesOrderTest, Basic) {
  EXPECT_TRUE(PermutationMatchesOrder({}, tensorstore::c_order));
  EXPECT_TRUE(PermutationMatchesOrder({}, tensorstore::fortran_order));
  EXPECT_TRUE(PermutationMatchesOrder({{0}}, tensorstore::c_order));
  EXPECT_TRUE(PermutationMatchesOrder({{0}}, tensorstore::fortran_order));
  EXPECT_TRUE(PermutationMatchesOrder({{0, 1}}, tensorstore::c_order));
  EXPECT_FALSE(PermutationMatchesOrder({{0, 1}}, tensorstore::fortran_order));
  EXPECT_TRUE(PermutationMatchesOrder({{0, 1, 2}}, tensorstore::c_order));
  EXPECT_FALSE(PermutationMatchesOrder({{1}}, tensorstore::c_order));
  EXPECT_FALSE(PermutationMatchesOrder({{1}}, tensorstore::fortran_order));
  EXPECT_FALSE(PermutationMatchesOrder({{1, 0}}, tensorstore::c_order));
  EXPECT_TRUE(PermutationMatchesOrder({{1, 0}}, tensorstore::fortran_order));
}

TEST(InvertPermutationTest, Rank0) {
  std::vector<DimensionIndex> source;
  std::vector<DimensionIndex> dest;
  tensorstore::InvertPermutation(0, source.data(), dest.data());
}

TEST(InvertPermutationTest, Rank1) {
  std::vector<DimensionIndex> source{0};
  std::vector<DimensionIndex> dest(1, 42);
  tensorstore::InvertPermutation(1, source.data(), dest.data());
  EXPECT_THAT(dest, ::testing::ElementsAre(0));
}

TEST(InvertPermutationTest, Rank2Identity) {
  std::vector<DimensionIndex> source{0, 1};
  std::vector<DimensionIndex> dest(2, 42);
  tensorstore::InvertPermutation(2, source.data(), dest.data());
  EXPECT_THAT(dest, ::testing::ElementsAre(0, 1));
}

TEST(InvertPermutationTest, Rank2Transpose) {
  std::vector<DimensionIndex> source{1, 0};
  std::vector<DimensionIndex> dest(2, 42);
  tensorstore::InvertPermutation(2, source.data(), dest.data());
  EXPECT_THAT(dest, ::testing::ElementsAre(1, 0));
}

TEST(InvertPermutationTest, Rank3) {
  std::vector<DimensionIndex> source{1, 2, 0};
  std::vector<DimensionIndex> dest(3, 42);
  tensorstore::InvertPermutation(3, source.data(), dest.data());
  EXPECT_THAT(dest, ::testing::ElementsAre(2, 0, 1));
  std::vector<DimensionIndex> source2(3, 42);
  tensorstore::InvertPermutation(3, dest.data(), source2.data());
  EXPECT_EQ(source, source2);
}

TEST(SetPermutationFromStridedLayoutTest, Rank0) {
  tensorstore::StridedLayout<> layout(0);
  std::vector<DimensionIndex> permutation(0);
  // No effects, but verify it does not crash.
  tensorstore::SetPermutationFromStridedLayout(layout, permutation);
}

TEST(SetPermutationFromStridedLayoutTest, Rank1) {
  tensorstore::StridedLayout<> layout({5}, {10});
  std::vector<DimensionIndex> permutation(1, 42);
  tensorstore::SetPermutationFromStridedLayout(layout, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0));
}

TEST(SetPermutationFromStridedLayoutTest, Rank2COrder) {
  tensorstore::StridedLayout<> layout({5, 6}, {10, 5});
  std::vector<DimensionIndex> permutation(2, 42);
  tensorstore::SetPermutationFromStridedLayout(layout, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0, 1));
}

TEST(SetPermutationFromStridedLayoutTest, Rank2FortranOrder) {
  tensorstore::StridedLayout<> layout({5, 6}, {5, 10});
  std::vector<DimensionIndex> permutation(2, 42);
  tensorstore::SetPermutationFromStridedLayout(layout, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(1, 0));
}

TEST(SetPermutationFromStridedLayoutTest, Rank2ZeroStride) {
  tensorstore::StridedLayout<> layout({5, 6}, {0, 0});
  std::vector<DimensionIndex> permutation(2, 42);
  tensorstore::SetPermutationFromStridedLayout(layout, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0, 1));
}

TEST(SetPermutationFromStridedLayoutTest, Rank4) {
  tensorstore::StridedLayout<> layout({5, 6, 7, 8}, {10, 5, 6, 6});
  std::vector<DimensionIndex> permutation(4, 42);
  tensorstore::SetPermutationFromStridedLayout(layout, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0, 2, 3, 1));
}

TEST(TransformOutputDimensionOrderTest, Rank0) {
  std::vector<DimensionIndex> source;
  std::vector<DimensionIndex> dest;
  tensorstore::TransformOutputDimensionOrder(tensorstore::IdentityTransform(0),
                                             source, dest);
}

TEST(TransformOutputDimensionOrderTest, Rank1Identity) {
  std::vector<DimensionIndex> source{0};
  std::vector<DimensionIndex> dest(1, 42);
  tensorstore::TransformOutputDimensionOrder(tensorstore::IdentityTransform(1),
                                             source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(0));
}

TEST(TransformOutputDimensionOrderTest, Rank2COrderIdentity) {
  std::vector<DimensionIndex> source{0, 1};
  std::vector<DimensionIndex> dest(2, 42);
  std::vector<DimensionIndex> source2(2, 42);
  auto transform = tensorstore::IdentityTransform(2);
  tensorstore::TransformOutputDimensionOrder(transform, source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(0, 1));
  tensorstore::TransformInputDimensionOrder(transform, dest, source2);
  EXPECT_EQ(source, source2);
}

TEST(TransformOutputDimensionOrderTest, Rank2FortranOrderIdentity) {
  std::vector<DimensionIndex> source{1, 0};
  std::vector<DimensionIndex> dest(2, 42);
  std::vector<DimensionIndex> source2(2, 42);
  auto transform = tensorstore::IdentityTransform(2);
  tensorstore::TransformOutputDimensionOrder(transform, source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(1, 0));
  tensorstore::TransformInputDimensionOrder(transform, dest, source2);
  EXPECT_EQ(source, source2);
}

TEST(TransformOutputDimensionOrderTest, Rank2COrderTranspose) {
  std::vector<DimensionIndex> source{0, 1};
  std::vector<DimensionIndex> dest(2, 42);
  std::vector<DimensionIndex> source2(2, 42);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      tensorstore::IdentityTransform(2) | Dims(1, 0).Transpose());
  tensorstore::TransformOutputDimensionOrder(transform, source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(1, 0));
  tensorstore::TransformInputDimensionOrder(transform, dest, source2);
  EXPECT_EQ(source, source2);
}

TEST(TransformOutputDimensionOrderTest, Rank2FortranOrderTranspose) {
  std::vector<DimensionIndex> source{1, 0};
  std::vector<DimensionIndex> dest(2, 42);
  std::vector<DimensionIndex> source2(2, 42);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      tensorstore::IdentityTransform(2) | Dims(1, 0).Transpose());
  tensorstore::TransformOutputDimensionOrder(transform, source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(0, 1));
  tensorstore::TransformInputDimensionOrder(transform, dest, source2);
  EXPECT_EQ(source, source2);
}

}  // namespace
