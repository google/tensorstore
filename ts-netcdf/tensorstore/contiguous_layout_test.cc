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

#include "tensorstore/contiguous_layout.h"

#include <array>
#include <sstream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index.h"
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::ComputeStrides;
using ::tensorstore::ContiguousLayoutOrder;
using ::tensorstore::ContiguousLayoutPermutation;
using ::tensorstore::DimensionIndex;
using ::tensorstore::GetContiguousIndices;
using ::tensorstore::GetContiguousOffset;
using ::tensorstore::Index;

TEST(ContiguousLayoutOrderTest, PrintToOstream) {
  {
    std::ostringstream ostr;
    ostr << ContiguousLayoutOrder::c;
    EXPECT_EQ("C", ostr.str());
  }
  {
    std::ostringstream ostr;
    ostr << ContiguousLayoutOrder::fortran;
    EXPECT_EQ("F", ostr.str());
  }
}

TEST(ComputeStridesTest, COrder) {
  {
    std::array<Index, 3> strides;
    ComputeStrides(ContiguousLayoutOrder::c, /*element_stride=*/1,
                   tensorstore::span<const Index>({3l, 4l, 5l}), strides);
    EXPECT_THAT(strides, ::testing::ElementsAre(20, 5, 1));
  }
  {
    std::array<Index, 3> strides;
    ComputeStrides(ContiguousLayoutOrder::c, /*element_stride=*/2,
                   tensorstore::span<const Index>({3l, 4l, 5l}), strides);
    EXPECT_THAT(strides, ::testing::ElementsAre(40, 10, 2));
  }
}

TEST(ComputeStridesTest, FOrder) {
  std::array<Index, 3> strides;
  ComputeStrides(ContiguousLayoutOrder::fortran, /*element_stride=*/1,
                 tensorstore::span<const Index>({3l, 4l, 5l}), strides);
  EXPECT_THAT(strides, ::testing::ElementsAre(1, 3, 12));
}

TEST(ComputeStridesFromLayoutPermutationTest, Basic) {
  {
    std::array<Index, 3> strides;
    ComputeStrides(ContiguousLayoutPermutation<>({{2, 0, 1}}),
                   /*element_stride=*/1,
                   tensorstore::span<const Index>({3l, 4l, 5l}), strides);
    EXPECT_THAT(strides, ::testing::ElementsAre(4, 1, 3 * 4));
  }
  {
    std::array<Index, 3> strides;
    ComputeStrides(ContiguousLayoutPermutation<>({{1, 2, 0}}),
                   /*element_stride=*/2,
                   tensorstore::span<const Index>({3l, 4l, 5l}), strides);
    EXPECT_THAT(strides, ::testing::ElementsAre(2, 2 * 3 * 5, 2 * 3));
  }
}

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
  EXPECT_TRUE(IsValidPermutation(tensorstore::span<const DimensionIndex>()));
  EXPECT_TRUE(IsValidPermutation(tensorstore::span<const DimensionIndex>({0})));
  EXPECT_FALSE(
      IsValidPermutation(tensorstore::span<const DimensionIndex>({1})));
  EXPECT_FALSE(
      IsValidPermutation(tensorstore::span<const DimensionIndex>({-1})));
  EXPECT_TRUE(
      IsValidPermutation(tensorstore::span<const DimensionIndex>({0, 1})));
  EXPECT_TRUE(
      IsValidPermutation(tensorstore::span<const DimensionIndex>({1, 0})));
  EXPECT_FALSE(
      IsValidPermutation(tensorstore::span<const DimensionIndex>({1, 1})));
  EXPECT_FALSE(
      IsValidPermutation(tensorstore::span<const DimensionIndex>({0, 0})));
  EXPECT_TRUE(
      IsValidPermutation(tensorstore::span<const DimensionIndex>({1, 2, 0})));
  EXPECT_FALSE(
      IsValidPermutation(tensorstore::span<const DimensionIndex>({1, 2, 1})));
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

TEST(SetPermutationFromStrides, Rank0) {
  std::vector<DimensionIndex> permutation(0);
  // No effects, but verify it does not crash.
  tensorstore::SetPermutationFromStrides({}, permutation);
}

TEST(SetPermutationFromStridesTest, Rank1) {
  Index strides[] = {10};
  std::vector<DimensionIndex> permutation(1, 42);
  tensorstore::SetPermutationFromStrides(strides, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0));
}

TEST(SetPermutationFromStridesTest, Rank2COrder) {
  Index strides[] = {10, 5};
  std::vector<DimensionIndex> permutation(2, 42);
  tensorstore::SetPermutationFromStrides(strides, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0, 1));
}

TEST(SetPermutationFromStridesTest, Rank2FortranOrder) {
  Index strides[] = {5, 10};
  std::vector<DimensionIndex> permutation(2, 42);
  tensorstore::SetPermutationFromStrides(strides, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(1, 0));
}

TEST(SetPermutationFromStridesTest, Rank2ZeroStride) {
  Index strides[] = {0, 0};
  std::vector<DimensionIndex> permutation(2, 42);
  tensorstore::SetPermutationFromStrides(strides, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0, 1));
}

TEST(SetPermutationFromStridesTest, Rank4) {
  Index strides[] = {10, 5, 6, 6};
  std::vector<DimensionIndex> permutation(4, 42);
  tensorstore::SetPermutationFromStrides(strides, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0, 2, 3, 1));
}

TEST(GetContiguousOffsetTest, Basic) {
  Index indices[2];
  EXPECT_EQ(3 * 11 + 4,
            GetContiguousOffset<ContiguousLayoutOrder::c>({{7, 11}}, {{3, 4}}));
  GetContiguousIndices<ContiguousLayoutOrder::c, Index>(3 * 11 + 4, {{7, 11}},
                                                        indices);
  EXPECT_THAT(indices, ::testing::ElementsAre(3, 4));
  EXPECT_EQ(3 + 4 * 7, GetContiguousOffset<ContiguousLayoutOrder::fortran>(
                           {{7, 11}}, {{3, 4}}));
  GetContiguousIndices<ContiguousLayoutOrder::fortran, Index>(
      3 + 4 * 7, {{7, 11}}, indices);
  EXPECT_THAT(indices, ::testing::ElementsAre(3, 4));
  EXPECT_EQ(
      2 * (7 * 11) + 3 * 11 + 4,
      GetContiguousOffset<ContiguousLayoutOrder::c>({{5, 7, 11}}, {{2, 3, 4}}));
  EXPECT_EQ(2 + 5 * 3 + (5 * 7) * 4,
            GetContiguousOffset<ContiguousLayoutOrder::fortran>({{5, 7, 11}},
                                                                {{2, 3, 4}}));
  EXPECT_EQ(0, GetContiguousOffset<ContiguousLayoutOrder::c>({}, {}));
  EXPECT_EQ(0, GetContiguousOffset<ContiguousLayoutOrder::fortran>({}, {}));
}

}  // namespace
