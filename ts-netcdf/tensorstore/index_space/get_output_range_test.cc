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

/// Tests for GetOutputRange.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Box;
using ::tensorstore::BoxView;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::IndexInterval;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::IndexTransformView;
using ::tensorstore::kMaxFiniteIndex;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_index_space::TransformAccess;

TEST(GetOutputRangeTest, Constant) {
  Box<> output_range(1);
  // Transform maps [-inf,inf] -> {5}
  auto result = GetOutputRange(
      IndexTransformBuilder<>(1, 1).output_constant(0, 5).Finalize().value(),
      output_range);
  ASSERT_TRUE(result);
  // Bounds are exactly equal to the range of the transform.
  EXPECT_TRUE(*result);
  EXPECT_EQ(BoxView({5}, {1}), output_range);
}

TEST(GetOutputRangeTest, SingleInputDimensionExactOneDimensional) {
  Box<> output_range(1);
  // Transform maps {1,2,3,4,5} -> {3,4,5,6,7}
  auto result = GetOutputRange(IndexTransformBuilder<>(1, 1)
                                   .input_origin({1})
                                   .input_shape({5})
                                   .output_single_input_dimension(0, 2, 1, 0)
                                   .Finalize()
                                   .value(),
                               output_range);
  ASSERT_TRUE(result);
  // Bounds are exactly equal to the range of the transform.
  EXPECT_TRUE(*result);
  EXPECT_EQ(BoxView({3}, {5}), output_range);
}

TEST(GetOutputRangeTest, SingleInputDimensionStrideZeroOneDimensional) {
  Box<> output_range(1);
  // Transform maps {1,2,3,4,5} -> {8,8,8,8,8}
  auto t = IndexTransformBuilder<>(1, 1)
               .input_origin({1})
               .input_shape({5})
               .output_single_input_dimension(0, 8, 1, 0)
               .Finalize()
               .value();
  // IndexTransformBuilder can't be used to create a transform with a stride of
  // 0.  Instead, we have to modify the transform representation directly.
  tensorstore::internal_index_space::TransformAccess::rep(t)
      ->output_index_maps()[0]
      .stride() = 0;
  auto result = GetOutputRange(t, output_range);
  ASSERT_TRUE(result);
  // Bounds are exactly equal to the range of the transform.
  EXPECT_TRUE(*result);
  EXPECT_EQ(BoxView({8}, {1}), output_range);
}

TEST(GetOutputRangeTest, SingleInputDimensionStrideZeroTwoDimensional) {
  Box<> output_range(2);
  // Transform maps {1,2,3,4,5} -> {{8,9},{8,9},{8,9},{8,9},{8,9}}
  auto t = IndexTransformBuilder<>(1, 2)
               .input_origin({1})
               .input_shape({5})
               .output_single_input_dimension(0, 8, 1, 0)
               .output_single_input_dimension(1, 9, 1, 0)
               .Finalize()
               .value();
  // IndexTransformBuilder can't be used to create a transform with a stride of
  // 0.  Instead, we have to modify the transform representation directly.
  tensorstore::internal_index_space::TransformAccess::rep(t)
      ->output_index_maps()[0]
      .stride() = 0;
  tensorstore::internal_index_space::TransformAccess::rep(t)
      ->output_index_maps()[1]
      .stride() = 0;
  auto result = GetOutputRange(t, output_range);
  ASSERT_TRUE(result);
  // Bounds are exactly equal to the range of the transform.
  EXPECT_TRUE(*result);
  EXPECT_EQ(BoxView({8, 9}, {1, 1}), output_range);
}

TEST(GetOutputRangeTest,
     SingleInputDimensionExactOneDimensionalNegativeOneStride) {
  Box<> output_range(1);
  // Transform maps {1,2,3,4,5} -> {1,0,-1,-2,-3}
  auto result = GetOutputRange(IndexTransformBuilder<>(1, 1)
                                   .input_origin({1})
                                   .input_shape({5})
                                   .output_single_input_dimension(0, 2, -1, 0)
                                   .Finalize()
                                   .value(),
                               output_range);
  ASSERT_TRUE(result);
  // Bounds are exactly equal to the range of the transform.
  EXPECT_TRUE(*result);
  EXPECT_EQ(BoxView({-3}, {5}), output_range);
}

TEST(GetOutputRangeTest, SingleInputDimensionExactTwoDimensional) {
  Box<> output_range(2);
  // Transform maps [1,5] * [8,13] -> [1+2,5+2 * [8+3,13+3]
  auto result = GetOutputRange(IndexTransformBuilder<>(2, 2)
                                   .input_origin({1, 8})
                                   .input_shape({5, 6})
                                   .output_single_input_dimension(0, 2, 1, 0)
                                   .output_single_input_dimension(1, 3, 1, 1)
                                   .Finalize()
                                   .value(),
                               output_range);
  ASSERT_TRUE(result);
  // Bounds are exactly equal to the range of the transform.
  EXPECT_TRUE(*result);
  EXPECT_EQ(BoxView({3, 11}, {5, 6}), output_range);
}

TEST(GetOutputRangeTest, SingleInputDimensionNotExactBecauseStride2) {
  Box<> output_range(1);
  // Transform maps {1,2,3,4,5} -> {4,6,8,10,12}
  auto result = GetOutputRange(IndexTransformBuilder<>(1, 1)
                                   .input_origin({1})
                                   .input_shape({5})
                                   .output_single_input_dimension(0, 2, 2, 0)
                                   .Finalize()
                                   .value(),
                               output_range);
  ASSERT_TRUE(result);
  // Bounds are not exact range because {5,7,9,11} are contained in the box but
  // not in the range of the transform.
  EXPECT_FALSE(*result);
  EXPECT_EQ(BoxView({4}, {9}), output_range);
}

TEST(GetOutputRangeTest, SingleInputDimensionNotExactBecauseStrideNegative2) {
  Box<> output_range(1);
  // Transform maps {1,2,3,4,5} -> {0,-2,-4,-6,-8}
  auto result = GetOutputRange(IndexTransformBuilder<>(1, 1)
                                   .input_origin({1})
                                   .input_shape({5})
                                   .output_single_input_dimension(0, 2, -2, 0)
                                   .Finalize()
                                   .value(),
                               output_range);
  ASSERT_TRUE(result);
  // Bounds are not exact because {-1,-3,-5,-7} are not in the range of the
  // transform.
  EXPECT_FALSE(*result);
  EXPECT_EQ(BoxView({-8}, {9}), output_range);
}

TEST(GetOutputRangeTest, SingleInputDimensionNotExactBecauseDiagonal) {
  Box<> output_range(2);
  // Transform maps {1,2,3,4,5} -> {{3,6},{4,7},{5,8},{6,9},{7,10}}
  auto result = GetOutputRange(IndexTransformBuilder<>(1, 2)
                                   .input_origin({1})
                                   .input_shape({5})
                                   .output_single_input_dimension(0, 2, 1, 0)
                                   .output_single_input_dimension(1, 5, 1, 0)
                                   .Finalize()
                                   .value(),
                               output_range);
  ASSERT_TRUE(result);
  // Bounds are not exact because off-diagonal positions, like {3,7}, are not in
  // the range of the transform.
  EXPECT_FALSE(*result);
  EXPECT_EQ(BoxView({3, 6}, {5, 5}), output_range);
}

TEST(GetOutputRangeTest, ArrayUnbounded) {
  Box<> output_range(1);
  // Transform maps {1,2,3,4,5} -> {17,-4,23,29,-10}
  // index_range associated with index array output index map is [-inf,+inf].
  auto result = GetOutputRange(
      IndexTransformBuilder<>(1, 1)
          .input_origin({1})
          .input_shape({5})
          .output_index_array(0, 2, 3, MakeArray<Index>({5, -2, 7, 9, -4}))
          .Finalize()
          .value(),
      output_range);
  ASSERT_TRUE(result);
  // Bounds are not exact because of the use of an index array.
  EXPECT_FALSE(*result);
  // output_range is [-inf,+inf].  For index array output index maps, the bounds
  // are computed based on the stored index_range, which is unbounded in this
  // case, not the actual index values.
  EXPECT_EQ(tensorstore::BoxView<>(1), output_range);
}

TEST(GetOutputRangeTest, ArrayBounded) {
  Box<> output_range(1);
  // Transform maps {1,2,3,4,5} -> {17,-4,23,29,-10}
  // index_range is [-4,9].
  // index_range transformed by offset of 2 and stride of 3 is [-10,29].
  auto result = GetOutputRange(
      IndexTransformBuilder<>(1, 1)
          .input_origin({1})
          .input_shape({5})
          .output_index_array(0, 2, 3, MakeArray<Index>({5, -2, 7, 9, -4}),
                              IndexInterval::Closed(-4, 9))
          .Finalize()
          .value(),
      output_range);
  ASSERT_TRUE(result);
  // Bounds are not exact because of the use of an index array.
  EXPECT_FALSE(*result);
  const auto interval = IndexInterval::UncheckedClosed(-4 * 3 + 2, 9 * 3 + 2);
  EXPECT_EQ(BoxView({interval.inclusive_min()}, {interval.size()}),
            output_range);
}

TEST(GetOutputRangeTest, InvalidConstantOffset) {
  Box<> output_range(1);
  // Transform maps [-inf,+inf] -> std::numeric_limits<Index>::max()
  auto result =
      GetOutputRange(IndexTransformBuilder<>(0, 1)
                         .output_constant(0, std::numeric_limits<Index>::max())
                         .Finalize()
                         .value(),
                     output_range);
  EXPECT_THAT(result, MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(GetOutputRangeTest, InvalidSingleInputDimension) {
  Box<> output_range(1);
  // Transform maps {kMaxFiniteIndex} -> {kMaxFiniteIndex+2}
  auto result = GetOutputRange(IndexTransformBuilder<>(1, 1)
                                   .input_origin({kMaxFiniteIndex})
                                   .input_shape({1})
                                   .output_single_input_dimension(0, 2, 1, 0)
                                   .Finalize()
                                   .value(),
                               output_range);
  EXPECT_THAT(result, MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(GetOutputRangeTest, InvalidIndexRange) {
  Box<> output_range(1);
  // Transform maps {0} -> {10}
  // index_range is [kMaxFiniteIndex,kMaxFiniteIndex]
  // index_range transformed by offset of 10 is
  // [kMaxFiniteIndex+10,kMaxFiniteIndex+10].
  auto result = GetOutputRange(
      IndexTransformBuilder<>(1, 1)
          .input_origin({0})
          .input_shape({1})
          .output_index_array(0, 10, 1, MakeArray<Index>({0}),
                              IndexInterval::Sized(kMaxFiniteIndex, 1))
          .Finalize()
          .value(),
      output_range);
  EXPECT_THAT(result, MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
