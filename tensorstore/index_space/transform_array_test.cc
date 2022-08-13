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

#include "tensorstore/index_space/internal/transform_array.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::IdentityTransform;
using ::tensorstore::Index;
using ::tensorstore::IndexInterval;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::IndexTransformView;
using ::tensorstore::MakeArray;
using ::tensorstore::MakeOffsetArray;
using ::tensorstore::MatchesStatus;

TEST(TransformArrayTest, OneDimensionalIdentity) {
  auto original_array = tensorstore::MakeArray<int>({1, 2, 3, 4});
  auto new_array =
      tensorstore::TransformArray(original_array, IdentityTransform<1>())
          .value();
  EXPECT_EQ(original_array, new_array);
}

TEST(TransformArrayTest, OneDimensionalIdentityWithOrigin) {
  auto original_array = tensorstore::MakeOffsetArray<int>({5}, {1, 2, 3, 4});
  auto new_array =
      tensorstore::TransformArray(original_array, IdentityTransform<1>())
          .value();
  EXPECT_EQ(original_array, new_array);
}

TEST(TransformArrayTest, OneDimensionalSliceUnstrided) {
  auto original_array = tensorstore::MakeArray<int>({1, 2, 3, 4});
  auto new_array = tensorstore::TransformArray(
                       original_array, IndexTransformBuilder<1, 1>()
                                           .input_origin({1})
                                           .input_shape({2})
                                           .output_single_input_dimension(0, 0)
                                           .Finalize()
                                           .value())
                       .value();
  EXPECT_EQ(&original_array(1), &new_array(1));
  EXPECT_EQ(MakeOffsetArray<int>({1}, {2, 3}), new_array);
}

TEST(TransformArrayTest, OneDimensionalSliceUnstridedWithOrigin) {
  auto original_array = tensorstore::MakeOffsetArray<int>({5}, {1, 2, 3, 4});
  auto new_array =
      tensorstore::TransformArray(original_array,
                                  IndexTransformBuilder<1, 1>()
                                      .input_origin({1})
                                      .input_shape({2})
                                      .output_single_input_dimension(0, 5, 1, 0)
                                      .Finalize()
                                      .value())
          .value();
  EXPECT_EQ(&original_array(6), &new_array(1));
  EXPECT_EQ(MakeOffsetArray<int>({1}, {2, 3}), new_array);
}

TEST(TransformArrayTest, OneDimensionalSliceStrided) {
  auto original_array = tensorstore::MakeArray<int>({1, 2, 3, 4});
  auto new_array =
      tensorstore::TransformArray(
          original_array, IndexTransformBuilder<1, 1>()
                              .input_origin({1})
                              .input_shape({2})
                              .output_single_input_dimension(0, -1, 2, 0)
                              .Finalize()
                              .value())
          .value();
  EXPECT_EQ(&original_array(1), &new_array(1));
  EXPECT_EQ(MakeOffsetArray<int>({1}, {2, 4}), new_array);
}

TEST(TransformArrayTest, OneDimensionalSliceStridedWithOrigin) {
  auto original_array = tensorstore::MakeOffsetArray<int>({5}, {1, 2, 3, 4});
  auto new_array =
      tensorstore::TransformArray(original_array,
                                  IndexTransformBuilder<1, 1>()
                                      .input_origin({1})
                                      .input_shape({2})
                                      .output_single_input_dimension(0, 4, 2, 0)
                                      .Finalize()
                                      .value())
          .value();
  EXPECT_EQ(&original_array(6), &new_array(1));
  EXPECT_EQ(MakeOffsetArray<int>({1}, {2, 4}), new_array);
}

TEST(TransformArrayTest, OneDArrayOneDIndexArray) {
  auto original_array = tensorstore::MakeArray<int>({1, 2, 3, 4});
  auto new_array =
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<1, 1>()
              .input_origin({2})
              .input_shape({4})
              .output_index_array(0, 1, 1, MakeArray<Index>({0, 2, 2, 1}))
              .Finalize()
              .value())
          .value();
  EXPECT_EQ(MakeOffsetArray<int>({2}, {2, 4, 4, 3}), new_array);
}

// Tests that TransformArray correctly handles index arrays that exceed the
// internal buffer size of 1024.
TEST(TransformArrayTest, OneDArrayOneDIndexArray1025) {
  constexpr Index kSize = 1025;
  auto index_array = tensorstore::AllocateArray<Index>({kSize});
  for (Index i = 0; i < kSize; ++i) index_array(i) = i;
  auto new_array =
      tensorstore::TransformArray(index_array,
                                  IndexTransformBuilder<1, 1>()
                                      .input_shape({kSize})
                                      .output_index_array(0, 0, 1, index_array)
                                      .Finalize()
                                      .value())
          .value();
  EXPECT_EQ(index_array, new_array);
}

// Tests that TransformArray retains zero-stride dimensions of the array as
// zero-stride when `skip_repeated_elements` is specified.
TEST(TransformArrayTest, TwoDArrayOneDIndexArrayRetainZeroStride) {
  auto index_array = tensorstore::MakeArray<Index>({0, 1, 2, 3, 4});
  tensorstore::SharedArray<Index, 2> index_array2;
  index_array2.element_pointer() = index_array.element_pointer();
  index_array2.shape()[0] = 5;
  index_array2.shape()[1] = 2;
  index_array2.byte_strides()[0] = index_array.byte_strides()[0];
  index_array2.byte_strides()[1] = 0;
  EXPECT_EQ(index_array2,
            MakeArray<Index>({{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}));
  auto new_array =
      tensorstore::TransformArray(index_array2,
                                  IndexTransformBuilder<2, 2>()
                                      .input_shape({5, 2})
                                      .output_index_array(0, 0, 1, index_array2)
                                      .output_single_input_dimension(1, 1)
                                      .Finalize()
                                      .value())
          .value();
  EXPECT_EQ(index_array2, new_array);
  EXPECT_EQ(index_array2.layout(), new_array.layout());
}

TEST(TransformArrayTest, IndexArrayBoundsOverflow) {
  auto original_array = tensorstore::MakeOffsetArray<int>({5}, {1, 2, 3, 4});
  EXPECT_THAT(tensorstore::TransformArray(
                  original_array,
                  IndexTransformBuilder<1, 1>()
                      .input_origin({2})
                      .input_shape({4})
                      .output_index_array(0, std::numeric_limits<Index>::min(),
                                          1, MakeArray<Index>({0, 2, 2, 1}))
                      .Finalize()
                      .value())
                  .status(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*Integer overflow propagating range.*"));
}

TEST(TransformArrayTest, OneDArrayOneDIndexArrayWithOrigin) {
  auto original_array = tensorstore::MakeOffsetArray<int>({5}, {1, 2, 3, 4});
  auto new_array =
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<1, 1>()
              .input_origin({2})
              .input_shape({4})
              .output_index_array(0, 6, 1, MakeArray<Index>({0, 2, 2, 1}))
              .Finalize()
              .value())
          .value();
  EXPECT_EQ(MakeOffsetArray<int>({2}, {2, 4, 4, 3}), new_array);
}

TEST(TransformArrayTest, TwoDArrayOneDIndexArray) {
  auto original_array =
      tensorstore::MakeArray<int>({{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto new_array =
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<2, 2>()
              .input_origin({1, 2})
              .input_shape({2, 4})
              .output_single_input_dimension(0, -1, 1, 0)
              .output_index_array(1, 1, 1, MakeArray<Index>({{0, 2, 2, 1}}))
              .Finalize()
              .value())
          .value();
  EXPECT_EQ(MakeOffsetArray<int>({1, 2}, {{2, 4, 4, 3}, {6, 8, 8, 7}}),
            new_array);
}

TEST(TransformArrayTest, TwoDArrayOneDIndexArrayWithOrigin) {
  auto original_array =
      tensorstore::MakeOffsetArray<int>({5, 6}, {{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto new_array =
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<2, 2>()
              .input_origin({1, 2})
              .input_shape({2, 4})
              .output_single_input_dimension(0, 4, 1, 0)
              .output_index_array(1, 7, 1, MakeArray<Index>({{0, 2, 2, 1}}))
              .Finalize()
              .value())
          .value();
  EXPECT_EQ(MakeOffsetArray<int>({1, 2}, {{2, 4, 4, 3}, {6, 8, 8, 7}}),
            new_array);
}

TEST(TransformArrayTest, TwoDArrayOneDIndexArrayStrided) {
  auto original_array =
      tensorstore::MakeArray<int>({{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto new_array =
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<2, 2>()
              .input_origin({1, 2})
              .input_shape({2, 4})
              .output_single_input_dimension(0, 2, -1, 0)
              .output_index_array(1, 1, 2, MakeArray<Index>({{0, 1, 1, 0}}))
              .Finalize()
              .value())
          .value();
  EXPECT_EQ(MakeOffsetArray<int>({1, 2}, {{6, 8, 8, 6}, {2, 4, 4, 2}}),
            new_array);
}

TEST(TransformArrayTest, ArrayIndexOutOfBounds) {
  auto original_array =
      tensorstore::MakeArray<int>({{1, 2, 3, 4}, {5, 6, 7, 8}});
  EXPECT_THAT(
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<2, 2>()
              .input_origin({1, 2})
              .input_shape({2, 4})
              .output_single_input_dimension(0, 2, -1, 0)
              .output_index_array(1, 1, 2, MakeArray<Index>({{0, 2, 1, 0}}))
              .Finalize()
              .value())
          .status(),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    ".*Index 2 is outside valid range \\[0, 2\\).*"));

  EXPECT_THAT(
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<2, 2>()
              .input_origin({1, 2})
              .input_shape({2, 4})
              .output_single_input_dimension(0, 2, -1, 0)
              .output_index_array(1, 1, 2, MakeArray<Index>({{0, -1, 1, 0}}))
              .Finalize()
              .value())
          .status(),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    ".*Index -1 is outside valid range \\[0, 2\\).*"));
}

TEST(TransformArrayTest, TwoDArrayOneDIndexArrayStridedWithOrigin) {
  auto original_array =
      tensorstore::MakeOffsetArray<int>({5, 6}, {{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto new_array =
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<2, 2>()
              .input_origin({1, 2})
              .input_shape({2, 4})
              .output_single_input_dimension(0, 7, -1, 0)
              .output_index_array(1, 7, 2, MakeArray<Index>({{0, 1, 1, 0}}))
              .Finalize()
              .value())
          .value();
  EXPECT_EQ(MakeOffsetArray<int>({1, 2}, {{6, 8, 8, 6}, {2, 4, 4, 2}}),
            new_array);
  EXPECT_THAT(new_array.byte_strides(),
              ::testing::ElementsAre(sizeof(int), sizeof(int) * 2));
}

TEST(TransformArrayTest, IncludeRepeated) {
  auto original_array =
      tensorstore::MakeOffsetArray<int>({5, 6}, {{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto new_array =
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<3, 2>()
              .input_origin({1, 2, 3})
              .input_shape({2, 4, 2})
              .output_single_input_dimension(0, 7, -1, 0)
              .output_index_array(1, 7, 2,
                                  MakeArray<Index>({{{0}, {1}, {1}, {0}}}))
              .Finalize()
              .value(),
          tensorstore::include_repeated_elements)
          .value();
  EXPECT_EQ(MakeOffsetArray<int>({1, 2, 3}, {{{6, 6}, {8, 8}, {8, 8}, {6, 6}},
                                             {{2, 2}, {4, 4}, {4, 4}, {2, 2}}}),
            new_array);
  EXPECT_THAT(
      new_array.byte_strides(),
      ::testing::ElementsAre(sizeof(int) * 2, sizeof(int) * 4, sizeof(int)));
}

TEST(TransformArrayTest, SkipSingleton) {
  auto original_array =
      tensorstore::MakeOffsetArray<int>({5, 6}, {{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto new_array =
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<3, 2>()
              .input_origin({1, 2, 3})
              .input_shape({2, 4, 1})
              .output_single_input_dimension(0, 7, -1, 0)
              .output_index_array(1, 7, 2,
                                  MakeArray<Index>({{{0}, {1}, {1}, {0}}}))
              .Finalize()
              .value(),
          tensorstore::skip_repeated_elements)
          .value();
  EXPECT_EQ(MakeOffsetArray<int>({1, 2, 3},
                                 {{{6}, {8}, {8}, {6}}, {{2}, {4}, {4}, {2}}}),
            new_array);
  EXPECT_THAT(new_array.byte_strides(),
              ::testing::ElementsAre(sizeof(int), sizeof(int) * 2, 0));
}

TEST(TransformArrayTest, SkipRepeated) {
  auto original_array =
      tensorstore::MakeOffsetArray<int>({5, 6}, {{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto new_array =
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<3, 2>()
              .input_origin({1, 2, 3})
              .input_shape({2, 4, 2})
              .output_single_input_dimension(0, 7, -1, 0)
              .output_index_array(1, 7, 2,
                                  MakeArray<Index>({{{0}, {1}, {1}, {0}}}))
              .Finalize()
              .value(),
          tensorstore::skip_repeated_elements)
          .value();
  EXPECT_EQ(MakeOffsetArray<int>({1, 2, 3}, {{{6, 6}, {8, 8}, {8, 8}, {6, 6}},
                                             {{2, 2}, {4, 4}, {4, 4}, {2, 2}}}),
            new_array);
  EXPECT_THAT(new_array.byte_strides(),
              ::testing::ElementsAre(sizeof(int), sizeof(int) * 2, 0));
}

TEST(TransformArrayTest, OrderConstraint) {
  auto original_array =
      tensorstore::MakeOffsetArray<int>({5, 6}, {{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto new_array =
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<2, 2>()
              .input_origin({1, 2})
              .input_shape({2, 4})
              .output_single_input_dimension(0, 7, -1, 0)
              .output_index_array(1, 7, 2, MakeArray<Index>({{0, 1, 1, 0}}))
              .Finalize()
              .value(),
          tensorstore::c_order)
          .value();
  EXPECT_EQ(MakeOffsetArray<int>({1, 2}, {{6, 8, 8, 6}, {2, 4, 4, 2}}),
            new_array);
  EXPECT_THAT(new_array.byte_strides(),
              ::testing::ElementsAre(sizeof(int) * 4, sizeof(int)));
}

TEST(TransformArrayTest, OrderConstraintIncludeRepeated) {
  auto original_array =
      tensorstore::MakeOffsetArray<int>({5, 6}, {{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto new_array =
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<3, 2>()
              .input_origin({1, 2, 3})
              .input_shape({2, 4, 2})
              .output_single_input_dimension(0, 7, -1, 0)
              .output_index_array(1, 7, 2,
                                  MakeArray<Index>({{{0}, {1}, {1}, {0}}}))
              .Finalize()
              .value(),
          {tensorstore::c_order, tensorstore::include_repeated_elements})
          .value();
  EXPECT_EQ(MakeOffsetArray<int>({1, 2, 3}, {{{6, 6}, {8, 8}, {8, 8}, {6, 6}},
                                             {{2, 2}, {4, 4}, {4, 4}, {2, 2}}}),
            new_array);
  EXPECT_THAT(
      new_array.byte_strides(),
      ::testing::ElementsAre(sizeof(int) * 8, sizeof(int) * 2, sizeof(int)));
}

TEST(TransformArrayTest, OrderConstraintSkipRepeated) {
  auto original_array =
      tensorstore::MakeOffsetArray<int>({5, 6}, {{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto new_array =
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<3, 2>()
              .input_origin({1, 2, 3})
              .input_shape({2, 4, 2})
              .output_single_input_dimension(0, 7, -1, 0)
              .output_index_array(1, 7, 2,
                                  MakeArray<Index>({{{0}, {1}, {1}, {0}}}))
              .Finalize()
              .value(),
          {tensorstore::c_order, tensorstore::skip_repeated_elements})
          .value();
  EXPECT_EQ(MakeOffsetArray<int>({1, 2, 3}, {{{6, 6}, {8, 8}, {8, 8}, {6, 6}},
                                             {{2, 2}, {4, 4}, {4, 4}, {2, 2}}}),
            new_array);
  EXPECT_THAT(new_array.byte_strides(),
              ::testing::ElementsAre(sizeof(int) * 4, sizeof(int), 0));
}

TEST(TransformArrayTest, MultipleArrayIndexedDimensions) {
  auto original_array = tensorstore::MakeArray<int>({{1, 2}, {5, 6}});
  auto new_array =
      tensorstore::TransformArray(
          original_array,
          IndexTransformBuilder<2, 2>()
              .input_origin({0, 0})
              .input_shape({2, 2})
              .output_index_array(0, 0, 1, MakeArray<Index>({{0, 1}}))
              .output_index_array(1, 0, 1, MakeArray<Index>({{0}, {1}}))
              .Finalize()
              .value())
          .value();
  EXPECT_EQ(MakeArray<int>({{1, 5}, {2, 6}}), new_array);
}

TEST(TransformArrayTest, EmptyDomain) {
  auto original_array = tensorstore::MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(  //
      auto transform,                //
      (IndexTransformBuilder<2, 2>()
           .input_shape({0, 3})
           .implicit_upper_bounds({1, 0})
           .output_single_input_dimension(0, 0)
           .output_index_array(0, 0, 1, MakeArray<Index>({{0, 1, 2}}))
           .Finalize()));
  EXPECT_THAT(tensorstore::TransformArray(original_array, transform),
              ::testing::Optional(tensorstore::AllocateArray<int>({0, 3})));
}

}  // namespace
