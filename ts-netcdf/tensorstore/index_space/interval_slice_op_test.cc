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

/// Tests for the
/// DimExpression::{{Translate,}{Closed,HalfOpen,Sized}Interval,Stride}
/// operations.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/box.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/dim_expression_testutil.h"
#include "tensorstore/util/status.h"

namespace {

using ::tensorstore::AllDims;
using ::tensorstore::BoxView;
using ::tensorstore::Dims;
using ::tensorstore::Index;
using ::tensorstore::IndexInterval;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::kImplicit;
using ::tensorstore::kInfIndex;
using ::tensorstore::kInfSize;
using ::tensorstore::MakeArray;
using ::tensorstore::StrCat;
using ::tensorstore::internal_index_space::EquivalentIndices;
using ::tensorstore::internal_index_space::TestDimExpression;
using ::tensorstore::internal_index_space::TestDimExpressionError;

TEST(ClosedIntervalTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({0, 2, 0})
                                      .input_shape({7, 4, 10})
                                      .input_labels({"x", "y", "z"})
                                      .output_identity_transform()
                                      .Finalize()
                                      .value();
  const auto expected_new_transform =
      IndexTransformBuilder<3, 3>()
          .input_origin({1, 2, -4})
          .input_shape({4, 4, 3})
          .input_labels({"x", "y", "z"})
          .output_single_input_dimension(0, 0)
          .output_single_input_dimension(1, 1)
          .output_single_input_dimension(2, 0, -2, 2)
          .Finalize()
          .value();
  const EquivalentIndices equivalent_indices = {
      {{2, 3, 6}, {2, 3, -3}},
  };
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/Dims(0, 2).ClosedInterval({1, 8}, {4, 3}, {1, -2}),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices);

  // Test using labels to select dimensions.
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/Dims("x", "z").ClosedInterval({1, 8}, {4, 3}, {1, -2}),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices);
}

TEST(ClosedIntervalTest, ExampleWithOffset) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({0, 2, 0})
                                      .input_shape({7, 4, 10})
                                      .input_labels({"x", "y", "z"})
                                      .output_identity_transform()
                                      .Finalize()
                                      .value();
  const auto expected_new_transform =
      IndexTransformBuilder<3, 3>()
          .input_origin({1, 2, -4})
          .input_shape({4, 4, 4})
          .input_labels({"x", "y", "z"})
          .output_single_input_dimension(0, 0)
          .output_single_input_dimension(1, 1)
          .output_single_input_dimension(2, 1, -2, 2)
          .Finalize()
          .value();
  const EquivalentIndices equivalent_indices = {
      {{2, 3, 7}, {2, 3, -3}},
  };
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/Dims(0, 2).ClosedInterval({1, 9}, {4, 3}, {1, -2}),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices);
}

TEST(HalfOpenIntervalTest, Example) {
  const auto expected_new_transform =
      IndexTransformBuilder<3, 3>()
          .input_origin({1, 2, -4})
          .input_shape({3, 4, 3})
          .input_labels({"x", "y", "z"})
          .output_single_input_dimension(0, 0)
          .output_single_input_dimension(1, 1)
          .output_single_input_dimension(2, 0, -2, 2)
          .Finalize()
          .value();
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<3, 3>()
          .input_origin({0, 2, 0})
          .input_shape({7, 4, 10})
          .input_labels({"x", "y", "z"})
          .output_identity_transform()
          .Finalize()
          .value(),
      /*expression=*/Dims(0, 2).HalfOpenInterval({1, 8}, {4, 3}, {1, -2}),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/{{{2, 3, 6}, {2, 3, -3}}});
}

TEST(SizedIntervalTest, Example) {
  const auto expected_new_transform =
      IndexTransformBuilder<3, 3>()
          .input_origin({1, 2, -4})
          .input_shape({3, 4, 2})
          .input_labels({"x", "y", "z"})
          .output_single_input_dimension(0, 0)
          .output_single_input_dimension(1, 1)
          .output_single_input_dimension(2, 0, -2, 2)
          .Finalize()
          .value();
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<3, 3>()
          .input_origin({0, 2, 0})
          .input_shape({7, 4, 10})
          .input_labels({"x", "y", "z"})
          .output_identity_transform()
          .Finalize()
          .value(),
      /*expression=*/Dims(0, 2).SizedInterval({1, 8}, {3, 2}, {1, -2}),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/{{{2, 3, 6}, {2, 3, -3}}});
}

TEST(ClosedIntervalTest, OneDimensionalConstantNonStrided) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 1>()
                        .input_origin({-5})
                        .input_shape({10})
                        .output_constant(0, 3)
                        .Finalize()
                        .value(),
                    /*expression=*/AllDims().ClosedInterval(-3, 4),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-3})
                        .input_shape({8})
                        .output_identity_transform()
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-3})
                        .input_shape({8})
                        .output_constant(0, 3)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{1}, {1}}});
}

TEST(ClosedIntervalTest, OneDimensionalConstantPositiveStrided) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 1>()
                        .input_origin({-5})
                        .input_shape({12})
                        .output_constant(0, 3)
                        .Finalize()
                        .value(),
                    /*expression=*/AllDims().ClosedInterval(-3, 5, 2),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-1})
                        .input_shape({5})
                        .output_single_input_dimension(0, -1, 2, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-1})
                        .input_shape({5})
                        .output_constant(0, 3)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{1}, {1}}});
}

TEST(ClosedIntervalTest, OneDimensionalConstantNegativeStrided) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 1>()
                        .input_origin({-5})
                        .input_shape({12})
                        .output_constant(0, 3)
                        .Finalize()
                        .value(),
                    /*expression=*/AllDims().ClosedInterval(5, -3, -2),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-2})
                        .input_shape({5})
                        .output_single_input_dimension(0, 1, -2, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-2})
                        .input_shape({5})
                        .output_constant(0, 3)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{1}, {1}}});
}

TEST(ClosedIntervalTest, OneDimensionalSingleInputDimensionNonStrided) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 1>()
                        .input_origin({-5})
                        .input_shape({10})
                        .output_single_input_dimension(0, 3, 2, 0)
                        .Finalize()
                        .value(),
                    /*expression=*/AllDims().ClosedInterval(-3, 4),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-3})
                        .input_shape({8})
                        .output_identity_transform()
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-3})
                        .input_shape({8})
                        .output_single_input_dimension(0, 3, 2, 0)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{1}, {1}}});
}

TEST(ClosedIntervalTest, OneDimensionalSingleInputDimensionPositiveStrided) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 1>()
                        .input_origin({-5})
                        .input_shape({12})
                        .output_single_input_dimension(0, 3, 2, 0)
                        .Finalize()
                        .value(),
                    /*expression=*/AllDims().ClosedInterval(-3, 5, 2),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-1})
                        .input_shape({5})
                        .output_single_input_dimension(0, -1, 2, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-1})
                        .input_shape({5})
                        .output_single_input_dimension(0, 1, 4, 0)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{-3}, {-1}}, {{-1}, {0}}});
}

TEST(ClosedIntervalTest, OneDimensionalArrayNonStrided) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<1, 1>()
          .input_origin({-2})
          .input_shape({4})
          .output_index_array(0, 3, 2, MakeArray<Index>({6, 5, 4, 3}))
          .Finalize()
          .value(),
      /*expression=*/AllDims().ClosedInterval(-1, 1),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<1, 1>()
          .input_origin({-1})
          .input_shape({3})
          .output_identity_transform()
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<1, 1>()
          .input_origin({-1})
          .input_shape({3})
          .output_index_array(0, 3, 2, MakeArray<Index>({5, 4, 3}))
          .Finalize()
          .value(),
      /*equivalent_indices=*/{{{1}, {1}}});
}

TEST(ClosedIntervalTest, OneDimensionalArrayNonStridedZeroElements) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<1, 1>()
          .input_origin({-2})
          .input_shape({4})
          .output_index_array(0, 3, 2, MakeArray<Index>({6, 5, 4, 3}))
          .Finalize()
          .value(),
      /*expression=*/AllDims().ClosedInterval(-1, -2),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<1, 1>()
          .input_origin({-1})
          .input_shape({0})
          .output_identity_transform()
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<1, 1>()
          .input_origin({-1})
          .input_shape({0})
          .output_constant(0, 0)
          .Finalize()
          .value(),
      /*equivalent_indices=*/{});
}

TEST(ClosedIntervalTest, OneDimensionalArrayNonStridedOneElement) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<1, 1>()
          .input_origin({-2})
          .input_shape({4})
          .output_index_array(0, 3, 2, MakeArray<Index>({6, 5, 4, 3}))
          .Finalize()
          .value(),
      /*expression=*/AllDims().ClosedInterval(-1, -1),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<1, 1>()
          .input_origin({-1})
          .input_shape({1})
          .output_identity_transform()
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<1, 1>()
          .input_origin({-1})
          .input_shape({1})
          .output_constant(0, 13)
          .Finalize()
          .value(),
      /*equivalent_indices=*/{{{-1}, {-1}}});
}

TEST(ClosedIntervalTest, OneDimensionalArrayNonStridedInvalidOneElement) {
  TestDimExpressionError(
      IndexTransformBuilder<1, 1>()
          .input_origin({-2})
          .input_shape({4})
          .output_index_array(0, 3, 2, MakeArray<Index>({6, 5, 4, 3}),
                              IndexInterval::Closed(3, 4))
          .Finalize()
          .value(),
      AllDims().ClosedInterval(-1, -1), absl::StatusCode::kOutOfRange,
      "Index 5 is outside valid range \\[3, 5\\)");
}

TEST(SliceTranslateClosedIntervalTest, OneDimensionalArrayNonStrided) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<1, 1>()
          .input_origin({-2})
          .input_shape({4})
          .output_index_array(0, 3, 2, MakeArray<Index>({6, 5, 4, 3}))
          .Finalize()
          .value(),
      /*expression=*/AllDims().TranslateClosedInterval(-1, 1),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<1, 1>()
          .input_origin({0})
          .input_shape({3})
          .output_single_input_dimension(0, -1, 1, 0)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<1, 1>()
          .input_origin({0})
          .input_shape({3})
          .output_index_array(0, 3, 2, MakeArray<Index>({5, 4, 3}))
          .Finalize()
          .value(),
      /*equivalent_indices=*/{{{1}, {2}}});
}

TEST(SliceTranslateClosedIntervalTest, OneDimensionalArrayStrided) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<1, 1>()
          .input_origin({-2})
          .input_shape({4})
          .output_index_array(0, 3, 2, MakeArray<Index>({6, 5, 4, 3}))
          .Finalize()
          .value(),
      /*expression=*/AllDims().TranslateClosedInterval(-1, 1, 2),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<1, 1>()
          .input_origin({0})
          .input_shape({2})
          .output_single_input_dimension(0, -1, 2, 0)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<1, 1>()
          .input_origin({0})
          .input_shape({2})
          .output_index_array(0, 3, 2, MakeArray<Index>({5, 3}))
          .Finalize()
          .value(),
      /*equivalent_indices=*/{{{-1}, {0}}});
}

TEST(ClosedIntervalTest, DimSubset) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<4, 4>()
                        .input_origin({-10, 1, 2, -kInfIndex})
                        .input_shape({kInfIndex + 1, 4, 5, kInfIndex + 7})
                        .output_single_input_dimension(0, 1, 4, 1)
                        .output_single_input_dimension(1, 2, 3, 3)
                        .output_constant(2, 3)
                        .output_index_array(
                            3, 4, 1,
                            MakeArray<Index>(
                                {{{{5}, {6}, {7}, {8}, {9}},
                                  {{15}, {16}, {17}, {18}, {19}},
                                  {{25}, {26}, {27}, {28}, {29}},
                                  {{35}, {36}, {37}, {38}, {39}}}}))
                        .Finalize()
                        .value(),
                    /*expression=*/
                    Dims(1, 2, 0).ClosedInterval({2, 2, -5}, {3, 4, 10}),
                    /*expected_new_dimension_selection=*/{1, 2, 0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<4, 4>()
                        .input_origin({-5, 2, 2, -kInfIndex})
                        .input_shape({16, 2, 3, kInfIndex + 7})
                        .output_identity_transform()
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<4, 4>()
                        .input_origin({-5, 2, 2, -kInfIndex})
                        .input_shape({16, 2, 3, kInfIndex + 7})
                        .output_single_input_dimension(0, 1, 4, 1)
                        .output_single_input_dimension(1, 2, 3, 3)
                        .output_constant(2, 3)
                        .output_index_array(
                            3, 4, 1,
                            MakeArray<Index>(
                                {{{{15}, {16}, {17}}, {{25}, {26}, {27}}}}))
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{1, 2, 3, 4}, {1, 2, 3, 4}}});
}

TEST(SliceClosedIntervalTest, DimSubsetStriding) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<4, 4>()
                        .input_origin({-10, 1, 2, -kInfIndex})
                        .input_shape({kInfIndex + 1, 4, 5, kInfIndex + 7})
                        .output_single_input_dimension(0, 1, 4, 1)
                        .output_single_input_dimension(1, 2, 3, 3)
                        .output_constant(2, 3)
                        .output_index_array(
                            3, 4, 1,
                            MakeArray<Index>(
                                {{{{5}, {6}, {7}, {8}, {9}},
                                  {{15}, {16}, {17}, {18}, {19}},
                                  {{25}, {26}, {27}, {28}, {29}},
                                  {{35}, {36}, {37}, {38}, {39}}}}))
                        .Finalize()
                        .value(),
                    /*expression=*/
                    Dims(1, 2, 0, 3)
                        .ClosedInterval({3, 2, 10, 1}, {2, 4, -5, kImplicit},
                                        {-1, 2, -2, 4}),
                    /*expected_new_dimension_selection=*/{1, 2, 0, 3},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<4, 4>()
                        .input_origin({-5, -3, 1, 0})
                        .input_shape({8, 2, 2, 2})
                        .output_single_input_dimension(0, 0, -2, 0)
                        .output_single_input_dimension(1, 0, -1, 1)
                        .output_single_input_dimension(2, 0, 2, 2)
                        .output_single_input_dimension(3, 1, 4, 3)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<4, 4>()
                        .input_origin({-5, -3, 1, 0})
                        .input_shape({8, 2, 2, 2})
                        .output_single_input_dimension(0, 1, -4, 1)
                        .output_single_input_dimension(1, 5, 12, 3)
                        .output_constant(2, 3)
                        .output_index_array(
                            3, 4, 1,
                            MakeArray<Index>({{{{25}, {27}}, {{15}, {17}}}}))
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{2, 2, 2, 5}, {-1, -2, 1, 1}}});
}

TEST(SliceClosedIntervalTest, UnboundedStart) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).ClosedInterval(kImplicit, 9),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({5})
                        .input_shape({5})
                        .output_identity_transform()
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({5})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(SliceClosedIntervalTest, OneDimensionalNegativeStridedUnboundedStop) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).ClosedInterval(12, kImplicit, -1),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-12})
                        .input_shape({8})
                        .output_single_input_dimension(0, 0, -1, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({-12})
                        .input_shape({8})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(SliceHalfOpenIntervalTest, OneDimensionalUnstrided) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).HalfOpenInterval(6, 10),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({6})
                        .input_shape({4})
                        .output_identity_transform()
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({6})
                        .input_shape({4})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(SliceHalfOpenIntervalTest, OneDimensionalUnstridedUnboundedStart) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).HalfOpenInterval(kImplicit, 10),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({5})
                        .input_shape({5})
                        .output_identity_transform()
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({5})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(SliceHalfOpenIntervalTest, OneDimensionalUnstridedUnboundedStop) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).HalfOpenInterval(6, kImplicit),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({6})
                        .input_shape({9})
                        .output_identity_transform()
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({6})
                        .input_shape({9})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(SliceHalfOpenIntervalTest, OneDimensionalNegativeStridedUnboundedStop) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).HalfOpenInterval(12, kImplicit, -1),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-12})
                        .input_shape({8})
                        .output_single_input_dimension(0, 0, -1, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({-12})
                        .input_shape({8})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(SliceHalfOpenIntervalTest, ErrorHandling) {
  TestDimExpressionError(
      IndexTransformBuilder<1, 0>().Finalize().value(),
      Dims(0).HalfOpenInterval(6, std::numeric_limits<Index>::min() + 1),
      absl::StatusCode::kInvalidArgument,
      StrCat(".* do not specify a valid closed index interval"));
}

TEST(SliceClosedIntervalTest, ErrorHandling) {
  TestDimExpressionError(IndexTransformBuilder<1, 0>()
                             .input_origin({5})
                             .input_shape({10})
                             .Finalize()
                             .value(),
                         Dims(0).ClosedInterval(6, 10, 0),
                         absl::StatusCode::kInvalidArgument,
                         ".*Invalid stride 0");

  TestDimExpressionError(
      IndexTransformBuilder<1, 0>()
          .input_origin({5})
          .input_shape({10})
          .Finalize()
          .value(),
      Dims(0).ClosedInterval(6, 4), absl::StatusCode::kInvalidArgument,
      ".*\\(6, 4\\) do not specify a valid closed index interval");

  TestDimExpressionError(
      IndexTransformBuilder<1, 0>().input_shape({10}).Finalize().value(),
      Dims(0).ClosedInterval(-kInfIndex, 4, 2),
      absl::StatusCode::kInvalidArgument,
      ".*Slicing with non-unit stride of 2 requires a finite start index");

  TestDimExpressionError(
      IndexTransformBuilder<1, 0>()
          .input_origin({2})
          .input_shape({kInfIndex - 2 + 1})
          .Finalize()
          .value(),
      Dims(0).ClosedInterval(kInfIndex, 4, -2),
      absl::StatusCode::kInvalidArgument,
      ".*Slicing with non-unit stride of -2 requires a finite start index");

  TestDimExpressionError(IndexTransformBuilder<1, 0>()
                             .input_origin({5})
                             .input_shape({10})
                             .Finalize()
                             .value(),
                         Dims(0).ClosedInterval(6, 15),
                         absl::StatusCode::kOutOfRange,
                         ".*Slice interval \\[6, 16\\) is not "
                         "contained within domain \\[5, 15\\)");

  TestDimExpressionError(
      IndexTransformBuilder<1, 1>()
          .input_origin({5})
          .input_shape({10})
          .output_single_input_dimension(0, 0,
                                         std::numeric_limits<Index>::max(), 0)
          .Finalize()
          .value(),
      Dims(0).ClosedInterval(5, 10, 3), absl::StatusCode::kInvalidArgument,
      "Integer overflow computing offset for output dimension 0");

  TestDimExpressionError(
      IndexTransformBuilder<1, 1>()
          .input_origin({5})
          .input_shape({10})
          .output_single_input_dimension(0, std::numeric_limits<Index>::max(),
                                         1, 0)
          .Finalize()
          .value(),
      Dims(0).ClosedInterval(5, 10, 3), absl::StatusCode::kInvalidArgument,
      "Integer overflow computing offset for output dimension 0");

  TestDimExpressionError(
      IndexTransformBuilder<1, 1>()
          .input_origin({5})
          .input_shape({10})
          .output_single_input_dimension(0, 0,
                                         std::numeric_limits<Index>::max(), 0)
          .Finalize()
          .value(),
      Dims(0).ClosedInterval(5, 10, 2), absl::StatusCode::kInvalidArgument,
      "Integer overflow computing stride for output dimension 0");
}

TEST(SliceTranslateClosedIntervalTest, ErrorHandling) {
  TestDimExpressionError(IndexTransformBuilder<1, 0>().Finalize().value(),
                         Dims(0).TranslateClosedInterval(-kInfIndex, 100),
                         absl::StatusCode::kInvalidArgument,
                         ".*Interval \\(-inf, 101\\) is not bounded below");
}

TEST(SliceSizedIntervalTest, ErrorHandling) {
  TestDimExpressionError(IndexTransformBuilder<1, 0>()
                             .input_origin({5})
                             .input_shape({10})
                             .Finalize()
                             .value(),
                         Dims(0).SizedInterval(6, -2),
                         absl::StatusCode::kInvalidArgument,
                         ".*Negative size -2 specified for sized interval");
  TestDimExpressionError(IndexTransformBuilder<1, 0>().Finalize().value(),
                         Dims(0).SizedInterval(6, kInfIndex - 1, 100),
                         absl::StatusCode::kOutOfRange,
                         ".*Integer overflow computing slice result");
  TestDimExpressionError(IndexTransformBuilder<1, 0>().Finalize().value(),
                         Dims(0).SizedInterval(6, kInfSize - 1),
                         absl::StatusCode::kOutOfRange,
                         ".*Integer overflow computing slice result");
}

TEST(SliceSizedIntervalTest, OneDimensionalUnstrided) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).SizedInterval(6, 3),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({6})
                        .input_shape({3})
                        .output_identity_transform()
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({6})
                        .input_shape({3})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(SliceSizedIntervalTest, OneDimensionalUnstridedUnboundedMin) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).SizedInterval(kImplicit, 3),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({5})
                        .input_shape({3})
                        .output_identity_transform()
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({3})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(SliceSizedIntervalTest, OneDimensionalUnstridedUnboundedSize) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).SizedInterval(6, kImplicit),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({6})
                        .input_shape({9})
                        .output_identity_transform()
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({6})
                        .input_shape({9})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(SliceSizedIntervalTest, OneDimensionalPositiveStrided) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).SizedInterval(7, 3, 2),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({3})
                        .input_shape({3})
                        .output_single_input_dimension(0, 1, 2, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({3})
                        .input_shape({3})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(SliceSizedIntervalTest, OneDimensionalPositiveStridedUnboundedSize) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).SizedInterval(6, kImplicit, 2),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({3})
                        .input_shape({5})
                        .output_single_input_dimension(0, 0, 2, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({3})
                        .input_shape({5})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});

  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).SizedInterval(7, kImplicit, 2),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({3})
                        .input_shape({4})
                        .output_single_input_dimension(0, 1, 2, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({3})
                        .input_shape({4})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(SliceSizedIntervalTest, OneDimensionalNegativeStrided) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).SizedInterval(13, 3, -2),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-6})
                        .input_shape({3})
                        .output_single_input_dimension(0, 1, -2, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({-6})
                        .input_shape({3})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(SliceSizedIntervalTest, OneDimensionalNegativeStridedUnboundedSize) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 0>()
                        .input_origin({5})
                        .input_shape({10})
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).SizedInterval(13, kImplicit, -2),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({-6})
                        .input_shape({5})
                        .output_single_input_dimension(0, 1, -2, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 0>()
                        .input_origin({-6})
                        .input_shape({5})
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(StrideTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({0, 2, 1})
                                      .input_inclusive_max({6, 5, 8})
                                      .input_labels({"x", "y", "z"})
                                      .output_identity_transform()
                                      .Finalize()
                                      .value();
  const auto expected_new_transform =
      IndexTransformBuilder<3, 3>()
          .input_origin({-3, 2, 1})
          .input_inclusive_max({0, 5, 2})
          .input_labels({"x", "y", "z"})
          .output_single_input_dimension(0, 0, -2, 0)
          .output_single_input_dimension(1, 1)
          .output_single_input_dimension(2, 0, 3, 2)
          .Finalize()
          .value();
  const EquivalentIndices equivalent_indices = {
      {{4, 3, 3}, {-2, 3, 1}},
  };
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/Dims(0, 2).Stride({-2, 3}),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices);

  // Test using labels to select dimensions.
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/Dims("x", "z").Stride({-2, 3}),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices);
}

TEST(StrideTest, ErrorZeroStride) {
  TestDimExpressionError(
      IndexTransformBuilder<1, 0>().Finalize().value(), Dims(0).Stride(0),
      absl::StatusCode::kInvalidArgument,
      StrCat("Applying stride to input dimension 0: Stride must be non-zero"));
}

TEST(StrideTest, ErrorStrideOverflow) {
  TestDimExpressionError(
      IndexTransformBuilder<1, 1>()
          .output_single_input_dimension(0, 0,
                                         std::numeric_limits<Index>::min(), 0)
          .Finalize()
          .value(),
      Dims(0).Stride(std::numeric_limits<Index>::min()),
      absl::StatusCode::kInvalidArgument,
      StrCat("Integer overflow computing stride for output dimension 0"));
}

TEST(BoxSliceTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({0, 2, 0})
                                      .input_inclusive_max({6, 5, 9})
                                      .input_labels({"x", "y", "z"})
                                      .output_identity_transform()
                                      .Finalize()
                                      .value();
  const auto expected_new_transform = IndexTransformBuilder<3, 3>()
                                          .input_origin({1, 2, 4})
                                          .input_inclusive_max({3, 5, 7})
                                          .input_labels({"x", "y", "z"})
                                          .output_identity_transform()
                                          .Finalize()
                                          .value();
  const EquivalentIndices equivalent_indices = {
      {{1, 3, 4}, {1, 3, 4}},
  };
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/Dims(0, 2).BoxSlice(BoxView({1, 4}, {3, 4})),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices);

  // Test using labels to select dimensions.
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/Dims("x", "z").BoxSlice(BoxView({1, 4}, {3, 4})),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices);
}

TEST(TranslateBoxSliceTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({0, 2, 0})
                                      .input_inclusive_max({6, 5, 9})
                                      .input_labels({"x", "y", "z"})
                                      .output_identity_transform()
                                      .Finalize()
                                      .value();
  const auto expected_new_transform =
      IndexTransformBuilder<3, 3>()
          .input_origin({0, 2, 0})
          .input_inclusive_max({2, 5, 3})
          .input_labels({"x", "y", "z"})
          .output_single_input_dimension(0, 1, 1, 0)
          .output_single_input_dimension(1, 1)
          .output_single_input_dimension(2, 4, 1, 2)
          .Finalize()
          .value();
  const EquivalentIndices equivalent_indices = {
      {{1, 3, 4}, {0, 3, 0}},
  };
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/Dims(0, 2).TranslateBoxSlice(BoxView({1, 4}, {3, 4})),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices);

  // Test using labels to select dimensions.
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/Dims("x", "z").TranslateBoxSlice(BoxView({1, 4}, {3, 4})),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices);
}

}  // namespace
