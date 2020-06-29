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

/// Tests for the DimExpression::IndexArraySlice(index_array...),
/// DimExpression::IndexVectorArraySlice(index_vector_array), and
/// DimExpression::OuterIndexArraySlice(index_array...) operations.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/dim_expression_testutil.h"
#include "tensorstore/util/status.h"

namespace {

using tensorstore::AllDims;
using tensorstore::DimensionIndex;
using tensorstore::Dims;
using tensorstore::Index;
using tensorstore::IndexInterval;
using tensorstore::IndexTransformBuilder;
using tensorstore::MakeArray;
using tensorstore::SharedArrayView;
using tensorstore::span;
using tensorstore::internal_index_space::EquivalentIndices;
using tensorstore::internal_index_space::TestDimExpression;

TEST(IndexArraySliceTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({0, 2, 0})
                                      .input_shape({7, 4, 10})
                                      .input_labels({"x", "y", "z"})
                                      .output_single_input_dimension(0, 0, 1, 0)
                                      .output_single_input_dimension(1, 0, 1, 1)
                                      .output_single_input_dimension(2, 0, 1, 2)
                                      .Finalize()
                                      .value();
  const auto expected_new_transform =
      IndexTransformBuilder<3, 3>()
          .input_origin({0, 0, 2})
          .input_shape({2, 3, 4})
          .input_labels({"", "", "y"})
          .output_index_array(
              0, 0, 1, MakeArray<Index>({{{1}, {2}, {3}}, {{4}, {5}, {6}}}),
              IndexInterval::Sized(0, 7))
          .output_single_input_dimension(1, 0, 1, 2)
          .output_index_array(
              2, 0, 1, MakeArray<Index>({{{7}, {8}, {9}}, {{0}, {1}, {2}}}),
              IndexInterval::Sized(0, 10))
          .Finalize()
          .value();
  const EquivalentIndices equivalent_indices = {
      {{1, 3, 7}, {0, 0, 3}},
      {{2, 3, 8}, {0, 1, 3}},
      {{3, 3, 9}, {0, 2, 3}},
      {{6, 3, 2}, {1, 2, 3}},
  };
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/
      Dims(0, 2).IndexArraySlice(MakeArray<Index>({{1, 2, 3}, {4, 5, 6}}),
                                 MakeArray<Index>({{7, 8, 9}, {0, 1, 2}})),
      /*expected_new_dimension_selection=*/{0, 1},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices,
      /*can_operate_in_place=*/false);

  // Test using labels to select dimensions.
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/
      Dims("x", "z").IndexArraySlice(MakeArray<Index>({{1, 2, 3}, {4, 5, 6}}),
                                     MakeArray<Index>({{7, 8, 9}, {0, 1, 2}})),
      /*expected_new_dimension_selection=*/{0, 1},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices,
      /*can_operate_in_place=*/false);
}

TEST(IndexVectorArraySliceTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({0, 2, 0})
                                      .input_shape({7, 4, 10})
                                      .input_labels({"x", "y", "z"})
                                      .output_single_input_dimension(0, 0, 1, 0)
                                      .output_single_input_dimension(1, 0, 1, 1)
                                      .output_single_input_dimension(2, 0, 1, 2)
                                      .Finalize()
                                      .value();
  const auto expected_new_transform =
      IndexTransformBuilder<3, 3>()
          .input_origin({0, 0, 2})
          .input_shape({2, 3, 4})
          .input_labels({"", "", "y"})
          .output_index_array(
              0, 0, 1, MakeArray<Index>({{{1}, {2}, {3}}, {{4}, {5}, {6}}}),
              IndexInterval::Sized(0, 7))
          .output_single_input_dimension(1, 0, 1, 2)
          .output_index_array(
              2, 0, 1, MakeArray<Index>({{{7}, {8}, {9}}, {{0}, {1}, {2}}}),
              IndexInterval::Sized(0, 10))
          .Finalize()
          .value();
  const EquivalentIndices equivalent_indices = {
      {{2, 3, 8}, {0, 1, 3}},
      {{6, 3, 2}, {1, 2, 3}},
  };
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/
                    Dims(0, 2).IndexVectorArraySlice(
                        MakeArray<Index>({{{1, 7}, {2, 8}, {3, 9}},
                                          {{4, 0}, {5, 1}, {6, 2}}}),
                        -1),
                    /*expected_new_dimension_selection=*/{0, 1},
                    /*expected_identity_new_transform=*/
                    expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices,
                    /*can_operate_in_place=*/false);

  // Test using labels to select dimensions.
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/
                    Dims("x", "z").IndexVectorArraySlice(
                        MakeArray<Index>({{{1, 7}, {2, 8}, {3, 9}},
                                          {{4, 0}, {5, 1}, {6, 2}}}),
                        -1),
                    /*expected_new_dimension_selection=*/{0, 1},
                    /*expected_identity_new_transform=*/
                    expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices,
                    /*can_operate_in_place=*/false);
}

TEST(IndexArrayOuterIndexArraySliceTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({4, 2, 0})
                                      .input_shape({5, 4, 10})
                                      .input_labels({"x", "y", "z"})
                                      .output_single_input_dimension(0, 0, 1, 0)
                                      .output_single_input_dimension(1, 0, 1, 1)
                                      .output_single_input_dimension(2, 0, 1, 2)
                                      .Finalize()
                                      .value();
  const auto expected_new_transform =
      IndexTransformBuilder<4, 3>()
          .input_origin({0, 2, 0, 0})
          .input_shape({2, 4, 2, 2})
          .input_labels({"", "y", "", ""})
          .output_index_array(0, 0, 1, MakeArray<Index>({{{{6}}}, {{{7}}}}),
                              IndexInterval::Sized(4, 5))
          .output_single_input_dimension(1, 0, 1, 1)
          .output_index_array(2, 0, 1, MakeArray<Index>({{{{2, 3}, {4, 5}}}}),
                              IndexInterval::Sized(0, 10))
          .Finalize()
          .value();
  const EquivalentIndices equivalent_indices = {
      {{6, 3, 3}, {0, 3, 0, 1}},
      {{7, 3, 4}, {1, 3, 1, 0}},
  };
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/
      Dims(2, 0).OuterIndexArraySlice(MakeArray<Index>({{2, 3}, {4, 5}}),
                                      MakeArray<Index>({6, 7})),
      /*expected_new_dimension_selection=*/{2, 3, 0},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices,
      /*can_operate_in_place=*/false);

  // Test using labels to select dimensions.
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/
      Dims("z", "x").OuterIndexArraySlice(MakeArray<Index>({{2, 3}, {4, 5}}),
                                          MakeArray<Index>({6, 7})),
      /*expected_new_dimension_selection=*/{2, 3, 0},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices,
      /*can_operate_in_place=*/false);
}

TEST(IndexArraySliceTest, OneDOutputOneDArray) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<2, 2>()
          .input_origin({-10, -100})
          .input_shape({20, 200})
          .output_single_input_dimension(0, -2, -3, 0)
          .output_single_input_dimension(1, 10, 11, 1)
          .Finalize()
          .value(),
      /*expression=*/Dims(0).IndexArraySlice(MakeArray<Index>({1, 2})),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<2, 2>()
          .input_origin({0, -100})
          .input_shape({2, 200})
          .output_index_array(0, 0, 1, MakeArray<Index>({{1}, {2}}),
                              IndexInterval::Sized(-10, 20))
          .output_single_input_dimension(1, 0, 1, 1)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<2, 2>()
          .input_origin({0, -100})
          .input_shape({2, 200})
          .output_index_array(0, -2, -3, MakeArray<Index>({{1}, {2}}),
                              IndexInterval::Sized(-10, 20))
          .output_single_input_dimension(1, 10, 11, 1)
          .Finalize()
          .value(),
      /*equivalent_indices=*/{{{1, 5}, {0, 5}}, {{2, 5}, {1, 5}}},
      /*can_operate_in_place=*/false);
}

TEST(IndexArraySliceTest, ZeroElementIndexArray) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({-10, -100})
                        .input_shape({20, 200})
                        .output_single_input_dimension(0, -2, -3, 0)
                        .output_single_input_dimension(1, 10, 11, 1)
                        .Finalize()
                        .value(),
                    /*expression=*/
                    Dims(0).IndexArraySlice(
                        tensorstore::AllocateArray<Index>({5, 0, 3})),
                    /*expected_new_dimension_selection=*/{0, 1, 2},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<4, 2>()
                        .input_origin({0, 0, 0, -100})
                        .input_shape({5, 0, 3, 200})
                        .output_constant(0, 0)
                        .output_single_input_dimension(1, 3)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<4, 2>()
                        .input_origin({0, 0, 0, -100})
                        .input_shape({5, 0, 3, 200})
                        .output_constant(0, -2)
                        .output_single_input_dimension(1, 10, 11, 3)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{},
                    /*can_operate_in_place=*/false);
}

TEST(IndexArraySliceTest, OneElementIndexArray) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<2, 2>()
          .input_origin({-10, -100})
          .input_shape({20, 200})
          .output_single_input_dimension(0, -2, -3, 0)
          .output_single_input_dimension(1, 10, 11, 1)
          .Finalize()
          .value(),
      /*expression=*/Dims(0).IndexArraySlice(MakeArray<Index>({{5}})),
      /*expected_new_dimension_selection=*/{0, 1},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<3, 2>()
          .input_origin({0, 0, -100})
          .input_shape({1, 1, 200})
          .output_constant(0, 5)
          .output_single_input_dimension(1, 0, 1, 2)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<3, 2>()
          .input_origin({0, 0, -100})
          .input_shape({1, 1, 200})
          .output_constant(0, -17)
          .output_single_input_dimension(1, 10, 11, 2)
          .Finalize()
          .value(),
      /*equivalent_indices=*/{{{5, 6}, {0, 0, 6}}},
      /*can_operate_in_place=*/false);
}

TEST(IndexArraySliceTest, OneDOutputOneDArrayLabeled) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({-10, -100})
                        .input_shape({20, 200})
                        .input_labels({"x", "y"})
                        .output_single_input_dimension(0, -2, -3, 0)
                        .output_single_input_dimension(1, 10, 11, 1)
                        .Finalize()
                        .value(),
                    /*expression=*/
                    Dims(0)
                        .IndexArraySlice(MakeArray<Index>({1, 2}))
                        .Label("index"),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({0, -100})
                        .input_shape({2, 200})
                        .input_labels({"index", "y"})
                        .output_index_array(0, 0, 1,
                                            MakeArray<Index>({{1}, {2}}),
                                            IndexInterval::Sized(-10, 20))
                        .output_single_input_dimension(1, 0, 1, 1)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({0, -100})
                        .input_shape({2, 200})
                        .input_labels({"index", "y"})
                        .output_index_array(0, -2, -3,
                                            MakeArray<Index>({{1}, {2}}),
                                            IndexInterval::Sized(-10, 20))
                        .output_single_input_dimension(1, 10, 11, 1)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/
                    {{{1, 5}, {0, 5}}, {{2, 5}, {1, 5}}},
                    /*can_operate_in_place=*/false);
}

TEST(IndexArraySliceTest, TwoDOutputOneDArray) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({-10, -2})
                        .input_shape({20, 15})
                        .output_single_input_dimension(0, -2, -3, 0)
                        .output_single_input_dimension(1, -4, 2, 1)
                        .Finalize()
                        .value(),
                    /*expression=*/
                    AllDims()
                        .IndexArraySlice(MakeArray<Index>({1, 2}),
                                         MakeArray<Index>({3, 4}))
                        .Label("index"),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 2>()
                        .input_origin({0})
                        .input_shape({2})
                        .input_labels({"index"})
                        .output_index_array(0, 0, 1, MakeArray<Index>({1, 2}),
                                            IndexInterval::Sized(-10, 20))
                        .output_index_array(1, 0, 1, MakeArray<Index>({3, 4}),
                                            IndexInterval::Sized(-2, 15))
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 2>()
                        .input_origin({0})
                        .input_shape({2})
                        .input_labels({"index"})
                        .output_index_array(0, -2, -3, MakeArray<Index>({1, 2}),
                                            IndexInterval::Sized(-10, 20))
                        .output_index_array(1, -4, 2, MakeArray<Index>({3, 4}),
                                            IndexInterval::Sized(-2, 15))
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{1, 3}, {0}}, {{2, 4}, {1}}},
                    /*can_operate_in_place=*/false);
}

TEST(IndexArraySliceTest, TwoDOutputOneDArrayBroadcast) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<2, 2>()
          .input_origin({-10, -2})
          .input_shape({20, 15})
          .output_single_input_dimension(0, -2, -3, 0)
          .output_single_input_dimension(1, -4, 2, 1)
          .Finalize()
          .value(),
      /*expression=*/
      AllDims().IndexArraySlice(MakeArray<Index>({{1, 2}}),
                                MakeArray<Index>({{3}, {4}})),
      /*expected_new_dimension_selection=*/{0, 1},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<2, 2>()
          .input_origin({0, 0})
          .input_shape({2, 2})
          .output_index_array(0, 0, 1, MakeArray<Index>({{1, 2}}),
                              IndexInterval::Sized(-10, 20))
          .output_index_array(1, 0, 1, MakeArray<Index>({{3}, {4}}),
                              IndexInterval::Sized(-2, 15))
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<2, 2>()
          .input_origin({0, 0})
          .input_shape({2, 2})
          .output_index_array(0, -2, -3, MakeArray<Index>({{1, 2}}),
                              IndexInterval::Sized(-10, 20))
          .output_index_array(1, -4, 2, MakeArray<Index>({{3}, {4}}),
                              IndexInterval::Sized(-2, 15))
          .Finalize()
          .value(),
      /*equivalent_indices=*/
      {{{1, 3}, {0, 0}}, {{1, 4}, {1, 0}}, {{2, 4}, {1, 1}}},
      /*can_operate_in_place=*/false);
}

TEST(IndexArraySliceTest, ErrorHandling) {
  TestDimExpressionError(
      IndexTransformBuilder<2, 0>().Finalize().value(),
      Dims(span<const DimensionIndex>({0}))
          .IndexArraySlice(MakeArray<Index>({1, 2}), MakeArray<Index>({3, 4})),
      absl::StatusCode::kInvalidArgument,
      "Number of selected dimensions \\(1\\) does not equal number of index "
      "arrays \\(2\\)");

  TestDimExpressionError(
      IndexTransformBuilder<1, 0>().Finalize().value(),
      Dims(span<const DimensionIndex>())
          .IndexArraySlice(span<const SharedArrayView<const Index>>()),
      absl::StatusCode::kInvalidArgument,
      "At least one index array must be specified");
  TestDimExpressionError(
      IndexTransformBuilder<2, 0>().Finalize().value(),
      Dims(0, 1).IndexArraySlice(MakeArray<Index>({1, 2}),
                                 MakeArray<Index>({3, 4, 5})),
      absl::StatusCode::kInvalidArgument,
      "Index arrays with shapes \\{2\\}, \\{3\\} cannot be broadcast "
      "to a common shape");
}

TEST(IndexVectorArraySliceTest, OneDOutputOneDArray) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({-10, -100})
                        .input_shape({20, 200})
                        .output_single_input_dimension(0, -2, -3, 0)
                        .output_single_input_dimension(1, 10, 11, 1)
                        .Finalize()
                        .value(),
                    /*expression=*/
                    Dims(0).IndexVectorArraySlice(MakeArray<Index>({{1, 2}}),
                                                  0),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({0, -100})
                        .input_shape({2, 200})
                        .output_index_array(0, 0, 1,
                                            MakeArray<Index>({{1}, {2}}),
                                            IndexInterval::Sized(-10, 20))
                        .output_single_input_dimension(1, 0, 1, 1)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({0, -100})
                        .input_shape({2, 200})
                        .output_index_array(0, -2, -3,
                                            MakeArray<Index>({{1}, {2}}),
                                            IndexInterval::Sized(-10, 20))
                        .output_single_input_dimension(1, 10, 11, 1)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/
                    {{{1, 5}, {0, 5}}, {{2, 5}, {1, 5}}},
                    /*can_operate_in_place=*/false);
}

TEST(IndexVectorArraySliceTest, OneElementIndexArray) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<2, 2>()
          .input_origin({-10, -100})
          .input_shape({20, 200})
          .output_single_input_dimension(0, -2, -3, 0)
          .output_single_input_dimension(1, 10, 11, 1)
          .Finalize()
          .value(),
      /*expression=*/Dims(0).IndexVectorArraySlice(MakeArray<Index>({{1}}), 0),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<2, 2>()
          .input_origin({0, -100})
          .input_shape({1, 200})
          .output_constant(0, 1)
          .output_single_input_dimension(1, 0, 1, 1)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<2, 2>()
          .input_origin({0, -100})
          .input_shape({1, 200})
          .output_constant(0, -5)
          .output_single_input_dimension(1, 10, 11, 1)
          .Finalize()
          .value(),
      /*equivalent_indices=*/{{{1, 5}, {0, 5}}},
      /*can_operate_in_place=*/false);
}

TEST(IndexVectorArraySliceTest, OneDOutputOneDArrayLabeled) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<2, 2>()
          .input_origin({-10, -100})
          .input_shape({20, 200})
          .input_labels({"x", "y"})
          .output_single_input_dimension(0, -2, -3, 0)
          .output_single_input_dimension(1, 10, 11, 1)
          .Finalize()
          .value(),
      /*expression=*/
      Dims(0)
          .IndexVectorArraySlice(MakeArray<Index>({{1, 2}}), 0)
          .Label("index"),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<2, 2>()
          .input_origin({0, -100})
          .input_shape({2, 200})
          .input_labels({"index", "y"})
          .output_index_array(0, 0, 1, MakeArray<Index>({{1}, {2}}),
                              IndexInterval::Sized(-10, 20))
          .output_single_input_dimension(1, 0, 1, 1)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<2, 2>()
          .input_origin({0, -100})
          .input_shape({2, 200})
          .input_labels({"index", "y"})
          .output_index_array(0, -2, -3, MakeArray<Index>({{1}, {2}}),
                              IndexInterval::Sized(-10, 20))
          .output_single_input_dimension(1, 10, 11, 1)
          .Finalize()
          .value(),
      /*equivalent_indices=*/{{{1, 5}, {0, 5}}, {{2, 5}, {1, 5}}},
      /*can_operate_in_place=*/false);
}

TEST(IndexVectorArraySliceTest, TwoDOutputOneDArray) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({-10, -2})
                        .input_shape({20, 15})
                        .output_single_input_dimension(0, -2, -3, 0)
                        .output_single_input_dimension(1, -4, 2, 1)
                        .Finalize()
                        .value(),
                    /*expression=*/
                    AllDims()
                        .IndexVectorArraySlice(
                            MakeArray<Index>({{1, 3}, {2, 4}}), -1)
                        .Label("index"),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 2>()
                        .input_origin({0})
                        .input_shape({2})
                        .input_labels({"index"})
                        .output_index_array(0, 0, 1, MakeArray<Index>({1, 2}),
                                            IndexInterval::Sized(-10, 20))
                        .output_index_array(1, 0, 1, MakeArray<Index>({3, 4}),
                                            IndexInterval::Sized(-2, 15))
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 2>()
                        .input_origin({0})
                        .input_shape({2})
                        .input_labels({"index"})
                        .output_index_array(0, -2, -3, MakeArray<Index>({1, 2}),
                                            IndexInterval::Sized(-10, 20))
                        .output_index_array(1, -4, 2, MakeArray<Index>({3, 4}),
                                            IndexInterval::Sized(-2, 15))
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{1, 3}, {0}}, {{2, 4}, {1}}},
                    /*can_operate_in_place=*/false);
}

TEST(IndexVectorArraySliceTest, ErrorHandling) {
  TestDimExpressionError(
      IndexTransformBuilder<2, 0>().Finalize().value(),
      Dims(0).IndexVectorArraySlice(MakeArray<Index>({1, 2}), 0),
      absl::StatusCode::kInvalidArgument,
      "Number of selected dimensions \\(1\\) does not equal index vector "
      "length \\(2\\)");

  TestDimExpressionError(
      IndexTransformBuilder<2, 0>().Finalize().value(),
      Dims(0).IndexVectorArraySlice(MakeArray<Index>({1, 2}), 1),
      absl::StatusCode::kInvalidArgument,
      "Dimension index 1 is outside valid range \\[-1, 1\\)");
}

TEST(OuterIndexArraySliceTest, Integration) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<3, 3>()
                        .input_origin({-10, -100, -2})
                        .input_shape({21, 200, 15})
                        .output_single_input_dimension(0, -2, -3, 0)
                        .output_single_input_dimension(1, 6, 5, 1)
                        .output_single_input_dimension(2, -4, 2, 2)
                        .Finalize()
                        .value(),
                    /*expression=*/
                    Dims(0, 2).OuterIndexArraySlice(
                        MakeArray<Index>({{3, 4, 5}, {8, 9, 10}}),
                        MakeArray<Index>({1, 2})),
                    /*expected_new_dimension_selection=*/{0, 1, 3},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<4, 3>()
                        .input_origin({0, 0, -100, 0})
                        .input_shape({2, 3, 200, 2})
                        .output_index_array(
                            0, 0, 1,
                            MakeArray<Index>({{{{3}}, {{4}}, {{5}}},
                                              {{{8}}, {{9}}, {{10}}}}),
                            IndexInterval::Sized(-10, 21))
                        .output_single_input_dimension(1, 0, 1, 2)
                        .output_index_array(2, 0, 1,
                                            MakeArray<Index>({{{{1, 2}}}}),
                                            IndexInterval::Sized(-2, 15))
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<4, 3>()
                        .input_origin({0, 0, -100, 0})
                        .input_shape({2, 3, 200, 2})
                        .output_index_array(
                            0, -2, -3,
                            MakeArray<Index>({{{{3}}, {{4}}, {{5}}},
                                              {{{8}}, {{9}}, {{10}}}}),
                            IndexInterval::Sized(-10, 21))
                        .output_single_input_dimension(1, 6, 5, 2)
                        .output_index_array(2, -4, 2,
                                            MakeArray<Index>({{{{1, 2}}}}),
                                            IndexInterval::Sized(-2, 15))
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/
                    {{{3, 5, 1}, {0, 0, 5, 0}},
                     {{9, 5, 2}, {1, 1, 5, 1}},
                     {{8, 5, 2}, {1, 0, 5, 1}},
                     {{10, 5, 2}, {1, 2, 5, 1}}},
                    /*can_operate_in_place=*/false);
}

TEST(OuterIndexArraySliceTest, OneElementIndexArray) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<3, 3>()
                        .input_origin({-10, -100, -2})
                        .input_shape({21, 200, 15})
                        .output_single_input_dimension(0, -2, -3, 0)
                        .output_single_input_dimension(1, 6, 5, 1)
                        .output_single_input_dimension(2, -4, 2, 2)
                        .Finalize()
                        .value(),
                    /*expression=*/
                    Dims(0, 2).OuterIndexArraySlice(
                        MakeArray<Index>({{3, 4, 5}, {8, 9, 10}}),
                        MakeArray<Index>({1})),
                    /*expected_new_dimension_selection=*/{0, 1, 3},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<4, 3>()
                        .input_origin({0, 0, -100, 0})
                        .input_shape({2, 3, 200, 1})
                        .output_index_array(
                            0, 0, 1,
                            MakeArray<Index>({{{{3}}, {{4}}, {{5}}},
                                              {{{8}}, {{9}}, {{10}}}}),
                            IndexInterval::Sized(-10, 21))
                        .output_single_input_dimension(1, 0, 1, 2)
                        .output_constant(2, 1)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<4, 3>()
                        .input_origin({0, 0, -100, 0})
                        .input_shape({2, 3, 200, 1})
                        .output_index_array(
                            0, -2, -3,
                            MakeArray<Index>({{{{3}}, {{4}}, {{5}}},
                                              {{{8}}, {{9}}, {{10}}}}),
                            IndexInterval::Sized(-10, 21))
                        .output_single_input_dimension(1, 6, 5, 2)
                        .output_constant(2, -2)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{3, 5, 1}, {0, 0, 5, 0}}},
                    /*can_operate_in_place=*/false);
}

TEST(OuterIndexArraySliceTest, ErrorHandling) {
  TestDimExpressionError(
      IndexTransformBuilder<2, 0>().Finalize().value(),
      Dims(span<const DimensionIndex>({0}))
          .OuterIndexArraySlice(MakeArray<Index>({1, 2}),
                                MakeArray<Index>({3, 4})),
      absl::StatusCode::kInvalidArgument,
      "Number of selected dimensions \\(1\\) does not equal number of index "
      "arrays \\(2\\)");
}

}  // namespace
