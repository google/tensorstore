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

/// Tests for the DimExpression::Diagonal operation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/dim_expression_testutil.h"

namespace {

using tensorstore::Dims;
using tensorstore::Index;
using tensorstore::IndexTransformBuilder;
using tensorstore::kInfIndex;
using tensorstore::kInfSize;
using tensorstore::MakeArray;
using tensorstore::internal_index_space::EquivalentIndices;
using tensorstore::internal_index_space::TestDimExpression;

TEST(DiagonalTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({5, 4, 5})
                                      .input_labels({"x", "y", "z"})
                                      .output_single_input_dimension(0, 0, 1, 0)
                                      .output_single_input_dimension(1, 0, 1, 1)
                                      .output_single_input_dimension(2, 0, 1, 2)
                                      .Finalize()
                                      .value();
  const auto expected_new_transform =
      IndexTransformBuilder<2, 3>()
          .input_origin({3, 2})
          .input_shape({3, 4})
          .input_labels({"", "y"})
          .output_single_input_dimension(0, 0, 1, 0)
          .output_single_input_dimension(1, 0, 1, 1)
          .output_single_input_dimension(2, 0, 1, 0)
          .Finalize()
          .value();
  const EquivalentIndices equivalent_indices = {
      {{4, 3, 4}, {4, 3}},
  };
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(0, 2).Diagonal(),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices);

  // Test using labels to select dimensions.
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims("x", "z").Diagonal(),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices);
}

// `Diagonal()` with zero selected dimensions adds a new dimension.
TEST(DiagonalTest, ZeroDimensional) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<2, 2>()
                        .input_origin({1, 2})
                        .input_shape({5, 4})
                        .input_labels({"x", "y"})
                        .output_single_input_dimension(0, 5, 1, 0)
                        .output_single_input_dimension(1, 0, 1, 1)
                        .Finalize()
                        .value(),
                    /*expression=*/Dims().Diagonal(),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<3, 2>()
                        .input_origin({-kInfIndex, 1, 2})
                        .input_shape({kInfSize, 5, 4})
                        .implicit_lower_bounds({1, 0, 0})
                        .implicit_upper_bounds({1, 0, 0})
                        .input_labels({"", "x", "y"})
                        .output_single_input_dimension(0, 0, 1, 1)
                        .output_single_input_dimension(1, 0, 1, 2)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<3, 2>()
                        .input_origin({-kInfIndex, 1, 2})
                        .input_shape({kInfSize, 5, 4})
                        .implicit_lower_bounds({1, 0, 0})
                        .implicit_upper_bounds({1, 0, 0})
                        .input_labels({"", "x", "y"})
                        .output_single_input_dimension(0, 5, 1, 1)
                        .output_single_input_dimension(1, 0, 1, 2)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{3, 4}, {8, 3, 4}}},
                    /*can_operate_in_place=*/false);
}

// `Diagonal()` with a single selected dimension is equivalent to
// `MoveToFront().Label("")`.
TEST(DiagonalTest, OneDimensional) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<3, 3>()
                        .input_origin({1, 2, 3})
                        .input_shape({5, 4, 5})
                        .input_labels({"x", "y", "z"})
                        .output_single_input_dimension(0, 5, 1, 0)
                        .output_single_input_dimension(1, 0, 1, 1)
                        .output_single_input_dimension(2, 0, 1, 2)
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(1).Diagonal(),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<3, 3>()
                        .input_origin({2, 1, 3})
                        .input_shape({4, 5, 5})
                        .input_labels({"", "x", "z"})
                        .output_single_input_dimension(0, 0, 1, 1)
                        .output_single_input_dimension(1, 0, 1, 0)
                        .output_single_input_dimension(2, 0, 1, 2)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<3, 3>()
                        .input_origin({2, 1, 3})
                        .input_shape({4, 5, 5})
                        .input_labels({"", "x", "z"})
                        .output_single_input_dimension(0, 5, 1, 1)
                        .output_single_input_dimension(1, 0, 1, 0)
                        .output_single_input_dimension(2, 0, 1, 2)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{4, 3, 5}, {3, 4, 5}}});
}

TEST(DiagonalTest, TwoDimensionalSimple) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<3, 3>()
                        .input_origin({5, 6, 7})
                        .input_shape({10, 9, 15})
                        .output_single_input_dimension(0, 1, 1, 1)
                        .output_single_input_dimension(1, 2, 2, 0)
                        .output_single_input_dimension(2, 3, 3, 2)
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(2, 0).Diagonal(),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<2, 3>()
                        .input_origin({7, 6})
                        .input_shape({8, 9})
                        .output_single_input_dimension(0, 0, 1, 0)
                        .output_single_input_dimension(1, 0, 1, 1)
                        .output_single_input_dimension(2, 0, 1, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<2, 3>()
                        .input_origin({7, 6})
                        .input_shape({8, 9})
                        .output_single_input_dimension(0, 1, 1, 1)
                        .output_single_input_dimension(1, 2, 2, 0)
                        .output_single_input_dimension(2, 3, 3, 0)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{10, 11, 10}, {10, 11}}});
}

TEST(DiagonalTest, TwoDimensionalSimpleImplicitLower) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<3, 3>()
          .input_origin({5, 6, 7})
          .input_shape({10, 9, 15})
          .implicit_lower_bounds({1, 0, 1})
          .output_single_input_dimension(0, 1, 1, 1)
          .output_single_input_dimension(1, 2, 2, 0)
          .output_single_input_dimension(2, 3, 3, 2)
          .Finalize()
          .value(),
      /*expression=*/Dims(2, 0).Diagonal(),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<2, 3>()
          .input_origin({7, 6})
          .input_shape({8, 9})
          .implicit_lower_bounds({1, 0})
          .output_single_input_dimension(0, 0, 1, 0)
          .output_single_input_dimension(1, 0, 1, 1)
          .output_single_input_dimension(2, 0, 1, 0)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<2, 3>()
          .input_origin({7, 6})
          .input_shape({8, 9})
          .implicit_lower_bounds({1, 0})
          .output_single_input_dimension(0, 1, 1, 1)
          .output_single_input_dimension(1, 2, 2, 0)
          .output_single_input_dimension(2, 3, 3, 0)
          .Finalize()
          .value(),
      /*equivalent_indices=*/{{{10, 11, 10}, {10, 11}}});
}

TEST(DiagonalTest, TwoDimensionalSimpleImplicitUpper) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<3, 3>()
          .input_origin({5, 6, 7})
          .input_shape({10, 9, 15})
          .implicit_upper_bounds({1, 0, 1})
          .output_single_input_dimension(0, 1, 1, 1)
          .output_single_input_dimension(1, 2, 2, 0)
          .output_single_input_dimension(2, 3, 3, 2)
          .Finalize()
          .value(),
      /*expression=*/Dims(2, 0).Diagonal(),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<2, 3>()
          .input_origin({7, 6})
          .input_shape({8, 9})
          .implicit_upper_bounds({1, 0})
          .output_single_input_dimension(0, 0, 1, 0)
          .output_single_input_dimension(1, 0, 1, 1)
          .output_single_input_dimension(2, 0, 1, 0)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<2, 3>()
          .input_origin({7, 6})
          .input_shape({8, 9})
          .implicit_upper_bounds({1, 0})
          .output_single_input_dimension(0, 1, 1, 1)
          .output_single_input_dimension(1, 2, 2, 0)
          .output_single_input_dimension(2, 3, 3, 0)
          .Finalize()
          .value(),
      /*equivalent_indices=*/{{{10, 11, 10}, {10, 11}}});
}

TEST(DiagonalTest, IndexArray) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<3, 2>()
                        .input_origin({5, 6, 6})
                        .input_shape({4, 5, 2})
                        .output_index_array(
                            0, 2, 3,
                            MakeArray<Index>(
                                {{{1, 4}}, {{2, 5}}, {{3, 6}}, {{4, 7}}}))
                        .output_constant(1, 0)
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0, 2).Diagonal(),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<2, 3>()
                        .input_origin({6, 6})
                        .input_shape({2, 5})
                        .output_single_input_dimension(0, 0, 1, 0)
                        .output_single_input_dimension(1, 0, 1, 1)
                        .output_single_input_dimension(2, 0, 1, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({6, 6})
                        .input_shape({2, 5})
                        .output_index_array(0, 2, 3,
                                            MakeArray<Index>({{2}, {6}}))
                        .output_constant(1, 0)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{6, 8, 6}, {6, 8}}});
}

TEST(DiagonalTest, Labeled) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<3, 2>()
                        .input_origin({5, 6, 6})
                        .input_shape({4, 5, 2})
                        .input_labels({"a", "b", "c"})
                        .output_index_array(
                            0, 2, 3,
                            MakeArray<Index>(
                                {{{1, 4}}, {{2, 5}}, {{3, 6}}, {{4, 7}}}))
                        .output_constant(1, 0)
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0, 2).Diagonal().Label("diag"),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<2, 3>()
                        .input_origin({6, 6})
                        .input_shape({2, 5})
                        .input_labels({"diag", "b"})
                        .output_single_input_dimension(0, 0, 1, 0)
                        .output_single_input_dimension(1, 0, 1, 1)
                        .output_single_input_dimension(2, 0, 1, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({6, 6})
                        .input_shape({2, 5})
                        .input_labels({"diag", "b"})
                        .output_index_array(0, 2, 3,
                                            MakeArray<Index>({{2}, {6}}))
                        .output_constant(1, 0)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{6, 8, 6}, {6, 8}}});
}

}  // namespace
