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

/// Tests for the DimExpression::AddNew operation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/dim_expression_testutil.h"

namespace {

using tensorstore::Dims;
using tensorstore::Index;
using tensorstore::IndexInterval;
using tensorstore::IndexTransformBuilder;
using tensorstore::kInfIndex;
using tensorstore::kInfSize;
using tensorstore::MakeArray;
using tensorstore::internal_index_space::TestDimExpression;

TEST(AddNewTest, Example) {
  const auto expected_new_transform =
      IndexTransformBuilder<3, 1>()
          .input_origin({-kInfIndex, 1, -kInfIndex})
          .input_shape({kInfSize, 5, kInfSize})
          .implicit_lower_bounds({1, 0, 1})
          .implicit_upper_bounds({1, 0, 1})
          .input_labels({"", "x", ""})
          .output_single_input_dimension(0, 0, 1, 1)
          .Finalize()
          .value();
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<1, 1>()
          .input_origin({1})
          .input_shape({5})
          .input_labels({"x"})
          .output_single_input_dimension(0, 0, 1, 0)
          .Finalize()
          .value(),
      /*expression=*/Dims(0, -1).AddNew(),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/
      {
          {{2}, {1, 2, 8}},
          {{2}, {5, 2, 9}},
      },
      /*can_operate_in_place=*/false);
}

TEST(AddNewTest, Simple) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<2, 3>()
                        .input_origin({2, 3})
                        .input_shape({3, 4})
                        .output_single_input_dimension(0, 1, 3, 1)
                        .output_single_input_dimension(1, 2, 4, 0)
                        .output_index_array(2, 3, 5,
                                            MakeArray<Index>({{1, 2, 3, 4}}),
                                            IndexInterval::Closed(-1, 10))
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0, -1).AddNew(),
                    /*expected_new_dimension_selection=*/{0, 3},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<4, 2>()
                        .input_origin({-kInfIndex, 2, 3, -kInfIndex})
                        .input_shape({kInfSize, 3, 4, kInfSize})
                        .implicit_lower_bounds({1, 0, 0, 1})
                        .implicit_upper_bounds({1, 0, 0, 1})
                        .output_single_input_dimension(0, 0, 1, 1)
                        .output_single_input_dimension(1, 0, 1, 2)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<4, 3>()
                        .input_origin({-kInfIndex, 2, 3, -kInfIndex})
                        .input_shape({kInfSize, 3, 4, kInfSize})
                        .implicit_lower_bounds({1, 0, 0, 1})
                        .implicit_upper_bounds({1, 0, 0, 1})
                        .output_single_input_dimension(0, 1, 3, 2)
                        .output_single_input_dimension(1, 2, 4, 1)
                        .output_index_array(
                            2, 3, 5, MakeArray<Index>({{{{1}, {2}, {3}, {4}}}}),
                            IndexInterval::Closed(-1, 10))
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/
                    {
                        {{3, 4}, {100, 3, 4, 500}},
                        {{3, 4}, {-100, 3, 4, -500}},
                    },
                    /*can_operate_in_place=*/false);
}

TEST(AddNewTest, Constant) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 1>()
                        .input_origin({1})
                        .input_shape({5})
                        .output_constant(0, 1)
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0).AddNew(),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<2, 1>()
                        .input_origin({-kInfIndex, 1})
                        .input_shape({kInfSize, 5})
                        .implicit_lower_bounds({1, 0})
                        .implicit_upper_bounds({1, 0})
                        .output_single_input_dimension(0, 0, 1, 1)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<2, 1>()
                        .input_origin({-kInfIndex, 1})
                        .input_shape({kInfSize, 5})
                        .implicit_lower_bounds({1, 0})
                        .implicit_upper_bounds({1, 0})
                        .output_constant(0, 1)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/
                    {
                        {{1}, {-100, 1}},
                        {{1}, {100, 1}},
                    },
                    /*can_operate_in_place=*/false);
}

TEST(AddNewTest, Labeled) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 1>()
                        .input_origin({1})
                        .input_shape({5})
                        .input_labels({"a"})
                        .output_constant(0, 1)
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(-1, 0).AddNew().Label("x", "y"),
                    /*expected_new_dimension_selection=*/{2, 0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<3, 1>()
                        .input_origin({-kInfIndex, 1, -kInfIndex})
                        .input_shape({kInfSize, 5, kInfSize})
                        .implicit_lower_bounds({1, 0, 1})
                        .implicit_upper_bounds({1, 0, 1})
                        .input_labels({"y", "a", "x"})
                        .output_single_input_dimension(0, 0, 1, 1)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<3, 1>()
                        .input_origin({-kInfIndex, 1, -kInfIndex})
                        .input_shape({kInfSize, 5, kInfSize})
                        .implicit_lower_bounds({1, 0, 1})
                        .implicit_upper_bounds({1, 0, 1})
                        .input_labels({"y", "a", "x"})
                        .output_constant(0, 1)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/
                    {
                        {{2}, {1, 2, 8}},
                        {{2}, {5, 2, 9}},
                    },
                    /*can_operate_in_place=*/false);
}

TEST(AddNewTest, EmptyDimensionSelection) {
  const auto transform = IndexTransformBuilder<1, 1>()
                             .input_origin({1})
                             .input_shape({5})
                             .input_labels({"x"})
                             .output_single_input_dimension(0, 0, 1, 0)
                             .Finalize()
                             .value();
  TestDimExpression(
      /*original_transform=*/transform,
      /*expression=*/Dims().AddNew(),
      /*expected_new_dimension_selection=*/{},
      /*expected_identity_new_transform=*/transform,
      /*expected_new_transform=*/transform,
      /*equivalent_indices=*/
      {
          {{2}, {2}},
          {{3}, {3}},
      },
      /*can_operate_in_place=*/true);
}

}  // namespace
