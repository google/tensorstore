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

/// Tests for the DimExpression::{Unsafe,}Mark{Bounds,Labels}{Explicit,Implicit}
/// operations.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/dim_expression_testutil.h"

namespace {

using tensorstore::DimensionIndex;
using tensorstore::Dims;
using tensorstore::IndexTransformBuilder;
using tensorstore::internal_index_space::TestDimExpression;

TEST(MarkBoundsExplicitTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({3, 4, 2})
                                      .implicit_lower_bounds({0, 1, 1})
                                      .implicit_upper_bounds({1, 0, 0})
                                      .input_labels({"x", "y", "z"})
                                      .output_single_input_dimension(0, 0)
                                      .output_single_input_dimension(1, 1)
                                      .output_single_input_dimension(2, 2)
                                      .Finalize()
                                      .value();
  const auto expected_new_transform = IndexTransformBuilder<3, 3>()
                                          .input_origin({1, 2, 3})
                                          .input_shape({3, 4, 2})
                                          .implicit_lower_bounds({0, 1, 0})
                                          .implicit_upper_bounds({0, 0, 0})
                                          .input_labels({"x", "y", "z"})
                                          .output_single_input_dimension(0, 0)
                                          .output_single_input_dimension(1, 1)
                                          .output_single_input_dimension(2, 2)
                                          .Finalize()
                                          .value();
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(0, 2).MarkBoundsExplicit(),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/{});

  // Test with explicit `lower` and `upper` arguments to MarkBoundExplicit.
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(0, 2).MarkBoundsExplicit(true, true),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/{});

  // Test using labels to select dimensions.
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims("x", "z").MarkBoundsExplicit(),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/{});
}

TEST(UnsafeMarkBoundsImplicitTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({3, 4, 2})
                                      .implicit_lower_bounds({0, 0, 0})
                                      .implicit_upper_bounds({0, 0, 0})
                                      .input_labels({"x", "y", "z"})
                                      .output_single_input_dimension(0, 0)
                                      .output_single_input_dimension(1, 1)
                                      .output_single_input_dimension(2, 2)
                                      .Finalize()
                                      .value();
  const auto expected_new_transform = IndexTransformBuilder<3, 3>()
                                          .input_origin({1, 2, 3})
                                          .input_shape({3, 4, 2})
                                          .implicit_lower_bounds({1, 0, 1})
                                          .implicit_upper_bounds({1, 0, 1})
                                          .input_labels({"x", "y", "z"})
                                          .output_single_input_dimension(0, 0)
                                          .output_single_input_dimension(1, 1)
                                          .output_single_input_dimension(2, 2)
                                          .Finalize()
                                          .value();
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(0, 2).UnsafeMarkBoundsImplicit(),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/{},
                    /*can_operate_in_place=*/true,
                    /*test_compose=*/false);

  // Test with explicit `lower` and `upper` arguments to MarkBoundExplicit.
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/Dims(0, 2).UnsafeMarkBoundsImplicit(true, true),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/{},
      /*can_operate_in_place=*/true,
      /*test_compose=*/false);

  // Test using labels to select dimensions.
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims("x", "z").UnsafeMarkBoundsImplicit(),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/{},
                    /*can_operate_in_place=*/true,
                    /*test_compose=*/false);
}

TEST(MarkBoundsExplicitTest, LowerOnly) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({3, 4, 2})
                                      .implicit_lower_bounds({0, 1, 1})
                                      .implicit_upper_bounds({1, 0, 0})
                                      .output_single_input_dimension(0, 0)
                                      .output_single_input_dimension(1, 1)
                                      .output_single_input_dimension(2, 2)
                                      .Finalize()
                                      .value();
  const auto expected_new_transform = IndexTransformBuilder<3, 3>()
                                          .input_origin({1, 2, 3})
                                          .input_shape({3, 4, 2})
                                          .implicit_lower_bounds({0, 1, 0})
                                          .implicit_upper_bounds({1, 0, 0})
                                          .output_single_input_dimension(0, 0)
                                          .output_single_input_dimension(1, 1)
                                          .output_single_input_dimension(2, 2)
                                          .Finalize()
                                          .value();
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(0, 2).MarkBoundsExplicit(true, false),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/{});
}

TEST(MarkBoundsExplicitTest, UpperOnly) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({3, 4, 2})
                                      .implicit_lower_bounds({0, 1, 1})
                                      .implicit_upper_bounds({1, 0, 0})
                                      .output_single_input_dimension(0, 0)
                                      .output_single_input_dimension(1, 1)
                                      .output_single_input_dimension(2, 2)
                                      .Finalize()
                                      .value();
  const auto expected_new_transform = IndexTransformBuilder<3, 3>()
                                          .input_origin({1, 2, 3})
                                          .input_shape({3, 4, 2})
                                          .implicit_lower_bounds({0, 1, 1})
                                          .implicit_upper_bounds({0, 0, 0})
                                          .output_single_input_dimension(0, 0)
                                          .output_single_input_dimension(1, 1)
                                          .output_single_input_dimension(2, 2)
                                          .Finalize()
                                          .value();
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(0, 2).MarkBoundsExplicit(false, true),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/{});
}

TEST(MarkBoundsExplicitTest, None) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({3, 4, 2})
                                      .implicit_lower_bounds({0, 1, 1})
                                      .implicit_upper_bounds({1, 0, 0})
                                      .output_single_input_dimension(0, 0)
                                      .output_single_input_dimension(1, 1)
                                      .output_single_input_dimension(2, 2)
                                      .Finalize()
                                      .value();
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(0, 2).MarkBoundsExplicit(false, false),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/original_transform,
                    /*expected_new_transform=*/original_transform,
                    /*equivalent_indices=*/{});
}

}  // namespace
