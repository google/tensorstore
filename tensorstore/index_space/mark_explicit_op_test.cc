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
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/dim_expression_testutil.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::Dims;
using ::tensorstore::Index;
using ::tensorstore::IndexDomainBuilder;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::MakeArray;
using ::tensorstore::internal_index_space::TestDimExpression;
using ::tensorstore::internal_index_space::TestDimExpressionErrorTransformOnly;

TEST(MarkBoundsExplicitTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({3, 4, 2})
                                      .implicit_lower_bounds({0, 1, 1})
                                      .implicit_upper_bounds({1, 0, 0})
                                      .input_labels({"x", "y", "z"})
                                      .output_identity_transform()
                                      .Finalize()
                                      .value();
  const auto expected_new_transform = IndexTransformBuilder<3, 3>()
                                          .input_origin({1, 2, 3})
                                          .input_shape({3, 4, 2})
                                          .implicit_lower_bounds({0, 1, 0})
                                          .implicit_upper_bounds({0, 0, 0})
                                          .input_labels({"x", "y", "z"})
                                          .output_identity_transform()
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

TEST(MarkBoundsExplicitTest, IndexArray) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder(2, 1)
          .input_shape({2, 3})
          .implicit_upper_bounds({1, 0})
          .output_index_array(0, 0, 1, MakeArray<Index>({{1, 2, 3}}))
          .Finalize()
          .value(),
      /*expression=*/Dims(0).MarkBoundsExplicit(),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder(2, 2)
          .input_shape({2, 3})
          .output_identity_transform()
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder(2, 1)
          .input_shape({2, 3})
          .output_index_array(0, 0, 1, MakeArray<Index>({{1, 2, 3}}))
          .Finalize()
          .value(),
      /*equivalent_indices=*/{});
}

TEST(MarkBoundsExplicitTest, IndexArrayZeroSize) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder(2, 1)
          .input_shape({0, 3})
          .implicit_upper_bounds({1, 0})
          .output_index_array(0, 0, 1, MakeArray<Index>({{1, 2, 3}}))
          .Finalize()
          .value(),
      /*expression=*/Dims(0).MarkBoundsExplicit(),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder(2, 2)
          .input_shape({0, 3})
          .output_identity_transform()
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder(2, 1)
          .input_shape({0, 3})
          .output_constant(0, 0)
          .Finalize()
          .value(),
      /*equivalent_indices=*/{});
}

TEST(UnsafeMarkBoundsImplicitTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({3, 4, 2})
                                      .implicit_lower_bounds({0, 0, 0})
                                      .implicit_upper_bounds({0, 0, 0})
                                      .input_labels({"x", "y", "z"})
                                      .output_identity_transform()
                                      .Finalize()
                                      .value();
  const auto expected_new_transform = IndexTransformBuilder<3, 3>()
                                          .input_origin({1, 2, 3})
                                          .input_shape({3, 4, 2})
                                          .implicit_lower_bounds({1, 0, 1})
                                          .implicit_upper_bounds({1, 0, 1})
                                          .input_labels({"x", "y", "z"})
                                          .output_identity_transform()
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

TEST(UnsafeMarkBoundsImplicitTest, IndexArray) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder(2, 1)
          .input_shape({2, 3})
          .output_index_array(0, 0, 1, MakeArray<Index>({{1, 2, 3}}))
          .Finalize()
          .value(),
      /*expression=*/
      Dims(0).UnsafeMarkBoundsImplicit(/*lower=*/false, /*upper=*/true),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder(2, 2)
          .input_shape({2, 3})
          .implicit_upper_bounds({1, 0})
          .output_identity_transform()
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder(2, 1)
          .input_shape({2, 3})
          .implicit_upper_bounds({1, 0})
          .output_index_array(0, 0, 1, MakeArray<Index>({{1, 2, 3}}))
          .Finalize()
          .value(),
      /*equivalent_indices=*/{},
      /*can_operate_in_place=*/true,
      /*test_compose=*/false);
}

TEST(UnsafeMarkBoundsImplicitTest, IndexArrayInvalid) {
  TestDimExpressionErrorTransformOnly(
      IndexTransformBuilder(2, 1)
          .input_shape({2, 3})
          .output_index_array(0, 0, 1, MakeArray<Index>({{1, 2, 3}}))
          .Finalize()
          .value(),
      Dims(1).UnsafeMarkBoundsImplicit(/*lower=*/false, /*upper=*/true),
      absl::StatusCode::kInvalidArgument,
      "Cannot mark input dimension 1 as having implicit bounds because it "
      "indexes the index array map for output dimension 0",
      /*expected_domain=*/
      IndexDomainBuilder(2)
          .shape({2, 3})
          .implicit_upper_bounds({0, 1})
          .Finalize()
          .value());
}

TEST(MarkBoundsExplicitTest, LowerOnly) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({3, 4, 2})
                                      .implicit_lower_bounds({0, 1, 1})
                                      .implicit_upper_bounds({1, 0, 0})
                                      .output_identity_transform()
                                      .Finalize()
                                      .value();
  const auto expected_new_transform = IndexTransformBuilder<3, 3>()
                                          .input_origin({1, 2, 3})
                                          .input_shape({3, 4, 2})
                                          .implicit_lower_bounds({0, 1, 0})
                                          .implicit_upper_bounds({1, 0, 0})
                                          .output_identity_transform()
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
                                      .output_identity_transform()
                                      .Finalize()
                                      .value();
  const auto expected_new_transform = IndexTransformBuilder<3, 3>()
                                          .input_origin({1, 2, 3})
                                          .input_shape({3, 4, 2})
                                          .implicit_lower_bounds({0, 1, 1})
                                          .implicit_upper_bounds({0, 0, 0})
                                          .output_identity_transform()
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
                                      .output_identity_transform()
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
