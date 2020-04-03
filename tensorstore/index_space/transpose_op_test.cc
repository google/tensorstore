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

/// Tests for the `DimExpression::Transpose()` operation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/dim_expression_testutil.h"
#include "tensorstore/util/status.h"

namespace {

using tensorstore::Dims;
using tensorstore::Index;
using tensorstore::IndexInterval;
using tensorstore::IndexTransformBuilder;
using tensorstore::MakeArray;
using tensorstore::internal_index_space::EquivalentIndices;
using tensorstore::internal_index_space::TestDimExpression;

TEST(TransposeTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({3, 4, 2})
                                      .implicit_lower_bounds({1, 0, 0})
                                      .implicit_upper_bounds({0, 1, 0})
                                      .input_labels({"x", "y", "z"})
                                      .output_single_input_dimension(0, 0, 1, 0)
                                      .output_single_input_dimension(1, 0, 1, 1)
                                      .output_single_input_dimension(2, 0, 1, 2)
                                      .Finalize()
                                      .value();
  const auto expected_new_transform =
      IndexTransformBuilder<3, 3>()
          .input_origin({3, 1, 2})
          .input_shape({2, 3, 4})
          .implicit_lower_bounds({0, 1, 0})
          .implicit_upper_bounds({0, 0, 1})
          .input_labels({"z", "x", "y"})
          .output_single_input_dimension(0, 0, 1, 1)
          .output_single_input_dimension(1, 0, 1, 2)
          .output_single_input_dimension(2, 0, 1, 0)
          .Finalize()
          .value();
  const EquivalentIndices equivalent_indices = {{{2, 3, 4}, {4, 2, 3}}};
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(2, 0, 1).Transpose(),
                    /*expected_new_dimension_selection=*/{0, 1, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices);

  // Test using labels to select dimensions.
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims("z", "x", "y").Transpose(),
                    /*expected_new_dimension_selection=*/{0, 1, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices);
}

TEST(TransposeTest, Simple) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<4, 2>()
                        .input_origin({1, 2, 3, 4})
                        .input_shape({5, 6, 4, 8})
                        .output_single_input_dimension(0, 1, 2, 1)
                        .output_index_array(
                            1, 2, 3, MakeArray<Index>({{{{1}, {2}, {3}, {4}}}}),
                            IndexInterval::Closed(-3, 10))
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(2, 0, 1, 3).Transpose(),
                    /*expected_new_dimension_selection=*/{0, 1, 2, 3},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<4, 4>()
                        .input_origin({3, 1, 2, 4})
                        .input_shape({4, 5, 6, 8})
                        .output_single_input_dimension(0, 0, 1, 1)
                        .output_single_input_dimension(1, 0, 1, 2)
                        .output_single_input_dimension(2, 0, 1, 0)
                        .output_single_input_dimension(3, 0, 1, 3)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<4, 2>()
                        .input_origin({3, 1, 2, 4})
                        .input_shape({4, 5, 6, 8})
                        .output_single_input_dimension(0, 1, 2, 2)
                        .output_index_array(
                            1, 2, 3,
                            MakeArray<Index>(
                                {{{{1}}}, {{{2}}}, {{{3}}}, {{{4}}}}),
                            IndexInterval::Closed(-3, 10))
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{2, 4, 3, 5}, {3, 2, 4, 5}}});
}

TEST(TransposeTest, Constant) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<2, 2>()
                        .input_origin({1, 2})
                        .input_shape({5, 6})
                        .output_constant(0, 1)
                        .output_constant(1, 2)
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(1, 0).Transpose(),
                    /*expected_new_dimension_selection=*/{0, 1},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({2, 1})
                        .input_shape({6, 5})
                        .output_single_input_dimension(0, 0, 1, 1)
                        .output_single_input_dimension(1, 0, 1, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({2, 1})
                        .input_shape({6, 5})
                        .output_constant(0, 1)
                        .output_constant(1, 2)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(TransposeTest, ErrorHandling) {
  TestDimExpressionError(
      IndexTransformBuilder<>(2, 2)
          .input_origin({1, 2})
          .input_shape({5, 6})
          .output_constant(0, 1)
          .output_constant(1, 2)
          .Finalize()
          .value(),
      Dims(1).Transpose(), absl::StatusCode::kInvalidArgument,
      "Number of dimensions \\(1\\) must equal input_rank \\(2\\)\\.");
}

TEST(TransposeTest, Labeled) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<4, 2>()
                        .input_origin({1, 2, 3, 4})
                        .input_shape({5, 6, 4, 8})
                        .input_labels({"a", "b", "c", "d"})
                        .output_single_input_dimension(0, 1, 2, 1)
                        .output_index_array(
                            1, 2, 3, MakeArray<Index>({{{{1}, {2}, {3}, {4}}}}),
                            IndexInterval::Closed(-3, 10))
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(2, 0, 1, 3).Transpose(),
                    /*expected_new_dimension_selection=*/{0, 1, 2, 3},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<4, 4>()
                        .input_origin({3, 1, 2, 4})
                        .input_shape({4, 5, 6, 8})
                        .input_labels({"c", "a", "b", "d"})
                        .output_single_input_dimension(0, 0, 1, 1)
                        .output_single_input_dimension(1, 0, 1, 2)
                        .output_single_input_dimension(2, 0, 1, 0)
                        .output_single_input_dimension(3, 0, 1, 3)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<4, 2>()
                        .input_origin({3, 1, 2, 4})
                        .input_shape({4, 5, 6, 8})
                        .input_labels({"c", "a", "b", "d"})
                        .output_single_input_dimension(0, 1, 2, 2)
                        .output_index_array(
                            1, 2, 3,
                            MakeArray<Index>(
                                {{{{1}}}, {{{2}}}, {{{3}}}, {{{4}}}}),
                            IndexInterval::Closed(-3, 10))
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{2, 4, 3, 5}, {3, 2, 4, 5}}});
}

}  // namespace
