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

/// Tests for the `DimExpression::Transpose(target_dimensions)` operation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/dim_expression_testutil.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::Dims;
using ::tensorstore::Index;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::span;
using ::tensorstore::internal_index_space::EquivalentIndices;
using ::tensorstore::internal_index_space::TestDimExpression;
using ::tensorstore::internal_index_space::TestDimExpressionError;

TEST(TransposeToTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_inclusive_max({3, 5, 4})
                                      .implicit_lower_bounds({1, 0, 0})
                                      .implicit_upper_bounds({0, 1, 0})
                                      .input_labels({"x", "y", "z"})
                                      .output_identity_transform()
                                      .Finalize()
                                      .value();
  const auto expected_new_transform = IndexTransformBuilder<3, 3>()
                                          .input_origin({2, 3, 1})
                                          .input_inclusive_max({5, 4, 3})
                                          .implicit_lower_bounds({0, 0, 1})
                                          .implicit_upper_bounds({1, 0, 0})
                                          .input_labels({"y", "z", "x"})
                                          .output_single_input_dimension(0, 2)
                                          .output_single_input_dimension(1, 0)
                                          .output_single_input_dimension(2, 1)
                                          .Finalize()
                                          .value();
  const EquivalentIndices equivalent_indices = {{{2, 3, 4}, {3, 4, 2}}};
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(2, 0).Transpose({1, 2}),
                    /*expected_new_dimension_selection=*/{1, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices);

  // Test using labels to select dimensions and `span` to specify
  // `target_dimensions`, and using a negative target dimension.
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/
      Dims("z", "x").Transpose(span<const DimensionIndex, 2>({1, -1})),
      /*expected_new_dimension_selection=*/{1, 2},
      /*expected_identity_new_transform=*/expected_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/equivalent_indices);
}

TEST(TransposeToTest, ErrorHandling) {
  TestDimExpressionError(IndexTransformBuilder<>(2, 2)
                             .input_origin({1, 2})
                             .input_shape({5, 6})
                             .output_constant(0, 1)
                             .output_constant(1, 2)
                             .Finalize()
                             .value(),
                         Dims(1).Transpose(span<const DimensionIndex>({1, 2})),
                         absl::StatusCode::kInvalidArgument,
                         "Number of selected dimensions \\(1\\) must equal "
                         "number of target dimensions \\(2\\)");
  TestDimExpressionError(IndexTransformBuilder<>(2, 2)
                             .input_origin({1, 2})
                             .input_shape({5, 6})
                             .output_constant(0, 1)
                             .output_constant(1, 2)
                             .Finalize()
                             .value(),
                         Dims(0, 1).Transpose({1, 1}),
                         absl::StatusCode::kInvalidArgument,
                         "Target dimension 1 occurs more than once");
  TestDimExpressionError(
      IndexTransformBuilder<>(2, 2)
          .input_origin({1, 2})
          .input_shape({5, 6})
          .output_constant(0, 1)
          .output_constant(1, 2)
          .Finalize()
          .value(),
      Dims(0).Transpose({2}), absl::StatusCode::kInvalidArgument,
      "Dimension index 2 is outside valid range \\[-2, 2\\)");
}

}  // namespace
