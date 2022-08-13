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

/// Tests for the DimExpression::Label operation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/dim_expression_testutil.h"
#include "tensorstore/util/status.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::Dims;
using ::tensorstore::IdentityTransform;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::span;
using ::tensorstore::internal_index_space::TestDimExpression;

TEST(LabelTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({3, 4, 2})
                                      .input_labels({"x", "y", "z"})
                                      .output_identity_transform()
                                      .Finalize()
                                      .value();
  const auto expected_new_transform = IndexTransformBuilder<3, 3>()
                                          .input_origin({1, 2, 3})
                                          .input_shape({3, 4, 2})
                                          .input_labels({"a", "y", "b"})
                                          .output_identity_transform()
                                          .Finalize()
                                          .value();
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(0, 2).Label("a", "b"),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/{});

  // Test using labels to select dimensions.
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims("x", "z").Label("a", "b"),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/{});
}

TEST(LabelTest, MultipleArguments) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<3, 1>()
                        .output_constant(0, 1)
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(1, 0).Label("x", "y"),
                    /*expected_new_dimension_selection=*/{1, 0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<3, 3>()
                        .input_labels({"y", "x", ""})
                        .output_identity_transform()
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<3, 1>()
                        .input_labels({"y", "x", ""})
                        .output_constant(0, 1)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(LabelTest, ErrorHandling) {
  TestDimExpressionError(
      IdentityTransform(1),
      Dims(span<const DimensionIndex>({0})).Label("x", "y"),
      absl::StatusCode::kInvalidArgument,
      "Number of dimensions \\(1\\) does not match number of "
      "labels \\(2\\)\\.");
}

}  // namespace
