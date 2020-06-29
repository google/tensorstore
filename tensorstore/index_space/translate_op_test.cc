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

/// Tests for the DimExpression::Translate* operations.

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
using tensorstore::IndexTransformBuilder;
using tensorstore::kImplicit;
using tensorstore::kInfIndex;
using tensorstore::kInfSize;
using tensorstore::kMaxFiniteIndex;
using tensorstore::kMinFiniteIndex;
using tensorstore::MakeArray;
using tensorstore::span;
using tensorstore::internal_index_space::EquivalentIndices;
using tensorstore::internal_index_space::TestDimExpression;

TEST(TranslateByTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({3, 4, 2})
                                      .input_labels({"x", "y", "z"})
                                      .output_single_input_dimension(0, 0, 1, 0)
                                      .output_single_input_dimension(1, 0, 1, 1)
                                      .output_single_input_dimension(2, 0, 1, 2)
                                      .Finalize()
                                      .value();
  const auto expected_new_transform =
      IndexTransformBuilder<3, 3>()
          .input_origin({11, 2, 23})
          .input_shape({3, 4, 2})
          .input_labels({"x", "y", "z"})
          .output_single_input_dimension(0, -10, 1, 0)
          .output_single_input_dimension(1, 0, 1, 1)
          .output_single_input_dimension(2, -20, 1, 2)
          .Finalize()
          .value();
  const EquivalentIndices equivalent_indices = {
      {{2, 3, 3}, {12, 3, 23}},
  };
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(0, 2).TranslateBy({10, 20}),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices);

  // Test using labels to select dimensions.
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims("x", "z").TranslateBy({10, 20}),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices);
}

TEST(TranslateToTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({3, 4, 2})
                                      .input_labels({"x", "y", "z"})
                                      .output_single_input_dimension(0, 0, 1, 0)
                                      .output_single_input_dimension(1, 0, 1, 1)
                                      .output_single_input_dimension(2, 0, 1, 2)
                                      .Finalize()
                                      .value();
  const auto expected_new_transform =
      IndexTransformBuilder<3, 3>()
          .input_origin({10, 2, 20})
          .input_shape({3, 4, 2})
          .input_labels({"x", "y", "z"})
          .output_single_input_dimension(0, -9, 1, 0)
          .output_single_input_dimension(1, 0, 1, 1)
          .output_single_input_dimension(2, -17, 1, 2)
          .Finalize()
          .value();
  const EquivalentIndices equivalent_indices = {
      {{2, 3, 3}, {11, 3, 20}},
  };
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(0, 2).TranslateTo({10, 20}),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices);

  // Test using labels to select dimensions.
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(0, 2).TranslateTo({10, 20}),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices);
}

TEST(TranslateByTest, OneDimensionalConstant) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .output_constant(0, 2)
                        .Finalize()
                        .value(),
                    /*expression=*/AllDims().TranslateBy(5),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .output_single_input_dimension(0, -5, 1, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .output_constant(0, 2)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{4}, {9}}});
}

TEST(TranslateByTest, OneDimensionalSingleInputDimension) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .output_single_input_dimension(0, 2, 3, 0)
                        .Finalize()
                        .value(),
                    /*expression=*/AllDims().TranslateBy(5),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .output_single_input_dimension(0, -5, 1, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .output_single_input_dimension(0, 2 - 3 * 5, 3, 0)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{4}, {9}}});
}

TEST(TranslateByTest, OneDimensionalSingleInputDimensionImplicit) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .output_single_input_dimension(0, 2, 3, 0)
                        .Finalize()
                        .value(),
                    /*expression=*/AllDims().TranslateBy(kImplicit),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .output_single_input_dimension(0, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .output_single_input_dimension(0, 2, 3, 0)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{4}, {4}}});
}

TEST(TranslateByTest, OneDimensionalIndexArray) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<1, 1>()
          .input_origin({-2})
          .input_shape({5})
          .output_index_array(0, 2, 3, MakeArray<Index>({6, 7, 8, 9, 10}))
          .Finalize()
          .value(),
      /*expression=*/AllDims().TranslateBy(5),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<1, 1>()
          .input_origin({3})
          .input_shape({5})
          .output_single_input_dimension(0, -5, 1, 0)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<1, 1>()
          .input_origin({3})
          .input_shape({5})
          .output_index_array(0, 2, 3, MakeArray<Index>({6, 7, 8, 9, 10}))
          .Finalize()
          .value(),
      /*equivalent_indices=*/{{{1}, {6}}});
}

TEST(TranslateByTest, AllDimsUniform) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<3, 5>()
          .input_origin({-kInfIndex, 5, -kInfIndex})
          .input_shape({kInfSize, 30, kInfIndex + 10})
          .output_single_input_dimension(0, 1, 4, 0)
          .output_single_input_dimension(1, 2, 5, 0)
          .output_constant(2, 3)
          .output_single_input_dimension(3, 4, 7, 1)
          .output_single_input_dimension(4, 5, 8, 2)
          .Finalize()
          .value(),
      /*expression=*/AllDims().TranslateBy(5),
      /*expected_new_dimension_selection=*/{0, 1, 2},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<3, 3>()
          .input_origin({-kInfIndex, 10, -kInfIndex})
          .input_shape({kInfSize, 30, kInfIndex + 15})
          .output_single_input_dimension(0, -5, 1, 0)
          .output_single_input_dimension(1, -5, 1, 1)
          .output_single_input_dimension(2, -5, 1, 2)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<3, 5>()
          .input_origin({-kInfIndex, 10, -kInfIndex})
          .input_shape({kInfSize, 30, kInfIndex + 15})
          .output_single_input_dimension(0, 1 - 4 * 5, 4, 0)
          .output_single_input_dimension(1, 2 - 5 * 5, 5, 0)
          .output_constant(2, 3)
          .output_single_input_dimension(3, 4 - 7 * 5, 7, 1)
          .output_single_input_dimension(4, 5 - 8 * 5, 8, 2)
          .Finalize()
          .value(),
      /*equivalent_indices=*/{{{4, 5, 6}, {4 + 5, 5 + 5, 6 + 5}}});
}

TEST(TranslateByTest, ErrorHandling) {
  TestDimExpressionError(
      IndexTransformBuilder<1, 1>().Finalize().value(),
      AllDims().TranslateBy(span<const Index>({1, 2})),
      absl::StatusCode::kInvalidArgument,
      "Number of dimensions \\(1\\) does not match number of "
      "indices \\(2\\)");

  TestDimExpressionError(IndexTransformBuilder<1, 1>()
                             .input_origin({kMinFiniteIndex})
                             .input_shape({10})
                             .Finalize()
                             .value(),
                         AllDims().TranslateBy(-kInfIndex),
                         absl::StatusCode::kOutOfRange,
                         "Index offset .* is outside valid range .*");

  TestDimExpressionError(IndexTransformBuilder<1, 1>()
                             .input_origin({kMinFiniteIndex})
                             .input_shape({10})
                             .Finalize()
                             .value(),
                         AllDims().TranslateBy(kInfIndex),
                         absl::StatusCode::kOutOfRange,
                         "Index offset .* is outside valid range .*");

  TestDimExpressionError(
      IndexTransformBuilder<1, 1>()
          .input_origin({kMinFiniteIndex})
          .input_shape({10})
          .Finalize()
          .value(),
      AllDims().TranslateBy(-1), absl::StatusCode::kInvalidArgument,
      "Shifted inclusive_min value .* is outside valid range .*");

  TestDimExpressionError(
      IndexTransformBuilder<1, 1>()
          .input_origin({kMaxFiniteIndex - 1})
          .input_shape({2})
          .Finalize()
          .value(),
      AllDims().TranslateBy(1), absl::StatusCode::kInvalidArgument,
      "Shifted inclusive_max value .* is outside valid range .*");

  TestDimExpressionError(IndexTransformBuilder<1, 1>()
                             .output_single_input_dimension(
                                 0, std::numeric_limits<Index>::min(), 1, 0)
                             .Finalize()
                             .value(),
                         AllDims().TranslateBy(1),
                         absl::StatusCode::kInvalidArgument,
                         "Integer overflow computing output offset .*");
}

TEST(TranslateByTest, DimSubsetUniform) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<3, 2>()
                        .input_origin({1, 2, -kInfIndex})
                        .input_shape({4, 5, kInfIndex + 7})
                        .output_single_input_dimension(0, 1, 1, 1)
                        .output_single_input_dimension(1, 2, 2, 2)
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0, 2).TranslateBy(5),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<3, 3>()
                        .input_origin({6, 2, -kInfIndex})
                        .input_shape({4, 5, kInfIndex + 7 + 5})
                        .output_single_input_dimension(0, -5, 1, 0)
                        .output_single_input_dimension(1, 0, 1, 1)
                        .output_single_input_dimension(2, -5, 1, 2)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<3, 2>()
                        .input_origin({6, 2, -kInfIndex})
                        .input_shape({4, 5, kInfIndex + 7 + 5})
                        .output_single_input_dimension(0, 1, 1, 1)
                        .output_single_input_dimension(1, 2 - 2 * 5, 2, 2)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{4, 5, 6}, {4 + 5, 5, 6 + 5}}});
}

TEST(TranslateByTest, DimSubsetNonUniform) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<3, 2>()
                        .input_origin({1, 2, -kInfIndex})
                        .input_shape({4, 5, kInfIndex + 7})
                        .output_single_input_dimension(0, 1, 1, 1)
                        .output_single_input_dimension(1, 2, 2, 2)
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(0, 2).TranslateBy({5, 6}),
                    /*expected_new_dimension_selection=*/{0, 2},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<3, 3>()
                        .input_origin({6, 2, -kInfIndex})
                        .input_shape({4, 5, kInfIndex + 7 + 6})
                        .output_single_input_dimension(0, -5, 1, 0)
                        .output_single_input_dimension(1, 0, 1, 1)
                        .output_single_input_dimension(2, -6, 1, 2)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<3, 2>()
                        .input_origin({6, 2, -kInfIndex})
                        .input_shape({4, 5, kInfIndex + 7 + 6})
                        .output_single_input_dimension(0, 1, 1, 1)
                        .output_single_input_dimension(1, 2 - 2 * 6, 2, 2)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{3, 4, 5}, {3 + 5, 4, 5 + 6}}});
}

TEST(TranslateToTest, OneDimensionalConstant) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 1>()
                        .input_origin({5})
                        .input_shape({10})
                        .output_constant(0, 2)
                        .Finalize()
                        .value(),
                    /*expression=*/AllDims().TranslateTo(8),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({8})
                        .input_shape({10})
                        .output_single_input_dimension(0, -3, 1, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({8})
                        .input_shape({10})
                        .output_constant(0, 2)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{7}, {10}}});
}

TEST(TranslateToTest, OneDimensionalSingleInputDimension) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 1>()
                        .input_origin({4})
                        .input_shape({10})
                        .output_single_input_dimension(0, 2, 3, 0)
                        .Finalize()
                        .value(),
                    /*expression=*/AllDims().TranslateTo(5),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({5})
                        .input_shape({10})
                        .output_single_input_dimension(0, -1, 1, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({5})
                        .input_shape({10})
                        .output_single_input_dimension(0, 2 - 3, 3, 0)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{6}, {7}}});
}

TEST(TranslateToTest, OneDimensionalSingleInputDimensionImplicit) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<1, 1>()
                        .input_origin({4})
                        .input_shape({10})
                        .output_single_input_dimension(0, 2, 3, 0)
                        .Finalize()
                        .value(),
                    /*expression=*/AllDims().TranslateTo(kImplicit),
                    /*expected_new_dimension_selection=*/{0},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({4})
                        .input_shape({10})
                        .output_single_input_dimension(0, 0)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 1>()
                        .input_origin({4})
                        .input_shape({10})
                        .output_single_input_dimension(0, 2, 3, 0)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{6}, {6}}});
}

TEST(TranslateToTest, TwoDimensionalSingleInputDimensionOneImplicit) {
  TestDimExpression(/*original_transform=*/IndexTransformBuilder<2, 2>()
                        .input_origin({4, 5})
                        .input_shape({10, 11})
                        .output_single_input_dimension(0, 2, 3, 0)
                        .output_single_input_dimension(1, 4, 5, 1)
                        .Finalize()
                        .value(),
                    /*expression=*/AllDims().TranslateTo({kImplicit, 10}),
                    /*expected_new_dimension_selection=*/{0, 1},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({4, 10})
                        .input_shape({10, 11})
                        .output_single_input_dimension(0, 0)
                        .output_single_input_dimension(1, -5, 1, 1)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({4, 10})
                        .input_shape({10, 11})
                        .output_single_input_dimension(0, 2, 3, 0)
                        .output_single_input_dimension(1, -25 + 4, 5, 1)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{6, 7}, {6, 12}}});
}

TEST(TranslateToTest, ErrorHandling) {
  TestDimExpressionError(IndexTransformBuilder<1, 1>().Finalize().value(),
                         AllDims().TranslateTo(1),
                         absl::StatusCode::kInvalidArgument,
                         "Interval \\(-inf, \\+inf\\) is not bounded below");

  TestDimExpressionError(
      IndexTransformBuilder<1, 1>()
          .input_origin({-5})
          .input_shape({10})
          .Finalize()
          .value(),
      AllDims().TranslateTo(std::numeric_limits<Index>::max()),
      absl::StatusCode::kOutOfRange, "Origin [0-9]+ is outside valid range .*");
}

}  // namespace
