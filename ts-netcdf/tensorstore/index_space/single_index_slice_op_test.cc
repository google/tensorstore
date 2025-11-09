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

/// Tests for the DimExpression::IndexSlice(index_vector) operation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/dim_expression_testutil.h"
#include "tensorstore/util/status.h"

namespace {

using ::tensorstore::AllDims;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Dims;
using ::tensorstore::Index;
using ::tensorstore::IndexDomainBuilder;
using ::tensorstore::IndexInterval;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::kInfIndex;
using ::tensorstore::MakeArray;
using ::tensorstore::span;
using ::tensorstore::internal_index_space::EquivalentIndices;
using ::tensorstore::internal_index_space::TestDimExpression;

TEST(SingleIndexSliceTest, Example) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .input_shape({3, 4, 2})
                                      .input_labels({"x", "y", "z"})
                                      .output_identity_transform()
                                      .Finalize()
                                      .value();
  const auto expected_new_transform = IndexTransformBuilder<1, 3>()
                                          .input_origin({2})
                                          .input_shape({4})
                                          .input_labels({"y"})
                                          .output_constant(0, 2)
                                          .output_single_input_dimension(1, 0)
                                          .output_constant(2, 4)
                                          .Finalize()
                                          .value();
  const EquivalentIndices equivalent_indices = {
      {{2, 3, 4}, {3}},
  };
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(0, 2).IndexSlice({2, 4}),
                    /*expected_new_dimension_selection=*/{},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices);

  // Test using labels to select dimensions.
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims("x", "z").IndexSlice({2, 4}),
                    /*expected_new_dimension_selection=*/{},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices);
}

// Tests that implicit bounds do not constrain slicing.
TEST(SingleIndexSliceTest, ImplicitLowerBound) {
  const auto original_transform = IndexTransformBuilder<3, 3>()
                                      .input_origin({1, 2, 3})
                                      .implicit_lower_bounds({1, 1, 0})
                                      .input_shape({3, 4, 2})
                                      .input_labels({"x", "y", "z"})
                                      .output_identity_transform()
                                      .Finalize()
                                      .value();
  const auto expected_new_transform = IndexTransformBuilder<1, 3>()
                                          .input_origin({2})
                                          .implicit_lower_bounds({1})
                                          .input_shape({4})
                                          .input_labels({"y"})
                                          .output_constant(0, -7)
                                          .output_single_input_dimension(1, 0)
                                          .output_constant(2, 4)
                                          .Finalize()
                                          .value();
  const EquivalentIndices equivalent_indices = {
      {{-7, 3, 4}, {3}},
  };
  TestDimExpression(/*original_transform=*/original_transform,
                    /*expression=*/Dims(0, 2).IndexSlice({-7, 4}),
                    /*expected_new_dimension_selection=*/{},
                    /*expected_identity_new_transform=*/expected_new_transform,
                    /*expected_new_transform=*/expected_new_transform,
                    /*equivalent_indices=*/equivalent_indices);
}

TEST(SingleIndexSliceTest, DimSubsetUniformIndexArrayRetained) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<3, 4>()
                        .input_origin({1, 2, -kInfIndex})
                        .input_shape({4, 5, kInfIndex + 7})
                        .output_single_input_dimension(0, 1, 4, 0)
                        .output_single_input_dimension(1, 2, 3, 2)
                        .output_constant(2, 3)
                        .output_index_array(
                            3, 4, 1,
                            MakeArray<Index>({{{5}, {6}, {7}, {8}, {9}},
                                              {{15}, {16}, {17}, {18}, {19}},
                                              {{25}, {26}, {27}, {28}, {29}},
                                              {{35}, {36}, {37}, {38}, {39}}}))
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(1, 2).IndexSlice(3),
                    /*expected_new_dimension_selection=*/{},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 3>()
                        .input_origin({1})
                        .input_shape({4})
                        .output_single_input_dimension(0, 0)
                        .output_constant(1, 3)
                        .output_constant(2, 3)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 4>()
                        .input_origin({1})
                        .input_shape({4})
                        .output_single_input_dimension(0, 1, 4, 0)
                        .output_constant(1, 2 + 3 * 3)
                        .output_constant(2, 3)
                        .output_index_array(3, 4, 1,
                                            MakeArray<Index>({6, 16, 26, 36}))
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{4, 3, 3}, {4}}});
}

TEST(SingleIndexSliceTest, DimSubsetUniformIndexArrayEliminated) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<3, 4>()
                        .input_origin({1, 2, -kInfIndex})
                        .input_shape({4, 5, kInfIndex + 7})
                        .output_single_input_dimension(0, 1, 4, 0)
                        .output_single_input_dimension(1, 2, 3, 2)
                        .output_constant(2, 3)
                        .output_index_array(
                            3, 4, 1,
                            MakeArray<Index>({{{5}, {6}, {7}, {8}, {9}}}))
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(1, 2).IndexSlice(3),
                    /*expected_new_dimension_selection=*/{},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 3>()
                        .input_origin({1})
                        .input_shape({4})
                        .output_single_input_dimension(0, 0)
                        .output_constant(1, 3)
                        .output_constant(2, 3)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 4>()
                        .input_origin({1})
                        .input_shape({4})
                        .output_single_input_dimension(0, 1, 4, 0)
                        .output_constant(1, 2 + 3 * 3)
                        .output_constant(2, 3)
                        .output_constant(3, 4 + 1 * 6)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{4, 3, 3}, {4}}});
}

TEST(SingleIndexSliceTest, DimSubsetNonUniform) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<3, 4>()
                        .input_origin({1, 2, -kInfIndex})
                        .input_shape({4, 5, kInfIndex + 7})
                        .output_single_input_dimension(0, 1, 4, 0)
                        .output_single_input_dimension(1, 2, 3, 2)
                        .output_constant(2, 3)
                        .output_index_array(
                            3, 4, 1,
                            MakeArray<Index>({{{5}, {6}, {7}, {8}, {9}},
                                              {{15}, {16}, {17}, {18}, {19}},
                                              {{25}, {26}, {27}, {28}, {29}},
                                              {{35}, {36}, {37}, {38}, {39}}}))
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(1, 2).IndexSlice({3, 4}),
                    /*expected_new_dimension_selection=*/{},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 3>()
                        .input_origin({1})
                        .input_shape({4})
                        .output_single_input_dimension(0, 0)
                        .output_constant(1, 3)
                        .output_constant(2, 4)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 4>()
                        .input_origin({1})
                        .input_shape({4})
                        .output_single_input_dimension(0, 1, 4, 0)
                        .output_constant(1, 2 + 4 * 3)
                        .output_constant(2, 3)
                        .output_index_array(3, 4, 1,
                                            MakeArray<Index>({6, 16, 26, 36}))
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{4, 3, 4}, {4}}});
}

TEST(SingleIndexSliceTest, DimSubsetNonUniformLabeled) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<3, 4>()
                        .input_origin({1, 2, -kInfIndex})
                        .input_shape({4, 5, kInfIndex + 7})
                        .input_labels({"x", "y", "z"})
                        .output_single_input_dimension(0, 1, 4, 0)
                        .output_single_input_dimension(1, 2, 3, 2)
                        .output_constant(2, 3)
                        .output_index_array(
                            3, 4, 1,
                            MakeArray<Index>({{{5}, {6}, {7}, {8}, {9}},
                                              {{15}, {16}, {17}, {18}, {19}},
                                              {{25}, {26}, {27}, {28}, {29}},
                                              {{35}, {36}, {37}, {38}, {39}}}))
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(1, 2).IndexSlice({3, 4}),
                    /*expected_new_dimension_selection=*/{},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 3>()
                        .input_origin({1})
                        .input_shape({4})
                        .input_labels({"x"})
                        .output_single_input_dimension(0, 0)
                        .output_constant(1, 3)
                        .output_constant(2, 4)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 4>()
                        .input_origin({1})
                        .input_shape({4})
                        .input_labels({"x"})
                        .output_single_input_dimension(0, 1, 4, 0)
                        .output_constant(1, 2 + 4 * 3)
                        .output_constant(2, 3)
                        .output_index_array(3, 4, 1,
                                            MakeArray<Index>({6, 16, 26, 36}))
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{{{4, 3, 4}, {4}}});
}

// Tests applying an IndexSlice operation to an index transform with an empty
// domain.
TEST(SingleIndexSliceTest, EmptyDomain) {
  TestDimExpression(/*original_transform=*/
                    IndexTransformBuilder<2, 2>()
                        .input_origin({1, 2})
                        .input_shape({0, 3})
                        .input_labels({"x", "y"})
                        .output_single_input_dimension(0, 2, 7, 0)
                        .output_index_array(1, 4, 3,
                                            MakeArray<Index>({{1, 2, 3}}))
                        .Finalize()
                        .value(),
                    /*expression=*/Dims(1).IndexSlice({3}),
                    /*expected_new_dimension_selection=*/{},
                    /*expected_identity_new_transform=*/
                    IndexTransformBuilder<1, 2>()
                        .input_origin({1})
                        .input_shape({0})
                        .input_labels({"x"})
                        .output_single_input_dimension(0, 0)
                        .output_constant(1, 3)
                        .Finalize()
                        .value(),
                    /*expected_new_transform=*/
                    IndexTransformBuilder<1, 2>()
                        .input_origin({1})
                        .input_shape({0})
                        .input_labels({"x"})
                        .output_single_input_dimension(0, 2, 7, 0)
                        .output_constant(1, 0)
                        .Finalize()
                        .value(),
                    /*equivalent_indices=*/{});
}

TEST(ErrorHandlingTest, DimensionSelectionRankMismatch) {
  TestDimExpressionError(IndexTransformBuilder<1, 1>().Finalize().value(),
                         AllDims().IndexSlice(span<const Index>({1, 2})),
                         absl::StatusCode::kInvalidArgument,
                         "Number of dimensions .* does not match number of "
                         "indices .*");
}

TEST(ErrorHandlingTest, OutOfBounds) {
  TestDimExpressionError(IndexTransformBuilder<1, 1>()
                             .input_origin({-10})
                             .input_shape({15})
                             .Finalize()
                             .value(),
                         AllDims().IndexSlice({5}),
                         absl::StatusCode::kOutOfRange,
                         "Slice mismatch: .* is outside valid domain .*");
}

TEST(ErrorHandlingTest, OutOfBoundsInfinity) {
  TestDimExpressionError(IndexTransformBuilder<1, 1>()
                             .input_origin({-kInfIndex})
                             .input_shape({15})
                             .Finalize()
                             .value(),
                         AllDims().IndexSlice({-kInfIndex}),
                         absl::StatusCode::kOutOfRange,
                         "Slice mismatch: .* is outside valid domain .*");
}

TEST(ErrorHandlingTest, SingleInputDimensionMapIntegerOverflow) {
  TestDimExpressionErrorTransformOnly(
      IndexTransformBuilder<1, 1>()
          .input_origin({0})
          .input_shape({10})
          .output_single_input_dimension(0, std::numeric_limits<Index>::max(),
                                         1, 0)
          .Finalize()
          .value(),
      AllDims().IndexSlice({1}), absl::StatusCode::kInvalidArgument,
      "Integer overflow computing offset for output dimension.*",
      IndexDomainBuilder<0>().Finalize().value());
}

TEST(ErrorHandlingTest, IndexArrayMapIntegerOverflow) {
  TestDimExpressionErrorTransformOnly(
      IndexTransformBuilder<1, 1>()
          .input_origin({0})
          .input_shape({3})
          .output_index_array(0, std::numeric_limits<Index>::max(), 1,
                              MakeArray<Index>({0, 1, 2}))
          .Finalize()
          .value(),
      AllDims().IndexSlice({1}), absl::StatusCode::kInvalidArgument,
      "Integer overflow computing offset for output dimension.*",
      IndexDomainBuilder<0>().Finalize().value());
}

TEST(ErrorHandlingTest, IndexArrayMapOutOfBounds) {
  TestDimExpressionErrorTransformOnly(
      IndexTransformBuilder<1, 1>()
          .input_origin({0})
          .input_shape({3})
          .output_index_array(0, 0, 1, MakeArray<Index>({0, 1, 2}),
                              IndexInterval::Closed(-5, -3))
          .Finalize()
          .value(),
      AllDims().IndexSlice({1}), absl::StatusCode::kOutOfRange,
      "Index .* is outside valid range .*",
      IndexDomainBuilder<0>().Finalize().value());
}

}  // namespace
