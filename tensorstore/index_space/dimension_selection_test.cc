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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/dim_expression_testutil.h"
#include "tensorstore/util/status.h"

namespace {

using tensorstore::DimensionIndex;
using tensorstore::DimRangeSpec;
using tensorstore::Dims;
using tensorstore::dynamic_rank;
using tensorstore::DynamicDims;
using tensorstore::IndexTransformBuilder;
using tensorstore::span;
using tensorstore::internal_index_space::TestDimExpressionError;

TEST(DimsTest, ErrorHandling) {
  TestDimExpressionError(
      IndexTransformBuilder<2, 0>().Finalize().value(),
      Dims(span<const DimensionIndex>({0, 0, 1})).IndexSlice(0),
      absl::StatusCode::kInvalidArgument,
      "Number of dimensions .* exceeds input rank .*");
  TestDimExpressionError(IndexTransformBuilder<2, 0>().Finalize().value(),
                         Dims(2).Label("b"), absl::StatusCode::kInvalidArgument,
                         "Dimension index 2 is outside valid range .*");
  TestDimExpressionError(IndexTransformBuilder<2, 0>().Finalize().value(),
                         Dims(1, 1).Label("b", "c"),
                         absl::StatusCode::kInvalidArgument,
                         "Input dimensions \\{1\\} specified more than once.*");
}

TEST(DimsTest, SelectUsingLabels) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<2, 0>()
          .input_labels({"x", "y"})
          .Finalize()
          .value(),
      /*expression=*/Dims("x").Label("a"),
      /*expected_new_dimension_selection=*/{0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<2, 2>()
          .input_labels({"a", "y"})
          .output_single_input_dimension(0, 0, 1, 0)
          .output_single_input_dimension(1, 0, 1, 1)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<2, 0>().input_labels({"a", "y"}).Finalize().value(),
      /*equivalent_indices=*/{});

  TestDimExpressionError(
      IndexTransformBuilder<2, 0>().input_labels({"x", "y"}).Finalize().value(),
      Dims("a").Label("z"), absl::StatusCode::kInvalidArgument,
      "Label \"a\" does not match one of \\{\"x\", \"y\"\\}");

  TestDimExpressionError(
      IndexTransformBuilder<2, 0>().input_labels({"", ""}).Finalize().value(),
      Dims("").Label("z"), absl::StatusCode::kInvalidArgument,
      "Dimension cannot be specified by empty label");

  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<2, 0>()
          .input_labels({"x", "y"})
          .Finalize()
          .value(),
      /*expression=*/Dims({"x", -1}).Label("a", "b"),
      /*expected_new_dimension_selection=*/{0, 1},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<2, 2>()
          .input_labels({"a", "b"})
          .output_single_input_dimension(0, 0, 1, 0)
          .output_single_input_dimension(1, 0, 1, 1)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<2, 0>().input_labels({"a", "b"}).Finalize().value(),
      /*equivalent_indices=*/{});
}

TEST(DynamicDimsTest, Existing) {
  const auto original_transform = IndexTransformBuilder<4, 0>()
                                      .input_labels({"a", "b", "c", "d"})
                                      .Finalize()
                                      .value();
  const auto expected_identity_new_transform =
      IndexTransformBuilder<4, 4>()
          .input_labels({"a1", "b1", "c1", "d1"})
          .output_single_input_dimension(0, 0)
          .output_single_input_dimension(1, 1)
          .output_single_input_dimension(2, 2)
          .output_single_input_dimension(3, 3)
          .Finalize()
          .value();
  const auto expected_new_transform =
      IndexTransformBuilder<4, 0>()
          .input_labels({"a1", "b1", "c1", "d1"})
          .Finalize()
          .value();

  // Test using `Dims` with argument pack.
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/
      Dims(DimRangeSpec{1, 4, 2}, 0, "c").Label("b1", "d1", "a1", "c1"),
      /*expected_new_dimension_selection=*/{1, 3, 0, 2},
      /*expected_identity_new_transform=*/expected_identity_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/{});

  // Test using `DynamicDims` with braced list.
  TestDimExpression(
      /*original_transform=*/original_transform,
      /*expression=*/
      DynamicDims({DimRangeSpec{1, 4, 2}, 0, "c"})
          .Label("b1", "d1", "a1", "c1"),
      /*expected_new_dimension_selection=*/{1, 3, 0, 2},
      /*expected_identity_new_transform=*/expected_identity_new_transform,
      /*expected_new_transform=*/expected_new_transform,
      /*equivalent_indices=*/{});
}

TEST(DynamicDimsTest, CombinedNew) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      /*expression=*/
      Dims(DimRangeSpec{1, 4, 2}, 0, -1).AddNew().Label("e", "f", "g", "h"),
      /*expected_new_dimension_selection=*/{1, 3, 0, 7},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 4>(8, tensorstore::StaticRank<4>{})
          .input_labels({"g", "e", "a", "f", "b", "c", "d", "h"})
          .output_single_input_dimension(0, 2)
          .output_single_input_dimension(1, 4)
          .output_single_input_dimension(2, 5)
          .output_single_input_dimension(3, 6)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 0>(8)
          .input_labels({"g", "e", "a", "f", "b", "c", "d", "h"})
          .Finalize()
          .value(),
      /*equivalent_indices=*/{},
      /*can_operate_in_place=*/false);
}

TEST(DynamicDimsTest, InvalidNewLabel) {
  TestDimExpressionError(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      Dims(DimRangeSpec{1, 4, 2}, "x").AddNew(),
      absl::StatusCode::kInvalidArgument,
      "New dimensions cannot be specified by label");
}

TEST(DynamicDimsTest, InvalidDimRangeSpecNewUnbounded) {
  TestDimExpressionError(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      Dims(DimRangeSpec{absl::nullopt, absl::nullopt, 1}).AddNew(),
      absl::StatusCode::kInvalidArgument,
      "`:` is not a valid specification for new dimensions");
}

TEST(DynamicDimsTest, InvalidDimRangeSpecNewMissingStop) {
  TestDimExpressionError(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      Dims(DimRangeSpec{5, absl::nullopt, 1}).AddNew(),
      absl::StatusCode::kInvalidArgument,
      "`5:` is not a valid specification for new dimensions");
}

TEST(DynamicDimsTest, InvalidDimRangeSpecNewNegativeStop) {
  TestDimExpressionError(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      Dims(DimRangeSpec{absl::nullopt, -3, 1}).AddNew(),
      absl::StatusCode::kInvalidArgument,
      "`:-3` is not a valid specification for new dimensions");
}

TEST(DynamicDimsTest, InvalidDimRangeSpecNewNegativeStartNegativeStep) {
  TestDimExpressionError(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      Dims(DimRangeSpec{-5, absl::nullopt, -1}).AddNew(),
      absl::StatusCode::kInvalidArgument,
      "`-5::-1` is not a valid specification for new dimensions");
}

TEST(DynamicDimsTest, InvalidDimRangeSpecNewMissingStart) {
  TestDimExpressionError(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      Dims(DimRangeSpec{absl::nullopt, 5, -1}).AddNew(),
      absl::StatusCode::kInvalidArgument,
      "`:5:-1` is not a valid specification for new dimensions");
}

TEST(DynamicDimsTest, InvalidDimRangeSpecNewInvalidInterval) {
  TestDimExpressionError(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      Dims(DimRangeSpec{6, 5, 1}).AddNew(), absl::StatusCode::kInvalidArgument,
      "`6:5` is not a valid specification for new dimensions");
}

TEST(DynamicDimsTest, InvalidDimRangeSpecNewInvalidMixedSigns) {
  TestDimExpressionError(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      Dims(DimRangeSpec{-1, 4, 1}).AddNew(), absl::StatusCode::kInvalidArgument,
      "`-1:4` is not a valid specification for new dimensions");
}

TEST(DynamicDimsTest, InvalidDimRangeSpecNewZeroStep) {
  TestDimExpressionError(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      Dims(DimRangeSpec{1, 4, 0}).AddNew(), absl::StatusCode::kInvalidArgument,
      "step must not be 0");
}

TEST(DynamicDimsTest, InvalidDimRangeSpecNewInvalidIntervalNegativeStep) {
  TestDimExpressionError(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      Dims(DimRangeSpec{5, 6, -1}).AddNew(), absl::StatusCode::kInvalidArgument,
      "`5:6:-1` is not a valid specification for new dimensions");
}

TEST(DimsTest, DimRangeSpecNegativeStep) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      /*expression=*/
      Dims(DimRangeSpec{-4, -7, -2}).AddNew().Label("e", "f"),
      /*expected_new_dimension_selection=*/{2, 0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 4>(6)
          .input_labels({"f", "a", "e", "b", "c", "d"})
          .output_single_input_dimension(0, 1)
          .output_single_input_dimension(1, 3)
          .output_single_input_dimension(2, 4)
          .output_single_input_dimension(3, 5)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 0>(6)
          .input_labels({"f", "a", "e", "b", "c", "d"})
          .Finalize()
          .value(),
      /*equivalent_indices=*/{},
      /*can_operate_in_place=*/false);
}

TEST(DimsTest, DimRangeSpecNegativeIndicesNew) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      /*expression=*/
      Dims(DimRangeSpec{-6, -3, 2}).AddNew().Label("e", "f"),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 4>(6)
          .input_labels({"e", "a", "f", "b", "c", "d"})
          .output_single_input_dimension(0, 1)
          .output_single_input_dimension(1, 3)
          .output_single_input_dimension(2, 4)
          .output_single_input_dimension(3, 5)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 0>(6)
          .input_labels({"e", "a", "f", "b", "c", "d"})
          .Finalize()
          .value(),
      /*equivalent_indices=*/{},
      /*can_operate_in_place=*/false);
}

TEST(DimsTest, DimRangeSpecImplicitStopNew) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      /*expression=*/
      Dims(DimRangeSpec{-3, absl::nullopt, 2}).AddNew().Label("e", "f"),
      /*expected_new_dimension_selection=*/{3, 5},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 4>(6)
          .input_labels({"a", "b", "c", "e", "d", "f"})
          .output_single_input_dimension(0, 0)
          .output_single_input_dimension(1, 1)
          .output_single_input_dimension(2, 2)
          .output_single_input_dimension(3, 4)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 0>(6)
          .input_labels({"a", "b", "c", "e", "d", "f"})
          .Finalize()
          .value(),
      /*equivalent_indices=*/{},
      /*can_operate_in_place=*/false);
}

TEST(DimsTest, DimRangeSpecImplicitStopNegativeStepNew) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      /*expression=*/
      Dims(DimRangeSpec{1, absl::nullopt, -1}).AddNew().Label("e", "f"),
      /*expected_new_dimension_selection=*/{1, 0},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 4>(6)
          .input_labels({"f", "e", "a", "b", "c", "d"})
          .output_single_input_dimension(0, 2)
          .output_single_input_dimension(1, 3)
          .output_single_input_dimension(2, 4)
          .output_single_input_dimension(3, 5)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 0>(6)
          .input_labels({"f", "e", "a", "b", "c", "d"})
          .Finalize()
          .value(),
      /*equivalent_indices=*/{},
      /*can_operate_in_place=*/false);
}

TEST(DimsTest, DimRangeSpecImplicitStartNegativeStepNew) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      /*expression=*/
      Dims(DimRangeSpec{absl::nullopt, -4, -2}).AddNew().Label("e", "f"),
      /*expected_new_dimension_selection=*/{5, 3},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 4>(6)
          .input_labels({"a", "b", "c", "f", "d", "e"})
          .output_single_input_dimension(0, 0)
          .output_single_input_dimension(1, 1)
          .output_single_input_dimension(2, 2)
          .output_single_input_dimension(3, 4)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 0>(6)
          .input_labels({"a", "b", "c", "f", "d", "e"})
          .Finalize()
          .value(),
      /*equivalent_indices=*/{},
      /*can_operate_in_place=*/false);
}

TEST(DimsTest, DimRangeSpecImplicitStartNew) {
  TestDimExpression(
      /*original_transform=*/IndexTransformBuilder<4, 0>()
          .input_labels({"a", "b", "c", "d"})
          .Finalize()
          .value(),
      /*expression=*/
      Dims(DimRangeSpec{absl::nullopt, 3, 2}).AddNew().Label("e", "f"),
      /*expected_new_dimension_selection=*/{0, 2},
      /*expected_identity_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 4>(6)
          .input_labels({"e", "a", "f", "b", "c", "d"})
          .output_single_input_dimension(0, 1)
          .output_single_input_dimension(1, 3)
          .output_single_input_dimension(2, 4)
          .output_single_input_dimension(3, 5)
          .Finalize()
          .value(),
      /*expected_new_transform=*/
      IndexTransformBuilder<dynamic_rank, 0>(6)
          .input_labels({"e", "a", "f", "b", "c", "d"})
          .Finalize()
          .value(),
      /*equivalent_indices=*/{},
      /*can_operate_in_place=*/false);
}

}  // namespace
