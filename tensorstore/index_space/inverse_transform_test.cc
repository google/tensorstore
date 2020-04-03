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
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
namespace {

using tensorstore::Index;
using tensorstore::IndexTransform;
using tensorstore::IndexTransformBuilder;
using tensorstore::InverseTransform;
using tensorstore::kMaxFiniteIndex;
using tensorstore::MakeArray;
using tensorstore::MatchesStatus;

TEST(InverseTransformTest, Null) {
  auto result = InverseTransform(IndexTransform<>());
  ASSERT_EQ(absl::OkStatus(), GetStatus(result));
  EXPECT_FALSE(result->valid());

  auto result_static = InverseTransform(IndexTransform<3, 3>());
  ASSERT_EQ(absl::OkStatus(), GetStatus(result_static));
  EXPECT_FALSE(result_static->valid());
}

TEST(InverseTransformTest, Example) {
  auto t =  //
      IndexTransformBuilder<>(2, 2)
          .input_labels({"x", "y"})
          .input_origin({1, 2})
          .input_exclusive_max({5, 8})
          .output_single_input_dimension(0, 5, -1, 1)
          .output_single_input_dimension(1, 3, 1, 0)
          .Finalize()
          .value();
  auto expected_inv =  //
      IndexTransformBuilder<>(2, 2)
          .input_labels({"y", "x"})
          .input_origin({-2, 4})
          .input_exclusive_max({4, 8})
          .output_single_input_dimension(0, -3, 1, 1)
          .output_single_input_dimension(1, 5, -1, 0)
          .Finalize()
          .value();
  EXPECT_EQ(expected_inv, InverseTransform(t));
  EXPECT_EQ(t, InverseTransform(expected_inv));
}

TEST(InverseTransformTest, IdentityRank3) {
  auto t =  //
      IndexTransformBuilder<>(3, 3)
          .input_labels({"x", "y", "z"})
          .input_origin({3, 4, 5})
          .input_shape({10, 11, 12})
          .implicit_lower_bounds({1, 0, 1})
          .implicit_upper_bounds({0, 1, 1})
          .output_single_input_dimension(0, 0, 1, 0)
          .output_single_input_dimension(1, 0, 1, 1)
          .output_single_input_dimension(2, 0, 1, 2)
          .Finalize()
          .value();
  EXPECT_EQ(t, InverseTransform(t));
}

TEST(InverseTransformTest, Offsets) {
  auto t =  //
      IndexTransformBuilder<>(3, 3)
          .input_labels({"x", "y", "z"})
          .input_origin({3, 4, 5})
          .input_shape({10, 11, 12})
          .implicit_lower_bounds({1, 0, 1})
          .implicit_upper_bounds({0, 1, 1})
          .output_single_input_dimension(0, 6, 1, 0)
          .output_single_input_dimension(1, 7, 1, 1)
          .output_single_input_dimension(2, 8, 1, 2)
          .Finalize()
          .value();
  auto expected_inv =  //
      IndexTransformBuilder<>(3, 3)
          .input_labels({"x", "y", "z"})
          .input_origin({9, 11, 13})
          .input_shape({10, 11, 12})
          .implicit_lower_bounds({1, 0, 1})
          .implicit_upper_bounds({0, 1, 1})
          .output_single_input_dimension(0, -6, 1, 0)
          .output_single_input_dimension(1, -7, 1, 1)
          .output_single_input_dimension(2, -8, 1, 2)
          .Finalize()
          .value();
  EXPECT_EQ(expected_inv, InverseTransform(t));
  EXPECT_EQ(t, InverseTransform(expected_inv));
}

TEST(InverseTransformTest, Strides) {
  auto t = IndexTransformBuilder<>(3, 3)
               .input_labels({"x", "y", "z"})
               .input_origin({3, 4, 5})
               .input_shape({10, 11, 12})
               .implicit_lower_bounds({1, 0, 1})
               .implicit_upper_bounds({0, 1, 1})
               .output_single_input_dimension(0, 0, -1, 0)
               .output_single_input_dimension(1, 0, 1, 1)
               .output_single_input_dimension(2, 0, -1, 2)
               .Finalize()
               .value();
  auto expected_inv = IndexTransformBuilder<>(3, 3)
                          .input_labels({"x", "y", "z"})
                          .input_origin({-12, 4, -16})
                          .input_shape({10, 11, 12})
                          .implicit_lower_bounds({0, 0, 1})
                          .implicit_upper_bounds({1, 1, 1})
                          .output_single_input_dimension(0, 0, -1, 0)
                          .output_single_input_dimension(1, 0, 1, 1)
                          .output_single_input_dimension(2, 0, -1, 2)
                          .Finalize()
                          .value();
  EXPECT_EQ(expected_inv, InverseTransform(t));
  EXPECT_EQ(t, InverseTransform(expected_inv));
}

TEST(InverseTransformTest, Permutation) {
  auto t = IndexTransformBuilder<>(3, 3)
               .input_labels({"x", "y", "z"})
               .input_origin({3, 4, 5})
               .input_shape({10, 11, 12})
               .implicit_lower_bounds({1, 0, 1})
               .implicit_upper_bounds({0, 1, 1})
               .output_single_input_dimension(0, 0, 1, 1)
               .output_single_input_dimension(1, 0, 1, 2)
               .output_single_input_dimension(2, 0, 1, 0)
               .Finalize()
               .value();
  auto expected_inv = IndexTransformBuilder<>(3, 3)
                          .input_labels({"y", "z", "x"})
                          .input_origin({4, 5, 3})
                          .input_shape({11, 12, 10})
                          .implicit_lower_bounds({0, 1, 1})
                          .implicit_upper_bounds({1, 1, 0})
                          .output_single_input_dimension(1, 0, 1, 0)
                          .output_single_input_dimension(2, 0, 1, 1)
                          .output_single_input_dimension(0, 0, 1, 2)
                          .Finalize()
                          .value();
  EXPECT_EQ(expected_inv, InverseTransform(t));
  EXPECT_EQ(t, InverseTransform(expected_inv));
}

TEST(InverseTransformTest, OffsetsAndStrides) {
  auto t = IndexTransformBuilder<>(3, 3)
               .input_labels({"x", "y", "z"})
               .input_origin({9, 11, 13})
               .input_shape({10, 11, 12})
               .implicit_lower_bounds({1, 0, 1})
               .implicit_upper_bounds({0, 1, 1})
               .output_single_input_dimension(0, -6, -1, 0)
               .output_single_input_dimension(1, -7, 1, 1)
               .output_single_input_dimension(2, -8, -1, 2)
               .Finalize()
               .value();
  auto expected_inv = IndexTransformBuilder<>(3, 3)
                          .input_labels({"x", "y", "z"})
                          .input_origin({-24, 4, -32})
                          .input_shape({10, 11, 12})
                          .implicit_lower_bounds({0, 0, 1})
                          .implicit_upper_bounds({1, 1, 1})
                          .output_single_input_dimension(0, -6, -1, 0)
                          .output_single_input_dimension(1, 7, 1, 1)
                          .output_single_input_dimension(2, -8, -1, 2)
                          .Finalize()
                          .value();
  EXPECT_EQ(expected_inv, InverseTransform(t));
  EXPECT_EQ(t, InverseTransform(expected_inv));
}

TEST(InverseTransformTest, OffsetsAndStridesAndPermutation) {
  auto t = IndexTransformBuilder<>(3, 3)
               .input_labels({"x", "y", "z"})
               .input_origin({9, 11, 13})
               .input_shape({10, 11, 12})
               .implicit_lower_bounds({1, 0, 1})
               .implicit_upper_bounds({0, 1, 1})
               .output_single_input_dimension(0, -6, -1, 1)
               .output_single_input_dimension(1, -7, 1, 2)
               .output_single_input_dimension(2, -8, -1, 0)
               .Finalize()
               .value();
  auto expected_inv = IndexTransformBuilder<>(3, 3)
                          .input_labels({"y", "z", "x"})
                          .input_origin({-27, 6, -26})
                          .input_shape({11, 12, 10})
                          .implicit_lower_bounds({1, 1, 0})
                          .implicit_upper_bounds({0, 1, 1})
                          .output_single_input_dimension(1, -6, -1, 0)
                          .output_single_input_dimension(2, 7, 1, 1)
                          .output_single_input_dimension(0, -8, -1, 2)
                          .Finalize()
                          .value();
  EXPECT_EQ(expected_inv, InverseTransform(t));
  EXPECT_EQ(t, InverseTransform(expected_inv));
}

TEST(InverseTransformTest, RankMismatch) {
  EXPECT_THAT(
      InverseTransform(IndexTransformBuilder<>(3, 2).Finalize().value()),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Transform with input rank \\(3\\) != output rank \\(2\\) "
                    "is not invertible"));
}

TEST(InverseTransformTest, ConstantMap) {
  EXPECT_THAT(
      InverseTransform(IndexTransformBuilder<>(1, 1).Finalize().value()),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Transform is not invertible due to "
                    "non-`single_input_dimension` map for output dimension 0"));
}

TEST(InverseTransformTest, IndexArrayMap) {
  EXPECT_THAT(
      InverseTransform(
          IndexTransformBuilder<>(1, 1)
              .input_shape({2})
              .output_index_array(0, 0, 1, MakeArray<Index>({0, 1}))
              .Finalize()
              .value()),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Transform is not invertible due to "
                    "non-`single_input_dimension` map for output dimension 0"));
}

TEST(InverseTransformTest, NonUnitStride) {
  EXPECT_THAT(InverseTransform(IndexTransformBuilder<>(1, 1)
                                   .output_single_input_dimension(0, 0, 2, 0)
                                   .Finalize()
                                   .value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Transform is not invertible due to stride of 2 "
                            "for output dimension 0"));
}

TEST(InverseTransformTest, Diagonal) {
  EXPECT_THAT(InverseTransform(IndexTransformBuilder<>(2, 2)
                                   .output_single_input_dimension(0, 0, 1, 0)
                                   .output_single_input_dimension(1, 0, 1, 0)
                                   .Finalize()
                                   .value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Transform is not invertible because input "
                            "dimension 0 maps to output dimensions 0 and 1"));
}

TEST(InverseTransformTest, DomainOverflow) {
  EXPECT_THAT(InverseTransform(
                  IndexTransformBuilder<>(1, 1)
                      .input_origin({10})
                      .input_shape({5})
                      .output_single_input_dimension(0, kMaxFiniteIndex, 1, 0)
                      .Finalize()
                      .value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error inverting map from input dimension 0 -> "
                            "output dimension 0: Integer overflow .*"));
}

TEST(InverseTransformTest, OffsetOverflow) {
  EXPECT_THAT(
      InverseTransform(IndexTransformBuilder<>(1, 1)
                           .output_single_input_dimension(
                               0, std::numeric_limits<Index>::min(), 1, 0)
                           .Finalize()
                           .value()),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Integer overflow occurred while inverting map from input "
                    "dimension 0 -> output dimension 0"));
}

}  // namespace
