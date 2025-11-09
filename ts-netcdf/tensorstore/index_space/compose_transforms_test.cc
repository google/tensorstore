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

#include "tensorstore/index_space/internal/compose_transforms.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::IdentityTransform;
using ::tensorstore::Index;
using ::tensorstore::IndexInterval;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::IndexTransformView;
using ::tensorstore::kMaxFiniteIndex;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesStatus;

TEST(ComposeTransformsTest, EmptyDomain) {
  auto b_to_c = IndexTransformBuilder<3, 2>()
                    .input_origin({1, 2, 3})
                    .input_shape({5, 6, 7})
                    .output_identity_transform()
                    .Finalize()
                    .value();
  auto a_to_b = IndexTransformBuilder<2, 3>()
                    .input_origin({1, 2})
                    .input_shape({5, 0})
                    .output_identity_transform()
                    .output_constant(2, 5)
                    .Finalize()
                    .value();
  auto a_to_c = ComposeTransforms(b_to_c, a_to_b).value();
  auto expected_a_to_c = IndexTransformBuilder<2, 2>()
                             .input_origin({1, 2})
                             .input_shape({5, 0})
                             .output_identity_transform()
                             .Finalize()
                             .value();
  EXPECT_EQ(expected_a_to_c, a_to_c);
}

TEST(ComposeTransformsTest, TransformArrayError) {
  auto b_to_c = IndexTransformBuilder<1, 1>()
                    .input_origin({0})
                    .input_shape({2})
                    .output_index_array(0, 0, 1, MakeArray<Index>({1, 2}))
                    .Finalize()
                    .value();
  auto a_to_b = IndexTransformBuilder<1, 1>()
                    .input_origin({1})
                    .input_shape({1})
                    .output_index_array(0, 0, 1, MakeArray<Index>({1}),
                                        IndexInterval::Closed(4, 6))
                    .Finalize()
                    .value();
  EXPECT_THAT(ComposeTransforms(b_to_c, a_to_b),
              MatchesStatus(absl::StatusCode::kOutOfRange));
}

TEST(ComposeTransformsTest, BtoCIndexArrayWithSingleIndex) {
  auto b_to_c = IndexTransformBuilder<1, 1>()
                    .input_origin({0})
                    .input_shape({2})
                    .output_index_array(0, 0, 1, MakeArray<Index>({7, 8}))
                    .Finalize()
                    .value();
  auto a_to_b = IndexTransformBuilder<1, 1>()
                    .input_origin({1})
                    .input_shape({1})
                    .output_identity_transform()
                    .Finalize()
                    .value();
  auto a_to_c = ComposeTransforms(b_to_c, a_to_b).value();
  auto expected_a_to_c = IndexTransformBuilder<1, 1>()
                             .input_origin({1})
                             .input_shape({1})
                             .output_constant(0, 8)
                             .Finalize()
                             .value();
  EXPECT_EQ(expected_a_to_c, a_to_c);
}

TEST(ComposeTransformsTest, BtoCIndexArrayWithInvalidSingleIndex) {
  auto b_to_c = IndexTransformBuilder<1, 1>()
                    .input_origin({0})
                    .input_shape({2})
                    .output_index_array(0, 0, 1, MakeArray<Index>({7, 8}),
                                        IndexInterval::Closed(2, 3))
                    .Finalize()
                    .value();
  auto a_to_b = IndexTransformBuilder<1, 1>()
                    .input_origin({1})
                    .input_shape({1})
                    .output_identity_transform()
                    .Finalize()
                    .value();
  EXPECT_THAT(ComposeTransforms(b_to_c, a_to_b),
              MatchesStatus(absl::StatusCode::kOutOfRange));
}

TEST(ComposeTransformsTest, AtoBIndexArrayWithSingleIndex) {
  auto b_to_c = IndexTransformBuilder<1, 1>()
                    .output_identity_transform()
                    .Finalize()
                    .value();
  auto a_to_b = IndexTransformBuilder<1, 1>()
                    .input_origin({0})
                    .input_shape({1})
                    .output_index_array(0, 0, 1, MakeArray<Index>({7}))
                    .Finalize()
                    .value();
  auto a_to_c = ComposeTransforms(b_to_c, a_to_b).value();
  auto expected_a_to_c = IndexTransformBuilder<1, 1>()
                             .input_origin({0})
                             .input_shape({1})
                             .output_constant(0, 7)
                             .Finalize()
                             .value();
  EXPECT_EQ(expected_a_to_c, a_to_c);
}

TEST(ComposeTransformsTest, AtoBIndexArrayWithInvalidSingleIndex) {
  auto b_to_c = IndexTransformBuilder<1, 1>()
                    .output_identity_transform()
                    .Finalize()
                    .value();
  auto a_to_b = IndexTransformBuilder<1, 1>()
                    .input_origin({0})
                    .input_shape({1})
                    .output_index_array(0, 0, 1, MakeArray<Index>({7}),
                                        IndexInterval::Closed(2, 3))
                    .Finalize()
                    .value();
  EXPECT_THAT(ComposeTransforms(b_to_c, a_to_b),
              MatchesStatus(absl::StatusCode::kOutOfRange));
}

TEST(ComposeTransformsTest, ConstantOutOfDomain) {
  auto b_to_c = IndexTransformBuilder<3, 2>()
                    .input_origin({1, 2, 3})
                    .input_shape({5, 6, 7})
                    .output_identity_transform()
                    .Finalize()
                    .value();
  auto a_to_b = IndexTransformBuilder<2, 3>()
                    .input_origin({1, 2})
                    .input_shape({5, 4})
                    .output_identity_transform()
                    .output_constant(2, 2)
                    .Finalize()
                    .value();
  EXPECT_THAT(ComposeTransforms(b_to_c, a_to_b).status(),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            ".*Index 2 is outside valid range \\[3, 10\\)"));
}

TEST(ComposeTransformsTest, ConstantOverflow) {
  // Overflow computing product of a_to_b offset and b_to_c stride.
  EXPECT_THAT(ComposeTransforms(IndexTransformBuilder<1, 1>()
                                    .output_single_input_dimension(0, 0, 100, 0)
                                    .Finalize()
                                    .value(),
                                IndexTransformBuilder<0, 1>()
                                    .output_constant(0, kMaxFiniteIndex)
                                    .Finalize()
                                    .value())
                  .status(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Overflow computing addition of b_to_c offset and multiplied a_to_c offset.
  EXPECT_THAT(
      ComposeTransforms(IndexTransformBuilder<1, 1>()
                            .output_single_input_dimension(
                                0, std::numeric_limits<Index>::max(), 1, 0)
                            .Finalize()
                            .value(),
                        IndexTransformBuilder<0, 1>()
                            .output_constant(0, 100)
                            .Finalize()
                            .value())
          .status(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(ComposeTransformsTest, SingleInputDimensionOverflow) {
  // Overflow adding b_to_c offset to multiplied a_to_c offset.
  EXPECT_THAT(
      ComposeTransforms(IndexTransformBuilder<1, 1>()
                            .output_single_input_dimension(
                                0, std::numeric_limits<Index>::max(), 1, 0)
                            .Finalize()
                            .value(),
                        IndexTransformBuilder<1, 1>()
                            .input_origin({0})
                            .input_shape({10})
                            .output_single_input_dimension(0, 100, 1, 0)
                            .Finalize()
                            .value())
          .status(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Overflow multiplying a_to_c offset by b_to_c output stride.
  EXPECT_THAT(ComposeTransforms(
                  IndexTransformBuilder<1, 1>()
                      .output_single_input_dimension(0, 0, 100, 0)
                      .Finalize()
                      .value(),
                  IndexTransformBuilder<1, 1>()
                      .input_origin({0})
                      .input_shape({10})
                      .output_single_input_dimension(0, kMaxFiniteIndex, 1, 0)
                      .Finalize()
                      .value())
                  .status(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Overflow multiplying a_to_c output stride by b_to_c output stride.
  EXPECT_THAT(
      ComposeTransforms(IndexTransformBuilder<1, 1>()
                            .output_single_input_dimension(0, 0, 100, 0)
                            .Finalize()
                            .value(),
                        IndexTransformBuilder<1, 1>()
                            .input_origin({0})
                            .input_shape({10})
                            .output_single_input_dimension(
                                0, 0, std::numeric_limits<Index>::max() - 1, 0)
                            .Finalize()
                            .value())
          .status(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(ComposeTransformsTest, IndexArrayBoundsOverflow) {
  // Overflow propagating the bounds for the index array.
  EXPECT_THAT(ComposeTransforms(
                  IndexTransformBuilder<1, 1>()
                      .input_origin({2})
                      .input_shape({100})
                      .output_identity_transform()
                      .Finalize()
                      .value(),
                  IndexTransformBuilder<1, 1>()
                      .input_origin({0})
                      .input_shape({2})
                      .output_index_array(0, std::numeric_limits<Index>::min(),
                                          1, MakeArray<Index>({1, 2}),
                                          IndexInterval::Closed(0, 100))
                      .Finalize()
                      .value())
                  .status(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(ComposeTransformsTest, RankMismatch) {
  EXPECT_THAT(
      ComposeTransforms(IdentityTransform(2), IdentityTransform(3)).status(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Rank 2 -> 2 transform cannot be composed with rank 3 -> 3 "
                    "transform\\."));
}

/// Tests that IndexTransform::operator() can be used to compose transforms.
TEST(ComposeTransformsTest, FunctionCallOperator) {
  const auto t0 = IndexTransformBuilder<1, 1>()
                      .input_origin({0})
                      .input_shape({3})
                      .output_single_input_dimension(0, 10, 1, 0)
                      .Finalize()
                      .value();

  const auto t1 = IndexTransformBuilder<1, 1>()
                      .input_origin({10})
                      .input_shape({5})
                      .output_single_input_dimension(0, 20, 1, 0)
                      .Finalize()
                      .value();
  const auto expected_composed = IndexTransformBuilder<1, 1>()
                                     .input_origin({0})
                                     .input_shape({3})
                                     .output_single_input_dimension(0, 30, 1, 0)
                                     .Finalize()
                                     .value();
  const auto composed = t0(t1).value();
  EXPECT_EQ(expected_composed, composed);
  EXPECT_EQ(expected_composed, ComposeTransforms(t1, t0).value());
}

/// Tests that rank-0 transforms can be composed.
TEST(ComposeTransformsTest, RankZero) {
  auto t0 = IdentityTransform(0);
  EXPECT_EQ(t0, ComposeTransforms(t0, t0).value());
}

/// Tests that implicit bounds do not constrain transform composition.
TEST(ComposeTransformsTest, ImplicitOutOfBounds) {
  const auto t0 = IndexTransformBuilder<1, 1>()
                      .input_origin({0})
                      .input_shape({4})
                      .implicit_lower_bounds({1})
                      .output_identity_transform()
                      .Finalize()
                      .value();

  const auto t1 = IndexTransformBuilder<1, 1>()
                      .input_origin({-1})
                      .input_exclusive_max({2})
                      .output_identity_transform()
                      .Finalize()
                      .value();
  EXPECT_THAT(ComposeTransforms(t0, t1), ::testing::Optional(t1));
}

TEST(ComposeTransformsTest, TransformIndexArraySkipRepeatedElements) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto t0, IndexTransformBuilder(2, 2)
                   .input_shape({5, 2})
                   .output_index_array(
                       0, 0, 1, MakeArray<Index>({{0}, {1}, {2}, {3}, {4}}))
                   .output_single_input_dimension(1, 1)
                   .Finalize());
  EXPECT_THAT(t0.output_index_maps()[0].index_array().byte_strides(),
              ::testing::ElementsAre(8, 0));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto t1, ComposeTransforms(t0, t0));
  EXPECT_EQ(t0, t1);
  EXPECT_THAT(t1.output_index_maps()[0].index_array().byte_strides(),
              ::testing::ElementsAre(8, 0));
}

}  // namespace
