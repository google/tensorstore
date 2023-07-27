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

/// Tests for the `{IndexTransform,IndexDomain}::Transpose()` operations.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"

namespace {

using ::tensorstore::IndexDomainBuilder;
using ::tensorstore::IndexTransformBuilder;

TEST(TransposeTest, Reverse) {
  auto original_transform = IndexTransformBuilder<3, 3>()
                                .input_origin({1, 2, 3})
                                .input_shape({3, 4, 2})
                                .implicit_lower_bounds({1, 0, 0})
                                .implicit_upper_bounds({0, 1, 0})
                                .input_labels({"x", "y", "z"})
                                .output_identity_transform()
                                .Finalize()
                                .value();
  auto original_domain = IndexDomainBuilder<3>()
                             .origin({1, 2, 3})
                             .shape({3, 4, 2})
                             .implicit_lower_bounds({1, 0, 0})
                             .implicit_upper_bounds({0, 1, 0})
                             .labels({"x", "y", "z"})
                             .Finalize()
                             .value();

  const auto expected_new_transform = IndexTransformBuilder<3, 3>()
                                          .input_origin({3, 2, 1})
                                          .input_shape({2, 4, 3})
                                          .implicit_lower_bounds({0, 0, 1})
                                          .implicit_upper_bounds({0, 1, 0})
                                          .input_labels({"z", "y", "x"})
                                          .output_single_input_dimension(0, 2)
                                          .output_single_input_dimension(1, 1)
                                          .output_single_input_dimension(2, 0)
                                          .Finalize()
                                          .value();
  EXPECT_THAT(original_transform.Transpose(),
              ::testing::Eq(expected_new_transform));
  EXPECT_THAT(std::move(original_transform).Transpose(),
              ::testing::Eq(expected_new_transform));
  EXPECT_THAT(original_domain.Transpose(),
              ::testing::Eq(expected_new_transform.domain()));
  EXPECT_THAT(std::move(original_domain).Transpose(),
              ::testing::Eq(expected_new_transform.domain()));
}

TEST(TransposeTest, Permutation) {
  auto original_transform = IndexTransformBuilder(3, 3)
                                .input_origin({1, 2, 3})
                                .input_shape({3, 4, 2})
                                .implicit_lower_bounds({1, 0, 0})
                                .implicit_upper_bounds({0, 1, 0})
                                .input_labels({"x", "y", "z"})
                                .output_identity_transform()
                                .Finalize()
                                .value();
  auto original_domain = IndexDomainBuilder(3)
                             .origin({1, 2, 3})
                             .shape({3, 4, 2})
                             .implicit_lower_bounds({1, 0, 0})
                             .implicit_upper_bounds({0, 1, 0})
                             .labels({"x", "y", "z"})
                             .Finalize()
                             .value();
  const auto expected_new_transform = IndexTransformBuilder(3, 3)
                                          .input_origin({3, 1, 2})
                                          .input_shape({2, 3, 4})
                                          .implicit_lower_bounds({0, 1, 0})
                                          .implicit_upper_bounds({0, 0, 1})
                                          .input_labels({"z", "x", "y"})
                                          .output_single_input_dimension(0, 1)
                                          .output_single_input_dimension(1, 2)
                                          .output_single_input_dimension(2, 0)
                                          .Finalize()
                                          .value();
  EXPECT_THAT(original_transform.Transpose({{2, 0, 1}}),
              ::testing::Eq(expected_new_transform));
  EXPECT_THAT(std::move(original_transform).Transpose({{2, 0, 1}}),
              ::testing::Eq(expected_new_transform));
  EXPECT_THAT(original_domain.Transpose({{2, 0, 1}}),
              ::testing::Eq(expected_new_transform.domain()));
  EXPECT_THAT(std::move(original_domain).Transpose({{2, 0, 1}}),
              ::testing::Eq(expected_new_transform.domain()));
}

TEST(TransposeTest, ReverseOutput) {
  auto original_transform = IndexTransformBuilder<3, 3>()
                                .input_origin({1, 2, 3})
                                .input_shape({3, 4, 2})
                                .implicit_lower_bounds({1, 0, 0})
                                .implicit_upper_bounds({0, 1, 0})
                                .input_labels({"x", "y", "z"})
                                .output_identity_transform()
                                .Finalize()
                                .value();

  auto expected_new_transform = IndexTransformBuilder<3, 3>()
                                    .input_origin({1, 2, 3})
                                    .input_shape({3, 4, 2})
                                    .implicit_lower_bounds({1, 0, 0})
                                    .implicit_upper_bounds({0, 1, 0})
                                    .input_labels({"x", "y", "z"})
                                    .output_single_input_dimension(0, 2)
                                    .output_single_input_dimension(1, 1)
                                    .output_single_input_dimension(2, 0)
                                    .Finalize()
                                    .value();

  EXPECT_THAT(original_transform.TransposeOutput(),
              ::testing::Eq(expected_new_transform));
  EXPECT_THAT(std::move(original_transform).TransposeOutput(),
              ::testing::Eq(expected_new_transform));
}

TEST(TransposeTest, PermutationOutput) {
  auto original_transform = IndexTransformBuilder(3, 3)
                                .input_origin({1, 2, 3})
                                .input_shape({3, 4, 2})
                                .implicit_lower_bounds({1, 0, 0})
                                .implicit_upper_bounds({0, 1, 0})
                                .input_labels({"x", "y", "z"})
                                .output_identity_transform()
                                .Finalize()
                                .value();
  const auto expected_new_transform = IndexTransformBuilder(3, 3)
                                          .input_origin({1, 2, 3})
                                          .input_shape({3, 4, 2})
                                          .implicit_lower_bounds({1, 0, 0})
                                          .implicit_upper_bounds({0, 1, 0})
                                          .input_labels({"x", "y", "z"})
                                          .output_single_input_dimension(0, 2)
                                          .output_single_input_dimension(1, 0)
                                          .output_single_input_dimension(2, 1)
                                          .Finalize()
                                          .value();
  EXPECT_THAT(original_transform.TransposeOutput({{2, 0, 1}}),
              ::testing::Eq(expected_new_transform));
  EXPECT_THAT(std::move(original_transform).TransposeOutput({{2, 0, 1}}),
              ::testing::Eq(expected_new_transform));
}

}  // namespace
