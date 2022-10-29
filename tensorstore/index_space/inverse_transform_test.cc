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

#include <random>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/index_transform_testutil.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Index;
using ::tensorstore::IndexDomainBuilder;
using ::tensorstore::IndexTransform;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::InverseTransform;
using ::tensorstore::kMaxFiniteIndex;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesStatus;

TEST(InverseTransformTest, Null) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto inv,
                                   InverseTransform(IndexTransform<>()));
  EXPECT_FALSE(inv.valid());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto inv_static,
                                   InverseTransform(IndexTransform<3, 3>()));
  EXPECT_FALSE(inv_static.valid());
}

TEST(InverseTransformTest, Example) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto t,  //
      IndexTransformBuilder(3, 3)
          .input_labels({"x", "", "y"})
          .input_origin({1, 3, 2})
          .input_exclusive_max({5, 4, 8})
          .implicit_lower_bounds({1, 0, 0})
          .implicit_upper_bounds({0, 0, 1})
          .output_single_input_dimension(0, 5, -1, 2)
          .output_single_input_dimension(1, 3, 1, 0)
          .output_constant(2, 7)
          .Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_inv,  //
      IndexTransformBuilder(3, 3)
          .input_labels({"y", "x", ""})
          .input_origin({-2, 4, 7})
          .input_exclusive_max({4, 8, 8})
          .implicit_lower_bounds({1, 1, 0})
          .implicit_upper_bounds({0, 0, 0})
          .output_single_input_dimension(0, -3, 1, 1)
          .output_constant(1, 3)
          .output_single_input_dimension(2, 5, -1, 0)
          .Finalize());
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
          .output_identity_transform()
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
               .output_single_input_dimension(1, 1)
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
                          .output_single_input_dimension(1, 1)
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
               .output_single_input_dimension(0, 1)
               .output_single_input_dimension(1, 2)
               .output_single_input_dimension(2, 0)
               .Finalize()
               .value();
  auto expected_inv = IndexTransformBuilder<>(3, 3)
                          .input_labels({"y", "z", "x"})
                          .input_origin({4, 5, 3})
                          .input_shape({11, 12, 10})
                          .implicit_lower_bounds({0, 1, 1})
                          .implicit_upper_bounds({1, 1, 0})
                          .output_single_input_dimension(1, 0)
                          .output_single_input_dimension(2, 1)
                          .output_single_input_dimension(0, 2)
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

TEST(InverseTransformTest, ErrorNonSingletonUnmappedInputDimension) {
  EXPECT_THAT(
      InverseTransform(IndexTransformBuilder<>(3, 2)
                           .output_identity_transform()
                           .Finalize()
                           .value()),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Transform is not invertible due to non-singleton "
                    "input dimension 2 with domain \\(-inf\\*, \\+inf\\*\\) "
                    "that is not mapped by an output dimension"));
  EXPECT_THAT(InverseTransform(IndexTransformBuilder(1, 0)
                                   .input_origin({0})
                                   .input_shape({2})
                                   .Finalize()
                                   .value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Transform is not invertible due to non-singleton "
                            "input dimension 0 with domain \\[0, 2\\) "
                            "that is not mapped by an output dimension"));
  EXPECT_THAT(InverseTransform(IndexTransformBuilder(1, 0)
                                   .input_origin({0})
                                   .input_shape({1})
                                   .implicit_lower_bounds({1})
                                   .Finalize()
                                   .value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Transform is not invertible due to non-singleton "
                            "input dimension 0 with domain \\[0\\*, 1\\) "
                            "that is not mapped by an output dimension"));
  EXPECT_THAT(InverseTransform(IndexTransformBuilder(1, 0)
                                   .input_origin({0})
                                   .input_shape({1})
                                   .implicit_upper_bounds({1})
                                   .Finalize()
                                   .value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Transform is not invertible due to non-singleton "
                            "input dimension 0 with domain \\[0, 1\\*\\) "
                            "that is not mapped by an output dimension"));
}

TEST(InverseTransformTest, ConstantMap) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto t, IndexTransformBuilder(0, 1).output_constant(0, 42).Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_inv,
                                   IndexTransformBuilder(1, 0)
                                       .input_origin({42})
                                       .input_shape({1})
                                       .Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_inv_with_label,
                                   IndexTransformBuilder(1, 0)
                                       .input_origin({42})
                                       .input_labels({"x"})
                                       .input_shape({1})
                                       .Finalize());
  EXPECT_THAT(InverseTransform(t), ::testing::Optional(expected_inv));
  EXPECT_THAT(InverseTransform(expected_inv), ::testing::Optional(t));
  // Input dimension label is lost, since IndexTransform does not support output
  // dimension labels and singleton input dimensions don't correspond to an
  // input dimension in the inverse transform.
  EXPECT_THAT(InverseTransform(expected_inv_with_label),
              ::testing::Optional(t));
}

TEST(InverseTransformTest, IndexArrayMap) {
  EXPECT_THAT(InverseTransform(
                  IndexTransformBuilder<>(1, 1)
                      .input_shape({2})
                      .output_index_array(0, 0, 1, MakeArray<Index>({0, 1}))
                      .Finalize()
                      .value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Transform is not invertible due to "
                            "index array map for output dimension 0"));
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
                                   .output_single_input_dimension(0, 0)
                                   .output_single_input_dimension(1, 0)
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

TEST(InverseTransformTest, RandomFromOutputSpace) {
  constexpr size_t kNumIterations = 100;
  for (size_t i = 0; i < kNumIterations; ++i) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_INTERNAL_INVERSE_TRANSFORM_TEST_SEED")};
    auto box = tensorstore::internal::MakeRandomBox(gen);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto domain, IndexDomainBuilder(box.rank()).bounds(box).Finalize());
    auto transform =
        tensorstore::internal::MakeRandomStridedIndexTransformForOutputSpace(
            gen, domain);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto inv_transform,
                                     InverseTransform(transform));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto inv_inv_transform,
                                     InverseTransform(inv_transform));
    EXPECT_EQ(transform, inv_inv_transform);
  }
}

TEST(InverseTransformTest, RandomFromInputSpace) {
  constexpr size_t kNumIterations = 100;
  for (size_t i = 0; i < kNumIterations; ++i) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_INTERNAL_INVERSE_TRANSFORM_TEST_SEED")};
    auto box = tensorstore::internal::MakeRandomBox(gen);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto domain, IndexDomainBuilder(box.rank()).bounds(box).Finalize());
    auto transform =
        tensorstore::internal::MakeRandomStridedIndexTransformForInputSpace(
            gen, domain);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto inv_transform,
                                     InverseTransform(transform));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto inv_inv_transform,
                                     InverseTransform(inv_transform));
    EXPECT_EQ(transform, inv_inv_transform);
  }
}

}  // namespace
