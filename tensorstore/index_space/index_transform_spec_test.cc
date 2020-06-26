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

#include "tensorstore/index_space/index_transform_spec.h"

#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using tensorstore::DimensionIndex;
using tensorstore::dynamic_rank;
using tensorstore::IdentityTransform;
using tensorstore::IndexTransform;
using tensorstore::IndexTransformBuilder;
using tensorstore::IndexTransformSpec;
using tensorstore::kInfIndex;
using tensorstore::MatchesStatus;
using tensorstore::StrCat;

TEST(IndexTransformSpecTest, DefaultConstruct) {
  IndexTransformSpec t;
  EXPECT_EQ(dynamic_rank, t.input_rank());
  EXPECT_EQ(dynamic_rank, t.output_rank());
  EXPECT_FALSE(t.transform().valid());
}

TEST(IndexTransformSpecTest, ConstructWithRank) {
  IndexTransformSpec t{2};
  EXPECT_EQ(2, t.input_rank());
  EXPECT_EQ(2, t.output_rank());
  EXPECT_FALSE(t.transform().valid());
}

TEST(IndexTransformSpecTest, ConstructWithTransform) {
  auto transform = IndexTransformBuilder<>(2, 1).Finalize().value();
  IndexTransformSpec t{transform};
  EXPECT_EQ(2, t.input_rank());
  EXPECT_EQ(1, t.output_rank());
  EXPECT_EQ(transform, t.transform());
}

TEST(IndexTransformSpecTest, AssignTransform) {
  IndexTransformSpec t;
  auto transform = IndexTransformBuilder<2, 1>().Finalize().value();
  t = transform;
  EXPECT_EQ(2, t.input_rank());
  EXPECT_EQ(1, t.output_rank());
  EXPECT_EQ(transform, t.transform());
}

TEST(IndexTransformSpecTest, AssignRank) {
  auto transform = IndexTransformBuilder<2, 1>().Finalize().value();
  IndexTransformSpec t(transform);
  t = 3;
  EXPECT_EQ(3, t.input_rank());
  EXPECT_EQ(3, t.output_rank());
  EXPECT_FALSE(t.transform().valid());
}

TEST(IndexTransformSpecTest, Comparison) {
  IndexTransformSpec t1{2};
  auto transform1 = IndexTransformBuilder<2, 1>().Finalize().value();
  auto transform2 = IndexTransformBuilder<2, 0>().Finalize().value();
  IndexTransformSpec t2{transform1};
  IndexTransformSpec t3{transform2};
  IndexTransformSpec t4{3};

  EXPECT_EQ(t1, t1);
  EXPECT_EQ(t2, t2);
  EXPECT_NE(t1, t2);
  EXPECT_NE(t1, t3);
  EXPECT_NE(t2, t3);
  EXPECT_NE(t1, t4);
}

TEST(IndexTransformSpecTest, PrintToOstream) {
  IndexTransformSpec t1{2};
  EXPECT_EQ("2", StrCat(t1));
  auto transform = IndexTransformBuilder<2, 1>().Finalize().value();
  IndexTransformSpec t2{transform};
  EXPECT_EQ(StrCat(transform), StrCat(t2));
}

TEST(IndexTransformSpecTest, ComposeUnknownUnknown) {
  IndexTransformSpec t1;
  IndexTransformSpec t2;
  EXPECT_EQ(ComposeIndexTransformSpecs(t1, t2), IndexTransformSpec());
}

TEST(IndexTransformSpecTest, ComposeRankRank) {
  IndexTransformSpec t1{2};
  IndexTransformSpec t2{2};
  EXPECT_EQ(ComposeIndexTransformSpecs(t1, t2), IndexTransformSpec(2));
}

TEST(IndexTransformSpecTest, ComposeRankUnknown) {
  IndexTransformSpec t1{2};
  IndexTransformSpec t2;
  EXPECT_EQ(ComposeIndexTransformSpecs(t1, t2), IndexTransformSpec(2));
}

TEST(IndexTransformSpecTest, ComposeUnknownRank) {
  IndexTransformSpec t1;
  IndexTransformSpec t2{2};
  EXPECT_EQ(ComposeIndexTransformSpecs(t1, t2), IndexTransformSpec(2));
}

TEST(IndexTransformSpecTest, ComposeRankRankError) {
  IndexTransformSpec t1{2};
  IndexTransformSpec t2{3};
  EXPECT_THAT(ComposeIndexTransformSpecs(t1, t2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot compose transform of rank 2 -> 2 with "
                            "transform of rank 3 -> 3"));
}

TEST(IndexTransformSpecTest, ComposeTransformRank) {
  auto transform = IndexTransformBuilder<>(2, 1).Finalize().value();
  IndexTransformSpec t1{transform};
  IndexTransformSpec t2{2};
  EXPECT_EQ(ComposeIndexTransformSpecs(t1, t2), IndexTransformSpec(transform));
}

TEST(IndexTransformSpecTest, ComposeRankTransform) {
  auto transform = IndexTransformBuilder<>(2, 1).Finalize().value();
  IndexTransformSpec t1{1};
  IndexTransformSpec t2{transform};
  EXPECT_EQ(ComposeIndexTransformSpecs(t1, t2), IndexTransformSpec(transform));
}

TEST(IndexTransformSpecTest, ComposeTransformRankError) {
  auto transform = IndexTransformBuilder<>(2, 1).Finalize().value();
  IndexTransformSpec t1{transform};
  IndexTransformSpec t2{3};
  EXPECT_THAT(ComposeIndexTransformSpecs(t1, t2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot compose transform of rank 2 -> 1 with "
                            "transform of rank 3 -> 3"));
}

TEST(IndexTransformSpecTest, ComposeRankTransformError) {
  auto transform = IndexTransformBuilder<>(2, 1).Finalize().value();
  IndexTransformSpec t1{3};
  IndexTransformSpec t2{transform};
  EXPECT_THAT(ComposeIndexTransformSpecs(t1, t2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot compose transform of rank 3 -> 3 with "
                            "transform of rank 2 -> 1"));
}

TEST(IndexTransformSpecTest, ComposeTransformTransform) {
  auto transform1 = IndexTransformBuilder<>(1, 3).Finalize().value();
  auto transform2 = IndexTransformBuilder<>(2, 1).Finalize().value();
  IndexTransformSpec t1{transform1};
  IndexTransformSpec t2{transform2};
  EXPECT_EQ(
      ComposeIndexTransformSpecs(t1, t2),
      IndexTransformSpec(IndexTransformBuilder<>(2, 3).Finalize().value()));
}

TEST(IndexTransformSpecTest, ComposeTransformTransformRankMismatch) {
  auto transform1 = IndexTransformBuilder<>(1, 3).Finalize().value();
  auto transform2 = IndexTransformBuilder<>(2, 2).Finalize().value();
  IndexTransformSpec t1{transform1};
  IndexTransformSpec t2{transform2};
  EXPECT_THAT(ComposeIndexTransformSpecs(t1, t2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot compose transform of rank 1 -> 3 with "
                            "transform of rank 2 -> 2"));
}

TEST(IndexTransformSpecTest, ComposeTransformTransformError) {
  auto transform1 = tensorstore::IdentityTransform({5, 5});
  auto transform2 = tensorstore::IdentityTransform({10, 10});
  IndexTransformSpec t1{transform1};
  IndexTransformSpec t2{transform2};
  EXPECT_THAT(ComposeIndexTransformSpecs(t1, t2),
              MatchesStatus(absl::StatusCode::kOutOfRange));
}

TEST(IndexTransformSpecTest, ApplyIndexTransform) {
  IndexTransformSpec spec_with_unknown_rank;
  IndexTransformSpec spec_with_known_rank{3};
  IndexTransformSpec spec_with_transform{tensorstore::IdentityTransform(3)};
  auto expr = tensorstore::Dims(0).SizedInterval(0, 10);
  EXPECT_THAT(tensorstore::ChainResult(spec_with_unknown_rank, expr),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Transform is unspecified"));
  EXPECT_THAT(tensorstore::ChainResult(spec_with_known_rank, expr),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Transform is unspecified"));
  EXPECT_THAT(tensorstore::ChainResult(spec_with_transform, expr),
              ::testing::Optional(IndexTransformSpec{
                  tensorstore::IndexTransformBuilder<>(3, 3)
                      .input_origin({0, -kInfIndex, -kInfIndex})
                      .input_exclusive_max({10, kInfIndex + 1, kInfIndex + 1})
                      .implicit_lower_bounds({false, true, true})
                      .implicit_upper_bounds({false, true, true})
                      .output_single_input_dimension(0, 0)
                      .output_single_input_dimension(1, 1)
                      .output_single_input_dimension(2, 2)
                      .Finalize()
                      .value()}));
}

}  // namespace
