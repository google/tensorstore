// Copyright 2021 The TensorStore Authors
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
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Box;
using ::tensorstore::Index;
using ::tensorstore::IndexDomainBuilder;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesStatus;

TEST(IndexTransformSliceByBoxTest, Simple) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto transform,
                                   IndexTransformBuilder(3, 3)
                                       .input_origin({0, 1, 2})
                                       .input_exclusive_max({4, 5, 6})
                                       .input_labels({"x", "y", "z"})
                                       .output_identity_transform()
                                       .Finalize());
  Box<> box({1, 1, 3}, {2, 2, 1});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_transform, transform | box);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_transform,
                                   IndexTransformBuilder(3, 3)
                                       .input_bounds(box)
                                       .input_labels({"x", "y", "z"})
                                       .output_identity_transform()
                                       .Finalize());
  EXPECT_EQ(expected_transform, new_transform);
}

TEST(IndexTransformSliceByBoxTest, ImplicitBounds) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto transform,
                                   IndexTransformBuilder(3, 3)
                                       .input_origin({0, 1, 2})
                                       .implicit_lower_bounds({0, 1, 0})
                                       .input_exclusive_max({4, 5, 6})
                                       .input_labels({"x", "y", "z"})
                                       .output_identity_transform()
                                       .Finalize());
  // Box is outside the implicit but not explicit bounds of
  // `transform.domain()`.
  Box<> box({1, -1, 3}, {2, 2, 1});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_transform, transform | box);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_transform,
                                   IndexTransformBuilder(3, 3)
                                       .input_bounds(box)
                                       .input_labels({"x", "y", "z"})
                                       .output_identity_transform()
                                       .Finalize());
  EXPECT_EQ(expected_transform, new_transform);
}

TEST(IndexTransformSliceByBoxTest, OutOfBounds) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto transform,
                                   IndexTransformBuilder(3, 3)
                                       .input_origin({0, 1, 2})
                                       .input_exclusive_max({4, 5, 6})
                                       .input_labels({"x", "y", "z"})
                                       .output_identity_transform()
                                       .Finalize());
  // Box is outside the explicit bounds of `transform.domain()`.
  Box<> box({1, -1, 3}, {2, 2, 1});
  EXPECT_THAT(
      transform | box,
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    "Cannot slice dimension 1 \\{\"y\": \\[1, 5\\)\\} with "
                    "interval \\{\\[-1, 1\\)\\}"));
}

TEST(IndexTransformSliceByBoxTest, RankMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto transform,
                                   IndexTransformBuilder(3, 3)
                                       .input_origin({0, 1, 2})
                                       .input_exclusive_max({4, 5, 6})
                                       .output_identity_transform()
                                       .Finalize());
  // Box is outside the explicit bounds of `transform.domain()`.
  Box<> box({1, 1}, {2, 2});
  EXPECT_THAT(
      transform | box,
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Rank of index domain \\(3\\) must match rank of box \\(2\\)"));
}

TEST(IndexTransformSliceByBoxTest, IndexArray) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(  //
      auto transform,                //
      IndexTransformBuilder(1, 1)
          .input_shape({5})
          .output_index_array(0, 0, 1, MakeArray<Index>({1, 2, 3, 4, 5}))
          .Finalize());
  Box<> box({1}, {2});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(  //
      auto expected_transform,       //
      IndexTransformBuilder(1, 1)
          .input_origin({1})
          .input_shape({2})
          .output_index_array(0, 0, 1, MakeArray<Index>({2, 3}))
          .Finalize());
  EXPECT_THAT(transform | box, ::testing::Optional(expected_transform));
}

TEST(IndexTransformSliceByBoxTest, IndexArrayZeroSize) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(  //
      auto transform,                //
      IndexTransformBuilder(1, 1)
          .input_shape({5})
          .output_index_array(0, 0, 1, MakeArray<Index>({1, 2, 3, 4, 5}))
          .Finalize());
  Box<> box({1}, {0});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(  //
      auto expected_transform,       //
      IndexTransformBuilder(1, 1)
          .input_origin({1})
          .input_shape({0})
          .output_constant(0, 0)
          .Finalize());
  EXPECT_THAT(transform | box, ::testing::Optional(expected_transform));
}

TEST(IndexDomainSliceByBoxTest, Simple) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain, IndexDomainBuilder(3)
                                                    .origin({0, 1, 2})
                                                    .exclusive_max({4, 5, 6})
                                                    .labels({"x", "y", "z"})
                                                    .Finalize());
  Box<> box({1, 1, 3}, {2, 2, 1});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_domain,
      IndexDomainBuilder(3).bounds(box).labels({"x", "y", "z"}).Finalize());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_domain,
                                   domain | tensorstore::BoxView<>(box));
  EXPECT_EQ(expected_domain, new_domain);
}

}  // namespace
