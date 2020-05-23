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
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::IndexDomain;
using tensorstore::IndexDomainBuilder;
using tensorstore::IndexTransformBuilder;
using tensorstore::MatchesStatus;

TEST(SliceByIndexDomainTest, BothFullyUnlabeled) {
  auto transform = IndexTransformBuilder<>(2, 2)
                       .input_origin({0, 1})
                       .input_exclusive_max({5, 7})
                       .output_single_input_dimension(0, 1)
                       .output_single_input_dimension(1, 0)
                       .Finalize()
                       .value();
  auto domain = IndexDomainBuilder(2)
                    .origin({2, 3})
                    .exclusive_max({4, 6})
                    .Finalize()
                    .value();
  // transform: [0, 5), [1, 7)
  // domain:    [2, 4), [3, 6)
  // result:    [2, 4), [3, 6)
  auto expected_new_transform = IndexTransformBuilder<>(2, 2)
                                    .input_origin({2, 3})
                                    .input_exclusive_max({4, 6})
                                    .output_single_input_dimension(0, 1)
                                    .output_single_input_dimension(1, 0)
                                    .Finalize()
                                    .value();
  auto new_transform = domain(std::move(transform));
  EXPECT_EQ(expected_new_transform, new_transform);
}

TEST(SliceByIndexDomainTest, FullyUnlabeledDomain) {
  auto transform = IndexTransformBuilder<>(2, 2)
                       .input_origin({0, 1})
                       .input_exclusive_max({5, 7})
                       .input_labels({"x", "y"})
                       .output_single_input_dimension(0, 1)
                       .output_single_input_dimension(1, 0)
                       .Finalize()
                       .value();
  auto domain = IndexDomainBuilder(2)
                    .origin({2, 3})
                    .exclusive_max({4, 6})
                    .Finalize()
                    .value();
  // transform: "x": [0, 5), "y": [1, 7)
  // domain:    [2, 4), [3, 6)
  // result:    "x": [2, 4), "y": [3, 6)
  // The transform labels are retained.
  auto expected_new_transform = IndexTransformBuilder<>(2, 2)
                                    .input_origin({2, 3})
                                    .input_exclusive_max({4, 6})
                                    .input_labels({"x", "y"})
                                    .output_single_input_dimension(0, 1)
                                    .output_single_input_dimension(1, 0)
                                    .Finalize()
                                    .value();
  auto new_transform = domain(std::move(transform));
  EXPECT_EQ(expected_new_transform, new_transform);
}

TEST(SliceByIndexDomainTest, FullyUnlabeledTransform) {
  auto transform = IndexTransformBuilder<>(2, 2)
                       .input_origin({0, 1})
                       .input_exclusive_max({5, 7})
                       .output_single_input_dimension(0, 1)
                       .output_single_input_dimension(1, 0)
                       .Finalize()
                       .value();
  auto domain = IndexDomainBuilder(2)
                    .origin({2, 3})
                    .exclusive_max({4, 6})
                    .labels({"x", "y"})
                    .Finalize()
                    .value();
  // transform: [0, 5), [1, 7)
  // domain:    "x": [2, 4), "y": [3, 6)
  // result:    "x": [2, 4), "y": [3, 6)
  // The domain labels are copied to the transform.
  auto expected_new_transform = IndexTransformBuilder<>(2, 2)
                                    .input_origin({2, 3})
                                    .input_exclusive_max({4, 6})
                                    .input_labels({"x", "y"})
                                    .output_single_input_dimension(0, 1)
                                    .output_single_input_dimension(1, 0)
                                    .Finalize()
                                    .value();
  auto new_transform = domain(std::move(transform));
  EXPECT_EQ(expected_new_transform, new_transform);
}

TEST(SliceByIndexDomainTest, LabeledFullMatch) {
  auto transform = IndexTransformBuilder<>(2, 2)
                       .input_origin({0, 1})
                       .input_exclusive_max({5, 7})
                       .input_labels({"x", "y"})
                       .output_single_input_dimension(0, 1)
                       .output_single_input_dimension(1, 0)
                       .Finalize()
                       .value();
  auto domain = IndexDomainBuilder(2)
                    .origin({2, 3})
                    .exclusive_max({6, 4})
                    .labels({"y", "x"})
                    .Finalize()
                    .value();
  // transform: "x": [0, 5), "y": [1, 7)
  // domain:    "y": [2, 6), "x": [3, 4)
  // result:    "x": [3, 4), "y": [2, 6)
  auto expected_new_transform = IndexTransformBuilder<>(2, 2)
                                    .input_origin({3, 2})
                                    .input_exclusive_max({4, 6})
                                    .input_labels({"x", "y"})
                                    .output_single_input_dimension(0, 1)
                                    .output_single_input_dimension(1, 0)
                                    .Finalize()
                                    .value();
  auto new_transform = domain(std::move(transform));
  EXPECT_EQ(expected_new_transform, new_transform);
}

TEST(SliceByIndexDomainTest, OutOfBounds) {
  auto transform = IndexTransformBuilder<>(2, 2)
                       .input_origin({0, 1})
                       .input_exclusive_max({5, 7})
                       .input_labels({"x", "y"})
                       .output_single_input_dimension(0, 1)
                       .output_single_input_dimension(1, 0)
                       .Finalize()
                       .value();
  auto domain = IndexDomainBuilder(2)
                    .origin({2, -1})
                    .exclusive_max({6, 4})
                    .labels({"y", "x"})
                    .Finalize()
                    .value();
  EXPECT_THAT(
      domain(transform),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    "Cannot slice target dimension 0 \\{\"x\": \\[0, 5\\)\\} "
                    "with index domain dimension 1 \\{\"x\": \\[-1, 4\\)\\}"));
}

// Tests that when the domain is fully labeled, the ranks need not be the same.
TEST(SliceByIndexDomainTest, LabeledPartialMatch) {
  auto transform = IndexTransformBuilder<>(3, 2)
                       .input_origin({0, 1, 2})
                       .input_exclusive_max({5, 7, 9})
                       .input_labels({"x", "y", "z"})
                       .output_single_input_dimension(0, 1)
                       .output_single_input_dimension(1, 0)
                       .Finalize()
                       .value();
  auto domain = IndexDomainBuilder(2)
                    .origin({2, 3})
                    .exclusive_max({6, 4})
                    .labels({"y", "x"})
                    .Finalize()
                    .value();
  // transform: "x": [0, 5), "y": [1, 7), "z": [2, 9)
  // domain:    "y": [2, 6), "x": [3, 4)
  // result:    "x": [3, 4), "y": [2, 6), "z": [2, 9)
  auto expected_new_transform = IndexTransformBuilder<>(3, 2)
                                    .input_origin({3, 2, 2})
                                    .input_exclusive_max({4, 6, 9})
                                    .input_labels({"x", "y", "z"})
                                    .output_single_input_dimension(0, 1)
                                    .output_single_input_dimension(1, 0)
                                    .Finalize()
                                    .value();
  auto new_transform = domain(std::move(transform));
  EXPECT_EQ(expected_new_transform, new_transform);
}

TEST(SliceByIndexDomainTest, PartiallyLabeled) {
  auto transform = IndexTransformBuilder<>(4, 2)
                       .input_labels({"x", "", "", "y"})
                       .output_single_input_dimension(0, 1)
                       .output_single_input_dimension(1, 0)
                       .Finalize()
                       .value();
  auto domain = IndexDomainBuilder(4)
                    .origin({1, 2, 3, 4})
                    .exclusive_max({6, 7, 8, 9})
                    .labels({"y", "", "x", ""})
                    .Finalize()
                    .value();
  // transform: "x": (inf), "": (inf), "": (inf), "y": (inf)
  // domain:    "y": [1, 6), "": [2, 7), "x": [3, 8), "": [4, 9)
  // result:    "x": [3, 8), "": [2, 7), "": [4, 9) "y": [1, 6)
  auto expected_new_transform = IndexTransformBuilder<>(4, 2)
                                    .input_origin({3, 2, 4, 1})
                                    .input_exclusive_max({8, 7, 9, 6})
                                    .input_labels({"x", "", "", "y"})
                                    .output_single_input_dimension(0, 1)
                                    .output_single_input_dimension(1, 0)
                                    .Finalize()
                                    .value();
  auto new_transform = domain(std::move(transform));
  EXPECT_EQ(expected_new_transform, new_transform);
}

TEST(SliceByIndexDomainTest, PartiallyLabeledRankMismatch) {
  auto transform = IndexTransformBuilder<>(4, 2)
                       .input_labels({"x", "", "", "y"})
                       .output_single_input_dimension(0, 1)
                       .output_single_input_dimension(1, 0)
                       .Finalize()
                       .value();
  auto domain = IndexDomainBuilder(3)
                    .origin({1, 2, 3})
                    .exclusive_max({6, 7, 8})
                    .labels({"y", "", "x"})
                    .Finalize()
                    .value();
  EXPECT_THAT(domain(std::move(transform)),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Rank \\(3\\) of index domain containing unlabeled "
                            "dimensions must equal slice target rank \\(4\\)"));
}

TEST(SliceByIndexDomainTest, LabelMismatch) {
  auto transform = IndexTransformBuilder<>(2, 2)
                       .input_labels({"x", "y"})
                       .output_single_input_dimension(0, 1)
                       .output_single_input_dimension(1, 0)
                       .Finalize()
                       .value();
  auto domain = IndexDomainBuilder(2)
                    .origin({1, 2})
                    .exclusive_max({6, 7})
                    .labels({"y", "z"})
                    .Finalize()
                    .value();
  EXPECT_THAT(
      domain(std::move(transform)),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Label \"z\" does not match one of \\{\"x\", \"y\"\\}"));
}

TEST(SliceByIndexDomainTest, PartiallyLabeledUnlabeledDimensionMismatch) {
  auto transform = IndexTransformBuilder<>(4, 2)
                       .input_labels({"x", "z", "", "y"})
                       .output_single_input_dimension(0, 1)
                       .output_single_input_dimension(1, 0)
                       .Finalize()
                       .value();
  auto domain = IndexDomainBuilder(4)
                    .origin({1, 2, 3, 4})
                    .exclusive_max({6, 7, 8, 9})
                    .labels({"y", "", "x", ""})
                    .Finalize()
                    .value();
  EXPECT_THAT(
      domain(std::move(transform)),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Number of unlabeled dimensions in index domain exceeds "
                    "number of unlabeled dimensions in slice target"));
}

TEST(SliceByIndexDomainTest, FullyUnlabeledRankMismatch) {
  auto transform = IndexTransformBuilder<>(2, 2)
                       .input_origin({0, 1})
                       .input_exclusive_max({5, 7})
                       .output_single_input_dimension(0, 1)
                       .output_single_input_dimension(1, 0)
                       .Finalize()
                       .value();
  auto domain =
      IndexDomainBuilder(1).origin({2}).exclusive_max({4}).Finalize().value();
  EXPECT_THAT(
      domain(std::move(transform)),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Rank of index domain \\(1\\) must match rank of slice target "
          "\\(2\\) when the index domain or slice target is unlabeled"));
}

}  // namespace
