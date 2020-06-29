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

#include "tensorstore/index_space/alignment.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::DimensionIndex;
using tensorstore::Index;
using tensorstore::IndexDomain;
using tensorstore::IndexDomainBuilder;
using tensorstore::IndexTransform;
using tensorstore::IndexTransformBuilder;
using tensorstore::MatchesStatus;
using tensorstore::Status;

using Dao = tensorstore::DomainAlignmentOptions;

TEST(AlignDimensionsToTest, AllUnlabeled) {
  auto source = IndexDomainBuilder(3)
                    .origin({3, 5, 4})
                    .exclusive_max({7, 6, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(3)
                    .origin({2, 0, 6})
                    .exclusive_max({6, 4, 12})
                    .Finalize()
                    .value();
  // source: [3, 7), [5, 6), [4, 10)
  // target: [2, 6), [0, 4), [6, 12)
  // alignment: rank 3 -> 3, with:
  //   output dimension 0 -> input dimension 0, offset 1
  //   output dimension 1 -> constant 5
  //   output dimension 2 -> input dimension 2, offset -2
  for (auto options : {Dao::all, Dao::translate | Dao::broadcast}) {
    std::vector<DimensionIndex> source_matches(source.rank());
    EXPECT_EQ(Status(),
              AlignDimensionsTo(source, target, source_matches, options));
    EXPECT_THAT(source_matches, ::testing::ElementsAre(0, -1, 2));
  }

  // Test aligning `source` to itself without any alignment transforms.
  {
    std::vector<DimensionIndex> source_matches(source.rank());
    EXPECT_EQ(Status(),
              AlignDimensionsTo(source, source, source_matches, Dao::none));
    EXPECT_THAT(source_matches, ::testing::ElementsAre(0, 1, 2));
  }

  {
    std::vector<DimensionIndex> source_matches(source.rank());
    EXPECT_THAT(
        AlignDimensionsTo(source, target, source_matches, Dao::translate),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      "Error aligning dimensions: "
                      "source dimension 1 \\[5, 6\\) mismatch with target "
                      "dimension 1 \\[0, 4\\)"));
  }

  {
    std::vector<DimensionIndex> source_matches(source.rank());
    EXPECT_THAT(
        AlignDimensionsTo(source, target, source_matches, Dao::broadcast),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      "Error aligning dimensions: "
                      "source dimension 0 \\[3, 7\\) mismatch with target "
                      "dimension 0 \\[2, 6\\), "
                      "source dimension 2 \\[4, 10\\) mismatch with target "
                      "dimension 2 \\[6, 12\\)"));
  }
}

// Tests that mismatched labels are allowed to match when permutation is
// disabled.
TEST(AlignDimensionsToTest, MismatchedLabelsNoPermute) {
  auto source = IndexDomainBuilder(3)
                    .labels({"x", "y", "z"})
                    .origin({3, 5, 4})
                    .exclusive_max({7, 6, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(3)
                    .labels({"a", "b", "c"})
                    .origin({2, 0, 6})
                    .exclusive_max({6, 4, 12})
                    .Finalize()
                    .value();
  std::vector<DimensionIndex> source_matches(source.rank());
  EXPECT_EQ(Status(), AlignDimensionsTo(source, target, source_matches,
                                        Dao::translate | Dao::broadcast));
  EXPECT_THAT(source_matches, ::testing::ElementsAre(0, -1, 2));

  EXPECT_THAT(AlignDimensionsTo(source, target, source_matches, Dao::all),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error aligning dimensions: "
                            "unmatched source dimension 0 \"x\": \\[3, 7\\) "
                            "does not have a size of 1, "
                            "unmatched source dimension 2 \"z\": \\[4, 10\\) "
                            "does not have a size of 1"));
}

TEST(AlignDimensionsToTest, SourceUnlabeled) {
  auto source = IndexDomainBuilder(3)
                    .origin({3, 5, 4})
                    .exclusive_max({7, 6, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(3)
                    .origin({4, 0, 6})
                    .labels({"x", "y", "z"})
                    .exclusive_max({8, 4, 12})
                    .Finalize()
                    .value();
  // source: [3, 7), [5, 6), [4, 10)
  // target: "x": [2, 6), "y": [0, 4), "z": [6, 12)
  // alignment: rank 3 -> 3, with:
  //   output dimension 0 -> input dimension 0, offset 1
  //   output dimension 1 -> constant 5
  //   output dimension 2 -> input dimension 2, offset -2
  for (auto options : {Dao::all, Dao::translate | Dao::broadcast}) {
    std::vector<DimensionIndex> source_matches(source.rank());
    EXPECT_EQ(Status(),
              AlignDimensionsTo(source, target, source_matches, options));
    EXPECT_THAT(source_matches, ::testing::ElementsAre(0, -1, 2));
  }
}

TEST(AlignDimensionsToTest, TargetUnlabeled) {
  auto source = IndexDomainBuilder(3)
                    .origin({3, 5, 4})
                    .labels({"x", "y", "z"})
                    .exclusive_max({7, 6, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(3)
                    .origin({4, 0, 6})
                    .exclusive_max({8, 4, 12})
                    .Finalize()
                    .value();
  // source: "x": [3, 7), "y": [5, 6), "z": [4, 10)
  // target: [2, 6), [0, 4), [6, 12)
  // alignment: rank 3 -> 3, with:
  //   output dimension 0 -> input dimension 0, offset 1
  //   output dimension 1 -> constant 5
  //   output dimension 2 -> input dimension 2, offset -2
  for (auto options : {Dao::all, Dao::translate | Dao::broadcast}) {
    std::vector<DimensionIndex> source_matches(source.rank());
    EXPECT_EQ(Status(),
              AlignDimensionsTo(source, target, source_matches, options));
    EXPECT_THAT(source_matches, ::testing::ElementsAre(0, -1, 2));
  }
}

TEST(AlignDimensionsToTest, AllLabeled) {
  auto source = IndexDomainBuilder(3)
                    .labels({"x", "y", "z"})
                    .origin({3, 5, 4})
                    .exclusive_max({7, 6, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(3)
                    .labels({"z", "x", "y"})
                    .origin({6, 4, 0})
                    .exclusive_max({12, 8, 4})
                    .Finalize()
                    .value();
  // source: "x": [3, 7), "y": [5, 6), "z": [4, 10)
  // target: "z": [6, 12), "x": [4, 8), "y": [0, 4)
  // alignment: rank 3 -> 3, with:
  //   output dimension 0 -> input dimension 1, offset -1
  //   output dimension 1 -> constant 5
  //   output dimension 2 -> input dimension 0, offset -2
  std::vector<DimensionIndex> source_matches(source.rank());
  EXPECT_EQ(Status(), AlignDimensionsTo(source, target, source_matches));
  EXPECT_THAT(source_matches, ::testing::ElementsAre(1, -1, 0));
}

TEST(AlignDimensionsToTest, AllLabeledPermuteOnly) {
  auto source = IndexDomainBuilder(3)
                    .labels({"x", "y", "z"})
                    .origin({3, 5, 4})
                    .exclusive_max({7, 6, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(3)
                    .labels({"z", "x", "y"})
                    .origin({4, 3, 5})
                    .exclusive_max({10, 7, 6})
                    .Finalize()
                    .value();
  for (auto options : {Dao::permute, Dao::permute | Dao::translate,
                       Dao::permute | Dao::broadcast, Dao::all}) {
    std::vector<DimensionIndex> source_matches(source.rank());
    EXPECT_EQ(Status(),
              AlignDimensionsTo(source, target, source_matches, options));
    EXPECT_THAT(source_matches, ::testing::ElementsAre(1, 2, 0));
  }

  for (auto options : {Dao::none, Dao::translate, Dao::broadcast,
                       Dao::translate | Dao::broadcast}) {
    std::vector<DimensionIndex> source_matches(source.rank());
    EXPECT_THAT(AlignDimensionsTo(source, target, source_matches, options),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              "Error aligning dimensions: .*"));
  }
}

TEST(AlignDimensionsToTest, AllLabeledPermuteTranslateOnly) {
  auto source = IndexDomainBuilder(3)
                    .labels({"x", "y", "z"})
                    .origin({3, 5, 4})
                    .exclusive_max({7, 9, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(3)
                    .labels({"z", "x", "y"})
                    .origin({6, 4, 0})
                    .exclusive_max({12, 8, 4})
                    .Finalize()
                    .value();
  // source: "x": [3, 7), "y": [5, 9), "z": [4, 10)
  // target: "z": [6, 12), "x": [4, 8), "y": [0, 4)
  // alignment: rank 3 -> 3, with:
  //   output dimension 0 -> input dimension 1, offset -1
  //   output dimension 1 -> input dimension 2, offset 1
  //   output dimension 2 -> input dimension 0, offset -2
  std::vector<DimensionIndex> source_matches(source.rank());
  EXPECT_EQ(Status(), AlignDimensionsTo(source, target, source_matches));
  EXPECT_THAT(source_matches, ::testing::ElementsAre(1, 2, 0));
}

TEST(AlignDimensionsToTest, PartiallyLabeled) {
  auto source = IndexDomainBuilder(3)
                    .labels({"x", "y", ""})
                    .origin({3, 5, 4})
                    .exclusive_max({7, 6, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(4)
                    .labels({"", "", "x", "y"})
                    .origin({0, 6, 4, 0})
                    .exclusive_max({10, 12, 8, 4})
                    .Finalize()
                    .value();
  // source: "x": [3, 7), "y": [5, 6), "": [4, 10)
  // target: "": [0, 10) "": [6, 12), "x": [4, 8), "y": [0, 4)
  // alignment: rank 4 -> 3, with:
  //   output dimension 0 -> input dimension 2, offset -1
  //   output dimension 1 -> constant 5
  //   output dimension 2 -> input dimension 1, offset -2
  std::vector<DimensionIndex> source_matches(source.rank());
  EXPECT_EQ(Status(), AlignDimensionsTo(source, target, source_matches));
  EXPECT_THAT(source_matches, ::testing::ElementsAre(2, -1, 1));

  EXPECT_THAT(AlignDimensionsTo(source, target, source_matches, Dao::none),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Aligning source domain of rank 3 to target "
                            "domain of rank 4 requires broadcasting"));
}

TEST(AlignDomainToTest, PartiallyLabeled) {
  auto source = IndexDomainBuilder(3)
                    .labels({"x", "y", ""})
                    .origin({3, 5, 4})
                    .exclusive_max({7, 6, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(4)
                    .labels({"", "", "x", "y"})
                    .origin({0, 6, 4, 0})
                    .exclusive_max({10, 12, 8, 4})
                    .Finalize()
                    .value();
  // source: "x": [3, 7), "y": [5, 6), "": [4, 10)
  // target: "": [0, 10) "": [6, 12), "x": [4, 8), "y": [0, 4)
  // alignment: rank 4 -> 3, with:
  //   output dimension 0 -> input dimension 2, offset -1
  //   output dimension 1 -> constant 5
  //   output dimension 2 -> input dimension 1, offset -2
  IndexTransform<> alignment = IndexTransformBuilder<>(4, 3)
                                   .input_domain(target)
                                   .output_single_input_dimension(0, -1, 1, 2)
                                   .output_constant(1, 5)
                                   .output_single_input_dimension(2, -2, 1, 1)
                                   .Finalize()
                                   .value();
  EXPECT_EQ(alignment, AlignDomainTo(source, target));
}

TEST(AlignDimensionsToTest, BroadcastOnly) {
  auto source = IndexDomainBuilder(2)
                    .origin({2, 3})
                    .exclusive_max({5, 6})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(3)
                    .origin({1, 2, 3})
                    .exclusive_max({4, 5, 6})
                    .Finalize()
                    .value();
  for (auto options : {Dao::broadcast, Dao::broadcast | Dao::translate,
                       Dao::broadcast | Dao::permute, Dao::all}) {
    std::vector<DimensionIndex> source_matches(source.rank());
    EXPECT_EQ(Status(),
              AlignDimensionsTo(source, target, source_matches, options));
    EXPECT_THAT(source_matches, ::testing::ElementsAre(1, 2));
  }

  for (auto options : {Dao::none, Dao::permute, Dao::translate,
                       Dao::permute | Dao::translate}) {
    std::vector<DimensionIndex> source_matches(source.rank());
    EXPECT_THAT(AlignDimensionsTo(source, target, source_matches, options),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              "Aligning source domain of rank 2 to target "
                              "domain of rank 3 requires broadcasting"));
  }
}

TEST(AlignDimensionsToTest, PermuteAndBroadcast) {
  auto source = IndexDomainBuilder(2)
                    .origin({2, 3})
                    .exclusive_max({5, 4})
                    .labels({"x", "y"})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(2)
                    .origin({2, 5})
                    .exclusive_max({5, 10})
                    .labels({"x", "z"})
                    .Finalize()
                    .value();
  for (auto options : {Dao::permute | Dao::broadcast, Dao::all}) {
    std::vector<DimensionIndex> source_matches(source.rank());
    EXPECT_EQ(Status(),
              AlignDimensionsTo(source, target, source_matches, options));
    EXPECT_THAT(source_matches, ::testing::ElementsAre(0, -1));
  }

  for (auto options : {Dao::permute, Dao::permute | Dao::translate}) {
    std::vector<DimensionIndex> source_matches(source.rank());
    EXPECT_THAT(
        AlignDimensionsTo(source, target, source_matches, options),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      "Error aligning dimensions: "
                      "unmatched source dimension 1 \"y\": \\[3, 4\\)"));
  }
}

TEST(AlignDimensionsToTest, UnmatchedUnlabeledSourceDimension) {
  auto source = IndexDomainBuilder(4)
                    .labels({"x", "y", "", ""})
                    .origin({3, 5, 7, 4})
                    .exclusive_max({7, 9, 8, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(3)
                    .labels({"", "x", "y"})
                    .origin({0, 4, 0})
                    .exclusive_max({6, 8, 4})
                    .Finalize()
                    .value();
  std::vector<DimensionIndex> source_matches(source.rank());
  EXPECT_EQ(Status(), AlignDimensionsTo(source, target, source_matches));
  EXPECT_THAT(source_matches, ::testing::ElementsAre(1, 2, -1, 0));
}

TEST(AlignDimensionsToTest, MismatchedLabeled) {
  auto source = IndexDomainBuilder(3)
                    .labels({"x", "y", "z"})
                    .origin({3, 5, 4})
                    .exclusive_max({7, 6, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(3)
                    .labels({"z", "w", "y"})
                    .origin({6, 4, 0})
                    .exclusive_max({12, 8, 4})
                    .Finalize()
                    .value();
  std::vector<DimensionIndex> source_matches(source.rank());
  EXPECT_THAT(AlignDimensionsTo(source, target, source_matches),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error aligning dimensions: "
                            "unmatched source dimension 0 \"x\": \\[3, 7\\) "
                            "does not have a size of 1"));
}

TEST(AlignDomainToTest, MismatchedLabeled) {
  auto source = IndexDomainBuilder(3)
                    .labels({"x", "y", "z"})
                    .origin({3, 5, 4})
                    .exclusive_max({7, 6, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(3)
                    .labels({"z", "w", "y"})
                    .origin({6, 4, 0})
                    .exclusive_max({12, 8, 4})
                    .Finalize()
                    .value();
  EXPECT_THAT(AlignDomainTo(source, target),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(AlignDimensionsToTest, MismatchedSizeLabeled) {
  auto source = IndexDomainBuilder(3)
                    .labels({"x", "y", "z"})
                    .origin({3, 5, 4})
                    .exclusive_max({7, 7, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(3)
                    .labels({"z", "x", "y"})
                    .origin({6, 4, 0})
                    .exclusive_max({12, 8, 4})
                    .Finalize()
                    .value();
  std::vector<DimensionIndex> source_matches(source.rank());
  EXPECT_THAT(AlignDimensionsTo(source, target, source_matches),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error aligning dimensions: "
                            "source dimension 1 \"y\": \\[5, 7\\) mismatch "
                            "with target dimension 2 \"y\": \\[0, 4\\)"));
}

TEST(AlignDimensionsToTest, MismatchedSizeUnlabeled) {
  auto source = IndexDomainBuilder(3)
                    .origin({3, 5, 4})
                    .exclusive_max({7, 7, 10})
                    .Finalize()
                    .value();
  auto target = IndexDomainBuilder(3)
                    .origin({4, 0, 6})
                    .exclusive_max({8, 4, 12})
                    .Finalize()
                    .value();
  std::vector<DimensionIndex> source_matches(source.rank());
  EXPECT_THAT(AlignDimensionsTo(source, target, source_matches),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error aligning dimensions: "
                            "source dimension 1 \\[5, 7\\) mismatch with "
                            "target dimension 1 \\[0, 4\\)"));
}

}  // namespace
