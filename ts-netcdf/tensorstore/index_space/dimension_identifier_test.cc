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

#include "tensorstore/index_space/dimension_identifier.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {
using ::tensorstore::DimensionIdentifier;
using ::tensorstore::DimensionIndexBuffer;
using ::tensorstore::DimRangeSpec;
using ::tensorstore::DynamicDimSpec;
using ::tensorstore::Index;
using ::tensorstore::MatchesStatus;
using ::tensorstore::NormalizeDimensionIdentifier;
using ::tensorstore::NormalizeDimensionIndex;
using ::tensorstore::span;
using ::tensorstore::StrCat;

TEST(DimensionIdentifierTest, ConstructDefault) {
  DimensionIdentifier d;
  EXPECT_EQ(std::numeric_limits<Index>::max(), d.index());
  EXPECT_EQ(nullptr, d.label().data());
}

TEST(DimensionIdentifierTest, ConstructDimensionIndex) {
  DimensionIdentifier d(5);
  EXPECT_EQ(5, d.index());
  EXPECT_EQ(nullptr, d.label().data());
}

TEST(DimensionIdentifierTest, ConstructStringView) {
  DimensionIdentifier d(std::string_view("hello"));
  EXPECT_EQ(std::numeric_limits<Index>::max(), d.index());
  EXPECT_EQ("hello", d.label());
}

TEST(DimensionIdentifierTest, ConstructCString) {
  DimensionIdentifier d("hello");
  EXPECT_EQ(std::numeric_limits<Index>::max(), d.index());
  EXPECT_EQ("hello", d.label());
}

TEST(DimensionIdentifierTest, ConstructStdString) {
  std::string s = "hello";
  DimensionIdentifier d(s);
  EXPECT_EQ(std::numeric_limits<Index>::max(), d.index());
  EXPECT_EQ("hello", d.label());
}

TEST(DimensionIdentifierTest, Compare) {
  EXPECT_EQ(DimensionIdentifier(3), DimensionIdentifier(3));
  EXPECT_EQ(DimensionIdentifier("a"), DimensionIdentifier("a"));
  EXPECT_NE(DimensionIdentifier("a"), DimensionIdentifier(2));
  EXPECT_NE(DimensionIdentifier("a"), DimensionIdentifier("b"));
  EXPECT_NE(DimensionIdentifier(2), DimensionIdentifier(3));
}

TEST(DimensionIdentifierTest, PrintToOstream) {
  EXPECT_EQ("3", StrCat(DimensionIdentifier(3)));
  EXPECT_EQ("\"a\"", StrCat(DimensionIdentifier("a")));
}

TEST(NormalizeDimensionIndexTest, ValidNonNegative) {
  EXPECT_EQ(0, NormalizeDimensionIndex(0, 5));
  EXPECT_EQ(3, NormalizeDimensionIndex(3, 5));
  EXPECT_EQ(4, NormalizeDimensionIndex(4, 5));
}

TEST(NormalizeDimensionIndexTest, ValidNegative) {
  EXPECT_EQ(0, NormalizeDimensionIndex(-5, 5));
  EXPECT_EQ(2, NormalizeDimensionIndex(-3, 5));
  EXPECT_EQ(4, NormalizeDimensionIndex(-1, 5));
}

TEST(NormalizeDimensionIndexTest, InvalidNegative) {
  EXPECT_THAT(NormalizeDimensionIndex(-6, 5),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(NormalizeDimensionIndex(-7, 5),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(NormalizeDimensionIndexTest, InvalidNonNegative) {
  EXPECT_THAT(NormalizeDimensionIndex(5, 5),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(NormalizeDimensionIndex(6, 5),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(NormalizeDimensionLabelTest, ValidLabel) {
  EXPECT_EQ(2, NormalizeDimensionLabel(
                   "x", span<const std::string>({"a", "b", "x", "y"})));
}

TEST(NormalizeDimensionLabelTest, MissingLabel) {
  EXPECT_THAT(NormalizeDimensionLabel(
                  "w", span<const std::string>({"a", "b", "x", "y"})),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(NormalizeDimensionLabelTest, EmptyLabel) {
  EXPECT_THAT(NormalizeDimensionLabel(
                  "", span<const std::string>({"a", "b", "x", "y"})),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(NormalizeDimensionIdentifierTest, ValidLabel) {
  EXPECT_EQ(2, NormalizeDimensionIdentifier(
                   "x", span<const std::string>({"a", "b", "x", "y"})));
}

TEST(NormalizeDimensionIdentifierTest, ValidPositiveIndex) {
  EXPECT_EQ(2, NormalizeDimensionIdentifier(
                   2, span<const std::string>({"a", "b", "x", "y"})));
  EXPECT_EQ(0, NormalizeDimensionIdentifier(
                   0, span<const std::string>({"a", "b", "x", "y"})));
  EXPECT_EQ(3, NormalizeDimensionIdentifier(
                   3, span<const std::string>({"a", "b", "x", "y"})));
}

TEST(NormalizeDimensionIdentifierTest, ValidNegativeIndex) {
  EXPECT_EQ(2, NormalizeDimensionIdentifier(
                   -2, span<const std::string>({"a", "b", "x", "y"})));
  EXPECT_EQ(3, NormalizeDimensionIdentifier(
                   -1, span<const std::string>({"a", "b", "x", "y"})));
  EXPECT_EQ(0, NormalizeDimensionIdentifier(
                   -4, span<const std::string>({"a", "b", "x", "y"})));
}

TEST(NormalizeDimensionIdentifierTest, InvalidIndex) {
  EXPECT_THAT(NormalizeDimensionIdentifier(
                  4, span<const std::string>({"a", "b", "x", "y"})),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(NormalizeDimensionIdentifier(
                  -5, span<const std::string>({"a", "b", "x", "y"})),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(DimRangeSpecTest, Comparison) {
  DimRangeSpec a{1, 5, 1};
  DimRangeSpec b{0, 5, 1};
  DimRangeSpec c{1, 6, 1};
  DimRangeSpec d{1, 6, 2};
  EXPECT_EQ(a, a);
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);
}

TEST(DimRangeSpecTest, PrintToOstream) {
  EXPECT_EQ("1:5", StrCat(DimRangeSpec{1, 5, 1}));
  EXPECT_EQ("1:5:2", StrCat(DimRangeSpec{1, 5, 2}));
  EXPECT_EQ(":5", StrCat(DimRangeSpec{std::nullopt, 5, 1}));
  EXPECT_EQ("1:", StrCat(DimRangeSpec{1, std::nullopt, 1}));
  EXPECT_EQ(":", StrCat(DimRangeSpec{std::nullopt, std::nullopt, 1}));
  EXPECT_EQ("::-1", StrCat(DimRangeSpec{std::nullopt, std::nullopt, -1}));
}

TEST(NormalizeDimRangeSpecTest, ValidFullySpecifiedStep1) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(absl::OkStatus(),
            NormalizeDimRangeSpec(DimRangeSpec{2, 10, 1}, 20, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(2, 3, 4, 5, 6, 7, 8, 9));
}

TEST(NormalizeDimRangeSpecTest, ValidFullySpecifiedStep2) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(absl::OkStatus(),
            NormalizeDimRangeSpec(DimRangeSpec{2, 10, 2}, 20, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(2, 4, 6, 8));
}

TEST(NormalizeDimRangeSpecTest, ValidFullySpecifiedStep2Floor) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(absl::OkStatus(),
            NormalizeDimRangeSpec(DimRangeSpec{2, 7, 3}, 20, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(2, 5));
}

TEST(NormalizeDimRangeSpecTest, ValidFullySpecifiedStepNeg1) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(absl::OkStatus(),
            NormalizeDimRangeSpec(DimRangeSpec{9, 1, -1}, 20, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(9, 8, 7, 6, 5, 4, 3, 2));
}

TEST(NormalizeDimRangeSpecTest, ValidFullySpecifiedStepNeg2) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(absl::OkStatus(),
            NormalizeDimRangeSpec(DimRangeSpec{9, 1, -2}, 20, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(9, 7, 5, 3));
}

TEST(NormalizeDimRangeSpecTest, ValidStartOnlyStep1) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(
      absl::OkStatus(),
      NormalizeDimRangeSpec(DimRangeSpec{15, std::nullopt, 1}, 20, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(15, 16, 17, 18, 19));
}

TEST(NormalizeDimRangeSpecTest, ValidStartOnlyStepNegative1) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(
      absl::OkStatus(),
      NormalizeDimRangeSpec(DimRangeSpec{5, std::nullopt, -1}, 20, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(5, 4, 3, 2, 1, 0));
}

TEST(NormalizeDimRangeSpecTest, ValidNegativeStartOnlyStep1) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(
      absl::OkStatus(),
      NormalizeDimRangeSpec(DimRangeSpec{-5, std::nullopt, 1}, 20, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(15, 16, 17, 18, 19));
}

TEST(NormalizeDimRangeSpecTest, ValidStopOnlyStep1) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(
      absl::OkStatus(),
      NormalizeDimRangeSpec(DimRangeSpec{std::nullopt, 5, 1}, 20, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(0, 1, 2, 3, 4));
}

TEST(NormalizeDimRangeSpecTest, ValidNegativeStopOnlyStep1) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(
      absl::OkStatus(),
      NormalizeDimRangeSpec(DimRangeSpec{std::nullopt, -15, 1}, 20, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(0, 1, 2, 3, 4));
}

TEST(NormalizeDimRangeSpecTest, ValidStopOnlyStepNeg1) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(
      absl::OkStatus(),
      NormalizeDimRangeSpec(DimRangeSpec{std::nullopt, 15, -1}, 20, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(19, 18, 17, 16));
}

TEST(NormalizeDimRangeSpecTest, ValidNoBoundsStep1) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(absl::OkStatus(),
            NormalizeDimRangeSpec(DimRangeSpec{std::nullopt, std::nullopt, 1},
                                  5, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(0, 1, 2, 3, 4));
}

TEST(NormalizeDimRangeSpecTest, ValidNoBoundsStep2) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(absl::OkStatus(),
            NormalizeDimRangeSpec(DimRangeSpec{std::nullopt, std::nullopt, 2},
                                  5, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(0, 2, 4));
}

TEST(NormalizeDimRangeSpecTest, ValidMaxStop) {
  DimensionIndexBuffer buffer;
  EXPECT_EQ(absl::OkStatus(),
            NormalizeDimRangeSpec(DimRangeSpec{1, 5, 1}, 5, &buffer));
  EXPECT_THAT(buffer, ::testing::ElementsAre(1, 2, 3, 4));
}

TEST(NormalizeDimRangeSpecTest, InvalidStep0) {
  DimensionIndexBuffer buffer;
  EXPECT_THAT(
      NormalizeDimRangeSpec(DimRangeSpec{std::nullopt, std::nullopt, 0}, 5,
                            &buffer),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "step must not be 0"));
}

TEST(NormalizeDimRangeSpecTest, InvalidIntervalStep1) {
  DimensionIndexBuffer buffer;
  EXPECT_THAT(NormalizeDimRangeSpec(DimRangeSpec{3, 1, 1}, 5, &buffer),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "3:1 is not a valid range"));
}

TEST(NormalizeDimRangeSpecTest, InvalidIntervalStepNeg1) {
  DimensionIndexBuffer buffer;
  EXPECT_THAT(NormalizeDimRangeSpec(DimRangeSpec{1, 3, -1}, 5, &buffer),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "1:3:-1 is not a valid range"));
}

TEST(NormalizeDimRangeSpecTest, InvalidIndex) {
  DimensionIndexBuffer buffer;
  EXPECT_THAT(NormalizeDimRangeSpec(DimRangeSpec{1, 8, 1}, 5, &buffer),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Dimension exclusive stop index 8 is outside valid "
                            "range \\[-6, 5\\]"));
}

}  // namespace
