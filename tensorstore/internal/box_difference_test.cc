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

#include "tensorstore/internal/box_difference.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/box.h"
#include "tensorstore/index.h"

namespace {
using tensorstore::Box;
using tensorstore::BoxView;
using tensorstore::Index;
using tensorstore::internal::BoxDifference;

std::vector<Box<>> Subtract(BoxView<> outer, BoxView<> inner) {
  BoxDifference difference(outer, inner);
  Index count = difference.num_sub_boxes();
  std::vector<Box<>> boxes(count);
  for (Index i = 0; i < count; ++i) {
    auto& out = boxes[i];
    out.set_rank(outer.rank());
    difference.GetSubBox(i, out);
  }
  return boxes;
}

TEST(BoxDifferenceTest, RankZero) {
  EXPECT_THAT(Subtract(BoxView<>(), BoxView<>()),
              ::testing::UnorderedElementsAre());
}

TEST(BoxDifferenceTest, RankOneEmptyResult) {
  EXPECT_THAT(Subtract(BoxView({1}, {5}), BoxView({1}, {5})),
              ::testing::UnorderedElementsAre());
}

TEST(BoxDifferenceTest, RankOneFullResult) {
  EXPECT_THAT(Subtract(BoxView({1}, {5}), BoxView({6}, {5})),
              ::testing::UnorderedElementsAre(BoxView({1}, {5})));
}

TEST(BoxDifferenceTest, RankOneBeforeOnly) {
  EXPECT_THAT(Subtract(BoxView({1}, {5}), BoxView({3}, {4})),
              ::testing::UnorderedElementsAre(BoxView({1}, {2})));
}

TEST(BoxDifferenceTest, RankOneAfterOnly) {
  EXPECT_THAT(Subtract(BoxView({1}, {5}), BoxView({0}, {3})),
              ::testing::UnorderedElementsAre(BoxView({3}, {3})));
}

TEST(BoxDifferenceTest, RankOneBeforeAndAfter) {
  EXPECT_THAT(
      Subtract(BoxView({1}, {5}), BoxView({2}, {2})),
      ::testing::UnorderedElementsAre(BoxView({1}, {1}), BoxView({4}, {2})));
}

TEST(BoxDifferenceTest, RankTwoDim0EmptyDim1Empty) {
  EXPECT_THAT(Subtract(BoxView({1, 2}, {5, 7}), BoxView({1, 2}, {5, 7})),
              ::testing::UnorderedElementsAre());
}

TEST(BoxDifferenceTest, RankTwoDim0FullDim1Empty) {
  EXPECT_THAT(Subtract(BoxView({1, 2}, {5, 7}), BoxView({6, 2}, {5, 7})),
              ::testing::UnorderedElementsAre(BoxView({1, 2}, {5, 7})));
}

TEST(BoxDifferenceTest, RankTwoDim0EmptyDim1Full) {
  EXPECT_THAT(Subtract(BoxView({1, 2}, {5, 7}), BoxView({1, 10}, {5, 7})),
              ::testing::UnorderedElementsAre(BoxView({1, 2}, {5, 7})));
}

TEST(BoxDifferenceTest, RankTwoDim0BeforeDim1Empty) {
  EXPECT_THAT(Subtract(BoxView({1, 2}, {5, 7}), BoxView({4, 2}, {3, 7})),
              ::testing::UnorderedElementsAre(BoxView({1, 2}, {3, 7})));
}

TEST(BoxDifferenceTest, RankTwoDim0AfterDim1Empty) {
  EXPECT_THAT(Subtract(BoxView({1, 2}, {5, 7}), BoxView({-1, 2}, {3, 7})),
              ::testing::UnorderedElementsAre(BoxView({2, 2}, {4, 7})));
}

TEST(BoxDifferenceTest, RankTwoDim0BeforeAfterDim1Empty) {
  EXPECT_THAT(Subtract(BoxView({1, 2}, {5, 7}), BoxView({2, 2}, {3, 7})),
              ::testing::UnorderedElementsAre(BoxView({1, 2}, {1, 7}),
                                              BoxView({5, 2}, {1, 7})));
}

TEST(BoxDifferenceTest, RankTwoDim0EmptyDim1Before) {
  EXPECT_THAT(Subtract(BoxView({2, 1}, {7, 5}), BoxView({2, 4}, {7, 3})),
              ::testing::UnorderedElementsAre(BoxView({2, 1}, {7, 3})));
}

TEST(BoxDifferenceTest, RankTwoDim0EmptyDim1After) {
  EXPECT_THAT(Subtract(BoxView({2, 1}, {7, 5}), BoxView({2, -1}, {7, 3})),
              ::testing::UnorderedElementsAre(BoxView({2, 2}, {7, 4})));
}

TEST(BoxDifferenceTest, RankTwoDim0EmptyDim1BeforeAfter) {
  EXPECT_THAT(Subtract(BoxView({2, 1}, {7, 5}), BoxView({2, 2}, {7, 3})),
              ::testing::UnorderedElementsAre(BoxView({2, 1}, {7, 1}),
                                              BoxView({2, 5}, {7, 1})));
}

TEST(BoxDifferenceTest, RankTwoDim0BeforeDim1Before) {
  EXPECT_THAT(Subtract(BoxView({1, 2}, {5, 7}), BoxView({4, 4}, {3, 7})),
              ::testing::UnorderedElementsAre(BoxView({1, 4}, {3, 5}),
                                              BoxView({4, 2}, {2, 2}),
                                              BoxView({1, 2}, {3, 2})));
}

TEST(BoxDifferenceTest, RankTwoDim0AfterDim1Before) {
  EXPECT_THAT(Subtract(BoxView({1, 2}, {5, 7}), BoxView({-1, 4}, {3, 7})),
              ::testing::UnorderedElementsAre(BoxView({2, 4}, {4, 5}),
                                              BoxView({1, 2}, {1, 2}),
                                              BoxView({2, 2}, {4, 2})));
}

TEST(BoxDifferenceTest, RankTwoDim0BeforeAfterDim1Before) {
  EXPECT_THAT(Subtract(BoxView({1, 2}, {5, 7}), BoxView({2, 4}, {3, 7})),
              ::testing::UnorderedElementsAre(
                  BoxView({1, 4}, {1, 5}), BoxView({5, 4}, {1, 5}),
                  BoxView({2, 2}, {3, 2}), BoxView({1, 2}, {1, 2}),
                  BoxView({5, 2}, {1, 2})));
}

TEST(BoxDifferenceTest, RankTwoDim0BeforeDim1After) {
  EXPECT_THAT(Subtract(BoxView({1, 2}, {5, 7}), BoxView({4, 2}, {3, 1})),
              ::testing::UnorderedElementsAre(BoxView({1, 2}, {3, 1}),
                                              BoxView({4, 3}, {2, 6}),
                                              BoxView({1, 3}, {3, 6})));
}

TEST(BoxDifferenceTest, RankTwoDim0AfterDim1After) {
  EXPECT_THAT(Subtract(BoxView({1, 2}, {5, 7}), BoxView({-1, 2}, {3, 1})),
              ::testing::UnorderedElementsAre(BoxView({2, 2}, {4, 1}),
                                              BoxView({1, 3}, {1, 6}),
                                              BoxView({2, 3}, {4, 6})));
}

TEST(BoxDifferenceTest, RankTwoDim0BeforeAfterDim1After) {
  EXPECT_THAT(Subtract(BoxView({1, 2}, {5, 7}), BoxView({2, 2}, {3, 1})),
              ::testing::UnorderedElementsAre(
                  BoxView({1, 2}, {1, 1}), BoxView({5, 2}, {1, 1}),
                  BoxView({2, 3}, {3, 6}), BoxView({1, 3}, {1, 6}),
                  BoxView({5, 3}, {1, 6})));
}

TEST(BoxDifferenceTest, RankTwoDim0BeforeAfterDim1BeforeAfter) {
  EXPECT_THAT(Subtract(BoxView({1, 2}, {5, 7}), BoxView({2, 3}, {3, 1})),
              ::testing::UnorderedElementsAre(
                  BoxView({1, 3}, {1, 1}), BoxView({5, 3}, {1, 1}),
                  BoxView({2, 2}, {3, 1}), BoxView({1, 2}, {1, 1}),
                  BoxView({5, 2}, {1, 1}), BoxView({2, 4}, {3, 5}),
                  BoxView({1, 4}, {1, 5}), BoxView({5, 4}, {1, 5})));
}

}  // namespace
