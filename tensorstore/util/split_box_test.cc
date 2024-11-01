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

#include "tensorstore/util/split_box.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::tensorstore::Box;
using ::tensorstore::SplitBoxByGrid;

TEST(SplitBoxByGridTest, NoSplitRank1) {
  Box<> a(1), b(1);
  EXPECT_FALSE(SplitBoxByGrid(/*input=*/Box<1>({3}),
                              /*grid_cell_template=*/Box<1>({3}), {{a, b}}));
  EXPECT_FALSE(SplitBoxByGrid(/*input=*/Box<1>({3}),
                              /*grid_cell_template=*/Box<1>({4}), {{a, b}}));
  EXPECT_FALSE(SplitBoxByGrid(/*input=*/Box<1>({2}, {3}),
                              /*grid_cell_template=*/Box<1>({1}, {4}),
                              {{a, b}}));
  EXPECT_FALSE(SplitBoxByGrid(/*input=*/Box<1>({-2}, {3}),
                              /*grid_cell_template=*/Box<1>({-3}, {4}),
                              {{a, b}}));
  EXPECT_FALSE(SplitBoxByGrid(/*input=*/Box<1>({-2}, {3}),
                              /*grid_cell_template=*/Box<1>({1}, {4}),
                              {{a, b}}));
  EXPECT_FALSE(SplitBoxByGrid(/*input=*/Box<1>({2}, {3}),
                              /*grid_cell_template=*/Box<1>({0}, {5}),
                              {{a, b}}));
}

TEST(SplitBoxByGridTest, SplitRank1) {
  Box<> a(1), b(1);
  EXPECT_TRUE(SplitBoxByGrid(/*input=*/Box<1>({3}),
                             /*grid_cell_template=*/Box<1>({2}), {{a, b}}));
  EXPECT_EQ(Box<1>({0}, {2}), a);
  EXPECT_EQ(Box<1>({2}, {1}), b);
}

TEST(SplitBoxByGridTest, SplitRank2) {
  Box<> a(2), b(2);
  EXPECT_TRUE(SplitBoxByGrid(/*input=*/Box<2>({3, 10}),
                             /*grid_cell_template=*/Box<2>({2, 3}), {{a, b}}));
  EXPECT_EQ(Box<2>({0, 0}, {3, 6}), a);
  EXPECT_EQ(Box<2>({0, 6}, {3, 4}), b);
}

}  // namespace
