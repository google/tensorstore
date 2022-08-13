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

#include "tensorstore/driver/downsample/grid_occupancy_map.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/box.h"
#include "tensorstore/index.h"

namespace {

using ::tensorstore::Box;
using ::tensorstore::BoxView;
using ::tensorstore::Index;
using ::tensorstore::MakeArray;
using ::tensorstore::internal_downsample::GridOccupancyMap;
using ::tensorstore::internal_downsample::GridOccupancyTracker;

std::vector<Box<>> GetUnoccupiedBoxes(const GridOccupancyMap& map) {
  std::vector<Box<>> boxes;
  std::vector<Index> grid_cell(map.rank());
  map.InitializeCellIterator(grid_cell);
  Box<> box(map.rank());
  do {
    if (map.GetGridCellDomain(grid_cell, box)) {
      boxes.push_back(box);
    }
  } while (map.AdvanceCellIterator(grid_cell));
  return boxes;
}

TEST(GridOccupancyMapTest, Rank1) {
  GridOccupancyTracker tracker;
  // -1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
  //          XXXXXXXXX       XXXXXXXXXXXXX
  tracker.MarkOccupied(BoxView<1>({1}, {3}));
  tracker.MarkOccupied(BoxView<1>({5}, {4}));
  GridOccupancyMap map(std::move(tracker), BoxView<1>({-1}, {11}));
  EXPECT_THAT(
      map.partition_points,
      ::testing::ElementsAre(::testing::ElementsAre(-1, 1, 4, 5, 9, 10)));
  EXPECT_EQ(map.occupied_chunk_mask, MakeArray<bool>({0, 1, 0, 1, 0}));
  EXPECT_THAT(GetUnoccupiedBoxes(map),
              ::testing::ElementsAre(Box<>({-1}, {2}), Box<>({4}, {1}),
                                     Box<>({9}, {1})));
}

TEST(GridOccupancyMapTest, Rank2) {
  GridOccupancyTracker tracker;
  //     0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
  // -----------------------------------------
  // 0 | XXXXX               XXXXXXXXX
  //   | XXXXX               XXXXXXXXX
  // 1 | XXXXX               XXXXXXXXX
  //   | XXXXX
  // 2 | XXXXX
  //   |
  // 3 |             XXXXXXXXX
  tracker.MarkOccupied(BoxView<2>({0, 0}, {3, 2}));
  tracker.MarkOccupied(BoxView<2>({3, 3}, {1, 3}));
  tracker.MarkOccupied(BoxView<2>({0, 5}, {2, 3}));
  GridOccupancyMap map(std::move(tracker), BoxView<2>({4, 10}));
  EXPECT_THAT(
      map.partition_points,
      ::testing::ElementsAre(::testing::ElementsAre(0, 2, 3, 4),
                             ::testing::ElementsAre(0, 2, 3, 5, 6, 8, 10)));
  EXPECT_EQ(map.occupied_chunk_mask, MakeArray<bool>({
                                         {1, 0, 0, 1, 1, 0},
                                         {1, 0, 0, 0, 0, 0},
                                         {0, 0, 1, 1, 0, 0},
                                     }));
  EXPECT_THAT(
      GetUnoccupiedBoxes(map),
      ::testing::ElementsAre(
          Box<>({0, 2}, {2, 1}), Box<>({0, 3}, {2, 2}), Box<>({0, 8}, {2, 2}),
          Box<>({2, 2}, {1, 1}), Box<>({2, 3}, {1, 2}), Box<>({2, 5}, {1, 1}),
          Box<>({2, 6}, {1, 2}), Box<>({2, 8}, {1, 2}), Box<>({3, 0}, {1, 2}),
          Box<>({3, 2}, {1, 1}), Box<>({3, 6}, {1, 2}), Box<>({3, 8}, {1, 2})));
}

}  // namespace
