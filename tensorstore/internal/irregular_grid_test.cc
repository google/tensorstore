// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/irregular_grid.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/internal/grid_partition.h"
#include "tensorstore/internal/grid_partition_impl.h"
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::BoxView;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::IndexDomain;
using ::tensorstore::IndexInterval;
using ::tensorstore::kInfIndex;
using ::tensorstore::span;
using ::tensorstore::internal::IrregularGrid;
using ::testing::ElementsAre;

TEST(IrregularGridTest, Basic) {
  std::vector<Index> dimension0{2, 0, -3};
  std::vector<Index> dimension1{10, 45, 20, 30};
  auto grid = IrregularGrid({dimension0, dimension1});

  EXPECT_EQ(2, grid.rank());
  EXPECT_THAT(grid.shape(), ElementsAre(3, 4));
  EXPECT_THAT(grid.inclusive_min(0), ElementsAre(-3, 0, 2));
  EXPECT_THAT(grid.inclusive_min(1), ElementsAre(10, 20, 30, 45));

  IndexInterval grid_cell;
  EXPECT_EQ(grid(0, -4, &grid_cell), -1);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedClosed(-kInfIndex, -4));
  EXPECT_EQ(grid(0, -3, &grid_cell), 0);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedSized(-3, 3));
  EXPECT_EQ(grid(0, -2, &grid_cell), 0);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedSized(-3, 3));
  EXPECT_EQ(grid(0, -1, &grid_cell), 0);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedSized(-3, 3));
  EXPECT_EQ(grid(0, 0, &grid_cell), 1);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedSized(0, 2));
  EXPECT_EQ(grid(0, 1, &grid_cell), 1);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedSized(0, 2));
  EXPECT_EQ(grid(0, 2, &grid_cell), 2);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedClosed(2, kInfIndex));
  EXPECT_EQ(grid(0, 3, &grid_cell), 2);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedClosed(2, kInfIndex));

  EXPECT_EQ(grid(1, 7, &grid_cell), -1);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedClosed(-kInfIndex, 9));
  EXPECT_EQ(grid(1, 11, &grid_cell), 0);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedSized(10, 10));
  EXPECT_EQ(grid(1, 57, &grid_cell), 3);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedClosed(45, kInfIndex));
}

TEST(IrregularGridTest, IndexDomain) {
  const Index origin1[] = {-3, 10};
  const Index shape1[] = {3, 10};
  const Index origin2[] = {0, 20};
  const Index shape2[] = {2, 10};
  const Index origin3[] = {0, 30};
  const Index shape3[] = {2, 15};

  std::vector<IndexDomain<>> domains(
      {IndexDomain<>{BoxView<>{span(origin1), span(shape1)}},
       IndexDomain<>{BoxView<>{span(origin2), span(shape2)}},
       IndexDomain<>{BoxView<>{span(origin3), span(shape3)}}});

  auto grid = IrregularGrid::Make(domains);

  EXPECT_EQ(2, grid.rank());
  EXPECT_THAT(grid.shape(), ElementsAre(3, 4));
  EXPECT_THAT(grid.inclusive_min(0), ElementsAre(-3, 0, 2));
  EXPECT_THAT(grid.inclusive_min(1), ElementsAre(10, 20, 30, 45));

  IndexInterval grid_cell;
  EXPECT_EQ(grid(0, -4, &grid_cell), -1);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedClosed(-kInfIndex, -4));
  EXPECT_EQ(grid(0, -3, &grid_cell), 0);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedSized(-3, 3));
  EXPECT_EQ(grid(0, -2, &grid_cell), 0);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedSized(-3, 3));
  EXPECT_EQ(grid(0, -1, &grid_cell), 0);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedSized(-3, 3));
  EXPECT_EQ(grid(0, 0, &grid_cell), 1);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedSized(0, 2));
  EXPECT_EQ(grid(0, 1, &grid_cell), 1);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedSized(0, 2));
  EXPECT_EQ(grid(0, 2, &grid_cell), 2);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedClosed(2, kInfIndex));
  EXPECT_EQ(grid(0, 3, &grid_cell), 2);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedClosed(2, kInfIndex));

  EXPECT_EQ(grid(1, 7, &grid_cell), -1);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedClosed(-kInfIndex, 9));
  EXPECT_EQ(grid(1, 11, &grid_cell), 0);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedSized(10, 10));
  EXPECT_EQ(grid(1, 57, &grid_cell), 3);
  EXPECT_EQ(grid_cell, IndexInterval::UncheckedClosed(45, kInfIndex));
}

TEST(IrregularGridTest, Rank0) {
  std::vector<std::vector<Index>> inclusive_mins;
  auto grid = IrregularGrid(inclusive_mins);
  EXPECT_EQ(0, grid.rank());
  EXPECT_TRUE(grid.shape().empty());
  EXPECT_TRUE(grid.cell_origin({}).empty());
}

}  // namespace
