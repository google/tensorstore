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
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::BoxView;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::IndexDomain;
using ::tensorstore::IndexDomainBuilder;
using ::tensorstore::IndexInterval;
using ::tensorstore::kInfIndex;
using ::tensorstore::internal::IrregularGrid;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(IrregularGridTest, Basic) {
  std::vector<Index> dimension0{2, 0, -3};
  std::vector<Index> dimension1{10, 45, 20, 30};
  auto grid = IrregularGrid({dimension0, dimension1});

  EXPECT_EQ(2, grid.rank());
  EXPECT_THAT(grid.shape(), ElementsAre(2, 3));
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
  EXPECT_EQ(grid(0, 0, nullptr), 1);
}

TEST(IrregularGridTest, IndexDomain) {
  const Index origin1[] = {-3, 10};
  const Index shape1[] = {3, 10};
  const Index origin2[] = {0, 20};
  const Index shape2[] = {2, 10};
  const Index origin3[] = {0, 30};
  const Index shape3[] = {2, 15};

  std::vector<IndexDomain<>> domains(
      {IndexDomain<>{
           BoxView<>{tensorstore::span(origin1), tensorstore::span(shape1)}},
       IndexDomain<>{
           BoxView<>{tensorstore::span(origin2), tensorstore::span(shape2)}},
       IndexDomain<>{
           BoxView<>{tensorstore::span(origin3), tensorstore::span(shape3)}}});

  auto grid = IrregularGrid::Make(domains);

  EXPECT_EQ(2, grid.rank());
  EXPECT_THAT(grid.shape(), ElementsAre(2, 3));
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
  EXPECT_EQ(grid(0, 0, nullptr), 1);
}

TEST(IrregularGridTest, Rank0) {
  IrregularGrid default_grid;
  EXPECT_EQ(0, default_grid.rank());
  EXPECT_THAT(default_grid.shape(), IsEmpty());
  EXPECT_THAT(default_grid.cell_origin({}), IsEmpty());

  std::vector<std::vector<Index>> inclusive_mins;
  auto grid = IrregularGrid(inclusive_mins);
  EXPECT_EQ(0, grid.rank());
  EXPECT_THAT(grid.shape(), IsEmpty());
  EXPECT_THAT(grid.cell_origin({}), IsEmpty());
}

TEST(IrregularGridTest, CellOrigin) {
  auto grid = IrregularGrid({{2, 0, -3}, {10, 45, 20, 30}});
  ASSERT_THAT(grid.shape(), ElementsAre(2, 3));

  EXPECT_THAT(grid.cell_origin(std::vector<Index>{-1, -1}),
              ElementsAre(-kInfIndex, -kInfIndex));
  EXPECT_THAT(grid.cell_origin(std::vector<Index>{0, 0}), ElementsAre(-3, 10));
  EXPECT_THAT(grid.cell_origin(std::vector<Index>{1, 1}), ElementsAre(0, 20));
  EXPECT_THAT(grid.cell_origin(std::vector<Index>{1, 2}), ElementsAre(0, 30));
  EXPECT_THAT(grid.cell_origin(std::vector<Index>{2, 3}), ElementsAre(2, 45));
  EXPECT_THAT(grid.cell_origin(std::vector<Index>{0, 3}), ElementsAre(-3, 45));
  EXPECT_THAT(grid.cell_origin(std::vector<Index>{2, -1}),
              ElementsAre(2, -kInfIndex));
  EXPECT_THAT(grid.cell_origin(std::vector<Index>{3, 4}),
              ElementsAre(kInfIndex, kInfIndex));
}

TEST(IrregularGridTest, EmptyDimension) {
  auto grid = IrregularGrid({{}, {10, 20}});
  EXPECT_EQ(2, grid.rank());
  EXPECT_THAT(grid.shape(), ElementsAre(0, 1));
  EXPECT_THAT(grid.inclusive_min(0), IsEmpty());

  IndexInterval cell_bounds;
  EXPECT_EQ(grid(0, 5, &cell_bounds), -1);
  EXPECT_EQ(cell_bounds, IndexInterval::Infinite());
  EXPECT_THAT(grid.cell_origin(std::vector<Index>{-1, 0}),
              ElementsAre(-kInfIndex, 10));
  EXPECT_THAT(grid.cell_origin(std::vector<Index>{-1, 1}),
              ElementsAre(-kInfIndex, 20));
  EXPECT_THAT(grid.cell_origin(std::vector<Index>{0, 2}),
              ElementsAre(kInfIndex, kInfIndex));
}

TEST(IrregularGridTest, UnboundedIndexDomain) {
  IndexDomain<> domain = IndexDomainBuilder(2)
                             .origin({10, -kInfIndex})
                             .exclusive_max({kInfIndex + 1, 20})
                             .Finalize()
                             .value();
  auto grid =
      IrregularGrid::Make(tensorstore::span<const IndexDomain<>>(&domain, 1));
  EXPECT_THAT(grid.shape(), ElementsAre(1, 1));

  IndexInterval cell_bounds;
  EXPECT_EQ(grid(0, 5, &cell_bounds), -1);
  EXPECT_EQ(cell_bounds, IndexInterval::UncheckedHalfOpen(-kInfIndex, 10));
  EXPECT_EQ(grid(0, 100, &cell_bounds), 0);
  EXPECT_EQ(cell_bounds, IndexInterval::UncheckedClosed(10, kInfIndex));
  EXPECT_EQ(grid(1, 25, &cell_bounds), 1);
  EXPECT_EQ(cell_bounds, IndexInterval::UncheckedClosed(20, kInfIndex));

  EXPECT_THAT(grid.cell_origin(std::vector<Index>{-1, 0}),
              ElementsAre(-kInfIndex, -kInfIndex));
  EXPECT_THAT(grid.cell_origin(std::vector<Index>{0, 0}),
              ElementsAre(10, -kInfIndex));
  EXPECT_THAT(grid.cell_origin(std::vector<Index>{1, 1}),
              ElementsAre(kInfIndex, 20));
}

TEST(IrregularGridTest, HighRankDomainViews) {
  constexpr DimensionIndex kRank = 10;
  std::vector<Index> origin_a(kRank, 0);
  std::vector<Index> shape_a(kRank, 4);
  std::vector<Index> origin_b(kRank, 4);
  std::vector<Index> shape_b(kRank, 6);
  IndexDomain<> domain_a(BoxView<>(origin_a, shape_a));
  IndexDomain<> domain_b(BoxView<>(origin_b, shape_b));
  std::vector<tensorstore::IndexDomainView<>> views{domain_a, domain_b};

  auto grid = IrregularGrid::Make(tensorstore::span(views));
  EXPECT_EQ(kRank, grid.rank());
  EXPECT_EQ(std::vector<Index>(kRank, 2),
            std::vector<Index>(grid.shape().begin(), grid.shape().end()));
  EXPECT_THAT(grid.inclusive_min(9), ElementsAre(0, 4, 10));
  EXPECT_EQ(grid.cell_origin(std::vector<Index>(kRank, 1)),
            std::vector<Index>(kRank, 4));
}

}  // namespace
