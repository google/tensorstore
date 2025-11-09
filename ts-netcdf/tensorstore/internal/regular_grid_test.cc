// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/internal/regular_grid.h"

#include <array>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::IndexInterval;
using ::tensorstore::internal_grid_partition::RegularGridRef;
using ::testing::Eq;

TEST(RegularGridTest, Basic) {
  std::array<Index, 3> grid_cell_shape = {10, 20, 30};
  RegularGridRef regular_grid{grid_cell_shape};
  IndexInterval cell_bounds;

  // For dimensions [0..3) all indices in the range [0..9] map to the output
  // grid cell 0.
  for (Index i = 0; i < 10; i++) {
    EXPECT_THAT(regular_grid(0, i, &cell_bounds), Eq(0));
    EXPECT_THAT(cell_bounds, Eq(IndexInterval::UncheckedSized(0, 10)));

    EXPECT_THAT(regular_grid(1, i, &cell_bounds), Eq(0));
    EXPECT_THAT(cell_bounds, Eq(IndexInterval::UncheckedSized(0, 20)));

    EXPECT_THAT(regular_grid(2, i, &cell_bounds), Eq(0));
    EXPECT_THAT(cell_bounds, Eq(IndexInterval::UncheckedSized(0, 30)));
  }

  for (DimensionIndex i = 0; i < 3; i++) {
    Index j = (i + 1) * 10;

    EXPECT_THAT(regular_grid(i, j - 1, &cell_bounds), Eq(0));
    EXPECT_THAT(cell_bounds, Eq(IndexInterval::UncheckedSized(0, j)));

    EXPECT_THAT(regular_grid(i, j, &cell_bounds), Eq(1));
    EXPECT_THAT(cell_bounds, Eq(IndexInterval::UncheckedSized(j, j)));
  }
}

}  // namespace
