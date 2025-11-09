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

#include "tensorstore/internal/grid_partition.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/grid_partition_impl.h"
#include "tensorstore/internal/irregular_grid.h"
#include "tensorstore/internal/regular_grid.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace {
using ::tensorstore::Box;
using ::tensorstore::BoxView;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::IndexInterval;
using ::tensorstore::IndexTransform;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::IndexTransformView;
using ::tensorstore::MakeArray;
using ::tensorstore::Result;
using ::tensorstore::internal::GetGridCellRanges;
using ::tensorstore::internal::IrregularGrid;
using ::tensorstore::internal::OutputToGridCellFn;
using ::tensorstore::internal_grid_partition::IndexTransformGridPartition;
using ::tensorstore::internal_grid_partition::
    PrePartitionIndexTransformOverGrid;
using ::tensorstore::internal_grid_partition::RegularGridRef;
using ::testing::ElementsAre;

namespace get_grid_cell_ranges_tests {

using R = Box<>;
Result<std::vector<R>> GetRanges(
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    BoxView<> grid_bounds, OutputToGridCellFn output_to_grid_cell,
    IndexTransformView<> transform) {
  std::vector<R> results;
  IndexTransformGridPartition grid_partition;
  TENSORSTORE_RETURN_IF_ERROR(PrePartitionIndexTransformOverGrid(
      transform, grid_output_dimensions, output_to_grid_cell, grid_partition));
  TENSORSTORE_RETURN_IF_ERROR(GetGridCellRanges(
      grid_output_dimensions, grid_bounds, output_to_grid_cell, transform,
      [&](BoxView<> bounds) -> absl::Status {
        results.emplace_back(bounds);
        return absl::OkStatus();
      }));
  return results;
}

TEST(GetGridCellRangesTest, Rank0) {
  EXPECT_THAT(GetRanges(/*grid_output_dimensions=*/{}, /*grid_bounds=*/{},
                        /*output_to_grid_cell=*/RegularGridRef{{}},
                        IndexTransformBuilder(0, 0).Finalize().value()),
              ::testing::Optional(ElementsAre(R{})));
}

TEST(GetGridCellRangesTest, Rank1Unconstrained) {
  EXPECT_THAT(GetRanges(/*grid_output_dimensions=*/{{0}},
                        /*grid_bounds=*/Box<>{{0}, {10}},
                        /*output_to_grid_cell=*/RegularGridRef{{{5}}},
                        IndexTransformBuilder(1, 1)
                            .input_shape({50})
                            .output_identity_transform()
                            .Finalize()
                            .value()),
              ::testing::Optional(ElementsAre(R{{0}, {10}})));
}

TEST(GetGridCellRangesTest, Rank1Constrained) {
  // Grid dimension 0:
  //   Output range: [7, 36]
  //   Grid range: [1, 7]
  EXPECT_THAT(GetRanges(/*grid_output_dimensions=*/{{0}},
                        /*grid_bounds=*/Box<>{{0}, {10}},
                        /*output_to_grid_cell=*/RegularGridRef{{{5}}},
                        IndexTransformBuilder(1, 1)
                            .input_origin({7})
                            .input_shape({30})
                            .output_identity_transform()
                            .Finalize()
                            .value()),
              ::testing::Optional(ElementsAre(R({1}, {7}))));
}

TEST(GetGridCellRangesTest, Rank2ConstrainedBothDims) {
  // Grid dimension 0:
  //   Output range: [6, 13]
  //   Grid range: [1, 2]
  // Grid dimension 1:
  //   Output range: [7, 37)
  //   Grid range: [0, 3]
  EXPECT_THAT(GetRanges(/*grid_output_dimensions=*/{{0, 1}},
                        /*grid_bounds=*/Box<>{{0, 0}, {5, 10}},
                        /*output_to_grid_cell=*/RegularGridRef{{{5, 10}}},
                        IndexTransformBuilder(2, 2)
                            .input_origin({6, 7})
                            .input_shape({8, 30})
                            .output_identity_transform()
                            .Finalize()
                            .value()),
              ::testing::Optional(ElementsAre(  //
                  R{{1, 0}, {1, 4}},            //
                  R{{2, 0}, {1, 4}}             //
                  )));
}

TEST(GetGridCellRangesTest, Rank2ConstrainedFirstDimOnly) {
  // Grid dimension 0:
  //   Output range: [6, 13]
  //   Grid range: [1, 2]
  // Grid dimension 1:
  //   Output range: [0, 49]
  //   Grid range: [0, 9] (unconstrained)
  EXPECT_THAT(GetRanges(/*grid_output_dimensions=*/{{0, 1}},
                        /*grid_bounds=*/Box<>{{0, 0}, {5, 10}},
                        /*output_to_grid_cell=*/RegularGridRef{{{5, 5}}},
                        IndexTransformBuilder(2, 2)
                            .input_origin({6, 0})
                            .input_shape({8, 50})
                            .output_identity_transform()
                            .Finalize()
                            .value()),
              ::testing::Optional(ElementsAre(R{{1, 0}, {2, 10}})));
}

TEST(GetGridCellRangesTest, Rank2ConstrainedSecondDimOnly) {
  // Grid dimension 0:
  //   Output range: [0, 24]
  //   Grid range: [0, 4] (unconstrained)
  // Grid dimension 1:
  //   Output range: [7, 36]
  //   Grid range: [1, 7]
  EXPECT_THAT(GetRanges(/*grid_output_dimensions=*/{{0, 1}},
                        /*grid_bounds=*/Box<>{{0, 0}, {5, 10}},
                        /*output_to_grid_cell=*/RegularGridRef{{{5, 5}}},
                        IndexTransformBuilder(2, 2)
                            .input_origin({0, 7})
                            .input_shape({25, 30})
                            .output_identity_transform()
                            .Finalize()
                            .value()),
              ::testing::Optional(ElementsAre(  //
                  R{{0, 1}, {1, 7}},            //
                  R{{1, 1}, {1, 7}},            //
                  R{{2, 1}, {1, 7}},            //
                  R{{3, 1}, {1, 7}},            //
                  R{{4, 1}, {1, 7}}             //
                  )));
}

TEST(GetGridCellRangesTest, Rank2IndexArrayFirstDimUnconstrainedSecondDim) {
  // Grid dimension 0:
  //   Output range: {6, 15, 20}
  //   Grid range: {1, 3, 4}
  // Grid dimension 1:
  //   Output range: [0, 49]
  //   Grid range: [0, 9] (unconstrained)
  EXPECT_THAT(
      GetRanges(
          /*grid_output_dimensions=*/{{0, 1}},
          /*grid_bounds=*/Box<>{{0, 0}, {5, 10}},
          /*output_to_grid_cell=*/RegularGridRef{{{5, 5}}},
          IndexTransformBuilder(2, 2)
              .input_origin({0, 0})
              .input_shape({3, 50})
              .output_index_array(0, 0, 1, MakeArray<Index>({{6}, {15}, {20}}))
              .output_single_input_dimension(1, 1)
              .Finalize()
              .value()),
      ::testing::Optional(ElementsAre(  //
          R{{1, 0}, {1, 10}},           //
          R{{3, 0}, {2, 10}}            //
          )));
}

TEST(GetGridCellRangesTest, Rank2IndexArrayFirstDimConstrainedSecondDim) {
  // Grid dimension 0:
  //   Output range: {6, 15, 20}
  //   Grid range: {1, 3, 4}
  // Grid dimension 1:
  //   Output range: [7, 36]
  //   Grid range: [1, 7]
  EXPECT_THAT(
      GetRanges(
          /*grid_output_dimensions=*/{{0, 1}},
          /*grid_bounds=*/Box<>{{0, 0}, {5, 10}},
          /*output_to_grid_cell=*/RegularGridRef{{{5, 5}}},
          IndexTransformBuilder(2, 2)
              .input_origin({0, 7})
              .input_shape({3, 30})
              .output_index_array(0, 0, 1, MakeArray<Index>({{6}, {15}, {20}}))
              .output_single_input_dimension(1, 1)
              .Finalize()
              .value()),
      // Since grid dimension 1 is constrained, a separate range is required for
      // each grid dimension 0 index.
      ::testing::Optional(ElementsAre(  //
          R{{1, 1}, {1, 7}},            //
          R{{3, 1}, {1, 7}},            //
          R{{4, 1}, {1, 7}}             //
          )));
}

TEST(GetGridCellRangesTest, Rank2Diagonal) {
  // Grid dimension 0:
  //   Output range: [6, 13]
  //   Grid range: [1, 2]
  // Grid dimension 1:
  //   Output range: [6, 13]
  //   Grid range: [0, 1]
  EXPECT_THAT(GetRanges(/*grid_output_dimensions=*/{{0, 1}},
                        /*grid_bounds=*/Box<>{{0, 0}, {5, 10}},
                        /*output_to_grid_cell=*/RegularGridRef{{{5, 10}}},
                        IndexTransformBuilder(1, 2)
                            .input_origin({6})
                            .input_shape({8})
                            .output_single_input_dimension(0, 0)
                            .output_single_input_dimension(1, 0)
                            .Finalize()
                            .value()),
              ::testing::Optional(ElementsAre(  //
                  R{{1, 0}, {1, 1}},            //
                  R{{2, 1}, {1, 1}}             //
                  )));
}

}  // namespace get_grid_cell_ranges_tests

}  // namespace
