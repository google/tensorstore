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

#include "tensorstore/internal/grid_partition_impl.h"

#include <optional>
#include <ostream>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_grid_partition {

using tensorstore::MatchesStatus;

std::ostream& operator<<(std::ostream& os,
                         const IndexTransformGridPartition::StridedSet& s) {
  return os << "{grid_dimensions=" << s.grid_dimensions
            << ", input_dimension=" << s.input_dimension << "}";
}
bool operator==(const IndexTransformGridPartition::StridedSet& a,
                const IndexTransformGridPartition::StridedSet& b) {
  return a.input_dimension == b.input_dimension &&
         internal::RangesEqual(a.grid_dimensions, b.grid_dimensions);
}

bool operator!=(const IndexTransformGridPartition::StridedSet& a,
                const IndexTransformGridPartition::StridedSet& b) {
  return !(a == b);
}

std::ostream& operator<<(std::ostream& os,
                         const IndexTransformGridPartition::IndexArraySet& s) {
  return os << "IndexArraySet where:\n"
            << "  grid_dimensions=" << s.grid_dimensions << "\n"
            << "  input_dimensions=" << s.input_dimensions << "\n"
            << "  grid_cell_indices="
            << Array(s.grid_cell_indices.data(),
                     {s.num_partitions(), s.grid_dimensions.size()})
            << "\n"
            << "  partitioned_input_indices=" << s.partitioned_input_indices
            << "\n"
            << "  grid_cell_partition_offsets="
            << span(s.grid_cell_partition_offsets) << "\n";
}

bool operator==(const IndexTransformGridPartition::IndexArraySet& a,
                const IndexTransformGridPartition::IndexArraySet& b) {
  return internal::RangesEqual(a.input_dimensions, b.input_dimensions) &&
         internal::RangesEqual(a.grid_dimensions, b.grid_dimensions) &&
         a.grid_cell_indices == b.grid_cell_indices &&
         a.partitioned_input_indices == b.partitioned_input_indices &&
         a.grid_cell_partition_offsets == b.grid_cell_partition_offsets;
}
bool operator!=(const IndexTransformGridPartition::IndexArraySet& a,
                const IndexTransformGridPartition::IndexArraySet& b) {
  return !(a == b);
}

}  // namespace internal_grid_partition
}  // namespace tensorstore

namespace {
using tensorstore::DimensionIndex;
using tensorstore::Index;
using tensorstore::IndexInterval;
using tensorstore::IndexTransformBuilder;
using tensorstore::kInfIndex;
using tensorstore::MakeArray;
using tensorstore::MatchesStatus;
using tensorstore::Result;
using tensorstore::span;
using tensorstore::internal_grid_partition::IndexTransformGridPartition;
using tensorstore::internal_grid_partition::
    PrePartitionIndexTransformOverRegularGrid;
using ::testing::ElementsAre;

// Tests that if there are no grid dimensions, the result has no connected sets.
TEST(PrePartitionIndexTransformOverRegularGridTest, NoGridDimensions) {
  auto transform = tensorstore::IndexTransformBuilder<>(1, 1)
                       .input_origin({1})
                       .input_shape({5})
                       .output_single_input_dimension(0, 0)
                       .Finalize()
                       .value();
  span<const DimensionIndex> grid_output_dimensions;
  span<const Index> grid_cell_shape;
  absl::optional<IndexTransformGridPartition> partitioned;
  TENSORSTORE_CHECK_OK(PrePartitionIndexTransformOverRegularGrid(
      transform, grid_output_dimensions, grid_cell_shape, &partitioned));
  EXPECT_THAT(partitioned->strided_sets(), ElementsAre());
  EXPECT_THAT(partitioned->index_array_sets(), ElementsAre());
}

// Tests that a constant output index map leads to a result with no connected
// sets.
TEST(PrePartitionIndexTransformOverRegularGridTest, NoConnectedSets) {
  auto transform = tensorstore::IndexTransformBuilder<>(1, 1)
                       .input_origin({1})
                       .input_shape({5})
                       .output_constant(0, 3)
                       .Finalize()
                       .value();
  const DimensionIndex grid_output_dimensions[] = {0};
  const Index grid_cell_shape[] = {2};
  absl::optional<IndexTransformGridPartition> partitioned;
  TENSORSTORE_CHECK_OK(PrePartitionIndexTransformOverRegularGrid(
      transform, grid_output_dimensions, grid_cell_shape, &partitioned));
  EXPECT_THAT(partitioned->strided_sets(), ElementsAre());
  EXPECT_THAT(partitioned->index_array_sets(), ElementsAre());
}

// Tests that an identity transform to a grid dimension leads to a single
// strided connected set with a single grid dimension and a single input
// dimension.
TEST(PrePartitionIndexTransformOverRegularGridTest, StridedSingleSet) {
  auto transform = tensorstore::IndexTransformBuilder<>(1, 1)
                       .input_origin({1})
                       .input_shape({5})
                       .output_single_input_dimension(0, 0)
                       .Finalize()
                       .value();
  const DimensionIndex grid_output_dimensions[] = {0};
  const Index grid_cell_shape[] = {2};
  absl::optional<IndexTransformGridPartition> partitioned;
  TENSORSTORE_CHECK_OK(PrePartitionIndexTransformOverRegularGrid(
      transform, grid_output_dimensions, grid_cell_shape, &partitioned));
  EXPECT_THAT(partitioned->strided_sets(),
              ElementsAre(IndexTransformGridPartition::StridedSet{
                  /*.grid_dimensions=*/span<const DimensionIndex>({0}),
                  /*.input_dimension=*/0}));
  EXPECT_THAT(partitioned->index_array_sets(), ElementsAre());
}

// Tests that two independent `single_input_dimension` output index maps on grid
// dimensions lead to a result with two connected sets, each with a single input
// dimension and a single grid dimension.  Also tests that irrelevant output
// index maps are correctly ignored.
TEST(PrePartitionIndexTransformOverRegularGridTest,
     StridedSingleDimensionSets) {
  auto transform = tensorstore::IndexTransformBuilder<>(5, 4)
                       .input_origin({1, 2, 3, 4, 5})
                       .input_shape({6, 7, 8, 9, 10})
                       .output_single_input_dimension(0, 2)
                       .output_single_input_dimension(2, 4)
                       // Output dimension 3 is not in `grid_output_dimensions`,
                       // and therefore the following output index map should
                       // not affect the partitioning.
                       .output_single_input_dimension(3, 3)
                       .Finalize()
                       .value();
  const DimensionIndex grid_output_dimensions[] = {2, 0};
  const Index grid_cell_shape[] = {5, 10};
  absl::optional<IndexTransformGridPartition> partitioned;
  TENSORSTORE_CHECK_OK(PrePartitionIndexTransformOverRegularGrid(
      transform, grid_output_dimensions, grid_cell_shape, &partitioned));
  EXPECT_THAT(
      partitioned->strided_sets(),
      ElementsAre(IndexTransformGridPartition::StridedSet{
                      /*.grid_dimensions=*/span<const DimensionIndex>({0}),
                      /*.input_dimension=*/4},
                  IndexTransformGridPartition::StridedSet{
                      /*.grid_dimensions=*/span<const DimensionIndex>({1}),
                      /*.input_dimension=*/2}));
  EXPECT_THAT(partitioned->index_array_sets(), ElementsAre());
}

// Tests that an index transform where both output dimensions are
// `single_input_dimension` output index maps from the same input dimension, and
// both output dimensions are grid dimensions, leads to a single strided
// connected set containing the input dimension and both grid dimensions.
TEST(PrePartitionIndexTransformOverRegularGridTest, DiagonalStridedSet) {
  auto transform = tensorstore::IndexTransformBuilder<>(1, 2)
                       .input_origin({1})
                       .input_shape({6})
                       .output_single_input_dimension(0, 0)
                       .output_single_input_dimension(1, 0)
                       .Finalize()
                       .value();
  const DimensionIndex grid_output_dimensions[] = {0, 1};
  const Index grid_cell_shape[] = {5, 10};
  absl::optional<IndexTransformGridPartition> partitioned;
  TENSORSTORE_CHECK_OK(PrePartitionIndexTransformOverRegularGrid(
      transform, grid_output_dimensions, grid_cell_shape, &partitioned));
  EXPECT_THAT(partitioned->strided_sets(),
              ElementsAre(IndexTransformGridPartition::StridedSet{
                  /*.grid_dimensions=*/span<const DimensionIndex>({0, 1}),
                  /*.input_dimension=*/0}));
  EXPECT_THAT(partitioned->index_array_sets(), ElementsAre());
}

// Same as above, but with extra input and output dimensions, and adds an extra
// independent strided connected set.
TEST(PrePartitionIndexTransformOverRegularGridTest, DiagonalStridedSets) {
  auto transform = tensorstore::IndexTransformBuilder<>(5, 4)
                       .input_origin({1, 2, 3, 4, 5})
                       .input_shape({6, 7, 8, 9, 10})
                       .output_single_input_dimension(0, 2)
                       .output_single_input_dimension(1, 4)
                       .output_single_input_dimension(2, 4)
                       .output_single_input_dimension(3, 3)
                       .Finalize()
                       .value();
  const DimensionIndex grid_output_dimensions[] = {2, 0, 1};
  const Index grid_cell_shape[] = {5, 10, 15};
  absl::optional<IndexTransformGridPartition> partitioned;
  TENSORSTORE_CHECK_OK(PrePartitionIndexTransformOverRegularGrid(
      transform, grid_output_dimensions, grid_cell_shape, &partitioned));
  EXPECT_THAT(
      partitioned->strided_sets(),
      ElementsAre(IndexTransformGridPartition::StridedSet{
                      /*.grid_dimensions=*/span<const DimensionIndex>({0, 2}),
                      /*.input_dimension=*/4},
                  IndexTransformGridPartition::StridedSet{
                      /*.grid_dimensions=*/span<const DimensionIndex>({1}),
                      /*.input_dimension=*/2}));
  EXPECT_THAT(partitioned->index_array_sets(), ElementsAre());
}

// Tests that a single output dimension (included in grid_output_dimensions)
// with an `array` output index map that depends on a single input dimension
// leads to a single index array connected set.
TEST(PrePartitionIndexTransformOverRegularGridTest, SingleIndexArrayDimension) {
  auto transform =
      tensorstore::IndexTransformBuilder<>(1, 1)
          .input_origin({0})
          .input_shape({4})
          .output_index_array(0, 5, 2, MakeArray<Index>({1, 9, 8, 4}))
          .Finalize()
          .value();
  const DimensionIndex grid_output_dimensions[] = {0};
  const Index grid_cell_shape[] = {4};
  absl::optional<IndexTransformGridPartition> partitioned;
  TENSORSTORE_CHECK_OK(PrePartitionIndexTransformOverRegularGrid(
      transform, grid_output_dimensions, grid_cell_shape, &partitioned));
  EXPECT_THAT(
      partitioned->index_array_sets(),
      ElementsAre(IndexTransformGridPartition::IndexArraySet{
          /*.grid_dimensions=*/span<const DimensionIndex>({0}),
          /*.input_dimensions=*/span<const DimensionIndex>({0}),
          /*.grid_cell_indices=*/{1, 3, 5},
          /*.partitioned_input_indices=*/MakeArray<Index>({{0}, {3}, {1}, {2}}),
          /*.grid_cell_partition_offsets=*/{0, 1, 2}}));
  EXPECT_THAT(partitioned->strided_sets(), ElementsAre());
}

// Tests that two output dimensions (included in grid_output_dimensions), where
// one depends on the single input dimension using a `single_input_dimension`
// output index map, and the other depends on the single input dimension using
// an `array` output index map, leads to a single index array connected set
// containing both grid dimensions and the single output dimension.
TEST(PrePartitionIndexTransformOverRegularGridTest,
     IndexArrayAndStridedDimension) {
  auto transform =
      tensorstore::IndexTransformBuilder<>(1, 2)
          .input_origin({0})
          .input_shape({4})
          .output_index_array(0, 5, 2, MakeArray<Index>({1, 9, 8, 4}))
          .output_single_input_dimension(1, 3, 5, 0)
          .Finalize()
          .value();
  const DimensionIndex grid_output_dimensions[] = {1, 0};
  const Index grid_cell_shape[] = {10, 4};
  absl::optional<IndexTransformGridPartition> partitioned;
  TENSORSTORE_CHECK_OK(PrePartitionIndexTransformOverRegularGrid(
      transform, grid_output_dimensions, grid_cell_shape, &partitioned));
  // Input indices are:                         {0,  1,  2,  3}

  // Output indices for output dimension 1 are: {3,  8, 13, 18}
  // Note: output dimension 1 is grid dimension 0.

  // Output indices for output dimension 0 are: {7, 23, 21, 13}
  // Note: output dimension 0 is grid dimension 1.

  // Cell indices for grid dimension 0 are:     {0,  0,  1,  1}
  // Cell indices for grid dimension 1 are:     {1,  5,  5,  3}
  EXPECT_THAT(
      partitioned->index_array_sets(),
      ElementsAre(IndexTransformGridPartition::IndexArraySet{
          /*.grid_dimensions=*/span<const DimensionIndex>({0, 1}),
          /*.input_dimensions=*/span<const DimensionIndex>({0}),
          /*.grid_cell_indices=*/{0, 1, 0, 5, 1, 3, 1, 5},
          /*.partitioned_input_indices=*/MakeArray<Index>({{0}, {1}, {3}, {2}}),
          /*.grid_cell_partition_offsets=*/{0, 1, 2, 3}}));
  EXPECT_THAT(partitioned->strided_sets(), ElementsAre());
}

// Tests that two output dimensions (included in grid_output_dimensions) and two
// input dimensions, where one output dimension has a `single_input_dimension`
// map from input dimension 1, and the other output dimension has an index array
// map that depends only on input dimension 0, leads to two connected sets, each
// containing one grid dimension and one input dimension.  This also tests that
// transposed input dimensions are handled correctly.
TEST(PrePartitionIndexTransformOverRegularGridTest,
     IndexArrayAndStridedDimensionIndependent) {
  auto transform =
      tensorstore::IndexTransformBuilder<>(2, 2)
          .input_origin({0, 0})
          .input_shape({2, 3})
          .output_single_input_dimension(0, 1)
          .output_index_array(1, 0, 1, MakeArray<Index>({{0}, {0}}))
          .Finalize()
          .value();
  const DimensionIndex grid_output_dimensions[] = {0, 1};
  const Index grid_cell_shape[] = {3, 1};
  absl::optional<IndexTransformGridPartition> partitioned;
  TENSORSTORE_CHECK_OK(PrePartitionIndexTransformOverRegularGrid(
      transform, grid_output_dimensions, grid_cell_shape, &partitioned));
  // Input indices for input dimension 1 are:   {0,  1,  2}

  // Output indices for output dimension 0 are: {0,  1,  2}
  // Note: output dimension 0 is grid dimension 0.

  // Input indices for input dimension 0 are:   {0,  1}

  // Output indices for output dimension 1 are: {0,  0}
  // Note: output dimension 1 is grid dimension 1.

  // Cell indices for grid dimension 0 are:     {0}
  // Cell indices for grid dimension 1 are:     {0}
  EXPECT_THAT(partitioned->index_array_sets(),
              ElementsAre(IndexTransformGridPartition::IndexArraySet{
                  /*.grid_dimensions=*/span<const DimensionIndex>({1}),
                  /*.input_dimensions=*/span<const DimensionIndex>({0}),
                  /*.grid_cell_indices=*/{0},
                  /*.partitioned_input_indices=*/MakeArray<Index>({{0}, {1}}),
                  /*.grid_cell_partition_offsets=*/{0}}));
  EXPECT_THAT(partitioned->strided_sets(),
              ElementsAre(IndexTransformGridPartition::StridedSet{
                  /*.grid_dimensions=*/span<const DimensionIndex>({0}),
                  /*.input_dimension=*/1}));
}

// Tests that a connected set containing two index array output index maps is
// correctly handled.
TEST(PrePartitionIndexTransformOverRegularGridTest,
     TwoOutputsTwoDimensionalIndexArrays) {
  auto transform =
      tensorstore::IndexTransformBuilder<>(2, 2)
          .input_origin({-1, 2})
          .input_shape({2, 3})
          .output_index_array(0, 5, 2, MakeArray<Index>({{1, 2, 3}, {3, 4, 5}}))
          .output_index_array(1, 2, 1, MakeArray<Index>({{5, 9, 1}, {8, 2, 3}}))
          .Finalize()
          .value();
  const DimensionIndex grid_output_dimensions[] = {1, 0};
  const Index grid_cell_shape[] = {3, 5};
  absl::optional<IndexTransformGridPartition> partitioned;
  TENSORSTORE_CHECK_OK(PrePartitionIndexTransformOverRegularGrid(
      transform, grid_output_dimensions, grid_cell_shape, &partitioned));
  EXPECT_THAT(
      partitioned->index_array_sets(),
      ElementsAre(IndexTransformGridPartition::IndexArraySet{
          /*.grid_dimensions=*/span<const DimensionIndex>({0, 1}),
          /*.input_dimensions=*/span<const DimensionIndex>({0, 1}),
          /*.grid_cell_indices=*/{1, 2, 1, 3, 2, 1, 3, 1, 3, 2},
          /*.partitioned_input_indices=*/
          MakeArray<Index>({{-1, 4}, {0, 3}, {0, 4}, {-1, 2}, {-1, 3}, {0, 2}}),
          /*.grid_cell_partition_offsets=*/{0, 2, 3, 4, 5}}));
  EXPECT_THAT(partitioned->strided_sets(), ElementsAre());
}

// Tests that an unbounded input domain leads to an error.
TEST(PrePartitionIndexTransformOverRegularGridTest, UnboundedDomain) {
  auto transform = tensorstore::IndexTransformBuilder<>(1, 1)
                       .input_origin({-kInfIndex})
                       .input_shape({100})
                       .output_single_input_dimension(0, 0)
                       .Finalize()
                       .value();
  const DimensionIndex grid_output_dimensions[] = {0};
  const Index grid_cell_shape[] = {5};
  absl::optional<IndexTransformGridPartition> partitioned;
  auto status = PrePartitionIndexTransformOverRegularGrid(
      transform, grid_output_dimensions, grid_cell_shape, &partitioned);
  EXPECT_THAT(status,
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Input dimension 0 has unbounded domain .*"));
}

// Tests that an out-of-bounds index in an index array leads to an error.
TEST(PrePartitionIndexTransformOverRegularGridTest, IndexArrayOutOfBounds) {
  auto transform = tensorstore::IndexTransformBuilder<>(1, 1)
                       .input_origin({1})
                       .input_shape({3})
                       .output_index_array(0, 0, 1, MakeArray<Index>({2, 3, 4}),
                                           IndexInterval::Closed(3, 10))
                       .Finalize()
                       .value();
  const DimensionIndex grid_output_dimensions[] = {0};
  const Index grid_cell_shape[] = {5};
  absl::optional<IndexTransformGridPartition> partitioned;
  auto status = PrePartitionIndexTransformOverRegularGrid(
      transform, grid_output_dimensions, grid_cell_shape, &partitioned);
  EXPECT_THAT(status,
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Index 2 is outside valid range \\[3, 11\\)"));
}

// Tests that integer overflow due to a `single_input_dimension` mapping leads
// to an error.
TEST(PrePartitionIndexTransformOverRegularGridTest, StridedDimensionOverflow) {
  auto transform =
      tensorstore::IndexTransformBuilder<>(1, 2)
          .input_origin({0})
          .input_shape({4})
          .output_index_array(0, 5, 2, MakeArray<Index>({1, 9, 8, 4}))
          .output_single_input_dimension(1, -kInfIndex, -kInfIndex, 0)
          .Finalize()
          .value();
  const DimensionIndex grid_output_dimensions[] = {1, 0};
  const Index grid_cell_shape[] = {10, 4};
  absl::optional<IndexTransformGridPartition> partitioned;
  auto status = PrePartitionIndexTransformOverRegularGrid(
      transform, grid_output_dimensions, grid_cell_shape, &partitioned);
  EXPECT_THAT(status, MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
