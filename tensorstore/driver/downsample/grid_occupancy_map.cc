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

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_downsample {

GridOccupancyMap::GridOccupancyMap(GridOccupancyTracker&& tracker,
                                   BoxView<> domain)
    : partition_points(domain.rank()) {
  // To represent the occupied and unoccupied regions of `domain`, we construct
  // a general (non-regular) rectilinear grid over `domain`:
  //
  // For each dimension `i` of `domain`, the grid line positions
  // (`dim_partition_points`) are chosen to be the union of the start/end bounds
  // of `domain[i]` and the start/end bounds within dimension `i` of all boxes
  // recorded in `tracker`.  This ensures that all boxes are aligned to grid
  // lines, but a box may cover more than one grid cell.
  //
  // We then construct a boolean mask array `occupied_chunk_mask` and mark all
  // grid cells covered by a box.
  //
  // To iterate over unoccupied regions, we iterate over the entire grid and
  // skip cells that are marked in `occupied_chunk_mask`.
  const DimensionIndex rank = domain.rank();
  span<Index> occupied_chunks = tracker.occupied_chunks;
  {
    // Temporary map from partition point to index in `partition_points`.
    absl::flat_hash_map<Index, Index> partition_map;
    for (DimensionIndex dim = 0; dim < rank; ++dim) {
      // Use `partition_map` to compute set of partition points, with all-zero
      // values.
      partition_map.clear();
      IndexInterval bounds = domain[dim];
      partition_map.emplace(bounds.inclusive_min(), 0);
      partition_map.emplace(bounds.exclusive_max(), 0);
      for (ptrdiff_t i = dim; i < occupied_chunks.size(); i += 2 * rank) {
        Index begin = occupied_chunks[i];
        Index end = begin + occupied_chunks[i + rank];
        partition_map.emplace(begin, 0);
        partition_map.emplace(end, 0);
      }
      // Compute ordered list of partition points and update `partition_map`
      // to map each partition point to its index within the partition point
      // list.
      auto& dim_partition_points = partition_points[dim];
      dim_partition_points.reserve(partition_map.size());
      for (const auto& p : partition_map) {
        dim_partition_points.push_back(p.first);
      }
      std::sort(dim_partition_points.begin(), dim_partition_points.end());
      for (size_t i = 0, size = dim_partition_points.size(); i < size; ++i) {
        partition_map.at(dim_partition_points[i]) = i;
      }
      // Update the `dim` column of `occupied_chunks` to contain indices into
      // the partition point list rather than `(begin, end)` index values.
      for (ptrdiff_t i = dim; i < occupied_chunks.size(); i += 2 * rank) {
        Index& begin = occupied_chunks[i];
        Index& end = occupied_chunks[i + rank];
        end = partition_map.at(begin + end);
        begin = partition_map.at(begin);
      }
    }
  }
  absl::FixedArray<Index, internal::kNumInlinedDims> grid_cell(rank);
  {
    for (DimensionIndex dim = 0; dim < rank; ++dim) {
      grid_cell[dim] = partition_points[dim].size() - 1;
    }
    occupied_chunk_mask = AllocateArray<bool>(grid_cell, c_order, value_init);
  }

  // Mark all occupied chunks in `occupied_chunk_mask`.
  for (ptrdiff_t i = 0; i < occupied_chunks.size(); i += 2 * rank) {
    std::copy_n(&occupied_chunks[i], rank, grid_cell.begin());
    do {
      occupied_chunk_mask(grid_cell) = true;
    } while (internal::AdvanceIndices(rank, grid_cell.data(),
                                      &occupied_chunks[i],
                                      &occupied_chunks[i + rank]));
  }
}

bool GridOccupancyMap::GetGridCellDomain(
    span<const Index> grid_cell, MutableBoxView<> grid_cell_domain) const {
  assert(grid_cell.size() == grid_cell_domain.rank());
  assert(grid_cell.size() == rank());
  if (occupied_chunk_mask(grid_cell)) return false;
  for (DimensionIndex dim = 0; dim < grid_cell.size(); ++dim) {
    const Index partition_index = grid_cell[dim];
    grid_cell_domain[dim] = IndexInterval::UncheckedHalfOpen(
        partition_points[dim][partition_index],
        partition_points[dim][partition_index + 1]);
  }
  return true;
}

void GridOccupancyMap::InitializeCellIterator(span<Index> grid_cell) const {
  std::fill(grid_cell.begin(), grid_cell.end(), 0);
}

bool GridOccupancyMap::AdvanceCellIterator(span<Index> grid_cell) const {
  assert(grid_cell.size() == occupied_chunk_mask.rank());
  return internal::AdvanceIndices(grid_cell.size(), grid_cell.data(),
                                  occupied_chunk_mask.shape().data());
}

}  // namespace internal_downsample
}  // namespace tensorstore
