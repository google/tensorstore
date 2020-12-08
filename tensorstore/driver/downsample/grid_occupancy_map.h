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

#ifndef TENSORSTORE_DRIVER_DOWNSAMPLE_GRID_OCCUPANCY_MAP_H_
#define TENSORSTORE_DRIVER_DOWNSAMPLE_GRID_OCCUPANCY_MAP_H_

/// \file
///
/// Facility for iterating over the regions of an index domain not covered by a
/// set of boxes.
///
/// This is used by the downsample driver to handle reads where some, but not
/// all, of the chunks can be downsampled directly.
///
/// The boxes that are covered (i.e. excluded) are recorded using
/// `GridOccupancyTracker`.  Once all boxes are recorded, a `GridOccupancyMap`
/// may be constructed.
#include <vector>

#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_downsample {

/// Tracks which portions of an index domain are occupied.
class GridOccupancyTracker {
 public:
  /// Flattened list of `origin, shape` for every chunk added.
  std::vector<Index> occupied_chunks;

  void MarkOccupied(BoxView<> box) {
    occupied_chunks.insert(occupied_chunks.end(), box.origin().begin(),
                           box.origin().end());
    occupied_chunks.insert(occupied_chunks.end(), box.shape().begin(),
                           box.shape().end());
  }
};

/// Processed occupancy map derived from a `GridOccupancyTracker`.
class GridOccupancyMap {
 public:
  /// Constructs the occupancy map.
  ///
  /// \param tracker Occupancy tracker specifying the chunks to add to the map.
  /// \param domain Domain that contains all chunks added to `tracker`.
  explicit GridOccupancyMap(GridOccupancyTracker&& tracker, BoxView<> domain);

  DimensionIndex rank() const { return occupied_chunk_mask.rank(); }

  bool GetGridCellDomain(span<const Index> grid_cell,
                         MutableBoxView<> grid_cell_domain) const;

  void InitializeCellIterator(span<Index> grid_cell) const;

  bool AdvanceCellIterator(span<Index> grid_cell) const;

  /// Ordered list of partition points for each dimension.  Always includes the
  /// both the `inclusive_min` and `exclusive_max` values of the domain for each
  /// dimension.
  std::vector<std::vector<Index>> partition_points;

  /// Mask corresponding to the non-regular grid specified by
  /// `partition_points`, where `true` indicates an occupied cell.
  ///
  /// The extent of dimension `dim` is equal to
  /// `partition_points[dim].size() - 1`, and a given `grid_cell` corresponds to
  /// the half-open interval with inclusive min
  /// `partition_points[i][grid_cell[i]]` and exclusive max
  /// `partition_points[i][[grid_cell[i]+1]`, for each dimension `i`.
  SharedArray<bool> occupied_chunk_mask;
};

}  // namespace internal_downsample
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DOWNSAMPLE_GRID_OCCUPANCY_MAP_H_
