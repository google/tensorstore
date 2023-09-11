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

#ifndef TENSORSTORE_INTERNAL_REGULAR_GRID_H_
#define TENSORSTORE_INTERNAL_REGULAR_GRID_H_

#include "absl/functional/function_ref.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/span.h"

/// \file
///
/// Defines the `RegularGridRef` type for use with `grid_partition.h` and
/// `grid_partition_impl.h`.

namespace tensorstore {
namespace internal_grid_partition {

/// For a given DimensionIndex dimension, returns the cell index corresponding
/// to the output_index, optionally filling the bounds for the cell.
using OutputToGridCellFn = absl::FunctionRef<Index(
    DimensionIndex grid_dim, Index output_index, IndexInterval* cell_bounds)>;

/// RegularGridRef is a functor that supplies a grid cell index as well as grid
/// cell bounds for PartitionIndexTransformOverGrid.
///
/// Conceptually it describes a grid which extends from -inf to +inf over all
/// grid dimensions. The grid cell with index vector `v` corresponds to the
/// hyperrectangle with inclusive lower bound `v * grid_cell_shape` and
/// exclusive upper bound `(v + 1) * grid_cell_shape`.
struct RegularGridRef {
  span<const Index> grid_cell_shape;

  DimensionIndex rank() const { return grid_cell_shape.size(); }

  IndexInterval GetCellOutputInterval(DimensionIndex dim,
                                      Index cell_index) const {
    return IndexInterval::UncheckedSized(cell_index * grid_cell_shape[dim],
                                         grid_cell_shape[dim]);
  }

  /// Converts output indices to grid indices of a regular grid.
  /// Returns the cell index and cell bounds.
  Index operator()(DimensionIndex dim, Index output_index,
                   IndexInterval* cell_bounds) const {
    Index cell_index = FloorOfRatio(output_index, grid_cell_shape[dim]);
    if (cell_bounds) {
      *cell_bounds = GetCellOutputInterval(dim, cell_index);
    }
    return cell_index;
  }
};

}  // namespace internal_grid_partition
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_REGULAR_GRID_H_
