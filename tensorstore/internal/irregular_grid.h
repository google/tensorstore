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

#ifndef TENSORSTORE_INTERNAL_IRREGULAR_GRID_H_
#define TENSORSTORE_INTERNAL_IRREGULAR_GRID_H_

#include <assert.h>

#include <vector>

#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

/// Consolidated representation of an irregular grid for
/// PartitionIndexTransformOverGrid and other grid-based indexing methods used
/// by the "stack" driver.
///
/// The grid cells have index vectors of length `rank()` in the range
/// `{0...} .. shape()`
class IrregularGrid {
 public:
  IrregularGrid() = default;
  IrregularGrid(std::vector<std::vector<Index>> unsorted_inclusive_mins);

  static IrregularGrid Make(span<const IndexDomainView<>> domains);
  static IrregularGrid Make(span<const IndexDomain<>> domains);

  /// Converts output indices to grid indices of a regular grid.
  Index operator()(DimensionIndex dim, Index output_index,
                   IndexInterval* cell_bounds) const;

  /// The rank of the grid.
  DimensionIndex rank() const { return shape_.size(); }

  /// The number of cells along each dimension.
  /// Valid cell indices are from 0 .. shape()[dimension], exclusive.
  span<const Index> shape() const { return shape_; }

  /// Returns the points on the grid for dimension r
  span<const Index> inclusive_min(DimensionIndex r) const {
    assert(r >= 0);
    assert(r < rank());
    return inclusive_mins_[r];
  }

  std::vector<Index> cell_origin(span<const Index> indices) const {
    assert(indices.size() == rank());
    std::vector<Index> origin;
    origin.reserve(rank());
    for (size_t i = 0; i < indices.size(); i++) {
      auto x = indices[i];
      if (x < 0) {
        origin.push_back(-kInfIndex);
      } else if (x >= shape_[i]) {
        origin.push_back(kInfIndex);
      } else {
        origin.push_back(inclusive_mins_[i][x]);
      }
    }
    return origin;
  }

 private:
  std::vector<Index> shape_;
  std::vector<std::vector<Index>> inclusive_mins_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_IRREGULAR_GRID_H_
