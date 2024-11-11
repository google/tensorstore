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

#ifndef TENSORSTORE_INTERNAL_GRID_PARTITION_ITERATOR_H_
#define TENSORSTORE_INTERNAL_GRID_PARTITION_ITERATOR_H_

#include <stddef.h>

#include <cassert>

#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/internal/grid_partition_impl.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_grid_partition {

/// For a given DimensionIndex dimension, returns the grid cell index
/// corresponding to the output_index, optionally filling the bounds for the
/// cell.
/// Implemented by `RegularGrid` and `IrregularGrid`, for example.
using OutputToGridCellFn = absl::FunctionRef<Index(
    DimensionIndex grid_dim, Index output_index, IndexInterval* cell_bounds)>;

/// Iterator for constructing the `cell_transform` for each grid cell.
/// Requires IndexTransformGridPartition to be precomputed.
///
/// After construction, the iterator is in an initial state where AtEnd may
/// be true, otherwise  `output_grid_cell_indices()` and `cell_transform()`
/// are valid.
///
/// To advance to the next grid cell, call `Advance()`.
class PartitionIndexTransformIterator {
 public:
  PartitionIndexTransformIterator(
      internal_grid_partition::IndexTransformGridPartition&& partition_info,
      tensorstore::span<const DimensionIndex> grid_output_dimensions,
      OutputToGridCellFn output_to_grid_cell, IndexTransformView<> transform);

  // Indices to the current grid cell.
  tensorstore::span<const Index> output_grid_cell_indices() const {
    return output_grid_cell_indices_;
  }

  /// View of the current cell transform.
  IndexTransformView<> cell_transform() {
    return internal_index_space::TransformAccess::Make<IndexTransformView<>>(
        cell_transform_.get());
  }

  /// Indicates whether iteration has completed.
  /// When false, both cell_transform() and output_grid_cell_indices() are
  /// valid.
  bool AtEnd() const { return at_end_; }

  // Advance the iterator.
  void Advance();

 private:
  size_t rank() const {
    return partition_info_.index_array_sets_.size() +
           partition_info_.strided_sets_.size();
  }

  // Advance the iteration position for the index array set at index `i`.
  Index AdvanceIndexArraySet(size_t i) { return position_[i] + 1; }

  // Reset the iteration position for the index array set at index `i`.
  void ResetIndexArraySet(size_t i);

  // For grid cell, `i`, updates the `output_grid_cell_indices` for the given
  // index array as well as the cell_transform.
  void ApplyIndexArraySet(size_t i);

  // Advance the iteration position for the strided set at index `i`.
  // Assumes that ApplyStridedSet(i) was previously invoked.
  Index AdvanceStridedSet(size_t i) {
    ABSL_DCHECK_GE(i, partition_info_.index_array_sets().size());
    auto set_i = i - partition_info_.index_array_sets().size();
    ABSL_DCHECK_LT(set_i, partition_info_.strided_sets().size());
    return strided_next_position_[set_i];
  }

  // Reset the iteration position for the strided set at index `i`.
  void ResetStridedSet(size_t i);

  // For grid cell, `i`, updates the `output_grid_cell_indices` and the
  // `cell_transform` for the associated strided set.
  void ApplyStridedSet(size_t i);

  internal_grid_partition::IndexTransformGridPartition partition_info_;
  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims>
      grid_output_dimensions_;

  OutputToGridCellFn output_to_grid_cell_;
  IndexTransformView<> transform_;
  bool at_end_;

  // This stores the current value of `cell_transform[h]`, as defined in
  // grid_partition.h, for `h = grid_cell_indices_`.  This is modified in
  // place while iterating over all values for grid_cell_indices_.
  internal_index_space::TransformRep::Ptr<> cell_transform_;

  // Current grid cell index vector `h` as defined in grid_partition.h, modified
  // in place while iterating over all index vectors in `H`.
  absl::FixedArray<Index, internal::kNumInlinedDims> output_grid_cell_indices_;

  // Iteration position for each connected set.
  absl::FixedArray<Index, internal::kNumInlinedDims> position_;
  absl::FixedArray<Index, internal::kNumInlinedDims> upper_bound_;

  // The next start position for each strided set.
  absl::FixedArray<Index, internal::kNumInlinedDims> strided_next_position_;
};

}  // namespace internal_grid_partition
namespace internal {

/// Partitions the input domain of a given `transform` from an input space
/// "full" to an output space "output" based on the grid (potentially irregular)
/// specified by `output_to_grid_cell`, which maps from a given dimension and
/// output_index to a grid cell and optional cell bounds.
///
/// For each grid cell index vector `h` in `H`, calls
///   `func(h, cell_transform[h])`.
///
/// To partition over a regular grid, `output_to_grid_cell` can be
///   internal_grid_partition::RegularGridRef.
///
/// \param grid_output_dimensions The sequence of dimensions of the index space
///     "output" corresponding to the grid by which to partition "full".
/// \param output_to_grid_cell    Function returning, for the provided grid
///     dimension, the cell index corresponding to output_index, optionally
///     filling the bounds for the cell.
/// \param transform The index transform from "full" to "output".  Must be
///     valid.
/// \param func The function to be called for each partition.  May return an
///     error `absl::Status` to abort the iteration.
/// \returns `absl::Status()` on success, or the last error returned by `func`.
/// \error `absl::StatusCode::kInvalidArgument` if any input dimension of
///     `transform` has an unbounded domain.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs.
/// \error `absl::StatusCode::kOutOfRange` if an index array contains an
///     out-of-bounds index.
absl::Status PartitionIndexTransformOverGrid(
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    internal_grid_partition::OutputToGridCellFn output_to_grid_cell,
    IndexTransformView<> transform,
    absl::FunctionRef<
        absl::Status(tensorstore::span<const Index> grid_cell_indices,
                     IndexTransformView<> cell_transform)>
        func);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRID_PARTITION_ITERATOR_H_
