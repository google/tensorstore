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

#include <algorithm>
#include <cassert>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/internal/grid_partition_impl.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_grid_partition {

namespace {

using ::tensorstore::internal_index_space::OutputIndexMap;
using ::tensorstore::internal_index_space::TransformAccess;
using ::tensorstore::internal_index_space::TransformRep;

using IndexArraySet = IndexTransformGridPartition::IndexArraySet;
using StridedSet = IndexTransformGridPartition::StridedSet;

struct ConnectedSetIterateParameters {
  const IndexTransformGridPartition& info;
  span<const DimensionIndex> grid_output_dimensions;
  OutputToGridCellFn output_to_grid_cell;
  IndexTransformView<> transform;
  absl::FunctionRef<absl::Status(span<const Index> grid_cell_indices,
                                 IndexTransformView<> cell_transform)>
      func;
};

/// Allocates the `cell_transform` and initializes the portions that are the
/// same for all grid cells.
///
/// \param info The preprocessed partitioning data.
/// \param full_input_rank The rank of the "full" input space to be partitioned.
/// \returns A non-null pointer to a partially-initialized transform from the
///     synthetic "cell" index space, of rank `cell_input_rank`, to the "full"
///     index space, of rank `full_input_rank`.
internal_index_space::TransformRep::Ptr<> InitializeCellTransform(
    const IndexTransformGridPartition& info, TransformRep* full_transform) {
  const DimensionIndex full_input_rank = full_transform->input_rank;
  DimensionIndex num_index_array_dims = 0;
  for (const IndexArraySet& index_array_set : info.index_array_sets()) {
    num_index_array_dims += index_array_set.input_dimensions.size();
  }
  const DimensionIndex cell_input_rank =
      full_input_rank - num_index_array_dims + info.index_array_sets().size();

  internal_index_space::TransformRep::Ptr<> cell_transform =
      TransformRep::Allocate(cell_input_rank, full_input_rank);
  cell_transform->input_rank = cell_input_rank;
  cell_transform->output_rank = full_input_rank;
  cell_transform->implicit_lower_bounds = false;
  cell_transform->implicit_upper_bounds = false;

  const span<Index> input_origin =
      cell_transform->input_origin().first(cell_input_rank);
  const span<OutputIndexMap> output_maps =
      cell_transform->output_index_maps().first(full_input_rank);

  // Initialize the `cell_transform` output index maps for all input
  // dimensions of the original input space that do affect grid cell indices
  // (i.e. contained in a connected set).
  {
    // Next synthetic input dimension index, corresponding to a connected set.
    // The synthetic input dimensions for index array connected sets come before
    // those for strided connected sets, to match the order of the recursive
    // iteration.
    DimensionIndex cell_input_dim = 0;
    for (const IndexArraySet& index_array_set : info.index_array_sets()) {
      // The `input_origin` is always 0 for the synthetic input dimension
      // corresponding to an index array connected set (in fact the origin is
      // arbitrary and any origin could be used).  While iterating, the
      // `input_shape[cell_input_dim]` will be set appropriately for each
      // partition.
      input_origin[cell_input_dim] = 0;
      for (const DimensionIndex full_input_dim :
           index_array_set.input_dimensions) {
        auto& map = output_maps[full_input_dim];
        // Use an `offset` of `0` and stride of `1`, since the precomputed index
        // arrays correspond directly to the input domains.
        map.offset() = 0;
        map.stride() = 1;
        auto& index_array_data = map.SetArrayIndexing(cell_input_rank);
        std::fill_n(index_array_data.byte_strides, cell_input_rank, 0);
        // Initialize the index array `byte_strides`, which are the same for
        // all partitions.
        index_array_data.byte_strides[cell_input_dim] =
            index_array_set.partitioned_input_indices.byte_strides()[0];
      }
      ++cell_input_dim;
    }

    // The output index maps corresponding to the original input dimensions in
    // strided connected sets do not depend on the partition.
    for (const auto& strided_set : info.strided_sets()) {
      auto& map = output_maps[strided_set.input_dimension];
      map.SetSingleInputDimension(cell_input_dim);
      // Use an `offset` of `0`.  The actual starting index into the original
      // input dimension will be set as `input_origin[cell_input_dim]`.
      map.offset() = 0;
      // Use a `stride` of `1`, to not skip any part of the original input
      // domain.
      map.stride() = 1;
      ++cell_input_dim;
    }
  }

  // Set up the `cell_transform` output index maps corresponding to all input
  // dimensions of the original input space that do not affect grid cell
  // indices (i.e. not contained in a connected set).  These output index maps
  // will not be modified.
  for (DimensionIndex cell_input_dim = info.index_array_sets().size() +
                                       info.strided_sets().size(),
                      full_input_dim = 0;
       full_input_dim < full_input_rank; ++full_input_dim) {
    auto& map = output_maps[full_input_dim];
    if (map.method() != OutputIndexMethod::constant) continue;
    map.SetSingleInputDimension(cell_input_dim);
    map.offset() = 0;
    map.stride() = 1;
    cell_transform->input_dimension(cell_input_dim) =
        full_transform->input_dimension(full_input_dim);
    ++cell_input_dim;
  }

  // Invariants checked in InvokeCallback
  return cell_transform;
}

/// Helper class for iterating over the grid cell index vectors and computing
/// the `cell_transform` for each grid cell, based on precomputed
/// `IndexTransformGridPartition` data.
class ConnectedSetIterateHelper {
 public:
  explicit ConnectedSetIterateHelper(ConnectedSetIterateParameters params)
      : params_(std::move(params)),
        grid_cell_indices_(params_.grid_output_dimensions.size()),
        cell_transform_(InitializeCellTransform(
            params_.info,
            internal_index_space::TransformAccess::rep(params_.transform))) {
    InitializeGridCellIndices();
  }

  /// Iterates over all grid cells and invokes the iteration callback function.
  ///
  /// This is implemented by recursively iterating over the partitions of each
  /// connected set.
  absl::Status Iterate() { return IterateOverIndexArraySets(0); }

 private:
  /// Sets the fixed grid cell indices for all grid dimensions that do not
  /// depend on any input dimensions (i.e. not contained in a connected set).
  /// These grid cell indices will not be modified.
  ///
  /// This function is used only by the constructor.
  void InitializeGridCellIndices() {
    for (DimensionIndex grid_dim = 0;
         grid_dim < params_.grid_output_dimensions.size(); ++grid_dim) {
      const DimensionIndex output_dim =
          params_.grid_output_dimensions[grid_dim];
      const OutputIndexMapRef<> map =
          params_.transform.output_index_map(output_dim);
      if (map.method() != OutputIndexMethod::constant) continue;
      grid_cell_indices_[grid_dim] =
          params_.output_to_grid_cell(grid_dim, map.offset(), nullptr);
    }
  }

  /// Recursively iterates over the partial grid cells corresponding to the
  /// index array connected sets, starting with `set_i`.
  ///
  /// For each grid cell, updates the `grid_cell_indices` for all grid
  /// dimensions in the connected set and updates the `cell_transform` array
  /// output index maps corresponding to each original input dimension in the
  /// connected set.
  ///
  /// If there are no remaining index array connected sets over which to
  /// recurse, starts recusing over the strided connected sets.
  ///
  /// Iteration is aborted if `InvokeCallback` returns an error.
  ///
  /// \param set_i The next index array connected set over which to iterate, in
  ///     the range `[0, info.index_array_sets().size()]`.
  /// \returns The return value of the last recursively call.
  absl::Status IterateOverIndexArraySets(DimensionIndex set_i) {
    if (set_i == params_.info.index_array_sets().size()) {
      return IterateOverStridedSets(0);
    }
    const IndexArraySet& index_array_set =
        params_.info.index_array_sets()[set_i];
    const span<const DimensionIndex> grid_dimensions =
        index_array_set.grid_dimensions;
    // Iterate over the precomputed partitions.
    for (Index partition_i = 0,
               num_partitions = index_array_set.num_partitions();
         partition_i < num_partitions; ++partition_i) {
      // Assign the grid_cell_indices to the precomputed grid cell indices for
      // this partition.
      const Index grid_cell_indices_offset =
          partition_i * grid_dimensions.size();
      for (DimensionIndex grid_i = 0; grid_i < grid_dimensions.size();
           ++grid_i) {
        const DimensionIndex grid_dim = grid_dimensions[grid_i];
        grid_cell_indices_[grid_dim] =
            index_array_set
                .grid_cell_indices[grid_cell_indices_offset + grid_i];
      }

      // Update the output index maps for the original input dimensions in this
      // connected set to reference the precomputed index array of input indices
      // corresponding to this partition.
      const SharedArray<const Index, 2> partition_input_indices =
          index_array_set.partition_input_indices(partition_i);
      cell_transform_->input_shape()[set_i] =
          partition_input_indices.shape()[0];
      ByteStridedPointer<const Index> partition_input_indices_ptr =
          partition_input_indices.byte_strided_pointer();
      const Index vector_dimension_byte_stride =
          partition_input_indices.byte_strides()[1];
      const span<OutputIndexMap> output_maps =
          cell_transform_->output_index_maps();
      for (DimensionIndex full_input_dim : index_array_set.input_dimensions) {
        internal_index_space::IndexArrayData& index_array_data =
            output_maps[full_input_dim].index_array_data();
        index_array_data.element_pointer = std::shared_ptr<const Index>(
            partition_input_indices.pointer(), partition_input_indices_ptr);
        partition_input_indices_ptr += vector_dimension_byte_stride;
      }
      TENSORSTORE_RETURN_IF_ERROR(IterateOverIndexArraySets(set_i + 1));
    }
    return absl::OkStatus();
  }

  /// Recursively iterates over the partial grid cells corresponding to the
  /// strided connected sets, starting with `set_i`.
  ///
  /// For each grid cell, updates the `grid_cell_indices` for all grid
  /// dimensions in the connected set, and updates the input domain of the
  /// corresponding synthetic input dimension of `cell_transform`.  The output
  /// index maps do not need to be updated.
  ///
  /// If there are no remaining strided sets over which to recurse, just invokes
  /// the iteration callback function.
  ///
  /// Iteration is aborted if `InvokeCallback` returns an error.
  ///
  /// \param set_i The next strided connected set over which to iterate, in the
  ///     range `[0, info.strided_sets().size()]`.
  /// \returns The return value of the last recursive call, or the last call to
  ///     `InvokeCallback`.
  absl::Status IterateOverStridedSets(DimensionIndex set_i) {
    if (set_i == params_.info.strided_sets().size()) return InvokeCallback();
    const StridedSet strided_set = params_.info.strided_sets()[set_i];
    const IndexInterval domain =
        params_.transform.input_domain()[strided_set.input_dimension];
    const DimensionIndex cell_input_dim =
        set_i + params_.info.index_array_sets().size();
    // Compute the grid cells that intersect the original input domain, by
    // starting at the first input index and then iteratively advancing to the
    // next input index not contained in the same partial grid cell.
    for (Index input_index = domain.inclusive_min();
         input_index < domain.exclusive_max();) {
      // The subset of the original input domain that corresponds to the current
      // partial grid cell.
      IndexInterval restricted_domain = domain;
      // For each grid dimension in the connected set, compute the grid cell
      // index corresponding to `input_index`, and constrain `restricted_domain`
      // to the range of this grid cell.
      for (const DimensionIndex grid_dim : strided_set.grid_dimensions) {
        const DimensionIndex output_dim =
            params_.grid_output_dimensions[grid_dim];
        const OutputIndexMapRef<> map =
            params_.transform.output_index_map(output_dim);
        IndexInterval cell_range;
        grid_cell_indices_[grid_dim] = params_.output_to_grid_cell(
            grid_dim, input_index * map.stride() + map.offset(), &cell_range);
        // The check in PrePartitionIndexTransformOverRegularGrid guarantees
        // that GetAffineTransformDomain is successful.
        const IndexInterval cell_domain =
            GetAffineTransformDomain(cell_range, map.offset(), map.stride())
                .value();
        restricted_domain = Intersect(restricted_domain, cell_domain);
      }
      assert(restricted_domain.size() > 0);
      // Set the input domain for `cell_input_dim` for the duration of the
      // subsequent recursive call to IterateOverStridedSets.
      cell_transform_->input_origin()[cell_input_dim] =
          restricted_domain.inclusive_min();
      cell_transform_->input_shape()[cell_input_dim] = restricted_domain.size();
      // Recursively iterate over the next strided connected set.
      TENSORSTORE_RETURN_IF_ERROR(IterateOverStridedSets(set_i + 1));
      input_index = restricted_domain.exclusive_max();
    }
    return absl::OkStatus();
  }

  /// Calls the iteration callback function.
  ///
  /// If an error `absl::Status` is returned, iteration should stop.
  ///
  /// \error Any error returned by the iteration callback function.
  absl::Status InvokeCallback() {
    internal_index_space::DebugCheckInvariants(cell_transform_.get());
    auto status = params_.func(
        grid_cell_indices_,
        TransformAccess::Make<IndexTransformView<>>(cell_transform_.get()));
    // If `func` created and is still holding a reference to `cell_transform_`,
    // we need to make a copy before modifying it.
    cell_transform_ = MutableRep(std::move(cell_transform_));
    return status;
  }

  ConnectedSetIterateParameters params_;

  // Current grid cell index vector `h` as defined in grid_partition.h, modified
  // in place while iterating over all index vectors in `H`.
  absl::FixedArray<Index, internal::kNumInlinedDims> grid_cell_indices_;

  // This stores the current value of `cell_transform[h]`, as defined in
  // grid_partition.h, for `h = grid_cell_indices_`.  This is modified in
  // place while iterating over all values for grid_cell_indices_.
  internal_index_space::TransformRep::Ptr<> cell_transform_;
};

}  // namespace
}  // namespace internal_grid_partition

namespace internal {

absl::Status PartitionIndexTransformOverGrid(
    span<const DimensionIndex> grid_output_dimensions,
    absl::FunctionRef<Index(DimensionIndex, Index, IndexInterval*)>
        output_to_grid_cell,
    IndexTransformView<> transform,
    absl::FunctionRef<absl::Status(span<const Index> grid_cell_indices,
                                   IndexTransformView<> cell_transform)>
        func) {
  std::optional<internal_grid_partition::IndexTransformGridPartition>
      partition_info;
  auto status = internal_grid_partition::PrePartitionIndexTransformOverGrid(
      transform, grid_output_dimensions, output_to_grid_cell, &partition_info);

  if (!status.ok()) return status;
  return internal_grid_partition::ConnectedSetIterateHelper(
             {/*.info=*/*partition_info,
              /*.grid_output_dimensions=*/grid_output_dimensions,
              /*.output_to_grid_cell=*/output_to_grid_cell,
              /*.transform=*/transform,
              /*.func=*/std::move(func)})
      .Iterate();
}

absl::Status PartitionIndexTransformOverRegularGrid(
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> grid_cell_shape, IndexTransformView<> transform,
    absl::FunctionRef<absl::Status(span<const Index> grid_cell_indices,
                                   IndexTransformView<> cell_transform)>
        func) {
  assert(grid_cell_shape.size() == grid_output_dimensions.size());
  internal_grid_partition::RegularGridRef grid{grid_cell_shape};
  return PartitionIndexTransformOverGrid(grid_output_dimensions, grid,
                                         transform, std::move(func));
}

}  // namespace internal
}  // namespace tensorstore
