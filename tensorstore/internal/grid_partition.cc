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
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/index_space/output_index_map.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/internal/grid_partition_impl.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/regular_grid.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/dimension_set.h"
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

/// Sets the fixed grid cell indices for all grid dimensions that do not
/// depend on any input dimensions (i.e. not contained in a connected set).
void InitializeConstantGridCellIndices(
    IndexTransformView<> transform,
    span<const DimensionIndex> grid_output_dimensions,
    OutputToGridCellFn output_to_grid_cell, span<Index> grid_cell_indices) {
  for (DimensionIndex grid_dim = 0; grid_dim < grid_output_dimensions.size();
       ++grid_dim) {
    const DimensionIndex output_dim = grid_output_dimensions[grid_dim];
    const OutputIndexMapRef<> map = transform.output_index_map(output_dim);
    if (map.method() != OutputIndexMethod::constant) continue;
    grid_cell_indices[grid_dim] =
        output_to_grid_cell(grid_dim, map.offset(), nullptr);
  }
}

// Java-style iterator for computing the grid cells that intersect the original
// input domain for a given `StridedSet`.
//
// To simplify the implementation, this uses a Java-style interface rather than
// a C++ standard library iterator interface.
//
// This computes the grid cells that intersect the original input domain, by
// starting at the first input index and then iteratively advancing to the next
// input index not contained in the same partial grid cell.
class StridedSetGridCellIterator {
 public:
  explicit StridedSetGridCellIterator(
      IndexTransformView<> transform,
      span<const DimensionIndex> grid_output_dimensions,
      OutputToGridCellFn output_to_grid_cell, StridedSet strided_set)
      : transform_(transform),
        grid_output_dimensions_(grid_output_dimensions),
        output_to_grid_cell_(output_to_grid_cell),
        strided_set_(strided_set) {
    const IndexInterval domain =
        transform.input_domain()[strided_set.input_dimension];
    input_index_ = domain.inclusive_min();
    input_end_index_ = domain.exclusive_max();
  }

  bool AtEnd() const { return input_index_ == input_end_index_; }

  IndexInterval Next(span<Index> grid_cell_indices) {
    assert(!AtEnd());
    // The subset of the original input domain that corresponds to the current
    // partial grid cell.
    IndexInterval restricted_domain =
        IndexInterval::UncheckedHalfOpen(input_index_, input_end_index_);
    // For each grid dimension in the connected set, compute the grid cell
    // index corresponding to `input_index`, and constrain `restricted_domain`
    // to the range of this grid cell.
    for (const DimensionIndex grid_dim :
         strided_set_.grid_dimensions.index_view()) {
      const DimensionIndex output_dim = grid_output_dimensions_[grid_dim];
      const OutputIndexMapRef<> map = transform_.output_index_map(output_dim);
      IndexInterval cell_range;
      grid_cell_indices[grid_dim] = output_to_grid_cell_(
          grid_dim, input_index_ * map.stride() + map.offset(), &cell_range);
      // The check in PrePartitionIndexTransformOverRegularGrid guarantees
      // that GetAffineTransformDomain is successful.
      const IndexInterval cell_domain =
          GetAffineTransformDomain(cell_range, map.offset(), map.stride())
              .value();
      restricted_domain = Intersect(restricted_domain, cell_domain);
    }
    assert(!restricted_domain.empty());
    input_index_ = restricted_domain.exclusive_max();
    return restricted_domain;
  }

 private:
  IndexTransformView<> transform_;
  span<const DimensionIndex> grid_output_dimensions_;
  OutputToGridCellFn output_to_grid_cell_;
  StridedSet strided_set_;
  Index input_index_;
  Index input_end_index_;
};

/// Helper class for iterating over the grid cell index vectors and computing
/// the `cell_transform` for each grid cell, based on precomputed
/// `IndexTransformGridPartition` data.
class ConnectedSetIterateHelper {
 public:
  explicit ConnectedSetIterateHelper(ConnectedSetIterateParameters params)
      : params_(std::move(params)),
        grid_cell_indices_(params_.grid_output_dimensions.size()),
        cell_transform_(internal_grid_partition::InitializeCellTransform(
            params_.info, params_.transform)) {
    InitializeConstantGridCellIndices(
        params_.transform, params_.grid_output_dimensions,
        params_.output_to_grid_cell, grid_cell_indices_);
  }

  /// Iterates over all grid cells and invokes the iteration callback function.
  ///
  /// This is implemented by recursively iterating over the partitions of each
  /// connected set.
  absl::Status Iterate() { return IterateOverIndexArraySets(0); }

 private:
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
    const auto grid_dimensions = index_array_set.grid_dimensions;
    const DimensionIndex num_grid_dimensions = grid_dimensions.count();
    // Iterate over the precomputed partitions.
    for (Index partition_i = 0,
               num_partitions = index_array_set.num_partitions();
         partition_i < num_partitions; ++partition_i) {
      // Assign the grid_cell_indices to the precomputed grid cell indices for
      // this partition.
      const Index grid_cell_indices_offset = partition_i * num_grid_dimensions;
      DimensionIndex grid_i = 0;
      for (DimensionIndex grid_dim : grid_dimensions.index_view()) {
        grid_cell_indices_[grid_dim] =
            index_array_set
                .grid_cell_indices[grid_cell_indices_offset + grid_i++];
      }

      UpdateCellTransformForIndexArraySetPartition(
          index_array_set, set_i, partition_i, cell_transform_.get());
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
    StridedSetGridCellIterator iterator(
        params_.transform, params_.grid_output_dimensions,
        params_.output_to_grid_cell, params_.info.strided_sets()[set_i]);
    const DimensionIndex cell_input_dim =
        set_i + params_.info.index_array_sets().size();
    while (!iterator.AtEnd()) {
      auto restricted_domain = iterator.Next(grid_cell_indices_);
      // Set the input domain for `cell_input_dim` for the duration of the
      // subsequent recursive call to IterateOverStridedSets.
      cell_transform_->input_origin()[cell_input_dim] =
          restricted_domain.inclusive_min();
      cell_transform_->input_shape()[cell_input_dim] = restricted_domain.size();
      // Recursively iterate over the next strided connected set.
      TENSORSTORE_RETURN_IF_ERROR(IterateOverStridedSets(set_i + 1));
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

bool GetStridedGridCellRanges(
    IndexTransformView<> transform, OutputToGridCellFn output_to_grid_cell,
    DimensionIndex grid_dim, DimensionIndex output_dim,
    absl::FunctionRef<bool(IndexInterval grid_cell_range)> callback) {
  const auto output_map = transform.output_index_maps()[output_dim];
  assert(output_map.method() == OutputIndexMethod::single_input_dimension);
  const Index output_offset = output_map.offset();
  const Index output_stride = output_map.stride();
  const DimensionIndex input_dim = output_map.input_dimension();
  const IndexInterval input_domain = transform.domain().box()[input_dim];
  if (output_map.stride() == 1 || output_map.stride() == -1) {
    // In unit stride case, it is guaranteed that every cell is in the range.
    // Therefore, there is no need to iterate over grid cells.

    // The check in `PrePartitionIndexTransformOverGrid` guarantees that
    // `GetAffineTransformRange` is successful.
    auto output_range = tensorstore::GetAffineTransformRange(
                            input_domain, output_offset, output_stride)
                            .value();
    Index min_cell_index =
        output_to_grid_cell(grid_dim, output_range.inclusive_min(), nullptr);
    Index max_cell_index =
        output_to_grid_cell(grid_dim, output_range.inclusive_max(), nullptr);
    return callback(
        IndexInterval::UncheckedClosed(min_cell_index, max_cell_index));
  }

  IndexInterval prev_interval;

  // In general case, we must iterate over grid cells.
  for (Index input_index = input_domain.inclusive_min();
       input_index < input_domain.exclusive_max();) {
    IndexInterval output_range;
    Index grid_cell = output_to_grid_cell(
        grid_dim, input_index * output_stride + output_offset, &output_range);
    const IndexInterval cell_domain =
        GetAffineTransformDomain(output_range, output_offset, output_stride)
            .value();
    assert(!cell_domain.empty());
    if (grid_cell == prev_interval.exclusive_min() ||
        grid_cell == prev_interval.exclusive_max()) {
      prev_interval = IndexInterval::UncheckedClosed(
          std::min(prev_interval.inclusive_min(), grid_cell),
          std::max(prev_interval.inclusive_max(), grid_cell));
    } else {
      if (IsFinite(prev_interval)) {
        if (!callback(prev_interval)) return false;
      }
      prev_interval = IndexInterval::UncheckedClosed(grid_cell, grid_cell);
    }
    input_index = cell_domain.exclusive_max();
  }

  return callback(prev_interval);
}

struct GetGridCellRangesIterateParameters {
  const IndexTransformGridPartition& info;
  span<const DimensionIndex> grid_output_dimensions;
  OutputToGridCellFn output_to_grid_cell;
  IndexTransformView<> transform;
  absl::FunctionRef<absl::Status(BoxView<> bounds)> func;
  DimensionIndex outer_prefix_rank;
  BoxView<> grid_bounds;
  span<const IndexInterval> inner_intervals;
  span<const StridedSet*> strided_sets_in_prefix;
  span<const IndexArraySet*> index_array_sets_in_prefix;
};

class GetGridCellRangesIterateHelper {
 public:
  explicit GetGridCellRangesIterateHelper(
      GetGridCellRangesIterateParameters params)
      : params_(params) {
    InitializeConstantGridCellIndices(
        params_.transform, params_.grid_output_dimensions,
        params_.output_to_grid_cell,
        span<Index>(&grid_bounds_origin_[0], params_.transform.output_rank()));
    for (DimensionIndex i = 0; i < params.outer_prefix_rank; ++i) {
      grid_bounds_shape_[i] = 1;
    }
    for (DimensionIndex i = params.outer_prefix_rank + 1,
                        rank = params.grid_bounds.rank();
         i < rank; ++i) {
      grid_bounds_origin_[i] = params.grid_bounds.origin()[i];
      grid_bounds_shape_[i] = params.grid_bounds.shape()[i];
    }
    if (params.inner_intervals.size() == 1) {
      const auto& inner_interval = params.inner_intervals[0];
      grid_bounds_origin_[params.outer_prefix_rank] =
          inner_interval.inclusive_min();
      grid_bounds_shape_[params.outer_prefix_rank] = inner_interval.size();
    }
  }

  /// Iterates over all grid cells and invokes the iteration callback function.
  ///
  /// This is implemented by recursively iterating over the partitions of each
  /// connected set.
  absl::Status Iterate() { return IterateOverIndexArraySets(0); }

 private:
  GetGridCellRangesIterateParameters params_;
  Index grid_bounds_origin_[kMaxRank];
  Index grid_bounds_shape_[kMaxRank];

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
    if (set_i == params_.index_array_sets_in_prefix.size()) {
      return IterateOverStridedSets(0);
    }
    const IndexArraySet& index_array_set =
        *params_.index_array_sets_in_prefix[set_i];
    const auto grid_dimensions = index_array_set.grid_dimensions;
    const DimensionIndex num_grid_dimensions = grid_dimensions.count();
    // Iterate over the precomputed partitions.
    for (Index partition_i = 0,
               num_partitions = index_array_set.num_partitions();
         partition_i < num_partitions; ++partition_i) {
      // Assign the grid_cell_indices to the precomputed grid cell indices for
      // this partition.
      const Index grid_cell_indices_offset = partition_i * num_grid_dimensions;
      DimensionIndex grid_i = 0;
      for (DimensionIndex grid_dim : grid_dimensions.index_view()) {
        grid_bounds_origin_[grid_dim] =
            index_array_set
                .grid_cell_indices[grid_cell_indices_offset + grid_i++];
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
    if (set_i == params_.strided_sets_in_prefix.size()) return InvokeCallback();
    StridedSetGridCellIterator iterator(
        params_.transform, params_.grid_output_dimensions,
        params_.output_to_grid_cell, *params_.strided_sets_in_prefix[set_i]);
    while (!iterator.AtEnd()) {
      iterator.Next(grid_bounds_origin_);
      // Recursively iterate over the next strided connected set.
      TENSORSTORE_RETURN_IF_ERROR(IterateOverStridedSets(set_i + 1));
    }
    return absl::OkStatus();
  }

  /// Calls the iteration callback function.
  ///
  /// If an error `absl::Status` is returned, iteration should stop.
  ///
  /// \error Any error returned by the iteration callback function.
  absl::Status InvokeCallback() {
    MutableBoxView<> bounds(params_.grid_bounds.rank(), grid_bounds_origin_,
                            grid_bounds_shape_);
    if (params_.inner_intervals.size() == 1) {
      return params_.func(bounds);
    }
    DimensionIndex outer_prefix_rank = params_.outer_prefix_rank;
    for (const auto& inner_interval : params_.inner_intervals) {
      bounds[outer_prefix_rank] = inner_interval;
      TENSORSTORE_RETURN_IF_ERROR(params_.func(bounds));
    }
    return absl::OkStatus();
  }
};

}  // namespace
}  // namespace internal_grid_partition

namespace internal {

absl::Status PartitionIndexTransformOverGrid(
    span<const DimensionIndex> grid_output_dimensions,
    internal_grid_partition::OutputToGridCellFn output_to_grid_cell,
    IndexTransformView<> transform,
    absl::FunctionRef<absl::Status(span<const Index> grid_cell_indices,
                                   IndexTransformView<> cell_transform)>
        func) {
  internal_grid_partition::IndexTransformGridPartition partition_info;
  auto status = internal_grid_partition::PrePartitionIndexTransformOverGrid(
      transform, grid_output_dimensions, output_to_grid_cell, partition_info);

  if (!status.ok()) return status;
  return internal_grid_partition::ConnectedSetIterateHelper(
             {/*.info=*/partition_info,
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

namespace internal_grid_partition {
absl::Status GetGridCellRanges(
    const IndexTransformGridPartition& grid_partition,
    span<const DimensionIndex> grid_output_dimensions, BoxView<> grid_bounds,
    OutputToGridCellFn output_to_grid_cell, IndexTransformView<> transform,
    absl::FunctionRef<absl::Status(BoxView<> bounds)> callback) {
  assert(grid_output_dimensions.size() == grid_bounds.rank());

  if (transform.domain().box().is_empty()) {
    // Domain is empty, maps to no grid cells.
    return absl::OkStatus();
  }

  if (grid_output_dimensions.empty()) {
    // Only a single grid cell.
    return callback({});
  }

  std::array<DimensionIndex, kMaxRank> dim_to_indexed_set;
  dim_to_indexed_set.fill(-1);

  // Grid dimensions that are in a one-to-one correspondence with an input
  // dimension.
  DimensionSet one_to_one_grid_dims;
  for (const auto& strided_set : grid_partition.strided_sets()) {
    if (strided_set.grid_dimensions.count() != 1) {
      continue;
    }
    const DimensionIndex grid_dim =
        strided_set.grid_dimensions.index_view().front();
    one_to_one_grid_dims[grid_dim] = true;
  }

  for (size_t i = 0; i < grid_partition.index_array_sets().size(); ++i) {
    const auto& set = grid_partition.index_array_sets()[i];
    if (set.grid_dimensions.count() != 1) {
      continue;
    }
    const DimensionIndex grid_dim = set.grid_dimensions.index_view().front();
    one_to_one_grid_dims[grid_dim] = true;
    dim_to_indexed_set[grid_dim] = i;
  }

  absl::InlinedVector<IndexInterval, 1> inner_intervals;

  DimensionSet grid_dimensions_outside_prefix;

  DimensionIndex range_queryable_grid_dim = grid_output_dimensions.size() - 1;
  for (; range_queryable_grid_dim >= 0; --range_queryable_grid_dim) {
    // Check if `range_queryable_grid_dim` is constrained.
    //
    // If it is constrained, then it will have to be the dimension over which
    // the intervals vary, and all outer dimensions will have to be fixed for
    // each interval.
    //
    // If it is not constrained, then it can be ignored.

    const DimensionIndex grid_dim = range_queryable_grid_dim;
    const IndexInterval grid_interval = grid_bounds[grid_dim];
    if (grid_interval.size() == 1) {
      // Only a single grid cell, therefore always unconstrained.
      inner_intervals.clear();
      inner_intervals.push_back(grid_interval);
      continue;
    }

    if (!one_to_one_grid_dims[grid_dim]) {
      // `grid_dim` is not in a one-to-one relationship with an input dimension,
      // and therefore the bounds can't be represented by a simple interval.
      break;
    }

    grid_dimensions_outside_prefix[grid_dim] = true;

    const DimensionIndex output_dim = grid_output_dimensions[grid_dim];

    inner_intervals.clear();

    DimensionIndex indexed_set_i = dim_to_indexed_set[grid_dim];
    if (indexed_set_i == -1) {
      internal_grid_partition::GetStridedGridCellRanges(
          transform, output_to_grid_cell, grid_dim, output_dim,
          [&](IndexInterval grid_cell_range) {
            inner_intervals.push_back(grid_cell_range);
            return true;
          });
    } else {
      const auto& set = grid_partition.index_array_sets()[indexed_set_i];
      const auto& grid_cell_indices = set.grid_cell_indices;
      size_t i = 0;
      while (i < grid_cell_indices.size()) {
        size_t last_i = i;
        while (last_i + 1 < grid_cell_indices.size() &&
               grid_cell_indices[last_i] + 1 == grid_cell_indices[last_i + 1]) {
          ++last_i;
        }
        inner_intervals.push_back(IndexInterval::UncheckedClosed(
            grid_cell_indices[i], grid_cell_indices[last_i]));
        i = last_i + 1;
      }
    }
    if (inner_intervals.size() == 1 &&
        tensorstore::Contains(inner_intervals[0], grid_interval)) {
      // Dimension is unconstrained.
      inner_intervals.clear();
      inner_intervals.push_back(grid_interval);
      continue;
    }

    // Dimension is constrained.
    --range_queryable_grid_dim;
    break;
  }

  const StridedSet* strided_sets_in_prefix_storage[kMaxRank];
  const IndexArraySet* index_array_sets_in_prefix_storage[kMaxRank];

  // Get the reduced list of sets that are part of the outer prefix.
  const auto get_sets_in_prefix = [&](auto sets, auto* buffer) {
    ptrdiff_t i = 0;
    for (const auto& set : sets) {
      if (grid_dimensions_outside_prefix[set.grid_dimensions.index_view()
                                             .front()]) {
        continue;
      }
      buffer[i++] = &set;
    }
    return span(buffer, i);
  };

  auto strided_sets_in_prefix = get_sets_in_prefix(
      grid_partition.strided_sets(), strided_sets_in_prefix_storage);
  auto index_array_sets_in_prefix = get_sets_in_prefix(
      grid_partition.index_array_sets(), index_array_sets_in_prefix_storage);

  if (range_queryable_grid_dim == grid_output_dimensions.size() - 1) {
    inner_intervals.push_back(grid_bounds[range_queryable_grid_dim]);
  }

  internal_grid_partition::GetGridCellRangesIterateHelper iterate_helper(
      internal_grid_partition::GetGridCellRangesIterateParameters{
          grid_partition, grid_output_dimensions, output_to_grid_cell,
          transform, callback, range_queryable_grid_dim + 1, grid_bounds,
          inner_intervals, strided_sets_in_prefix, index_array_sets_in_prefix});
  return iterate_helper.Iterate();
}
}  // namespace internal_grid_partition

namespace internal {

absl::Status GetGridCellRanges(
    span<const DimensionIndex> grid_output_dimensions, BoxView<> grid_bounds,
    internal_grid_partition::OutputToGridCellFn output_to_grid_cell,
    IndexTransformView<> transform,
    absl::FunctionRef<absl::Status(BoxView<> bounds)> callback) {
  using internal_grid_partition::StridedSet;

  assert(grid_output_dimensions.size() == grid_bounds.rank());

  if (transform.domain().box().is_empty()) {
    // Domain is empty, maps to no grid cells.
    return absl::OkStatus();
  }

  if (grid_output_dimensions.empty()) {
    // Only a single grid cell.
    return callback({});
  }

  internal_grid_partition::IndexTransformGridPartition grid_partition;
  TENSORSTORE_RETURN_IF_ERROR(
      internal_grid_partition::PrePartitionIndexTransformOverGrid(
          transform, grid_output_dimensions, output_to_grid_cell,
          grid_partition));
  return internal_grid_partition::GetGridCellRanges(
      grid_partition, grid_output_dimensions, grid_bounds, output_to_grid_cell,
      transform, callback);
}

}  // namespace internal
}  // namespace tensorstore
