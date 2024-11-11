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

#include "tensorstore/internal/grid_partition_iterator.h"

#include <stddef.h>

#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/index_space/output_index_map.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/internal/grid_partition_impl.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_grid_partition {

using IndexArraySet = IndexTransformGridPartition::IndexArraySet;
using StridedSet = IndexTransformGridPartition::StridedSet;

PartitionIndexTransformIterator::PartitionIndexTransformIterator(
    internal_grid_partition::IndexTransformGridPartition&& partition_info,
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    OutputToGridCellFn output_to_grid_cell, IndexTransformView<> transform)
    : partition_info_(std::move(partition_info)),
      grid_output_dimensions_(grid_output_dimensions.begin(),
                              grid_output_dimensions.end()),
      output_to_grid_cell_(std::move(output_to_grid_cell)),
      transform_(std::move(transform)),
      at_end_(false),
      cell_transform_(internal_grid_partition::InitializeCellTransform(
          partition_info_, transform_)),
      output_grid_cell_indices_(grid_output_dimensions_.size()),
      position_(rank()),
      upper_bound_(rank()),
      strided_next_position_(partition_info_.strided_sets().size()) {
  // Initialize the output_grid_cell_indices for the constant outputs.
  for (DimensionIndex grid_dim = 0; grid_dim < grid_output_dimensions_.size();
       ++grid_dim) {
    const DimensionIndex output_dim = grid_output_dimensions_[grid_dim];
    const OutputIndexMapRef<> map = transform_.output_index_map(output_dim);
    if (map.method() != OutputIndexMethod::constant) continue;
    output_grid_cell_indices_[grid_dim] =
        output_to_grid_cell_(grid_dim, map.offset(), nullptr);
  }
  // Initialize the iteration positions.
  for (size_t i = 0; i < rank(); ++i) {
    if (i < partition_info_.index_array_sets().size()) {
      ResetIndexArraySet(i);
    } else {
      ResetStridedSet(i);
    }
    at_end_ = at_end_ || (position_[i] == upper_bound_[i]);
  }
  if (!at_end_) {
    for (size_t i = 0; i < rank(); ++i) {
      if (i < partition_info_.index_array_sets().size()) {
        ApplyIndexArraySet(i);
      } else {
        ApplyStridedSet(i);
      }
    }
  }
}

void PartitionIndexTransformIterator::Advance() {
  ABSL_DCHECK(!at_end_);
  // If callers of `cell_transform()` still hold a reference, then make a copy
  // before modifying it.
  cell_transform_ = MutableRep(std::move(cell_transform_));

  // Advance to the next iterator position; this is in c-order and
  // will update strided sets before index array sets.
  size_t i = rank();
  while (i--) {
    // Advance the iteration position for the set at index `i`.
    if (i < partition_info_.index_array_sets().size()) {
      position_[i] = AdvanceIndexArraySet(i);
    } else {
      position_[i] = AdvanceStridedSet(i);
    }
    if (position_[i] == upper_bound_[i]) {
      if (i == 0) break;
      // Reset the iteration position for the set at index `i`
      // and advance to the next set.
      if (i < partition_info_.index_array_sets().size()) {
        ResetIndexArraySet(i);
      } else {
        ResetStridedSet(i);
      }
      continue;
    }
    // Update cell transforms for all updated sets.
    for (; i < rank(); ++i) {
      if (i < partition_info_.index_array_sets().size()) {
        ApplyIndexArraySet(i);
      } else {
        ApplyStridedSet(i);
      }
    }
    return;
  }
  // Iteration has completed.
  at_end_ = true;
}

void PartitionIndexTransformIterator::ResetIndexArraySet(size_t i) {
  ABSL_CHECK_LT(i, partition_info_.index_array_sets().size());
  const IndexArraySet& index_array_set = partition_info_.index_array_sets()[i];
  position_[i] = 0;
  upper_bound_[i] = index_array_set.num_partitions();
}

void PartitionIndexTransformIterator::ApplyIndexArraySet(size_t i) {
  ABSL_CHECK_LT(position_[i], upper_bound_[i]);
  ABSL_CHECK_LT(i, partition_info_.index_array_sets().size());
  const IndexArraySet& index_array_set = partition_info_.index_array_sets()[i];

  // Assign the grid_cell_indices to the precomputed grid cell indices for
  // this partition.
  const Index grid_cell_indices_offset =
      (position_[i]) * index_array_set.grid_dimensions.count();

  DimensionIndex grid_i = 0;
  for (DimensionIndex grid_dim : index_array_set.grid_dimensions.index_view()) {
    output_grid_cell_indices_[grid_dim] =
        index_array_set.grid_cell_indices[grid_cell_indices_offset + grid_i++];
  }
  // Updates the cell_transform for the current index array.
  UpdateCellTransformForIndexArraySetPartition(index_array_set, i, position_[i],
                                               cell_transform_.get());
}

void PartitionIndexTransformIterator::ResetStridedSet(size_t i) {
  ABSL_DCHECK_GE(i, partition_info_.index_array_sets().size());
  auto set_i = i - partition_info_.index_array_sets().size();
  ABSL_DCHECK_LT(set_i, partition_info_.strided_sets().size());

  const auto& strided_set = partition_info_.strided_sets()[set_i];
  const IndexInterval domain =
      transform_.input_domain()[strided_set.input_dimension];
  position_[i] = domain.inclusive_min();
  upper_bound_[i] = domain.exclusive_max();
  strided_next_position_[set_i] = domain.inclusive_min();
}

void PartitionIndexTransformIterator::ApplyStridedSet(size_t i) {
  ABSL_DCHECK_LT(position_[i], upper_bound_[i]);
  ABSL_DCHECK_GE(i, partition_info_.index_array_sets().size());
  auto set_i = i - partition_info_.index_array_sets().size();
  ABSL_DCHECK_LT(set_i, partition_info_.strided_sets().size());

  const StridedSet& strided_set = partition_info_.strided_sets()[set_i];

  IndexInterval restricted_domain =
      IndexInterval::UncheckedHalfOpen(position_[i], upper_bound_[i]);

  // For each grid dimension in the connected set, compute the grid cell
  // index corresponding to `input_index`, and constrain `restricted_domain`
  // to the range of this grid cell.
  for (const DimensionIndex grid_dim :
       strided_set.grid_dimensions.index_view()) {
    const DimensionIndex output_dim = grid_output_dimensions_[grid_dim];
    const OutputIndexMapRef<> map = transform_.output_index_map(output_dim);
    IndexInterval cell_range;
    output_grid_cell_indices_[grid_dim] = output_to_grid_cell_(
        grid_dim, position_[i] * map.stride() + map.offset(), &cell_range);
    // The check in PrePartitionIndexTransformOverGrid guarantees
    // that GetAffineTransformDomain is successful.
    const IndexInterval cell_domain =
        GetAffineTransformDomain(cell_range, map.offset(), map.stride())
            .value();
    restricted_domain = Intersect(restricted_domain, cell_domain);
  }

  ABSL_DCHECK(!restricted_domain.empty());

  // Updates the cell transform input domain of `i`.
  cell_transform_->input_origin()[i] = restricted_domain.inclusive_min();
  cell_transform_->input_shape()[i] = restricted_domain.size();

  strided_next_position_[set_i] = restricted_domain.exclusive_max();
}

}  // namespace internal_grid_partition
namespace internal {

absl::Status PartitionIndexTransformOverGrid(
    tensorstore::span<const DimensionIndex> grid_output_dimensions,
    internal_grid_partition::OutputToGridCellFn output_to_grid_cell,
    IndexTransformView<> transform,
    absl::FunctionRef<
        absl::Status(tensorstore::span<const Index> grid_cell_indices,
                     IndexTransformView<> cell_transform)>
        func) {
  internal_grid_partition::IndexTransformGridPartition partition_info;
  auto status = internal_grid_partition::PrePartitionIndexTransformOverGrid(
      transform, grid_output_dimensions, output_to_grid_cell, partition_info);

  internal_grid_partition::PartitionIndexTransformIterator iterator(
      std::move(partition_info), grid_output_dimensions, output_to_grid_cell,
      transform);
  while (!iterator.AtEnd()) {
    TENSORSTORE_RETURN_IF_ERROR(
        func(iterator.output_grid_cell_indices(), iterator.cell_transform()));
    iterator.Advance();
  }
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tensorstore
