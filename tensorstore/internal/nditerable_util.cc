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

#include "tensorstore/internal/nditerable_util.h"

#include <stddef.h>

#include <algorithm>
#include <cassert>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

namespace {

#ifndef NDEBUG
bool nditerable_use_unit_block_size = false;
#endif

template <bool Full>
void GetNDIterationLayoutInfo(const NDIterableLayoutConstraint& iterable,
                              span<const Index> shape,
                              IterationConstraints constraints,
                              NDIterationLayoutInfo<Full>* info) {
  info->shape.assign(shape.begin(), shape.end());
  info->directions.resize(shape.size());
  info->iteration_dimensions.clear();
  info->iteration_shape.clear();

  if constexpr (Full) {
    info->full_iteration_dimensions.clear();
  }

  info->empty = false;

  using DirectionPref = NDIterableLayoutConstraint::DirectionPref;

  DirectionPref direction_prefs[kMaxRank];
  std::fill_n(
      direction_prefs, shape.size(),
      constraints.repeated_elements_constraint() == skip_repeated_elements
          ? DirectionPref::kCanSkip
          : DirectionPref::kEither);
  iterable.UpdateDirectionPrefs(direction_prefs);

  for (DimensionIndex dim_i = 0; dim_i < shape.size(); ++dim_i) {
    const Index size = shape[dim_i];
    if (size == 0) {
      info->empty = true;
    } else if ((size == 1 &&
                direction_prefs[dim_i] != DirectionPref::kForwardRequired) ||
               direction_prefs[dim_i] == DirectionPref::kCanSkip) {
      if constexpr (Full) {
        info->full_iteration_dimensions.push_back(dim_i);
      }
      continue;
    }
    info->iteration_dimensions.push_back(dim_i);
  }

  if (info->iteration_dimensions.empty()) {
    // Add an inert dimension for rank 0 case.
    info->iteration_dimensions.push_back(-1);
    info->iteration_dimensions.push_back(-1);
    info->iteration_shape.push_back(1);
    info->iteration_shape.push_back(1);
  } else {
    if (constraints.order_constraint() == ContiguousLayoutOrder::fortran) {
      std::reverse(info->iteration_dimensions.begin(),
                   info->iteration_dimensions.end());
    } else if (constraints.order_constraint() == unspecified_order) {
      std::sort(info->iteration_dimensions.begin(),
                info->iteration_dimensions.end(),
                [&](DimensionIndex dim_i, DimensionIndex dim_j) {
                  return iterable.GetDimensionOrder(dim_i, dim_j) < 0;
                });
    }

    DimensionIndex dim_i = info->iteration_dimensions[0];
    Index size_i = shape[dim_i];
    info->iteration_shape.push_back(size_i);
    int dir_i =
        NDIterableLayoutConstraint::GetDirection(direction_prefs[dim_i]);
    info->directions[dim_i] = dir_i;
    auto next_iteration_dim_it = info->iteration_dimensions.begin();
    if constexpr (Full) {
      info->full_iteration_dimensions.push_back(dim_i);
    }
    for (DimensionIndex i = 1;
         i < static_cast<DimensionIndex>(info->iteration_dimensions.size());
         ++i) {
      DimensionIndex dim_j = info->iteration_dimensions[i];
      Index size_j = shape[dim_j];
      int dir_j =
          NDIterableLayoutConstraint::GetDirection(direction_prefs[dim_j]);
      info->directions[dim_j] = dir_j;
      if constexpr (Full) {
        info->full_iteration_dimensions.push_back(dim_j);
      }
      Index size_combined;
      if (iterable.CanCombineDimensions(dim_i, dir_i, dim_j, dir_j, size_j) &&
          !MulOverflow(size_i, size_j, &size_combined)) {
        size_j = size_combined;
        info->iteration_shape.back() = size_combined;
      } else {
        info->iteration_shape.push_back(size_j);
        ++next_iteration_dim_it;
      }
      *next_iteration_dim_it = dim_j;
      dim_i = dim_j;
      size_i = size_j;
      dir_i = dir_j;
    }
    info->iteration_dimensions.erase(next_iteration_dim_it + 1,
                                     info->iteration_dimensions.end());
  }
  if (info->iteration_dimensions.size() < 2) {
    assert(info->iteration_dimensions.size() == 1);
    info->iteration_dimensions.insert(info->iteration_dimensions.begin(), -1);
    info->iteration_shape.insert(info->iteration_shape.begin(), 1);
  }
}
}  // namespace

void GetNDIterationLayoutInfo(const NDIterableLayoutConstraint& iterable,
                              span<const Index> shape,
                              IterationConstraints constraints,
                              NDIterationSimplifiedLayoutInfo* info) {
  GetNDIterationLayoutInfo<false>(iterable, shape, constraints, info);
}

void GetNDIterationLayoutInfo(const NDIterableLayoutConstraint& iterable,
                              span<const Index> shape,
                              IterationConstraints constraints,
                              NDIterationFullLayoutInfo* info) {
  GetNDIterationLayoutInfo<true>(iterable, shape, constraints, info);
}

IterationBufferShape GetNDIterationBlockShape(
    ptrdiff_t working_memory_bytes_per_element,
    span<const Index> iteration_shape) {
#ifdef TENSORSTORE_INTERNAL_NDITERABLE_TEST_UNIT_BLOCK_SIZE
  return {1, 1};
#else
#if !defined(NDEBUG)
  if (nditerable_use_unit_block_size) {
    return {1, 1};
  }
#endif
  // TODO(jbms): maybe choose based on actual L1 cache size.
  //
  // Note: Choose an amount smaller than the default arena size of `32 * 1024`.
  constexpr Index kTargetMemoryUsage = 24 * 1024;
  const Index penultimate_dimension_size =
      iteration_shape[iteration_shape.size() - 2];
  const Index last_dimension_size = iteration_shape[iteration_shape.size() - 1];
  if (working_memory_bytes_per_element == 0) {
    return {penultimate_dimension_size, last_dimension_size};
  } else {
    const Index target_size = std::max(
        Index(8), kTargetMemoryUsage / Index(working_memory_bytes_per_element));
    const Index block_inner_size =
        std::max(Index(1), std::min(last_dimension_size, target_size));
    Index block_outer_size = 1;
    if (block_inner_size < target_size) {
      block_outer_size =
          std::min(penultimate_dimension_size, target_size / block_inner_size);
    }
    return {block_outer_size, block_inner_size};
  }
#endif
}

IterationBufferShape GetNDIterationBlockShape(
    const NDIterableBufferConstraint& iterable,
    NDIterable::IterationLayoutView layout, IterationBufferKind buffer_kind) {
  return GetNDIterationBlockShape(
      iterable.GetWorkingMemoryBytesPerElement(layout, buffer_kind),
      layout.iteration_shape);
}

void GetNDIterationBufferInfo(const NDIterableBufferConstraint& iterable,
                              NDIterable::IterationLayoutView layout,
                              NDIterationBufferInfo* buffer_info) {
  buffer_info->buffer_kind =
      iterable.GetIterationBufferConstraint(layout).min_buffer_kind;
  buffer_info->block_shape =
      GetNDIterationBlockShape(iterable, layout, buffer_info->buffer_kind);
}

#ifndef NDEBUG
void SetNDIterableTestUnitBlockSize(bool value) {
  nditerable_use_unit_block_size = value;
}
#endif

Index UpdatePartialBlock(NDIterator& iterator, span<const Index> indices,
                         IterationBufferShape block_shape,
                         IterationBufferKind buffer_kind,
                         IterationBufferPointer buffer, Index modified_count,
                         absl::Status* status) {
  Index full_rows = modified_count / block_shape[1];
  Index final_row_count = modified_count % block_shape[1];
  Index updated = 0;
  if (full_rows != 0) {
    updated = iterator.UpdateBlock(indices, {full_rows, block_shape[1]}, buffer,
                                   status);
    if (ABSL_PREDICT_FALSE(updated != full_rows * block_shape[1])) {
      return updated;
    }
  }
  if (final_row_count != 0) {
    buffer.AddElementOffset(buffer_kind, full_rows, 0);
    Index final_row_indices[kMaxRank];
    std::copy(indices.begin(), indices.end(), final_row_indices);
    final_row_indices[indices.size() - 2] += full_rows;
    updated += iterator.UpdateBlock(
        span<const Index>(final_row_indices, indices.size()),
        {1, final_row_count}, buffer, status);
  }
  return updated;
}

}  // namespace internal
}  // namespace tensorstore
