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

#ifndef TENSORSTORE_INTERNAL_GRID_PARTITION_H_
#define TENSORSTORE_INTERNAL_GRID_PARTITION_H_

/// \file
/// Facilities for partitioning an index transform according to a grid defined
/// over a subset of the output dimensions.
///
/// This partitioning is defined based on:
///
/// - an index transform `original_transform` from an input space `I` to an
///   output space `O`;
///
/// - for each grid dimension `g` in the subset `G` of output dimensions in `O`,
///   a map `M[g]` that maps an output index to a grid index.  We define
///   `M[G](v)` to be the index vector defined by `g in G -> M[g](v[g])`.  In
///   the common case of a regular grid, `M[g](x) = floor(x / cell_size[g])`,
///   where `cell_size[g]` is the size along output dimension `g` of each grid
///   cell.
///
/// To partition the `original_transform`, we compute:
///
/// - the set of grid cell index vectors `H = range(M[G] * original_transform)`;
///
/// - for each grid cell index vector `h` in `H`, we compute an index transform
///   `cell_transform[h]` from a synthetic input domain `C` to `I` such that:
///
///     a. `cell_transform[h]` is one-to-one;
///     b. `domain(cell_transform[h]) = (M[G] * original_transform)^(-1)(h)`;
///     b. `range(M[G] * original_transform * cell_transform[h]) = { h }`.
///
/// To compute the partition efficiently, we decompose the problem based on the
/// "connected sets" of input dimensions (of `I`) and grid dimensions (in `G`).
/// These "connected sets" are defined based on the bipartite graph containing
/// input dimensions and grid dimensions, where there is an edge between a grid
/// dimension (corresponding to a given output dimension) and an input dimension
/// if, and only if, the output dimension depends on the input dimension by
/// either a `single_input_dimension` output index map or an `array` output
/// index map with a non-zero byte stride.  Note that output dimensions of
/// `original_transform` that are not contained in `G` have no effect on the
/// connected sets.
///
/// Each connected set is classified as one of two types:
///
/// - A "strided connected set" has only `single_input_dimension` edges, and
///   must therefore contain exactly one input dimension (but possibly more than
///   one grid dimension).  These connected sets can be handled very efficiently
///   and require no extra temporary memory.
///
/// - An "index array connected set" has at least one `array` edge and may
///   contain multiple input and grid dimensions.  These connected sets require
///   an amount of temporary memory proportional to the number of grid
///   dimensions in the set multiplied by the product of the sizes of the
///   domains of the input dimensions in the set.  While the partitioning
///   procedure would work even if all grid and input dimensions were placed in
///   a single index array set, the decomposition into connected sets can
///   greatly improve the efficiency by reducing the amount of temporary memory
///   required.
///
/// By definition, each connected set corresponds to a subset `Gs` of `G`, a
/// subset `Is` of input dimensions of `I`, and a set `Hs = { h[Gs] : h in H }`
/// of "partial" grid cell index vectors restricted to `Gs`.  Additionally, we
/// compute each `cell_transform[h]` such that each connected set corresponds to
/// a single input dimension `c` of the synthetic input domain `C`:
///
/// - For strided connected sets, `c` simply corresponds to the single input
///   dimension contained in the set.
///
/// - For index array connected sets, `c` corresponds to a flattening of all of
///   the input dimensions contained in the set.
///
/// For each connected set and partial grid cell index vector `hs`, we define
/// `cell_sub_transform[hs]` from `{ c }` to `Is`, such that
/// `cell_transform[h] = \prod_{Gs} cell_sub_transform[h[Gs]]`.
///
/// In practice, we compute the set of partial grid cell indices `H[Gs]` for
/// each connected set, which is represented implicitly for strided connected
/// sets and explicitly for index array connected sets, and then compute `H` as
/// the outer product of `H[Gs]` over all connected sets.
///
/// NOTE(jbms): Currently only regular grids are supported.  Support for
/// irregular grids will be added in order to support virtual concatenated views
/// of multiple tensorstores.

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

/// Partitions the input domain of a given `transform` from an input space
/// "full" to an output space "output" based on the specified regular grid over
/// "output".
///
/// For each grid cell index vector `h` in `H`, calls
/// `func(h, cell_transform[h])`.
///
/// \param grid_output_dimensions The sequence of dimensions of the index space
///     "output" corresponding to the grid by which to partition "full".
/// \param grid_cell_shape The shape of a grid cell.  Each
///     `grid_cell_shape[grid_dim]` value specifies the size of a grid cell in
///     output dimension `grid_output_dimensions[grid_i]`.
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
absl::Status PartitionIndexTransformOverRegularGrid(
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> grid_cell_shape, IndexTransformView<> transform,
    absl::FunctionRef<absl::Status(span<const Index> grid_cell_indices,
                                   IndexTransformView<> cell_transform)>
        func);

/// Partitions the input domain of a given `transform` from an input space
/// "full" to an output space "output" based on potentially irregular grid
/// specified by `output_to_grid_cell`, which maps from a given dimension and
/// output_index to a grid cell and optional cell bounds.
///
/// For each grid cell index vector `h` in `H`, calls
///   `func(h, cell_transform[h])`.
///
absl::Status PartitionIndexTransformOverGrid(
    span<const DimensionIndex> grid_output_dimensions,
    absl::FunctionRef<Index(DimensionIndex grid_dim, Index output_index,
                            IndexInterval* cell_bounds)>
        output_to_grid_cell,
    IndexTransformView<> transform,
    absl::FunctionRef<absl::Status(span<const Index> grid_cell_indices,
                                   IndexTransformView<> cell_transform)>
        func);

absl::Status GetGridCellRanges(
    span<const DimensionIndex> grid_output_dimensions, BoxView<> grid_bounds,
    absl::FunctionRef<Index(DimensionIndex grid_dim, Index output_index,
                            IndexInterval* cell_bounds)>
        output_to_grid_cell,
    IndexTransformView<> transform,
    absl::FunctionRef<absl::Status(BoxView<> bounds)> callback);

}  // namespace internal

namespace internal_grid_partition {
class IndexTransformGridPartition;

// Computes the set of grid cells that intersect the output range of
// `transform`, and returns them as a set of lexicographical ranges.
//
// This computes the same set of grid cells as
// `PartitionIndexTransformOverGrid`, but differs in that it does not compute
// the `cell_transform` for each grid cell, and combines grid cells into ranges
// when possible.
//
// The `cell_transform` may be obtained for a given grid cell by calling
// `grid_partition.GetCellTransform`.
//
// Args:
//   grid_partition: Must have been previously initialized by a call to
//     `PrePartitionIndexTransformOverGrid` with the same `transform`,
//     `grid_output_dimensions`, and `output_to_grid_cell`.
//   grid_output_dimensions: Output dimensions of `transform` corresponding to
//     each grid dimension.
//   grid_bounds: Bounds of grid indices along each dimension.
//   output_to_grid_cell: Computes the grid cell corresponding to a given output
//     index.
//   transform: Index transform.
//   callback: Called for each grid cell range.  Any error return aborts
//     iteration and is propagated.  While specified as an arbitrary box, the
//     bounds passed to the callback are guaranteed to specify a lexicographical
//     range that is a subset of `grid_bounds`, i.e. `bounds[i].size() == 1` for
//     all `0 < m`, and `bounds[i] == grid_bounds[i]` for all `i >= n`, where
//     `0 <= m <= n <= grid_bounds.rank()`.
absl::Status GetGridCellRanges(
    const IndexTransformGridPartition& grid_partition,
    span<const DimensionIndex> grid_output_dimensions, BoxView<> grid_bounds,
    absl::FunctionRef<Index(DimensionIndex grid_dim, Index output_index,
                            IndexInterval* cell_bounds)>
        output_to_grid_cell,
    IndexTransformView<> transform,
    absl::FunctionRef<absl::Status(BoxView<> bounds)> callback);
}  // namespace internal_grid_partition

}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRID_PARTITION_H_
