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

#include "tensorstore/internal/grid_partition_impl.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/macros.h"
#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_grid_partition {

IndexTransformGridPartition::IndexTransformGridPartition(
    DimensionIndex input_rank, DimensionIndex grid_rank)
    : temp_buffer_(input_rank + grid_rank) {}

SharedArray<const Index, 2>
IndexTransformGridPartition::IndexArraySet::partition_input_indices(
    Index partition_i) const {
  ABSL_ASSERT(partition_i >= 0 && partition_i < num_partitions());
  SharedArray<const Index, 2> result;
  const Index start = grid_cell_partition_offsets[partition_i];
  const Index end =
      static_cast<size_t>(partition_i + 1) == grid_cell_partition_offsets.size()
          ? partitioned_input_indices.shape()[0]
          : grid_cell_partition_offsets[partition_i + 1];
  ABSL_ASSERT(start >= 0 && start < partitioned_input_indices.shape()[0]);
  ABSL_ASSERT(end > start && end <= partitioned_input_indices.shape()[0]);
  result.pointer() =
      std::shared_ptr<const Index>(partitioned_input_indices.pointer(),
                                   &partitioned_input_indices(start, 0));
  result.layout() = partitioned_input_indices.layout();
  result.shape()[0] = end - start;
  return result;
}

span<const Index>
IndexTransformGridPartition::IndexArraySet::partition_grid_cell_indices(
    Index partition_i) const {
  assert(partition_i >= 0 && partition_i < num_partitions());
  assert(grid_cell_indices.size() ==
         static_cast<size_t>(num_partitions() * grid_dimensions.size()));
  return span(&grid_cell_indices[partition_i * grid_dimensions.size()],
              grid_dimensions.size());
}

namespace {

/// Invokes the specified callback for each connected sets of input and grid
/// dimensions of an index transform.
///
/// \param grid_output_dimensions Array that maps grid dimension indices to
///     output dimension indices of the index transform.
/// \param output_index_maps The output index maps of the index transform.
/// \param temp_buffer[out] Array of size
///     `>= output_index_maps.input_rank() + grid_output_dimensions` to be used
///     as a temporary buffer.  The spans passed to `set_callback` will refer to
///     memory in this buffer.
/// \param set_callback Function with a signature compatible with:
///     `Status (span<const DimensionIndex> input_dims,
///              span<const DimensionIndex> grid_dims,
///              bool has_array)`.  This function is called for each connected.
///     Any error returned causes iteration to stop.
/// \return The error value returned by the last call to `set_callback`, or
///     `Status()` on success.
template <typename SetCallbackFn>
Status ForEachConnectedSet(span<const DimensionIndex> grid_output_dimensions,
                           OutputIndexMapRange<> output_index_maps,
                           span<DimensionIndex> temp_buffer,
                           SetCallbackFn set_callback) {
  const DimensionIndex input_rank = output_index_maps.input_rank();
  ABSL_ASSERT(temp_buffer.size() >= input_rank + grid_output_dimensions.size());
  const span<DimensionIndex> input_dims = temp_buffer.first(input_rank);
  const span<DimensionIndex> grid_dims =
      temp_buffer.subspan(input_rank, grid_output_dimensions.size());

  std::iota(input_dims.begin(), input_dims.end(), DimensionIndex(0));
  std::iota(grid_dims.begin(), grid_dims.end(), DimensionIndex(0));

  DimensionIndex num_dependent_grid_dims = grid_output_dimensions.size();

  // Keep track of a current set of grid dimension indices
  // `(grid_dims[grid_dim_set_begin:grid_dim_set_end])` and input dimension
  // indices `(input_dims[input_dim_set_begin:input_dim_set_end])`.
  //
  // Any of the grid dimensions in `grid_dims[:grid_dim_set_begin]` and the
  // input dimensions in `input_dims[:input_dim_set_begin]` are already part of
  // a different connected set and will never be added to the current set.
  //
  // Grid dimensions in `grid_dims[grid_dim_set_end:]` and input dimensions in
  // `input_dims[input_dim_set_end:]` may be added to the current set by
  // swapping them with `grid_dims[grid_dim_set_end++]` or
  // `input_dims[input_dim_set_end++]`, respectively.
  DimensionIndex input_dim_set_end = 0;
  DimensionIndex input_dim_set_begin;
  bool current_set_has_array;

  /// Adds the input dimensions on which the output dimension
  /// `grid_output_dimensions[grid_dims[grid_i]]` depends to the current set.
  ///
  /// Each output dimension `grid_output_dimensions[grid_dim]` depends on zero
  /// or more input dimensions due to a `single_input_dimension` or `array`
  /// output index map.  This adds to the current set any such input dimensions
  /// that are not already in the current set.
  ///
  /// If the dependencies are via an `array` output index map, sets
  /// `current_set_has_array = true`.
  ///
  /// \returns `true` if, and only if, any additional input dimensions were
  ///     added to the current set.
  const auto add_grid_dim_to_current_set = [&](DimensionIndex grid_i) -> bool {
    ABSL_ASSERT(grid_i >= 0 && grid_i < num_dependent_grid_dims);
    const DimensionIndex grid_dim = grid_dims[grid_i];
    ABSL_ASSERT(grid_dim >= 0 && grid_dim < grid_output_dimensions.size());
    const DimensionIndex output_dim = grid_output_dimensions[grid_dim];
    const OutputIndexMapRef<> map = output_index_maps[output_dim];
    switch (map.method()) {
      case OutputIndexMethod::constant:
        return false;
      case OutputIndexMethod::single_input_dimension: {
        const auto it = std::find(input_dims.begin() + input_dim_set_end,
                                  input_dims.end(), map.input_dimension());
        if (it != input_dims.end()) {
          using std::swap;
          std::swap(*it, input_dims[input_dim_set_end++]);
          return true;
        }
        return false;
      }
      case OutputIndexMethod::array: {
        const OutputIndexMapRef<>::IndexArrayView index_array =
            map.index_array();
        bool has_edge = false;
        for (DimensionIndex input_i = input_dim_set_end; input_i < input_rank;
             ++input_i) {
          if (index_array.byte_strides()[input_dims[input_i]] != 0) {
            using std::swap;
            std::swap(input_dims[input_i], input_dims[input_dim_set_end++]);
            has_edge = true;
            current_set_has_array = true;
          }
        }
        return has_edge;
      }
    }
    TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
  };

  /// Returns `true` if, and only if, the output dimension
  /// `grid_output_dimensions[grid_dims[grid_i]]` depends on any of the input
  /// dimensions in the current set.
  ///
  /// If the dependency is due to an `array` output index map, also sets
  /// `current_set_has_array` to `true`.
  const auto is_grid_dim_in_set = [&](DimensionIndex grid_i) -> DimensionIndex {
    ABSL_ASSERT(grid_i >= 0 && grid_i < num_dependent_grid_dims);
    const DimensionIndex grid_dim = grid_dims[grid_i];
    ABSL_ASSERT(grid_dim >= 0 && grid_dim < grid_output_dimensions.size());
    const DimensionIndex output_dim = grid_output_dimensions[grid_dim];
    const OutputIndexMapRef<> map = output_index_maps[output_dim];
    switch (map.method()) {
      case OutputIndexMethod::constant:
        return false;
      case OutputIndexMethod::single_input_dimension:
        return std::find(input_dims.begin() + input_dim_set_begin,
                         input_dims.begin() + input_dim_set_end,
                         map.input_dimension()) !=
               input_dims.begin() + input_dim_set_end;
      case OutputIndexMethod::array: {
        const OutputIndexMapRef<>::IndexArrayView index_array =
            map.index_array();
        if (std::any_of(input_dims.begin() + input_dim_set_begin,
                        input_dims.begin() + input_dim_set_end,
                        [&](DimensionIndex input_dim) {
                          return index_array.byte_strides()[input_dim] != 0;
                        })) {
          current_set_has_array = true;
          return true;
        }
        return false;
      }
    }
    TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
  };

  // Loop until all grid dimensions are part of a connected set.
  DimensionIndex grid_dim_set_begin, grid_dim_set_end = 0;
  while (grid_dim_set_end < num_dependent_grid_dims) {
    // Create a new set.
    input_dim_set_begin = input_dim_set_end;
    grid_dim_set_begin = grid_dim_set_end;
    current_set_has_array = false;

    if (!add_grid_dim_to_current_set(grid_dim_set_end)) {
      // Grid dimension has no input dimension dependencies, exclude it from all
      // connected sets.
      using std::swap;
      std::swap(grid_dims[grid_dim_set_end],
                grid_dims[--num_dependent_grid_dims]);
      continue;
    }
    ++grid_dim_set_end;

    // Successively add any remaining grid dimensions that depend on any of the
    // input dimensions in the current set.
    for (DimensionIndex grid_i = grid_dim_set_end;
         grid_i < num_dependent_grid_dims;) {
      if (is_grid_dim_in_set(grid_i)) {
        add_grid_dim_to_current_set(grid_i);
        using std::swap;
        std::swap(grid_dims[grid_i], grid_dims[grid_dim_set_end++]);
        grid_i = grid_dim_set_end;
      } else {
        ++grid_i;
      }
    }
    TENSORSTORE_RETURN_IF_ERROR(set_callback(
        input_dims.subspan(input_dim_set_begin,
                           input_dim_set_end - input_dim_set_begin),
        grid_dims.subspan(grid_dim_set_begin,
                          grid_dim_set_end - grid_dim_set_begin),
        current_set_has_array));
  }
  return absl::OkStatus();
}

/// Copies a tiled strided ranged of integers to a strided output iterator.
///
/// Fills a row-major 3-d array `output` of shape
/// `[outer_count, size, inner_count]` such that
/// `output(i, j, k) = start + j * stride`.  The strided range is tiled over the
/// first and last dimensions.
///
/// \param start The start value of the strided integer range.
/// \param size The length of the dimension to be filled with the strided
///     integer range.
/// \param stride The stride of the strided integer range.
/// \param outer_count The outer dimension over which to tile the strided range.
/// \param inner_count The inner dimension over which to tile the strided range.
/// \param output[out] The base iterator of the 3-d array to be filled.
/// \param output_stride The stride (in elements, not bytes) of the inner
///     dimension of the `output` array.
template <typename T, typename Stride, typename OutputIt, typename OutputStride>
OutputIt FillWithTiledStridedRange(T start, T size, Stride stride,
                                   Index outer_count, Index inner_count,
                                   OutputIt output,
                                   OutputStride output_stride) {
  // NOTE(jbms): Add special case for `inner_count == 1` if indicated by
  // profiling.
  const T end = start + size * stride;
  for (Index outer_i = 0; outer_i < outer_count; ++outer_i) {
    for (Index i = start; i != end; i += stride) {
      for (Index inner_i = 0; inner_i < inner_count; ++inner_i) {
        *output = i;
        output += output_stride;
      }
    }
  }
  return output;
}

/// Creates a contiguous index array of output indices equivalent to a
/// `single_input_dimension` map.
///
/// The resultant output indices have already been transformed by `map.offset()`
/// and `map.stride()`.
///
/// \param map The `single_input_dimension` output index map.
/// \param input_dims The sequence of input dimensions by which the new index
///     array is indexed.  Must include `map.input_dimension()`.
/// \param index_transform The index transform containing `map`.
/// \param output_indices[out] Base pointer to row-major array of shape
///     `{index_transform.input_shape()[input_dim] : input_dim in input_dims}`
///     to be filled with the output indices.
/// \param output_stride Stride (in elements, not bytes) of the innermost
///     dimension of `output_indices`.
/// \returns `Status()` on success.
/// \error `absl::StatusCode::kOutOfRange` if `map` has an invalid `offset`.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs.
Status GenerateSingleInputDimensionOutputIndices(
    OutputIndexMapRef<> map, span<const DimensionIndex> input_dims,
    IndexTransformView<> index_transform, Index* output_indices,
    Index output_stride) {
  ABSL_ASSERT(map.method() == OutputIndexMethod::single_input_dimension);
  const DimensionIndex single_input_dim = map.input_dimension();
  const IndexInterval domain = index_transform.input_domain()[single_input_dim];
  const Index stride = map.stride();
  TENSORSTORE_RETURN_IF_ERROR(
      GetAffineTransformRange(domain, map.offset(), stride));
  const Index start = map.offset() + stride * domain.inclusive_min();
  // Compute an index array where the dimension corresponding to
  // `single_input_dim` is a range of `size` integers starting from `start` with
  // a step of `stride`, and this range is simply tiled over all of the other
  // dimensions.
  span<const Index> input_shape = index_transform.input_shape();
  DimensionIndex input_i;
  // Compute the product of the sizes of the dimensions of the index array
  // before the one corresponding to `single_input_dim`.
  Index outer_count = 1;
  for (input_i = 0; input_i < input_dims.size(); ++input_i) {
    const DimensionIndex cur_input_dim = input_dims[input_i];
    if (cur_input_dim == single_input_dim) break;
    outer_count *= input_shape[cur_input_dim];
  }
  ++input_i;
  // Compute the product of the sizes of the dimensions of the index array after
  // the one corresponding to `single_input_dim`.
  Index inner_count = 1;
  for (; input_i < input_dims.size(); ++input_i) {
    const DimensionIndex cur_input_dim = input_dims[input_i];
    inner_count *= input_shape[cur_input_dim];
  }

  FillWithTiledStridedRange(start, domain.size(), stride, outer_count,
                            inner_count, output_indices, output_stride);
  return absl::OkStatus();
}

/// Creates a contiguous index array of output indices from an existing `array`
/// map.
///
/// The resultant output indices have already been transformed by `map.offset()`
/// and `map.stride()`.
///
/// \param map The `array` output index map.
/// \param input_dims The sequence of input dimensions by which the new index
///     array is indexed.
/// \param index_transform The index transform containing `map`.
/// \param output_indices[out] Base pointer to row-major array of shape
///     `{index_transform.input_shape()[input_dim] : input_dim in input_dims}`
///     to be filled with the output indices.
/// \param output_stride Stride (in elements, not bytes) of the innermost
///     dimension of `output_indices`.
/// \returns `Status()` on success.
/// \error `absl::StatusCode::kOutOfRange` if `map` contains an invalid index.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs.
Status GenerateIndexArrayOutputIndices(OutputIndexMapRef<> map,
                                       span<const DimensionIndex> input_dims,
                                       IndexTransformView<> index_transform,
                                       Index* output_indices,
                                       Index output_stride) {
  ABSL_ASSERT(map.method() == OutputIndexMethod::array);
  absl::FixedArray<Index, internal::kNumInlinedDims> output_byte_strides(
      index_transform.input_rank(), 0);
  DimensionIndex byte_stride = sizeof(Index) * output_stride;
  for (DimensionIndex i = input_dims.size() - 1; i >= 0; --i) {
    const DimensionIndex input_dim = input_dims[i];
    output_byte_strides[input_dim] = byte_stride;
    byte_stride *= index_transform.input_shape()[input_dim];
  }
  const OutputIndexMapRef<>::IndexArrayView index_array = map.index_array();
  TENSORSTORE_RETURN_IF_ERROR(ValidateIndexArrayBounds(
      index_array.index_range(), index_array.array_ref()));
  const Index stride = map.stride();
  const Index offset = map.offset();
  // Transform the index array by the offset and stride, and store the result in
  // `output_indices`.
  IterateOverArrays(
      [stride, offset](const Index* source_ptr, Index* output_ptr) {
        const Index source_index = *source_ptr;
        *output_ptr = source_index * stride + offset;
        return true;
      },
      skip_repeated_elements,
      // source
      map.index_array().array_ref(),
      // destination
      ArrayView<Index>(output_indices,
                       StridedLayoutView<>(index_transform.input_shape(),
                                           output_byte_strides)));
  return absl::OkStatus();
}

/// Converts output indices to grid indices of a regular grid.
///
/// \param output_indices[in,out] Non-null pointer to array of shape
///     `{num_positions}` with stride (in elements, not bytes) `stride` between
///     consecutive elements.  On invocation, contains the output indices.  On
///     return, contains the corresponding grid indices.
/// \param stride Stride of `output_indices` array.
/// \param num_positions Extent of `output_indices` array.
/// \param cell_size The size of the grid cell.
void ConvertOutputIndicesToCellIndices(Index* output_indices, Index stride,
                                       Index num_positions, Index cell_size) {
  for (Index* end = output_indices + num_positions * stride;
       output_indices != end; output_indices += stride) {
    auto& x = *output_indices;
    x = FloorOfRatio(x, cell_size);
  }
}

/// Computes the product of `input_shape[d]` for `d` in `dims`.
///
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs.
Result<Index> ProductOfIndirectExtents(span<const Index> input_shape,
                                       span<const DimensionIndex> dims) {
  Index num_positions = 1;
  for (const DimensionIndex dim : dims) {
    if (internal::MulOverflow(num_positions, input_shape[dim],
                              &num_positions)) {
      return absl::InvalidArgumentError(
          "Overflow computing number of positions in domain.");
    }
  }
  return num_positions;
}

/// Iterates over every partial input index vector within the domain of the
/// `input_dims` of the connected set, and writes the corresponding partial grid
/// cell index vectors to a flat one-dimensional array (which can then be
/// partitioned using PartitionIndexArraySetGridCellIndexVectors).
///
/// \param grid_dims The list of grid dimensions (in the range
///     `[0, grid_output_dimensions.size())` that are contained in this
///     connected set.
/// \param input_dims The list of input dimensions of `index_transform`
///     contained in this connected set.
/// \param grid_output_dimensions The mapping from grid dimension indices to
///     output dimensions of `index_transform`.
/// \param grid_cell_shape Array of length `grid_output_dimensions.size()`
///     specifying the size of a grid cell along each grid dimension.
/// \param index_transform The index transform.
/// \param num_positions The product of `index_transform.input_size(d)` for `d`
///     in `input_dims`.
/// \returns A vector representing a row-major array of shape
///     `{num_positions, grid_dims.size()}` containing the partial grid cell
///     index vectors for each input position.
Result<std::vector<Index>> GenerateIndexArraySetGridCellIndices(
    span<const DimensionIndex> grid_dims, span<const DimensionIndex> input_dims,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> grid_cell_shape, IndexTransformView<> index_transform,
    Index num_positions) {
  // Logically represents a row-major array of shape `{num_positions,
  // grid_dims.size()}` containing the partial grid cell index vectors for each
  // position.
  std::vector<Index> temp_cell_indices(grid_dims.size() * num_positions);
  // Loop over the grid dimensions and fill in `temp_celll_indices` with the
  // grid cell index vectors for each position.
  for (DimensionIndex grid_i = 0; grid_i < grid_dims.size(); ++grid_i) {
    const DimensionIndex grid_dim = grid_dims[grid_i];
    const DimensionIndex output_dim = grid_output_dimensions[grid_dim];
    const OutputIndexMapRef<> map =
        index_transform.output_index_map(output_dim);
    Index* cur_cell_indices = temp_cell_indices.data() + grid_i;
    // First, compute the output indices for this grid dimension.  These output
    // indices will then be transformed to grid indices.
    if (map.method() == OutputIndexMethod::single_input_dimension) {
      TENSORSTORE_RETURN_IF_ERROR(GenerateSingleInputDimensionOutputIndices(
          map, input_dims, index_transform, cur_cell_indices,
          grid_dims.size()));
    } else {
      ABSL_ASSERT(map.method() == OutputIndexMethod::array);
      TENSORSTORE_RETURN_IF_ERROR(
          GenerateIndexArrayOutputIndices(map, input_dims, index_transform,
                                          cur_cell_indices, grid_dims.size()));
    }
    ConvertOutputIndicesToCellIndices(cur_cell_indices, grid_dims.size(),
                                      num_positions, grid_cell_shape[grid_dim]);
  }
  return temp_cell_indices;
}

/// Checks if two offsets into a cell indices array refer to the same cell
/// indices.
///
/// Serves as the equality function for the `IndirectVectorMap` hash map type.
struct IndirectIndicesEqual {
  /// Pointer to the row-major array of shape `{num_vectors,num_dims}`
  /// specifying the index vectors.
  const Index* index_vectors;

  /// Dimensionality (number of components) of the vectors.
  DimensionIndex num_dims;

  bool operator()(Index a, Index b) const {
    return std::equal(index_vectors + a * num_dims,
                      index_vectors + a * num_dims + num_dims,
                      index_vectors + b * num_dims);
  }
};

/// Same as `IndirectIndicesEqual`, but compares the index vectors
/// lexicographically.
///
/// This is used by PartitionIndexArraySetGridCellIndexVectors to sort grid cell
/// index vectors.
struct IndirectIndicesLess {
  const Index* index_vectors;
  DimensionIndex num_dims;
  bool operator()(Index a, Index b) const {
    return std::lexicographical_compare(
        index_vectors + a * num_dims, index_vectors + a * num_dims + num_dims,
        index_vectors + b * num_dims, index_vectors + b * num_dims + num_dims);
  }
};

/// Computes a hash code for an offset into an array of index vectors based on
/// the index vector.
///
/// This is consistent with the `IndirectIndicesEqual` function, and serves as
/// the hash function for the `IndirectVectorMap` hash map type.
struct IndirectHashIndices {
  const Index* index_vectors;
  DimensionIndex num_dims;

  size_t operator()(Index x) const {
    return absl::Hash<HashHelper>()(HashHelper{index_vectors, num_dims, x});
  }

 private:
  // Helper type that combines the fields of a IndirectHashIndices object with
  // the Index being hashed.
  //
  // This wrapper type is needed to workaround the fact that the Abseil hash
  // framework only supports stateless hash functions.
  struct HashHelper {
    const Index* index_vectors;
    DimensionIndex num_dims;
    Index index;

    template <typename H>
    friend H AbslHashValue(H h, HashHelper x) {
      return H::combine_contiguous(
          std::move(h), x.index_vectors + x.index * x.num_dims, x.num_dims);
    }
  };
};

using IndirectVectorMap = absl::flat_hash_map<Index, Index, IndirectHashIndices,
                                              IndirectIndicesEqual>;

/// Given an array `temp_cell_indices` of non-unique partial grid cell index
/// vectors, implicitly computes a sorted version of this array (possibly
/// containing duplicates), where the index vectors are ordered
/// lexicographically.  For each distinct grid cell index vector in the sorted
/// array, copies the index vector to `grid_cell_indices` and stores at the
/// corresponding position in `grid_cell_partition_offsets` the offset of the
/// first occurrence of that index vector in the sorted array.
///
/// \param temp_cell_indices Non-null pointer to row-major array of shape
///     `{num_positions, num_grid_dims}` specifying partial grid cell index
///     vectors, which may be non-unique and ordered arbitrarily.
/// \param num_positions First dimension of the `temp_cell_indices` array.
/// \param num_grid_dims Number of dimensions in the partial grid cell index
///     vectors.
/// \param grid_cell_indices[out] Non-null pointer to vector to be filled with
///     the row-major array of shape `{num_partitions, num_grid_dims}`
///     specifying the distinct partial grid cell index vectors in
///     `temp_cell_indices`, ordered lexicographically.  The vector is resized
///     to the correct size, and any existing contents are overwritten.
/// \param grid_cell_partition_offsets[out] Non-null pointer to vector to be
///     resized to a length of `num_partitions`, where
///     `(*grid_cell_partition_offsets)[partition_i]` will be set to the offset
///     of the first occurrence in the sorted array of the partial grid cell
///     index vector
///     `span(grid_cell_indices->data() + i * num_grid_dims, num_grid_dims)`.
/// \returns An IndirectVectorMap that maps each position `position_i` in the
///     range `[0,num_positions)`, representing the grid cell index vector
///     `span(temp_celll_indices + position_i * num_grid_dims, num_grid_dims)`,
///     to the offset in the sorted array of the first occurrence of that index
///     vector.  All positions that correspond to equivalent partial grid cell
///     index vectors map to the same slot in the hash table.  Note that these
///     offsets are also stored in `*grid_cell_partition_offsets`, but there
///     they are indexed by the sorted order of the unique index vectors in the
///     `grid_cell_indices` array, not the original order.  The returned hash
///     map references the `temp_cell_indices` array, which must remain valid as
///     long as the hash map is used.
/// \remark The sorted version of `temp_cell_indices` (possibly containing
///     duplicates) is never explicitly computed.  Instead, an IndirectVectorMap
///     is used to determine the set of unique index vectors and the number of
///     occurrences of each.
IndirectVectorMap PartitionIndexArraySetGridCellIndexVectors(
    const Index* temp_cell_indices, Index num_positions, Index num_grid_dims,
    std::vector<Index>* grid_cell_indices,
    std::vector<Index>* grid_cell_partition_offsets) {
  /// Initialize an empty hash map keyed by positions `position_i` in the range
  /// `[0,num_positions)`, corresponding to a vector:
  /// `span(temp_cell_indices + grid_dims * position_i, num_dims)`.  Two
  /// distinct positions that correspond to equivalent vectors map to the same
  /// slot in the hash map.
  IndirectVectorMap cells(
      1, IndirectHashIndices{temp_cell_indices, num_grid_dims},
      IndirectIndicesEqual{temp_cell_indices, num_grid_dims});
  // Compute the number of occurrences of each partial grid cell index vector.
  for (DimensionIndex i = 0; i < num_positions; ++i) {
    ++cells[i];
  }

  // The total number of distinct partial grid cell index vectors is the number
  // of partitions.
  grid_cell_indices->resize(num_grid_dims * cells.size());
  grid_cell_partition_offsets->resize(cells.size());

  // Sort the partial grid cell index vectors lexicographically in order to
  // ensure a deterministic result, using `grid_cell_partition_offsets` as a
  // temporary array.  Note that `cells` is keyed by the
  // range`[0,num_positions)` but `cells.size()` is in general much smaller than
  // `num_positions`, as many positions may correspond to the same partial grid
  // cell.
  std::transform(cells.begin(), cells.end(),
                 grid_cell_partition_offsets->begin(),
                 [](IndirectVectorMap::const_reference x) { return x.first; });
  std::sort(grid_cell_partition_offsets->begin(),
            grid_cell_partition_offsets->end(),
            IndirectIndicesLess{temp_cell_indices, num_grid_dims});

  // Update the values stored in `cells`, as well as
  // grid_cell_partition_offsets, to contain the offsets into
  // partitioned_input_indices, and fill grid_cell_indices.  The `cells` hash
  // map will be used to group by partition the input indices to be computed.
  {
    Index offset = 0;
    Index* grid_cell_indices_ptr = grid_cell_indices->data();
    for (Index& position_i_or_offset : *grid_cell_partition_offsets) {
      const Index position_i = position_i_or_offset;
      auto it = cells.find(position_i);
      ABSL_ASSERT(it != cells.end());
      auto& count_or_offset = it->second;
      const Index count = count_or_offset;
      position_i_or_offset = count_or_offset = offset;
      offset += count;
      grid_cell_indices_ptr =
          std::copy_n(temp_cell_indices + position_i * num_grid_dims,
                      num_grid_dims, grid_cell_indices_ptr);
    }
  }

  return cells;
}

/// Computes the partial input index vectors within the domain subset of
/// `full_input_domain` specified by `input_dims`, and writes them to an array
/// in a partitioned way according to the `cells` map.
///
/// \param input_dims The list of distinct input dimensions in the subset, each
///     in the range `[0, full_input_domain.rank())`.
/// \param full_input_domain The full input domain.  Only values at indices in
///     `input_dims` are used.
/// \param cells An IndirectVectorMap that maps each partial grid cell,
///     identified by a flat input position index, to the starting offset in the
///     output array at which to write the partial input index vectors.
/// \param num_positions The product of `input_shape[d]` for `d` in
///     `input_dims`.
/// \returns A newly allocated array of shape
///     `{num_positions, input_dims.size()}` containing the
SharedArray<Index, 2> GenerateIndexArraySetPartitionedInputIndices(
    span<const DimensionIndex> input_dims, BoxView<> full_input_domain,
    IndirectVectorMap cells, Index num_positions) {
  Box<dynamic_rank(internal::kNumInlinedDims)> partial_input_domain(
      input_dims.size());
  for (DimensionIndex i = 0; i < input_dims.size(); ++i) {
    partial_input_domain[i] = full_input_domain[input_dims[i]];
  }
  SharedArray<Index, 2> partitioned_input_indices =
      AllocateArray<Index>({num_positions, input_dims.size()});
  // Flat position index.
  Index position_i = 0;
  IterateOverIndexRange(partial_input_domain, [&](span<const Index> indices) {
    auto it = cells.find(position_i);
    ABSL_ASSERT(it != cells.end());
    auto& offset = it->second;
    std::copy(indices.begin(), indices.end(),
              partitioned_input_indices.data() + offset * input_dims.size());
    ++offset;
    ++position_i;
  });
  return partitioned_input_indices;
}

/// Fills an `IndexArraySet` structure for a given connected set containing at
/// least one `array` dependency.
///
/// Computes:
///
/// - the list of partial grid cell index vectors;
///
/// - for each partial grid cell, the index arrays that map one-to-one from the
///   new synthetic input dimension to the subset of the domains of the original
///   input dimensions that correspond to the partial grid cell.
///
/// In this context, a "partial grid cell" means a cell index is specified for
/// each grid dimension in the connected set, but the cell indices for grid
/// dimensions outside the set are unspecified.  Likewise a "partial input index
/// vector" means an input index is specified only for each input dimension in
/// the connected set.
///
/// \params index_array_set[in,out] Non-null pointer to IndexArraySet structure.
///     On invocation, `only `index_array_set->input_dimensions` and
///     `index_array_set->grid_dimensions` must be valid.  On return, all other
///     fields have been set.
/// \param grid_output_dimensions Maps grid dimension indices to output
///     dimension indices.
/// \param grid_cell_shape Array of size `grid_output_dimensions.size()`
///     specifying the extent of the grid cells.
/// \param index_transform The index transform.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs.
/// \error `absl::StatusCode::kOutOfRange` if an index array contains an
///     out-of-bounds index.
Status FillIndexArraySetData(
    IndexTransformGridPartition::IndexArraySet* index_array_set,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> grid_cell_shape, IndexTransformView<> index_transform) {
  // Compute the total number of distinct partial input index vectors in the
  // input domain subset.  This allows us to terminate early if it equals 0, and
  // avoids the need for computing it and checking for overflow in each of the
  // functions called below that depend on it.
  TENSORSTORE_ASSIGN_OR_RETURN(
      Index num_positions,
      ProductOfIndirectExtents(index_transform.input_shape(),
                               index_array_set->input_dimensions));
  if (num_positions == 0) {
    return absl::OkStatus();
  }

  // Logically represents a row-major array of shape `{num_positions,
  // grid_dims.size()}` containing the partial grid cell index vectors for each
  // position in the input domain subset.
  TENSORSTORE_ASSIGN_OR_RETURN(
      std::vector<Index> temp_cell_indices,
      GenerateIndexArraySetGridCellIndices(
          index_array_set->grid_dimensions, index_array_set->input_dimensions,
          grid_output_dimensions, grid_cell_shape, index_transform,
          num_positions));

  // Compute `index_array_set->grid_cell_indices`, the sorted array of the
  // distinct index vectors in `temp_cell_indices`, and
  // `index_array_set->grid_cell_partition_offsets`, which specifies the
  // corresponding offsets, for each of those distinct index vectors, into the
  // `partitioned_input_indices` array that will be generated.  Also compute a
  // map `cells` that is used to partition the partial input index vectors
  // corresponding to each partial grid cell index vector in
  // `temp_cell_indices`.
  IndirectVectorMap cells = PartitionIndexArraySetGridCellIndexVectors(
      temp_cell_indices.data(), num_positions,
      index_array_set->grid_dimensions.size(),
      &index_array_set->grid_cell_indices,
      &index_array_set->grid_cell_partition_offsets);

  // Compute the partial input index vectors corresponding to each partial grid
  // cell index vector in `temp_cell_indices`, and directly write them
  // partitioned by grid cell using the `cells` map.
  index_array_set->partitioned_input_indices =
      GenerateIndexArraySetPartitionedInputIndices(
          index_array_set->input_dimensions, index_transform.domain().box(),
          std::move(cells), num_positions);
  return absl::OkStatus();
}

/// Computes an IndexTransformGridPartition structure, which is used to iterate
/// over the grid cell index vectors and construct the corresponding
/// `cell_transform` index transforms.
///
/// \param grid_output_dimensions The sequence of output dimensions of
///     `index_transform` corresponding to the grid.
/// \param grid_cell_shape The shape of a grid cell.
/// \param index_transform The original index transform to be partitioned.  Must
///     be valid.
/// \param output[out] Non-null pointer to IndexTransformGridPartition object to
///     be initialized.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs.
/// \error `absl::StatusCode::kOutOfRange` if an index array contains an
///     out-of-bounds index.
Status GenerateIndexTransformGridPartitionData(
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> grid_cell_shape, IndexTransformView<> index_transform,
    IndexTransformGridPartition* output) {
  return ForEachConnectedSet(
      grid_output_dimensions, index_transform.output_index_maps(),
      output->temp_buffer_,
      [&](span<const DimensionIndex> input_dims,
          span<const DimensionIndex> grid_dims, bool has_array) -> Status {
        if (!has_array) {
          // The connected set contains only `single_input_dimension`
          // dependencies.
          ABSL_ASSERT(input_dims.size() == 1);
          output->strided_sets_.push_back({grid_dims, input_dims[0]});
          return absl::OkStatus();
        }
        // Otherwise the connected set contains at least one `array` dependency.
        output->index_array_sets_.emplace_back();
        auto* set = &output->index_array_sets_.back();
        set->input_dimensions = input_dims;
        set->grid_dimensions = grid_dims;
        return FillIndexArraySetData(set, grid_output_dimensions,
                                     grid_cell_shape, index_transform);
      });
}
}  // namespace

Status PrePartitionIndexTransformOverRegularGrid(
    IndexTransformView<> index_transform,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> grid_cell_shape,
    absl::optional<IndexTransformGridPartition>* result) {
  ABSL_ASSERT(result != nullptr);
  ABSL_ASSERT(grid_output_dimensions.size() == grid_cell_shape.size());
  const DimensionIndex input_rank = index_transform.input_rank();

  // Check that the input domains are all bounded.
  for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
    const IndexInterval domain = index_transform.input_domain()[input_dim];
    if (!IsFinite(domain)) {
      return absl::InvalidArgumentError(StrCat("Input dimension ", input_dim,
                                               " has unbounded domain ", domain,
                                               "."));
    }
  }

  // Check that the output ranges due to `single_input_dimension` maps are
  // valid.  This check ensures that integer overflow cannot occur later when
  // computing the output indices from `single_input_dimension` maps.
  for (const DimensionIndex output_dim : grid_output_dimensions) {
    const OutputIndexMapRef<> map =
        index_transform.output_index_map(output_dim);
    if (map.method() != OutputIndexMethod::single_input_dimension) continue;
    auto status = GetStatus(GetAffineTransformRange(
        index_transform.input_domain()[map.input_dimension()], map.offset(),
        map.stride()));
    if (!status.ok()) {
      return MaybeAnnotateStatus(
          status, StrCat("Computing range of output dimension ", output_dim));
    }
  }

  // Compute the IndexTransformGridPartition structure.
  result->emplace(index_transform.input_rank(), grid_cell_shape.size());
  return internal_grid_partition::GenerateIndexTransformGridPartitionData(
      grid_output_dimensions, grid_cell_shape, index_transform,
      &result->value());
}

}  // namespace internal_grid_partition
}  // namespace tensorstore
