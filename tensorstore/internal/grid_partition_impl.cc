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
#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/index_space/output_index_map.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/regular_grid.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_grid_partition {

using ::tensorstore::internal_index_space::OutputIndexMap;
using ::tensorstore::internal_index_space::TransformRep;

using IndexArraySet = IndexTransformGridPartition::IndexArraySet;
using StridedSet = IndexTransformGridPartition::StridedSet;

SharedArray<const Index, 2>
IndexTransformGridPartition::IndexArraySet::partition_input_indices(
    Index partition_i) const {
  assert(partition_i >= 0 && partition_i < num_partitions());
  SharedArray<const Index, 2> result;
  const Index start = grid_cell_partition_offsets[partition_i];
  const Index end =
      static_cast<size_t>(partition_i + 1) == grid_cell_partition_offsets.size()
          ? partitioned_input_indices.shape()[0]
          : grid_cell_partition_offsets[partition_i + 1];
  assert(start >= 0 && start < partitioned_input_indices.shape()[0]);
  assert(end > start && end <= partitioned_input_indices.shape()[0]);
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
         static_cast<size_t>(num_partitions() * grid_dimensions.count()));
  return span(&grid_cell_indices[partition_i * grid_dimensions.count()],
              grid_dimensions.count());
}

namespace {
struct GridCellIndicesIndirectPartialCompare {
  DimensionSet grid_dimensions;
  const Index* grid_cell_indices_for_partitions;

  Index operator()(Index partition_i, const Index* full_indices) const {
    const Index* other_grid_cell_indices =
        grid_cell_indices_for_partitions +
        partition_i * grid_dimensions.count();
    DimensionIndex j = 0;
    for (DimensionIndex grid_dim : grid_dimensions.index_view()) {
      Index diff = other_grid_cell_indices[j] - full_indices[grid_dim];
      if (diff != 0) {
        return diff;
      }
      ++j;
    }
    return 0;
  }
};
}  // namespace

Index IndexTransformGridPartition::IndexArraySet::FindPartition(
    span<const Index> grid_cell_indices) const {
  Index lower = 0, upper = num_partitions();
  GridCellIndicesIndirectPartialCompare compare{grid_dimensions,
                                                this->grid_cell_indices.data()};
  while (lower != upper) {
    Index mid = (lower + upper) / 2;
    Index c = compare(mid, grid_cell_indices.data());
    if (c == 0) return mid;
    if (c > 0) {
      upper = mid;
    } else {
      lower = mid + 1;
    }
  }
  return -1;
}

void UpdateCellTransformForIndexArraySetPartition(
    const IndexArraySet& index_array_set, DimensionIndex set_i,
    Index partition_i, internal_index_space::TransformRep* cell_transform) {
  // Update the output index maps for the original input dimensions in this
  // connected set to reference the precomputed index array of input indices
  // corresponding to this partition.
  const SharedArray<const Index, 2> partition_input_indices =
      index_array_set.partition_input_indices(partition_i);
  cell_transform->input_shape()[set_i] = partition_input_indices.shape()[0];
  ByteStridedPointer<const Index> partition_input_indices_ptr =
      partition_input_indices.byte_strided_pointer();
  const Index vector_dimension_byte_stride =
      partition_input_indices.byte_strides()[1];
  const span<OutputIndexMap> output_maps = cell_transform->output_index_maps();
  for (DimensionIndex full_input_dim :
       index_array_set.input_dimensions.index_view()) {
    internal_index_space::IndexArrayData& index_array_data =
        output_maps[full_input_dim].index_array_data();
    index_array_data.element_pointer = std::shared_ptr<const Index>(
        partition_input_indices.pointer(), partition_input_indices_ptr);
    partition_input_indices_ptr += vector_dimension_byte_stride;
  }
}

IndexTransform<> IndexTransformGridPartition::GetCellTransform(
    IndexTransformView<> full_transform, span<const Index> grid_cell_indices,
    span<const DimensionIndex> grid_output_dimensions,
    absl::FunctionRef<IndexInterval(DimensionIndex grid_dim,
                                    Index grid_cell_index)>
        get_grid_cell_output_interval) const {
  auto cell_transform = InitializeCellTransform(*this, full_transform);
  for (DimensionIndex set_i = 0, num_sets = index_array_sets().size();
       set_i < num_sets; ++set_i) {
    const IndexArraySet& index_array_set = index_array_sets()[set_i];
    const Index partition_i = index_array_set.FindPartition(grid_cell_indices);
    assert(partition_i != -1);
    UpdateCellTransformForIndexArraySetPartition(
        index_array_set, set_i, partition_i, cell_transform.get());
  }
  for (DimensionIndex set_i = 0, num_sets = strided_sets().size();
       set_i < num_sets; ++set_i) {
    const StridedSet& strided_set = strided_sets()[set_i];
    const DimensionIndex cell_input_dim = set_i + index_array_sets().size();
    IndexInterval restricted_domain =
        full_transform.input_domain()[strided_set.input_dimension];
    for (const DimensionIndex grid_dim :
         strided_set.grid_dimensions.index_view()) {
      const DimensionIndex output_dim = grid_output_dimensions[grid_dim];
      IndexInterval cell_range =
          get_grid_cell_output_interval(grid_dim, grid_cell_indices[grid_dim]);
      const OutputIndexMapRef<> map =
          full_transform.output_index_map(output_dim);
      const IndexInterval cell_domain =
          GetAffineTransformDomain(cell_range, map.offset(), map.stride())
              .value();
      restricted_domain = Intersect(restricted_domain, cell_domain);
    }
    assert(!restricted_domain.empty());
    cell_transform->input_origin()[cell_input_dim] =
        restricted_domain.inclusive_min();
    cell_transform->input_shape()[cell_input_dim] = restricted_domain.size();
  }
  return internal_index_space::TransformAccess::Make<IndexTransform<>>(
      std::move(cell_transform));
}

namespace {

/// Invokes the specified callback for each connected sets of input and grid
/// dimensions of an index transform.
///
/// \param grid_output_dimensions Array that maps grid dimension indices to
///     output dimension indices of the index transform.
/// \param transform The index transform.
/// \param set_callback Function with a signature compatible with:
///     `void (DimensionSet input_dims,
///            DimensionSet grid_dims,
///            bool has_array)`.  This function is called for each
///     connected set.
template <typename SetCallbackFn>
void ForEachConnectedSet(span<const DimensionIndex> grid_output_dimensions,
                         IndexTransformView<> transform,
                         SetCallbackFn set_callback) {
  // Set of input dimensions on which each grid dimension depends.
  DimensionSet input_dims_for_grid_dims[kMaxRank];
  // Indicates for each grid dimension whether it has an index array output
  // index map with at least one non-zero byte stride.
  DimensionSet grid_dims_with_array_dependence;
  for (DimensionIndex grid_dim = 0; grid_dim < grid_output_dimensions.size();
       ++grid_dim) {
    auto [input_dims, array_dependence] =
        internal::GetInputDimensionsForOutputDimension(
            transform, grid_output_dimensions[grid_dim]);
    input_dims_for_grid_dims[grid_dim] = input_dims;
    grid_dims_with_array_dependence[grid_dim] = array_dependence;
  }

  // State variables captured by `add_grid_dim_to_current_set` and
  // `is_grid_dim_in_set`.
  DimensionSet current_input_dims, current_grid_dims;
  DimensionSet remaining_grid_dims{
      DimensionSet::UpTo(grid_output_dimensions.size())};
  bool current_set_has_array;

  /// Adds the input dimensions on which the output dimension
  /// `grid_output_dimensions[grid_dim]` depends to the current set.
  ///
  /// Each output dimension `grid_output_dimensions[grid_dim]` depends on zero
  /// or more input dimensions due to a `single_input_dimension` or `array`
  /// output index map.  This adds to the current set any such input dimensions
  /// that are not already in the current set.
  ///
  /// If the dependencies are via an `array` output index map, sets
  /// `current_set_has_array = true`.
  ///
  /// \returns The set of associated input dimensions.
  const auto add_grid_dim_to_current_set =
      [&](DimensionIndex grid_dim) -> DimensionSet {
    assert(remaining_grid_dims.test(grid_dim));
    assert(grid_dim >= 0 && grid_dim < grid_output_dimensions.size());
    remaining_grid_dims.reset(grid_dim);
    current_grid_dims.set(grid_dim);

    auto input_dims = input_dims_for_grid_dims[grid_dim];
    current_set_has_array |= grid_dims_with_array_dependence[grid_dim];
    current_input_dims |= input_dims;
    return input_dims;
  };

  /// Returns `true` if, and only if, the output dimension
  /// `grid_output_dimensions[grid_dim]` depends on any of the input dimensions
  /// in the current set.
  ///
  /// If the dependency is due to an `array` output index map, also sets
  /// `current_set_has_array` to `true`.
  const auto is_grid_dim_in_set =
      [&](DimensionIndex grid_dim) -> DimensionIndex {
    assert(remaining_grid_dims.test(grid_dim));
    assert(grid_dim >= 0 && grid_dim < grid_output_dimensions.size());
    return !(input_dims_for_grid_dims[grid_dim] & current_input_dims).none();
  };

  // Loop until all grid dimensions are part of a connected set.
  while (!remaining_grid_dims.none()) {
    // Create a new set.
    current_input_dims = {};
    current_grid_dims = {};
    current_set_has_array = false;

    if (add_grid_dim_to_current_set(remaining_grid_dims.index_view().front())
            .none()) {
      // Grid dimension has no input dimension dependencies.
      continue;
    }

    // Successively add any remaining grid dimensions that depend on any of the
    // input dimensions in the current set.
    for (DimensionIndex grid_dim : remaining_grid_dims.index_view()) {
      if (is_grid_dim_in_set(grid_dim)) {
        add_grid_dim_to_current_set(grid_dim);
      }
    }
    set_callback(current_input_dims, current_grid_dims, current_set_has_array);
  }
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
/// \returns `absl::Status()` on success.
/// \error `absl::StatusCode::kOutOfRange` if `map` has an invalid `offset`.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs.
absl::Status GenerateSingleInputDimensionOutputIndices(
    OutputIndexMapRef<> map, DimensionSet input_dims,
    IndexTransformView<> index_transform, Index* output_indices,
    Index output_stride) {
  assert(map.method() == OutputIndexMethod::single_input_dimension);
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
  // Compute the product of the sizes of the dimensions of the index array
  // before and after the one corresponding to `single_input_dim`.
  Index inner_count = 1;
  Index outer_count = 1;
  for (DimensionIndex input_dim : input_dims.index_view()) {
    if (input_dim == single_input_dim) {
      outer_count = inner_count;
      inner_count = 1;
    } else {
      inner_count *= input_shape[input_dim];
    }
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
/// \returns `absl::Status()` on success.
/// \error `absl::StatusCode::kOutOfRange` if `map` contains an invalid index.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs.
absl::Status GenerateIndexArrayOutputIndices(
    OutputIndexMapRef<> map, DimensionSet input_dims,
    IndexTransformView<> index_transform, Index* output_indices,
    Index output_stride) {
  assert(map.method() == OutputIndexMethod::array);
  const DimensionIndex input_rank = index_transform.input_rank();
  Index output_byte_strides[kMaxRank];
  std::fill_n(&output_byte_strides[0], input_rank, static_cast<Index>(0));
  DimensionIndex byte_stride = sizeof(Index) * output_stride;

  // Copy `input_dims` in order to iterate in reverse order.
  Index input_dims_copy[kMaxRank];
  DimensionIndex num_input_dims = 0;
  for (DimensionIndex input_dim : input_dims.index_view()) {
    input_dims_copy[num_input_dims++] = input_dim;
  }

  for (DimensionIndex i = num_input_dims - 1; i >= 0; --i) {
    const DimensionIndex input_dim = input_dims_copy[i];
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
      ArrayView<Index>(
          output_indices,
          StridedLayoutView<>(
              index_transform.input_shape(),
              span<const Index>(&output_byte_strides[0], input_rank))));
  return absl::OkStatus();
}

/// Computes the product of `input_shape[d]` for `d` in `dims`.
///
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs.
Result<Index> ProductOfIndirectExtents(span<const Index> input_shape,
                                       DimensionSet dims) {
  Index num_positions = 1;
  for (const DimensionIndex dim : dims.index_view()) {
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
///     `{num_positions, grid_dims.count()}` containing the partial grid cell
///     index vectors for each input position.
Result<std::vector<Index>> GenerateIndexArraySetGridCellIndices(
    DimensionSet grid_dims, DimensionSet input_dims,
    span<const DimensionIndex> grid_output_dimensions,
    OutputToGridCellFn output_to_grid_cell,
    IndexTransformView<> index_transform, Index num_positions) {
  const DimensionIndex num_grid_dims = grid_dims.count();
  // Logically represents a row-major array of shape
  // `{num_positions, num_grid_dims}` containing the partial grid cell index
  // vectors for each position.
  std::vector<Index> temp_cell_indices(num_grid_dims * num_positions);
  // Loop over the grid dimensions and fill in `temp_celll_indices` with the
  // grid cell index vectors for each position.
  DimensionIndex grid_i = 0;
  for (DimensionIndex grid_dim : grid_dims.index_view()) {
    const DimensionIndex output_dim = grid_output_dimensions[grid_dim];
    const OutputIndexMapRef<> map =
        index_transform.output_index_map(output_dim);
    Index* cur_cell_indices = temp_cell_indices.data() + grid_i;
    // First, compute the output indices for this grid dimension.  These output
    // indices will then be transformed to grid indices.
    if (map.method() == OutputIndexMethod::single_input_dimension) {
      TENSORSTORE_RETURN_IF_ERROR(GenerateSingleInputDimensionOutputIndices(
          map, input_dims, index_transform, cur_cell_indices, num_grid_dims));
    } else {
      assert(map.method() == OutputIndexMethod::array);
      TENSORSTORE_RETURN_IF_ERROR(GenerateIndexArrayOutputIndices(
          map, input_dims, index_transform, cur_cell_indices, num_grid_dims));
    }

    // Convert the output indices to grid cell indices
    for (Index* end = cur_cell_indices + num_positions * num_grid_dims;
         cur_cell_indices != end; cur_cell_indices += num_grid_dims) {
      *cur_cell_indices =
          output_to_grid_cell(grid_dim, *cur_cell_indices, nullptr);
    }
    ++grid_i;
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
      assert(it != cells.end());
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
///     `{num_positions, input_dims.count()}` containing the
SharedArray<Index, 2> GenerateIndexArraySetPartitionedInputIndices(
    DimensionSet input_dims, BoxView<> full_input_domain,
    IndirectVectorMap cells, Index num_positions) {
  const DimensionIndex num_input_dims = input_dims.count();
  Box<dynamic_rank(internal::kNumInlinedDims)> partial_input_domain(
      num_input_dims);
  {
    DimensionIndex i = 0;
    for (DimensionIndex input_dim : input_dims.index_view()) {
      partial_input_domain[i] = full_input_domain[input_dim];
      ++i;
    }
  }
  SharedArray<Index, 2> partitioned_input_indices =
      AllocateArray<Index>({num_positions, num_input_dims});
  // Flat position index.
  Index position_i = 0;
  IterateOverIndexRange(partial_input_domain, [&](span<const Index> indices) {
    auto it = cells.find(position_i);
    assert(it != cells.end());
    auto& offset = it->second;
    std::copy(indices.begin(), indices.end(),
              partitioned_input_indices.data() + offset * num_input_dims);
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
/// \params index_array_set[in,out] On invocation,
///     `only `index_array_set.input_dimensions`
///     and `index_array_set.grid_dimensions`
///     must be valid.  On return, all other fields have been set.
/// \param grid_output_dimensions Maps grid dimension indices to output
///     dimension indices.
/// \param grid_cell_shape Array of size `grid_output_dimensions.size()`
///     specifying the extent of the grid cells.
/// \param index_transform The index transform.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs.
/// \error `absl::StatusCode::kOutOfRange` if an index array contains an
///     out-of-bounds index.
absl::Status FillIndexArraySetData(
    IndexTransformGridPartition::IndexArraySet& index_array_set,
    span<const DimensionIndex> grid_output_dimensions,
    OutputToGridCellFn output_to_grid_cell,
    IndexTransformView<> index_transform) {
  // Compute the total number of distinct partial input index vectors in the
  // input domain subset.  This allows us to terminate early if it equals 0, and
  // avoids the need for computing it and checking for overflow in each of the
  // functions called below that depend on it.
  TENSORSTORE_ASSIGN_OR_RETURN(
      Index num_positions,
      ProductOfIndirectExtents(index_transform.input_shape(),
                               index_array_set.input_dimensions));
  if (num_positions == 0) {
    return absl::OkStatus();
  }

  // Logically represents a row-major array of shape
  // `{num_positions, grid_dims.count()}` containing the partial grid cell index
  // vectors for each position in the input domain subset.
  TENSORSTORE_ASSIGN_OR_RETURN(
      std::vector<Index> temp_cell_indices,
      GenerateIndexArraySetGridCellIndices(
          index_array_set.grid_dimensions, index_array_set.input_dimensions,
          grid_output_dimensions, output_to_grid_cell, index_transform,
          num_positions));

  // Compute `index_array_set.grid_cell_indices`, the sorted array of the
  // distinct index vectors in `temp_cell_indices`, and
  // `index_array_set.grid_cell_partition_offsets`, which specifies the
  // corresponding offsets, for each of those distinct index vectors, into the
  // `partitioned_input_indices` array that will be generated.  Also compute a
  // map `cells` that is used to partition the partial input index vectors
  // corresponding to each partial grid cell index vector in
  // `temp_cell_indices`.
  IndirectVectorMap cells = PartitionIndexArraySetGridCellIndexVectors(
      temp_cell_indices.data(), num_positions,
      index_array_set.grid_dimensions.count(),
      &index_array_set.grid_cell_indices,
      &index_array_set.grid_cell_partition_offsets);

  // Compute the partial input index vectors corresponding to each partial grid
  // cell index vector in `temp_cell_indices`, and directly write them
  // partitioned by grid cell using the `cells` map.
  index_array_set.partitioned_input_indices =
      GenerateIndexArraySetPartitionedInputIndices(
          index_array_set.input_dimensions, index_transform.domain().box(),
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
/// \param grid_partition[out] `IndexTransformGridPartition` object to be
///     initialized.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs.
/// \error `absl::StatusCode::kOutOfRange` if an index array contains an
///     out-of-bounds index.
absl::Status GenerateIndexTransformGridPartitionData(
    span<const DimensionIndex> grid_output_dimensions,
    OutputToGridCellFn output_to_grid_cell,
    IndexTransformView<> index_transform,
    IndexTransformGridPartition& grid_partition) {
  IndexTransformGridPartition::StridedSet strided_sets[kMaxRank];
  DimensionIndex num_strided_sets = 0;

  // List of [grid_dims, input_dims] for each index array set.
  std::pair<DimensionSet, DimensionSet> index_array_sets[kMaxRank];
  DimensionIndex num_index_array_sets = 0;

  ForEachConnectedSet(
      grid_output_dimensions, index_transform,
      [&](DimensionSet input_dims, DimensionSet grid_dims, bool has_array) {
        if (!has_array) {
          // The connected set contains only
          // `single_input_dimension` dependencies.
          assert(input_dims.count() == 1);
          strided_sets[num_strided_sets++] = {grid_dims,
                                              input_dims.index_view().front()};
        } else {
          index_array_sets[num_index_array_sets++] = {grid_dims, input_dims};
        }
      });

  grid_partition.strided_sets_.assign(&strided_sets[0],
                                      &strided_sets[num_strided_sets]);

  grid_partition.index_array_sets_.resize(num_index_array_sets);
  for (DimensionIndex i = 0; i < num_index_array_sets; ++i) {
    auto& set = grid_partition.index_array_sets_[i];
    auto [grid_dims, input_dims] = index_array_sets[i];
    set.input_dimensions = input_dims;
    set.grid_dimensions = grid_dims;
    TENSORSTORE_RETURN_IF_ERROR(FillIndexArraySetData(
        set, grid_output_dimensions, output_to_grid_cell, index_transform));
  }

  return absl::OkStatus();
}
}  // namespace

internal_index_space::TransformRep::Ptr<> InitializeCellTransform(
    const IndexTransformGridPartition& info,
    IndexTransformView<> full_transform) {
  const DimensionIndex full_input_rank = full_transform.input_rank();
  DimensionIndex num_index_array_dims = 0;
  for (const IndexArraySet& index_array_set : info.index_array_sets()) {
    num_index_array_dims += index_array_set.input_dimensions.count();
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
           index_array_set.input_dimensions.index_view()) {
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
        internal_index_space::TransformAccess::rep(full_transform)
            ->input_dimension(full_input_dim);
    ++cell_input_dim;
  }

  // Invariants checked in InvokeCallback
  return cell_transform;
}

absl::Status PrePartitionIndexTransformOverGrid(
    IndexTransformView<> index_transform,
    span<const DimensionIndex> grid_output_dimensions,
    OutputToGridCellFn output_to_grid_cell,
    IndexTransformGridPartition& grid_partition) {
  const DimensionIndex input_rank = index_transform.input_rank();

  // Check that the input domains are all bounded.
  for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
    const IndexInterval domain = index_transform.input_domain()[input_dim];
    if (!IsFinite(domain)) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Input dimension ", input_dim,
                              " has unbounded domain ", domain, "."));
    }
  }

  // Check that the output ranges due to `single_input_dimension` maps are
  // valid.  This check ensures that integer overflow cannot occur later when
  // computing the output indices from `single_input_dimension` maps.
  for (const DimensionIndex output_dim : grid_output_dimensions) {
    const OutputIndexMapRef<> map =
        index_transform.output_index_map(output_dim);
    if (map.method() != OutputIndexMethod::single_input_dimension) continue;
    auto status = GetAffineTransformRange(
                      index_transform.input_domain()[map.input_dimension()],
                      map.offset(), map.stride())
                      .status();
    if (!status.ok()) {
      return MaybeAnnotateStatus(
          status, tensorstore::StrCat("Computing range of output dimension ",
                                      output_dim));
    }
  }

  // Compute the IndexTransformGridPartition structure.
  return internal_grid_partition::GenerateIndexTransformGridPartitionData(
      grid_output_dimensions, output_to_grid_cell, index_transform,
      grid_partition);
}

}  // namespace internal_grid_partition
}  // namespace tensorstore
