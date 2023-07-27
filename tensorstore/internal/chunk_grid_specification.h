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

#ifndef TENSORSTORE_INTERNAL_CHUNK_GRID_SPECIFICATION_H_
#define TENSORSTORE_INTERNAL_CHUNK_GRID_SPECIFICATION_H_

#include <vector>

#include "absl/container/inlined_vector.h"
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/async_write_array.h"
#include "tensorstore/internal/cache/cache.h"

namespace tensorstore {
namespace internal {

/// Specifies a common chunk grid that divides the component arrays.
///
/// Each component array has its own data type, and all of the component arrays
/// need not have the same number of dimensions.  Some subset of the dimensions
/// of each component array are of fixed size and not broken into chunks, and
/// are called "unchunked dimensions".  The remaining dimensions of each
/// component array are divided into fixed size chunks, and are called "chunked
/// dimensions".
///
/// While the unchunked dimensions of each component are independent, every
/// component must have the same number of chunked dimensions, and the chunked
/// dimensions of all components must correspond.  Specifically, there is a
/// bijection between the chunked dimensions of each component and a common list
/// of chunked dimensions shared by all components of the chunk grid, and the
/// chunk size must be the same.
///
/// The chunked dimensions of one component may, however, be in a different
/// order, and may be interleaved with unchunked dimensions differently, than
/// the chunked dimensions of another component.  The data for each component is
/// always stored within the cache contiguously in C order.  The dimensions of a
/// given component may be permuted to control the effective layout order of
/// each component independently.  Multiple component arrays are not interleaved
/// in memory, i.e. "columnar" storage is used, even if they are interleaved in
/// the underlying persistent storage format.
///
/// It is expected that the most common use will be with just a single
/// component.
///
/// A "cell" corresponds to a vector of integer chunk coordinates, of dimension
/// equal to the number of chunked dimensions.
///
/// For example, to specify a chunk cache with a common 3-d chunk shape of
/// `[25, 50, 30]` and two component arrays, one of data type uint16 with an
/// additional unchunked dimension of extent 2, and one of data type float32
/// with additional unchunked dimensions of extents 3 and 4, the following
/// `GridChunkSpecification` could be used:
///
///     components[0]:
///       dtype: uint16
///       shape: [25, 50, 30, 2]
///       chunked_to_cell_dimensions: [0, 1, 2]
///
///     components[1]:
///       dtype: float32
///       shape: [3, 30, 50, 25, 4]
///       chunked_to_cell_dimensions: [3, 2, 1]
///
///     chunk_shape: [25, 50, 30]
struct ChunkGridSpecification {
  DimensionIndex rank() const {
    return static_cast<DimensionIndex>(chunk_shape.size());
  }

  /// Specification of the data type, unchunked dimensions, and fill value of a
  /// single component array.
  ///
  /// The fill value specifies the default value to use when there is no
  /// existing data for a chunk.  When reading, if a missing chunk is
  /// encountered, the read is satisfied using the fill value.  When writing
  /// back a partially-written chunk for which there is no existing data, the
  /// fill value is substituted at unwritten positions.
  ///
  /// For chunked dimensions, the extent in `fill_value.shape()` must match the
  /// corresponding extent in `chunk_shape`.  For unchunked dimensions, the
  /// extent in `fill_value.shape()` specifies the full extent for that
  /// dimension of the component array.
  ///
  /// For each `chunked_to_cell_dimensions[i]`, it must be the case that
  /// `fill_value.shape()[chunked_to_cell_dimensions[i]] = chunk_shape[i]`,
  /// where `chunk_shape` is from the containing `ChunkGridSpecification`.
  struct Component : public AsyncWriteArray::Spec {
    /// Construct a component specification from a fill value.
    ///
    /// The `chunked_to_cell_dimensions` map is set to an identity map over
    /// `[0, fill_value.rank())`, meaning all dimensions are chunked.
    ///
    /// There are no constraints on the memory layout of `fill_value`.  To more
    /// efficiently represent the `fill_value` if the same value is used for all
    /// positions within a given dimension, you can specify a byte stride of 0
    /// for that dimension.  In particular, if the same value is used for all
    /// positions in the cell, you can specify all zero byte strides.
    Component(SharedArray<const void> fill_value, Box<> component_bounds);

    /// Constructs a component specification with the specified fill value and
    /// set of chunked dimensions.
    Component(SharedArray<const void> fill_value, Box<> component_bounds,
              std::vector<DimensionIndex> chunked_to_cell_dimensions);

    /// Mapping from chunked dimensions (corresponding to components of
    /// `chunk_shape`) to cell dimensions (corresponding to dimensions of
    /// `fill_value`).
    std::vector<DimensionIndex> chunked_to_cell_dimensions;
  };

  using ComponentList = absl::InlinedVector<Component, 1>;

  /// Constructs a grid specification with the specified components.
  ChunkGridSpecification(ComponentList components_arg);

  /// The list of components.
  ComponentList components;

  /// The dimensions that are chunked (must be common to all components).
  std::vector<Index> chunk_shape;

  /// Returns the number of chunked dimensions.
  DimensionIndex grid_rank() const { return chunk_shape.size(); }

  /// Computes the origin of a cell for a particular component array at the
  /// specified grid position.
  ///
  /// \param component_index Index of the component.
  /// \param cell_indices Pointer to array of length `rank()` specifying the
  ///     grid position.
  /// \param origin[out] Non-null pointer to array of length
  ///     `components[component_index].rank()`.
  /// \post `origin[i] == 0` for all unchunked dimensions `i`
  /// \post `origin[component_spec.chunked_to_cell_dimensions[j]]` equals
  ///     `cell_indices[j] * spec.chunk_shape[j]` for all grid dimensions `j`.
  void GetComponentOrigin(const size_t component_index,
                          span<const Index> cell_indices,
                          span<Index> origin) const;
};

/// Returns the entry for the specified grid cell.  If it does not already
/// exist, it will be created.
template <typename CacheType>
PinnedCacheEntry<CacheType> GetEntryForGridCell(
    CacheType& cache, span<const Index> grid_cell_indices) {
  static_assert(std::is_base_of_v<Cache, CacheType>);
  const std::string_view key(
      reinterpret_cast<const char*>(grid_cell_indices.data()),
      grid_cell_indices.size() * sizeof(Index));
  return GetCacheEntry(&cache, key);
}

/// Returns a simple chunk layout based on the specified grid.
Result<ChunkLayout> GetChunkLayoutFromGrid(
    const ChunkGridSpecification::Component& component_spec);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CHUNK_GRID_SPECIFICATION_H_
