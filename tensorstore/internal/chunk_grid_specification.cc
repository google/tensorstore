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

#include "tensorstore/internal/chunk_grid_specification.h"

#include <assert.h>

#include <vector>

#include "absl/container/inlined_vector.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_permutation.h"
#include "tensorstore/internal/async_write_array.h"

namespace tensorstore {
namespace internal {

ChunkGridSpecification::Component::Component(SharedArray<const void> fill_value,
                                             Box<> component_bounds)
    : internal::AsyncWriteArray::Spec(std::move(fill_value),
                                      std::move(component_bounds)) {
  chunked_to_cell_dimensions.resize(rank());
  std::iota(chunked_to_cell_dimensions.begin(),
            chunked_to_cell_dimensions.end(), static_cast<DimensionIndex>(0));
}

ChunkGridSpecification::Component::Component(
    SharedArray<const void> fill_value, Box<> component_bounds,
    std::vector<DimensionIndex> chunked_to_cell_dimensions)
    : internal::AsyncWriteArray::Spec(std::move(fill_value),
                                      std::move(component_bounds)),
      chunked_to_cell_dimensions(std::move(chunked_to_cell_dimensions)) {}

ChunkGridSpecification::ChunkGridSpecification(ComponentList components_arg)
    : components(std::move(components_arg)) {
  assert(!components.empty());
  // Extract the chunk shape from the cell shape of the first component.
  chunk_shape.resize(components[0].chunked_to_cell_dimensions.size());
  for (DimensionIndex i = 0;
       i < static_cast<DimensionIndex>(chunk_shape.size()); ++i) {
    chunk_shape[i] =
        components[0].shape()[components[0].chunked_to_cell_dimensions[i]];
  }
  // Verify that the extents of the chunked dimensions are the same for all
  // components.
#if !defined(NDEBUG)
  for (const auto& component : components) {
    assert(component.chunked_to_cell_dimensions.size() == chunk_shape.size());
    DimensionSet seen_dimensions;
    for (DimensionIndex i = 0;
         i < static_cast<DimensionIndex>(chunk_shape.size()); ++i) {
      const DimensionIndex cell_dim = component.chunked_to_cell_dimensions[i];
      assert(!seen_dimensions[cell_dim]);
      seen_dimensions[cell_dim] = true;
      assert(chunk_shape[i] == component.shape()[cell_dim]);
    }
  }
#endif  // !defined(NDEBUG)
}

void ChunkGridSpecification::GetComponentOrigin(const size_t component_index,
                                                span<const Index> cell_indices,
                                                span<Index> origin) const {
  assert(rank() == cell_indices.size());
  assert(component_index < components.size());
  const auto& component_spec = components[component_index];
  assert(component_spec.rank() == origin.size());
  std::fill_n(origin.begin(), origin.size(), Index(0));
  for (DimensionIndex chunk_dim_i = 0;
       chunk_dim_i < static_cast<DimensionIndex>(
                         component_spec.chunked_to_cell_dimensions.size());
       ++chunk_dim_i) {
    const DimensionIndex cell_dim_i =
        component_spec.chunked_to_cell_dimensions[chunk_dim_i];
    origin[cell_dim_i] = cell_indices[chunk_dim_i] * chunk_shape[chunk_dim_i];
  }
}

Result<ChunkLayout> GetChunkLayoutFromGrid(
    const ChunkGridSpecification::Component& component_spec) {
  ChunkLayout layout;
  DimensionIndex inner_order[kMaxRank];
  const DimensionIndex rank = component_spec.rank();
  tensorstore::SetPermutation(c_order, span(inner_order, rank));
  TENSORSTORE_RETURN_IF_ERROR(
      layout.Set(ChunkLayout::InnerOrder(span(inner_order, rank))));
  TENSORSTORE_RETURN_IF_ERROR(
      layout.Set(ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(rank))));
  TENSORSTORE_RETURN_IF_ERROR(
      layout.Set(ChunkLayout::WriteChunkShape(component_spec.shape())));
  TENSORSTORE_RETURN_IF_ERROR(layout.Finalize());
  return layout;
}

}  // namespace internal
}  // namespace tensorstore
