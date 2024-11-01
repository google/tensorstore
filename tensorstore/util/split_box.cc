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

#include "tensorstore/util/split_box.h"

#include <array>

#include "tensorstore/box.h"

namespace tensorstore {

bool SplitBoxByGrid(BoxView<> input, BoxView<> grid_cell_template,
                    std::array<MutableBoxView<>, 2> split_output) {
  const DimensionIndex rank = input.rank();
  assert(rank == grid_cell_template.rank());
  assert(rank == split_output[0].rank());
  assert(rank == split_output[1].rank());

  DimensionIndex split_dim = -1;
  Index split_dim_min_cell = 0;
  Index split_dim_max_cell = 1;
  for (DimensionIndex dim = 0; dim < rank; ++dim) {
    const IndexInterval input_interval = input[dim];
    const IndexInterval cell = grid_cell_template[dim];
    assert(tensorstore::IsFinite(input_interval) ||
           tensorstore::Contains(cell, input_interval));
    assert(!cell.empty());
    const Index min_cell = FloorOfRatio(
        input_interval.inclusive_min() - cell.inclusive_min(), cell.size());
    const Index max_cell = CeilOfRatio(
        input_interval.inclusive_max() - cell.inclusive_min() + 1, cell.size());
    if (max_cell - min_cell > split_dim_max_cell - split_dim_min_cell) {
      split_dim = dim;
      split_dim_max_cell = max_cell;
      split_dim_min_cell = min_cell;
    }
  }
  if (split_dim == -1) return false;
  const Index split_cell = (split_dim_min_cell + split_dim_max_cell) / 2;
  const Index split_index = grid_cell_template[split_dim].inclusive_min() +
                            split_cell * grid_cell_template[split_dim].size();
  split_output[0].DeepAssign(input);
  split_output[1].DeepAssign(input);
  split_output[0][split_dim] = IndexInterval::UncheckedHalfOpen(
      input[split_dim].inclusive_min(), split_index);
  split_output[1][split_dim] = IndexInterval::UncheckedHalfOpen(
      split_index, input[split_dim].exclusive_max());
  return true;
}

}  // namespace tensorstore
