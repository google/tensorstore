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

#ifndef TENSORSTORE_UTIL_SPLIT_BOX_H_
#define TENSORSTORE_UTIL_SPLIT_BOX_H_

#include <array>

#include "tensorstore/box.h"

namespace tensorstore {

/// Splits an input box into two approximately equal-sized halves aligned to a
/// grid cell boundary.
///
/// If multiple dimensions can be split, splits along the dimension `i` for
/// which the largest number of grid cells intersects `input[i]`.
///
/// \param input The input box to split.
/// \param grid_cell_template Specifies the grid to which the split must be
///     aligned.  Each grid cell has a shape of `grid_cell_template.shape()` and
///     extends infinitely in all directions from an origin of
///     `grid_cell_template.origin()`.
/// \param split_output[out] Location where split result is stored.
/// \dchecks `IsFinite(input[i]) || Contains(grid_cell_template[i], input[i])`
///     for all dimensions `i`.
/// \dchecks `input.rank() == grid_cell_template.rank()`
/// \dchecks `split_output[0].rank() == input.rank()`
/// \dchecks `split_output[1].rank() == input.rank()`
/// \returns `true` if `input` could be split, i.e. it intersects more than one
///     grid cell.  In this case, `split_output` is set to the split result.
///     Returns `false` if `input` intersects only a single grid cell.  In this
///     case `split_output` is unchanged.
bool SplitBoxByGrid(BoxView<> input, BoxView<> grid_cell_template,
                    std::array<MutableBoxView<>, 2> split_output);

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_SPLIT_BOX_H_
