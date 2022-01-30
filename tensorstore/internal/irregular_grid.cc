// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/irregular_grid.h"

#include <assert.h>
#include <stddef.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

IrregularGrid::IrregularGrid(std::vector<std::vector<Index>> inclusive_mins)
    : shape_(inclusive_mins.size(), 0),
      inclusive_mins_(std::move(inclusive_mins)) {
  // Sort and remove duplicate grid points.
  for (size_t i = 0; i < inclusive_mins_.size(); i++) {
    std::sort(inclusive_mins_[i].begin(), inclusive_mins_[i].end());
    auto new_it =
        std::unique(inclusive_mins_[i].begin(), inclusive_mins_[i].end());
    inclusive_mins_[i].resize(
        std::distance(inclusive_mins_[i].begin(), new_it));
    shape_[i] = inclusive_mins_[i].size();
  }
}

Index IrregularGrid::operator()(DimensionIndex dim, Index output_index,
                                IndexInterval* cell_bounds) const {
  auto points = inclusive_min(dim);
  auto it = std::upper_bound(points.begin(), points.end(), output_index);
  Index cell = std::distance(points.begin(), it) - 1;
  if (cell_bounds) {
    if (cell < 0) {
      *cell_bounds = IndexInterval::UncheckedHalfOpen(-kInfIndex, points[0]);
    } else if (cell < points.size() - 1) {
      *cell_bounds =
          IndexInterval::UncheckedHalfOpen(points[cell], points[cell + 1]);
    } else {
      *cell_bounds = IndexInterval::UncheckedClosed(points[cell], kInfIndex);
    }
  }
  return cell;
}

/*static*/
IrregularGrid IrregularGrid::Make(span<const IndexDomain<>> domains) {
  absl::InlinedVector<IndexDomainView<>, 16> views;
  views.reserve(domains.size());
  for (const auto& d : domains) views.push_back(d);
  return Make(span(views));
}

/*static*/
IrregularGrid IrregularGrid::Make(span<const IndexDomainView<>> domains) {
  assert(!domains.empty());
  DimensionIndex rank = domains[0].rank();

  std::vector<std::vector<Index>> inclusive_mins;
  inclusive_mins.resize(rank);
  for (auto& d : domains) {
    assert(d.rank() == rank);
    for (DimensionIndex i = 0; i < rank; i++) {
      if (inclusive_mins[i].empty() ||
          inclusive_mins[i].back() != d[i].inclusive_min()) {
        inclusive_mins[i].push_back(d[i].inclusive_min());
      }
      inclusive_mins[i].push_back(d[i].exclusive_max());
    }
  }
  return IrregularGrid(std::move(inclusive_mins));
}

}  // namespace internal
}  // namespace tensorstore
