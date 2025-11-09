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

#include "tensorstore/internal/box_difference.h"

#include <cassert>
#include <limits>

#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/internal/integer_overflow.h"

namespace tensorstore {
namespace internal {

/// Each dimension is divided into 3 parts:
///
/// - the "before" part, corresponding to the region of `outer` before the first
///   index in `inner`;
///
/// - the "intersection" part, equal to the intersection of `outer` and `inner`;
///
/// - the "after" part, corresponding to the region of `outer` after the last
///   index in `inner`.
///
/// Any of these 3 parts may be empty, but the general case where they are all
/// non-empty looks like:
///
///     [---------------------OUTER-------------------]
///                   [------INNER-------]
///     [---BEFORE---][---INTERSECTION---][---AFTER---]
///
/// The before part can be empty:
///
///               [---------OUTER---------------------]
///     [--------------INNER--------------]
///               [------INTERSECTION-----][---AFTER--]
///
/// The after part can be empty:
///
///     [---------OUTER---------------------]
///               [--------------INNER--------------]
///     [-BEFORE-][------INTERSECTION-------]
///
/// The intersection can be empty with a non-empty "before" part:
///
///     [-----OUTER-----]
///                        [---INNER---]
///     [-----BEFORE----]
///
/// The intersection can be empty with a non-empty "after" part:
///
///                    [-----OUTER-----]
///     [---INNER---]
///                    [-----AFTER-----]
///
/// The difference is represented as the sum of all possible sub-boxes obtained
/// by picking one of the non-empty parts for each dimension, with the
/// constraint that at least one part must not be the "intersection" part.
///
/// As a special case, if the intersection is empty in any dimension, the
/// difference is simply equal to `outer` (which is just a single box).
namespace {
Index GetNumSubtractionSubBoxes(BoxView<> outer, BoxView<> inner) {
  assert(outer.rank() == inner.rank());
  const DimensionIndex rank = outer.rank();
  Index total_count = 1;
  for (DimensionIndex i = 0; i < rank; ++i) {
    IndexInterval outer_interval = outer[i];
    IndexInterval inner_interval = inner[i];
    Index num_parts = 1;
    if (Intersect(outer_interval, inner_interval).empty()) {
      // Intersection in this dimension is empty, difference is simply equal to
      // `outer`.
      return 1;
    }
    if (outer_interval.inclusive_min() < inner_interval.inclusive_min()) {
      // "before" part is non-empty
      ++num_parts;
    }
    if (outer_interval.inclusive_max() > inner_interval.inclusive_max()) {
      // "after" part is non-empty
      ++num_parts;
    }
    // Note: total_count is bounded by `pow(3, kMaxRank)`, which cannot
    // overflow.
    total_count *= num_parts;
  }
  // Subtract 1 for the one box corresponding to the intersection interval in
  // all dimensions, which is not included in the difference.
  return total_count - 1;
}
}  // namespace

BoxDifference::BoxDifference(BoxView<> outer, BoxView<> inner)
    : outer_(outer),
      inner_(inner),
      num_sub_boxes_(GetNumSubtractionSubBoxes(outer, inner)) {}

void BoxDifference::GetSubBox(Index sub_box_index, MutableBoxView<> out) const {
  const DimensionIndex rank = out.rank();
  assert(rank == outer_.rank());
  assert(sub_box_index >= 0 && sub_box_index < num_sub_boxes_);
  // Increment by 1, because the all zero bit pattern corresponds to the
  // intersection interval of all dimensions, which is not part of the
  // subtraction result.
  ++sub_box_index;
  for (DimensionIndex i = 0; i < rank; ++i) {
    IndexInterval outer_interval = outer_[i];
    IndexInterval inner_interval = inner_[i];
    Index num_parts = 1;
    IndexInterval intersection = Intersect(outer_interval, inner_interval);
    if (intersection.empty()) {
      out.DeepAssign(outer_);
      return;
    }
    const bool has_before =
        outer_interval.inclusive_min() < inner_interval.inclusive_min();
    const bool has_after =
        outer_interval.inclusive_max() > inner_interval.inclusive_max();
    if (has_before) ++num_parts;
    if (has_after) ++num_parts;
    const Index part_i = sub_box_index % num_parts;
    switch (part_i) {
      case 0:
        out[i] = intersection;
        break;
      case 1:
        if (has_before) {
          out[i] = IndexInterval::UncheckedHalfOpen(
              outer_interval.inclusive_min(), inner_interval.inclusive_min());
          break;
        }
        [[fallthrough]];
      case 2:
        out[i] = IndexInterval::UncheckedHalfOpen(
            inner_interval.exclusive_max(), outer_interval.exclusive_max());
        break;
    }
    sub_box_index /= num_parts;
  }
}

}  // namespace internal
}  // namespace tensorstore
