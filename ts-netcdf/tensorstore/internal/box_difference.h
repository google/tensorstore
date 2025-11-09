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

#ifndef TENSORSTORE_INTERNAL_BOX_DIFFERENCE_H_
#define TENSORSTORE_INTERNAL_BOX_DIFFERENCE_H_

#include <limits>

#include "tensorstore/box.h"
#include "tensorstore/index.h"

namespace tensorstore {
namespace internal {

/// Represents the difference of two n-dimensional boxes/hyperrectangles as a
/// sequence of disjoint sub-boxes.
///
/// TODO(jbms): Consider implementing a C++ iterator interface.
class BoxDifference {
 public:
  /// Constructs a BoxDifference data structure representing the region of
  /// `outer` not contained in `inner`.
  ///
  /// \param outer The outer box to reference.  Must remain valid as long as the
  ///     `BoxDifference` object is used.
  /// \param inner The inner box to reference.  Must remain valid as long as the
  ///     `BoxDifference` object is used.
  /// \dchecks `outer.rank() == inner.rank()`.
  BoxDifference(BoxView<> outer, BoxView<> inner);

  /// Returns the rank of the boxes.
  DimensionIndex rank() const { return outer_.rank(); }

  /// Returns the number of sub-boxes (at most `pow(3, rank()) - 1`) in the
  /// sequence representing the difference.
  Index num_sub_boxes() const { return num_sub_boxes_; }

  /// Assigns `out` to the sub-box at index `sub_box_index` in the sequence
  /// representing the difference.
  ///
  /// \dchecks `out.rank() == rank()`.
  /// \dchecks `0 <= sub_box_index && sub_box_index < num_sub_boxes()`.
  void GetSubBox(Index sub_box_index, MutableBoxView<> out) const;

 private:
  BoxView<> outer_;
  BoxView<> inner_;
  Index num_sub_boxes_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_BOX_DIFFERENCE_H_
