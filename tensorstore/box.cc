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

#include "tensorstore/box.h"

#include <algorithm>
#include <ostream>

namespace tensorstore {
namespace internal_box {

std::string DescribeForCast(DimensionIndex rank) {
  return StrCat("box with ", StaticCastTraits<DimensionIndex>::Describe(rank));
}

std::ostream& PrintToOstream(std::ostream& os, const BoxView<>& view) {
  return os << "{origin=" << view.origin() << ", shape=" << view.shape() << "}";
}

bool AreEqual(const BoxView<>& box_a, const BoxView<>& box_b) {
  return box_a.rank() == box_b.rank() &&
         std::equal(box_a.shape().begin(), box_a.shape().end(),
                    box_b.shape().begin()) &&
         std::equal(box_a.origin().begin(), box_a.origin().end(),
                    box_b.origin().begin());
}

}  // namespace internal_box
}  // namespace tensorstore
