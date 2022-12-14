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

#include "tensorstore/strided_layout.h"

#include <cstddef>
#include <limits>
#include <ostream>

#include "tensorstore/util/str_cat.h"

namespace tensorstore {

namespace internal_strided_layout {

void PrintToOstream(
    std::ostream& os,
    const StridedLayoutView<dynamic_rank, offset_origin>& layout) {
  os << "{domain=" << layout.domain()
     << ", byte_strides=" << layout.byte_strides() << "}";
}

std::string DescribeForCast(DimensionIndex rank) {
  return tensorstore::StrCat("strided layout with ",
                             StaticCastTraits<DimensionIndex>::Describe(rank));
}

bool StridedLayoutsEqual(StridedLayoutView<dynamic_rank, offset_origin> a,
                         StridedLayoutView<dynamic_rank, offset_origin> b) {
  return a.domain() == b.domain() &&
         internal::RangesEqual(a.byte_strides(), b.byte_strides());
}

}  // namespace internal_strided_layout

std::ostream& operator<<(std::ostream& os, ArrayOriginKind origin_kind) {
  return os << (origin_kind == zero_origin ? "zero" : "offset");
}

}  // namespace tensorstore
