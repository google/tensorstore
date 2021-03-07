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

#include "tensorstore/contiguous_layout.h"

#include <cassert>
#include <cstddef>
#include <ostream>

#include "tensorstore/util/span.h"

namespace tensorstore {

void ComputeStrides(ContiguousLayoutOrder order, std::ptrdiff_t element_stride,
                    span<const Index> shape, span<Index> strides) {
  const DimensionIndex rank = shape.size();
  assert(strides.size() == rank);
  if (order == ContiguousLayoutOrder::right) {
    for (DimensionIndex i = rank - 1; i >= 0; --i) {
      strides[i] = element_stride;
      element_stride *= shape[i];
    }
  } else {
    for (DimensionIndex i = 0; i < rank; ++i) {
      strides[i] = element_stride;
      element_stride *= shape[i];
    }
  }
}

std::ostream& operator<<(std::ostream& os, ContiguousLayoutOrder order) {
  return os << (order == ContiguousLayoutOrder::c ? 'C' : 'F');
}

}  // namespace tensorstore
