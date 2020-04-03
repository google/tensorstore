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

#ifndef TENSORSTORE_CONTIGUOUS_LAYOUT_H_
#define TENSORSTORE_CONTIGUOUS_LAYOUT_H_

#include <cstddef>
#include <iosfwd>

#include "tensorstore/index.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

enum class ContiguousLayoutOrder {
  right = 0,
  c = 0,
  row_major = 0,
  left = 1,
  fortran = 1,
  column_major = 1
};

std::ostream& operator<<(std::ostream& os, ContiguousLayoutOrder order);

constexpr ContiguousLayoutOrder c_order = ContiguousLayoutOrder::c;
constexpr ContiguousLayoutOrder row_major_order =
    ContiguousLayoutOrder::row_major;
constexpr ContiguousLayoutOrder fortran_order = ContiguousLayoutOrder::fortran;
constexpr ContiguousLayoutOrder column_major_order =
    ContiguousLayoutOrder::column_major;

/// \brief Computes strides for the specified `ContiguousLayoutOrder`.
///
/// \param order The layout order.
/// \param element_stride The stride of the innermost dimension.
/// \param shape The extents of the array.
/// \param strides The output strides array.
/// \pre strides.size() == shape.size()
/// \post If order == ContiguousLayoutOrder::left, strides[0] = element_stride
///     and strides[i+1] = strides[i] * shape[i].  If order ==
///     ContiguousLayoutOrder::right, strides[shape.size()-1] = element_stride
///     and strides[i] = strides[i+1] * shape[i+1].
///
void ComputeStrides(ContiguousLayoutOrder order, std::ptrdiff_t element_stride,
                    span<const Index> shape, span<Index> strides);

}  // namespace tensorstore

#endif  // TENSORSTORE_CONTIGUOUS_LAYOUT_H_
