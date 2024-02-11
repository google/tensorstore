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

#include <stddef.h>

#include <cassert>
#include <iosfwd>

#include "tensorstore/index.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

/// Specifies a C-order or Fortran-order contiguous array layout.
///
/// \relates StridedLayout
enum class ContiguousLayoutOrder {
  right = 0,
  c = 0,
  row_major = 0,
  left = 1,
  fortran = 1,
  column_major = 1
};

/// Prints to an `std::ostream`.
/// \relates ContiguousLayoutOrder
///
/// \id ContiguousLayoutOrder
std::ostream& operator<<(std::ostream& os, ContiguousLayoutOrder order);

/// \relates ContiguousLayoutOrder
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
/// \relates ContiguousLayoutOrder
void ComputeStrides(ContiguousLayoutOrder order, ptrdiff_t element_stride,
                    span<const Index> shape, span<Index> strides);

/// Computes the offset of a given index vector in C or Fortran order.
///
/// This is the inverse of `GetContiguousIndices`.
///
/// \param shape Shape for which to compute the offset.
/// \param indices Indices for which to compute the offset.
/// \dchecks `shape.size() == indices.size()`
/// \relates ContiguousLayoutOrder
template <ContiguousLayoutOrder Order = c_order, typename I = Index>
inline I GetContiguousOffset(span<const I> shape, span<const I> indices) {
  assert(shape.size() == indices.size());
  I offset = 0;
  for (ptrdiff_t i = (Order == c_order) ? 0 : (indices.size() - 1);
       (Order == c_order) ? (i < indices.size()) : (i >= 0);
       (Order == c_order) ? ++i : --i) {
    assert(indices[i] >= 0 && indices[i] < shape[i]);
    offset *= shape[i];
    offset += indices[i];
  }
  return offset;
}

/// Computes the indices corresponding to a given linear offset for the
/// specified `shape` and `Order`.
///
/// This is the inverse of `GetContiguousOffset`.
///
/// \param offset Offset for which to the compute the indices.  Must satisfy
///     `0 <= offset && offset < ProductOfExtents(shape)`.
/// \param shape Shape for which to compute the indices.
/// \param indices[out] Indices to be set.
/// \dchecks `shape.size() == indices.size()`
/// \relates ContiguousLayoutOrder
template <ContiguousLayoutOrder Order = c_order, typename I = Index>
inline void GetContiguousIndices(I offset, span<const I> shape,
                                 span<I> indices) {
  assert(shape.size() == indices.size());
  assert(offset >= 0);
  ptrdiff_t rank = shape.size();
  for (ptrdiff_t i = (Order == c_order) ? (rank - 1) : 0;
       (Order == c_order) ? (i >= 0) : (i < rank);
       (Order == c_order) ? --i : ++i) {
    const I size = shape[i];
    indices[i] = offset % size;
    offset /= size;
  }
  assert(offset == 0);
}

}  // namespace tensorstore

#endif  // TENSORSTORE_CONTIGUOUS_LAYOUT_H_
