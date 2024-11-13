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

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <ostream>
#include <string>

#include "absl/status/status.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
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

namespace internal_strided_layout {
bool IsContiguousLayout(DimensionIndex rank, const Index* shape,
                        const Index* byte_strides, ContiguousLayoutOrder order,
                        Index element_size) {
  if (rank == 0) return true;
  Index stride = element_size;
  if (order == c_order) {
    for (DimensionIndex i = rank - 1; i != 0; --i) {
      if (byte_strides[i] != stride) return false;
      if (internal::MulOverflow(stride, shape[i], &stride)) {
        return false;
      }
    }
    if (byte_strides[0] != stride) return false;
  } else {
    for (DimensionIndex i = 0; i != rank - 1; ++i) {
      if (byte_strides[i] != stride) return false;
      if (i == rank - 1) break;
      if (internal::MulOverflow(stride, shape[i], &stride)) {
        return false;
      }
    }
    if (byte_strides[rank - 1] != stride) return false;
  }
  return true;
}

bool IsContiguousLayout(DimensionIndex rank, const Index* shape,
                        const Index* byte_strides,
                        ContiguousLayoutPermutation<> order,
                        Index element_size) {
  assert(order.size() == rank);
  Index stride = element_size;
  for (DimensionIndex j = rank; j--;) {
    DimensionIndex i = order[j];
    assert(i >= 0 && i < rank);
    if (byte_strides[i] != stride) return false;
    if (internal::MulOverflow(stride, shape[i], &stride)) {
      return false;
    }
  }
  return true;
}

bool IsBroadcastScalar(DimensionIndex rank, const Index* shape,
                       const Index* byte_strides) {
  for (DimensionIndex i = 0; i < rank; ++i) {
    if (shape[i] > 1 && byte_strides[i] != 0) return false;
  }
  return true;
}

Index GetByteExtent(StridedLayoutView<> layout, Index element_size) {
  Index byte_extent = element_size;
  for (DimensionIndex i = 0, rank = layout.rank(); i < rank; ++i) {
    const Index size = layout.shape()[i];
    if (size == 0) return 0;
    if (size == 1) continue;
    byte_extent =
        std::max(byte_extent, internal::wrap_on_overflow::Multiply(
                                  std::abs(layout.byte_strides()[i]), size));
  }
  return byte_extent;
}

}  // namespace internal_strided_layout

absl::Status ValidateShapeBroadcast(
    tensorstore::span<const Index> source_shape,
    tensorstore::span<const Index> target_shape) {
  for (DimensionIndex source_dim = 0; source_dim < source_shape.size();
       ++source_dim) {
    const Index source_size = source_shape[source_dim];
    if (source_size == 1) continue;
    const DimensionIndex target_dim =
        target_shape.size() - source_shape.size() + source_dim;
    if (target_dim < 0 || target_shape[target_dim] != source_size) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Cannot broadcast array of shape ", source_shape,
                              " to target shape ", target_shape));
    }
  }
  return absl::OkStatus();
}

absl::Status BroadcastStridedLayout(StridedLayoutView<> source,
                                    tensorstore::span<const Index> target_shape,
                                    Index* target_byte_strides) {
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateShapeBroadcast(source.shape(), target_shape));
  for (DimensionIndex target_dim = 0; target_dim < target_shape.size();
       ++target_dim) {
    const DimensionIndex source_dim =
        target_dim + source.rank() - target_shape.size();
    target_byte_strides[target_dim] =
        (source_dim < 0 || source.shape()[source_dim] == 1)
            ? 0
            : source.byte_strides()[source_dim];
  }
  return absl::OkStatus();
}

absl::Status BroadcastStridedLayout(StridedLayoutView<> source,
                                    tensorstore::span<const Index> target_shape,
                                    StridedLayout<>& target) {
  target.set_rank(target_shape.size());
  std::copy(target_shape.begin(), target_shape.end(), target.shape().begin());
  return BroadcastStridedLayout(source, target_shape,
                                target.byte_strides().data());
}

Result<Index> BroadcastStridedLayout(
    StridedLayoutView<dynamic_rank, offset_origin> source,
    BoxView<> target_domain,
    StridedLayout<dynamic_rank, offset_origin>& target) {
  target.set_rank(target_domain.rank());
  TENSORSTORE_RETURN_IF_ERROR(BroadcastStridedLayout(
      StridedLayoutView<>(source.shape(), source.byte_strides()),
      target_domain.shape(), target.byte_strides().data()));
  std::copy_n(target_domain.origin().begin(), target_domain.rank(),
              target.origin().begin());
  std::copy_n(target_domain.shape().begin(), target_domain.rank(),
              target.shape().begin());
  return internal::wrap_on_overflow::Subtract(source.origin_byte_offset(),
                                              target.origin_byte_offset());
}

}  // namespace tensorstore
