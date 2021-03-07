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

#include "tensorstore/array.h"

#include <algorithm>
#include <limits>

#include "tensorstore/data_type_conversion.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/util/internal/iterate_impl.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_array {

bool CompareArraysEqual(
    const ArrayView<const void, dynamic_rank, zero_origin>& a,
    const ArrayView<const void, dynamic_rank, zero_origin>& b) {
  if (a.dtype() != b.dtype()) return false;
  if (!internal::RangesEqual(a.shape(), b.shape())) return false;
  return internal::IterateOverArrays({&a.dtype()->compare_equal, nullptr},
                                     /*status=*/nullptr,
                                     /*constraints=*/skip_repeated_elements, a,
                                     b)
      .success;
}

void CopyArrayImplementation(
    const ArrayView<const void, dynamic_rank, offset_origin>& source,
    const ArrayView<void, dynamic_rank, offset_origin>& dest) {
  TENSORSTORE_CHECK(source.dtype() == dest.dtype());
  internal::IterateOverArrays({&source.dtype()->copy_assign, nullptr},
                              /*status=*/nullptr,
                              /*constraints=*/skip_repeated_elements, source,
                              dest);
}

Status CopyConvertedArrayImplementation(
    const ArrayView<const void, dynamic_rank, offset_origin>& source,
    const ArrayView<void, dynamic_rank, offset_origin>& dest) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto r, internal::GetDataTypeConverterOrError(
                                           source.dtype(), dest.dtype()));
  Status status;
  if (!internal::IterateOverArrays(r.closure,
                                   /*status=*/&status,
                                   /*constraints=*/skip_repeated_elements,
                                   source, dest)
           .success) {
    return internal::GetElementCopyErrorStatus(std::move(status));
  }
  return status;
}

bool CompareArraysEqual(
    const ArrayView<const void, dynamic_rank, offset_origin>& a,
    const ArrayView<const void, dynamic_rank, offset_origin>& b) {
  if (a.dtype() != b.dtype()) return false;
  if (a.domain() != b.domain()) return false;
  return internal::IterateOverArrays({&a.dtype()->compare_equal, nullptr},
                                     /*status=*/nullptr,
                                     /*constraints=*/skip_repeated_elements, a,
                                     b)
      .success;
}

void PrintArrayDimension(
    std::string* result,
    ArrayView<const void, dynamic_rank, offset_origin> array,
    const ArrayFormatOptions& options, bool summarize) {
  if (array.rank() == 0) {
    array.dtype()->append_to_string(result, array.data());
    return;
  }
  *result += options.prefix;

  const Index size = array.shape()[0];
  const Index origin = array.origin()[0];
  if (summarize && size > 2 * options.summary_edge_items) {
    for (Index i = 0; i < options.summary_edge_items; ++i) {
      PrintArrayDimension(result, array[origin + i], options, summarize);
      *result += options.separator;
    }
    *result += options.summary_ellipses;
    for (Index i = size - options.summary_edge_items; i < size; ++i) {
      PrintArrayDimension(result, array[origin + i], options, summarize);
      if (i + 1 != size) {
        *result += options.separator;
      }
    }
  } else {
    for (Index i = 0; i < size; ++i) {
      if (i != 0) *result += options.separator;
      PrintArrayDimension(result, array[origin + i], options, summarize);
    }
  }
  *result += options.suffix;
}

std::string DescribeForCast(DataType dtype, DimensionIndex rank) {
  return StrCat("array with ", StaticCastTraits<DataType>::Describe(dtype),
                " and ", StaticCastTraits<DimensionIndex>::Describe(rank));
}

Status ArrayOriginCastError(span<const Index> shape) {
  return absl::InvalidArgumentError(StrCat("Cannot translate array with shape ",
                                           shape, " to have zero origin."));
}

}  // namespace internal_array

namespace internal {

SharedElementPointer<void> AllocateArrayLike(
    DataType r, StridedLayoutView<dynamic_rank, zero_origin> source_layout,
    Index* byte_strides, IterationConstraints constraints,
    ElementInitialization initialization) {
  const auto dimension_order =
      internal_iterate::ComputeStridedLayoutDimensionIterationOrder(
          constraints, source_layout.shape(),
          span({source_layout.byte_strides().data()}));
  const DimensionIndex rank = source_layout.rank();
  std::fill_n(byte_strides, rank, Index(0));
  Index stride = r->size;
  Index num_elements = 1;
  for (auto order_i = dimension_order.size(); order_i--;) {
    const DimensionIndex source_dim = dimension_order[order_i];
    byte_strides[source_dim] = stride;
    stride *= source_layout.shape()[source_dim];
    if (internal::MulOverflow(num_elements, source_layout.shape()[source_dim],
                              &num_elements)) {
      num_elements = kInfSize;
    }
  }
  return internal::AllocateAndConstructSharedElements(num_elements,
                                                      initialization, r);
}

}  // namespace internal

void InitializeArray(
    const ArrayView<void, dynamic_rank, offset_origin>& array) {
  internal::IterateOverArrays({&array.dtype()->initialize, nullptr},
                              /*status=*/nullptr,
                              /*constraints=*/skip_repeated_elements, array);
}

const ArrayFormatOptions& ArrayFormatOptions::Default() {
  // We leak this pointer to avoid destruction order problems.
  static const ArrayFormatOptions* array_format_options =
      new ArrayFormatOptions;
  return *array_format_options;
}

void AppendToString(
    std::string* result,
    const ArrayView<const void, dynamic_rank, offset_origin>& array,
    const ArrayFormatOptions& options) {
  const bool summarize = array.num_elements() > options.summary_threshold;
  if (!array.valid()) {
    *result += "<null>";
  } else {
    internal_array::PrintArrayDimension(result, array, options, summarize);
  }
  const span<const Index> origin = array.origin();
  if (std::any_of(origin.begin(), origin.end(),
                  [](Index x) { return x != 0; })) {
    StrAppend(result, " @ ", origin);
  }
}

std::string ToString(
    const ArrayView<const void, dynamic_rank, offset_origin>& array,
    const ArrayFormatOptions& options) {
  std::string result;
  AppendToString(&result, array, options);
  return result;
}

namespace internal_array {
void PrintToOstream(
    std::ostream& os,
    const ArrayView<const void, dynamic_rank, offset_origin>& array) {
  os << ToString(array);
}
}  // namespace internal_array

absl::Status ValidateShapeBroadcast(span<const Index> source_shape,
                                    span<const Index> target_shape) {
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
                                    span<const Index> target_shape,
                                    Index* target_byte_strides) {
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateShapeBroadcast(source.shape(), target_shape));
  SharedArray<const void> target;
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

Result<SharedArray<const void>> BroadcastArray(
    SharedArrayView<const void> source, span<const Index> target_shape) {
  SharedArray<const void> target;
  target.layout().set_rank(target_shape.size());
  TENSORSTORE_RETURN_IF_ERROR(BroadcastStridedLayout(
      source.layout(), target_shape, target.byte_strides().data()));
  target.element_pointer() = std::move(source.element_pointer());
  std::copy(target_shape.begin(), target_shape.end(), target.shape().begin());
  return target;
}

Result<SharedOffsetArray<const void>> BroadcastArray(
    SharedOffsetArrayView<const void> source, BoxView<> target_domain) {
  SharedOffsetArray<const void> target;
  target.layout().set_rank(target_domain.rank());
  TENSORSTORE_RETURN_IF_ERROR(BroadcastStridedLayout(
      StridedLayoutView<>(source.shape(), source.byte_strides()),
      target_domain.shape(), target.byte_strides().data()));
  std::copy_n(target_domain.origin().begin(), target_domain.rank(),
              target.origin().begin());
  std::copy_n(target_domain.shape().begin(), target_domain.rank(),
              target.shape().begin());
  target.element_pointer() =
      AddByteOffset(std::move(source.element_pointer()),
                    internal::wrap_on_overflow::Subtract(
                        source.layout().origin_byte_offset(),
                        target.layout().origin_byte_offset()));
  return target;
}

SharedArray<const void> UnbroadcastArray(
    SharedOffsetArrayView<const void> source) {
  DimensionIndex new_rank = 0;
  for (DimensionIndex orig_dim = source.rank() - 1; orig_dim >= 0; --orig_dim) {
    if (source.shape()[orig_dim] != 1 && source.byte_strides()[orig_dim] != 0) {
      new_rank = source.rank() - orig_dim;
    }
  }

  SharedArray<const void> new_array;
  new_array.layout().set_rank(new_rank);
  for (DimensionIndex new_dim = 0; new_dim < new_rank; ++new_dim) {
    const DimensionIndex orig_dim = source.rank() + new_dim - new_rank;
    Index byte_stride = source.byte_strides()[orig_dim];
    Index size = source.shape()[orig_dim];
    if (byte_stride == 0) {
      size = 1;
    } else if (size == 1) {
      byte_stride = 0;
    }
    new_array.shape()[new_dim] = size;
    new_array.byte_strides()[new_dim] = byte_stride;
  }
  new_array.element_pointer() =
      AddByteOffset(std::move(source.element_pointer()),
                    source.layout().origin_byte_offset());
  return new_array;
}

}  // namespace tensorstore
