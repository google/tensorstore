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
  if (a.data_type() != b.data_type()) return false;
  if (!internal::RangesEqual(a.shape(), b.shape())) return false;
  return internal::IterateOverArrays({&a.data_type()->compare_equal, nullptr},
                                     /*status=*/nullptr,
                                     /*constraints=*/skip_repeated_elements, a,
                                     b)
      .success;
}

void CopyArrayImplementation(
    const ArrayView<const void, dynamic_rank, offset_origin>& source,
    const ArrayView<void, dynamic_rank, offset_origin>& dest) {
  TENSORSTORE_CHECK(source.data_type() == dest.data_type());
  internal::IterateOverArrays({&source.data_type()->copy_assign, nullptr},
                              /*status=*/nullptr,
                              /*constraints=*/skip_repeated_elements, source,
                              dest);
}

Status CopyConvertedArrayImplementation(
    const ArrayView<const void, dynamic_rank, offset_origin>& source,
    const ArrayView<void, dynamic_rank, offset_origin>& dest) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto r, internal::GetDataTypeConverterOrError(source.data_type(),
                                                    dest.data_type()));
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
  if (a.data_type() != b.data_type()) return false;
  if (a.domain() != b.domain()) return false;
  return internal::IterateOverArrays({&a.data_type()->compare_equal, nullptr},
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
    array.data_type()->append_to_string(result, array.data());
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

std::string DescribeForCast(DataType data_type, DimensionIndex rank) {
  return StrCat("array with ", StaticCastTraits<DataType>::Describe(data_type),
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
  internal::IterateOverArrays({&array.data_type()->initialize, nullptr},
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

}  // namespace tensorstore
