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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <ostream>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "riegeli/varint/varint_reading.h"
#include "riegeli/varint/varint_writing.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/unaligned_data_type_functions.h"
#include "tensorstore/rank.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/span.h"  // IWYU pragma: keep
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/internal/iterate_impl.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_array {

namespace {
template <ArrayOriginKind OKind>
bool CompareArraysEqualImpl(const ArrayView<const void, dynamic_rank, OKind>& a,
                            const ArrayView<const void, dynamic_rank, OKind>& b,
                            EqualityComparisonKind comparison_kind) {
  if (a.dtype() != b.dtype()) return false;
  const auto& funcs =
      a.dtype()->compare_equal[static_cast<size_t>(comparison_kind)];
  if (IsBroadcastScalar(a)) {
    return internal::IterateOverArrays(
        {&funcs.array_scalar, nullptr},
        /*arg=*/const_cast<void*>(a.byte_strided_origin_pointer().get()),
        /*constraints=*/skip_repeated_elements, b);
  }
  if (IsBroadcastScalar(b)) {
    return internal::IterateOverArrays(
        {&funcs.array_scalar, nullptr},
        /*arg=*/const_cast<void*>(b.byte_strided_origin_pointer().get()),
        /*constraints=*/skip_repeated_elements, a);
  }
  return internal::IterateOverArrays({&funcs.array_array, nullptr},
                                     /*arg=*/nullptr,
                                     /*constraints=*/skip_repeated_elements, a,
                                     b);
}
}  // namespace

bool CompareArraysEqual(
    const ArrayView<const void, dynamic_rank, zero_origin>& a,
    const ArrayView<const void, dynamic_rank, zero_origin>& b,
    EqualityComparisonKind comparison_kind) {
  if (!internal::RangesEqual(a.shape(), b.shape())) return false;
  return CompareArraysEqualImpl<zero_origin>(a, b, comparison_kind);
}

bool CompareArraysEqual(
    const ArrayView<const void, dynamic_rank, offset_origin>& a,
    const ArrayView<const void, dynamic_rank, offset_origin>& b,
    EqualityComparisonKind comparison_kind) {
  if (a.domain() != b.domain()) return false;
  return CompareArraysEqualImpl<offset_origin>(a, b, comparison_kind);
}

void CopyArrayImplementation(
    const ArrayView<const void, dynamic_rank, offset_origin>& source,
    const ArrayView<void, dynamic_rank, offset_origin>& dest) {
  ABSL_CHECK_EQ(source.dtype(), dest.dtype());
  internal::IterateOverArrays({&source.dtype()->copy_assign, nullptr},
                              /*arg=*/nullptr,
                              /*constraints=*/skip_repeated_elements, source,
                              dest);
}

absl::Status CopyConvertedArrayImplementation(
    const ArrayView<const void, dynamic_rank, offset_origin>& source,
    const ArrayView<void, dynamic_rank, offset_origin>& dest) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto r, internal::GetDataTypeConverterOrError(
                                           source.dtype(), dest.dtype()));
  absl::Status status;
  if (!internal::IterateOverArrays(r.closure,
                                   /*arg=*/&status,
                                   /*constraints=*/skip_repeated_elements,
                                   source, dest)) {
    return internal::GetElementCopyErrorStatus(std::move(status));
  }
  return status;
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
  return tensorstore::StrCat(
      "array with ", StaticCastTraits<DataType>::Describe(dtype), " and ",
      StaticCastTraits<DimensionIndex>::Describe(rank));
}

absl::Status ArrayOriginCastError(tensorstore::span<const Index> shape) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Cannot translate array with shape ", shape, " to have zero origin."));
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
          tensorstore::span({source_layout.byte_strides().data()}));
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
                              /*arg=*/nullptr,
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
  const tensorstore::span<const Index> origin = array.origin();
  if (std::any_of(origin.begin(), origin.end(),
                  [](Index x) { return x != 0; })) {
    tensorstore::StrAppend(result, " @ ", origin);
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

namespace internal_array {
void UnbroadcastStridedLayout(
    StridedLayoutView<> layout, tensorstore::span<Index> unbroadcast_shape,
    tensorstore::span<Index> unbroadcast_byte_strides) {
  assert(unbroadcast_shape.size() == layout.rank());
  assert(unbroadcast_byte_strides.size() == layout.rank());
  for (DimensionIndex i = 0; i < layout.rank(); ++i) {
    Index byte_stride = layout.byte_strides()[i];
    Index size = layout.shape()[i];
    if (byte_stride == 0) {
      size = 1;
    } else if (size == 1) {
      byte_stride = 0;
    }
    unbroadcast_shape[i] = size;
    unbroadcast_byte_strides[i] = byte_stride;
  }
}
void UnbroadcastStridedLayout(StridedLayoutView<> layout,
                              StridedLayout<>& unbroadcast_layout) {
  DimensionIndex new_rank = 0;
  for (DimensionIndex orig_dim = layout.rank() - 1; orig_dim >= 0; --orig_dim) {
    if (layout.shape()[orig_dim] != 1 && layout.byte_strides()[orig_dim] != 0) {
      new_rank = layout.rank() - orig_dim;
    }
  }

  unbroadcast_layout.set_rank(new_rank);
  internal_array::UnbroadcastStridedLayout(
      StridedLayoutView<>(
          new_rank, layout.shape().data() + layout.rank() - new_rank,
          layout.byte_strides().data() + layout.rank() - new_rank),
      unbroadcast_layout.shape(), unbroadcast_layout.byte_strides());
}

}  // namespace internal_array

namespace internal_array {

bool EncodeArray(serialization::EncodeSink& sink,
                 OffsetArrayView<const void> array,
                 ArrayOriginKind origin_kind) {
  if (!array.dtype().valid()) {
    sink.Fail(absl::InvalidArgumentError(
        "Cannot serialize array with unspecified data type"));
    return false;
  }
  if (!serialization::Encode(sink, array.dtype())) return false;
  if (!serialization::RankSerializer::Encode(sink, array.rank())) return false;
  if (!serialization::Encode(sink, array.shape())) return false;
  if (origin_kind == offset_origin) {
    if (!serialization::Encode(sink, array.origin())) return false;
  }
  // Record the zero byte_strides.
  const DimensionIndex rank = array.rank();
  DimensionSet zero_byte_strides(false);
  for (DimensionIndex i = 0; i < rank; i++) {
    zero_byte_strides[i] =
        (array.byte_strides()[i] == 0 && array.shape()[i] != 1);
  }
  if (!riegeli::WriteVarint32(zero_byte_strides.to_uint(), sink.writer()))
    return false;

  return internal::IterateOverArrays(
      {&internal::kUnalignedDataTypeFunctions[static_cast<size_t>(
                                                  array.dtype().id())]
            .write_native_endian,
       &sink.writer()},
      /*arg=*/nullptr, {c_order, skip_repeated_elements}, array);
}

template <ArrayOriginKind OriginKind>
bool DecodeArray<OriginKind>::Decode(
    serialization::DecodeSource& source,
    SharedArray<void, dynamic_rank, OriginKind>& array,
    DataType data_type_constraint, DimensionIndex rank_constraint) {
  DataType dtype;
  if (!serialization::Decode(source, dtype)) return false;
  if (!dtype.valid()) {
    source.Fail(absl::DataLossError(
        "Cannot deserialize array with unspecified data type"));
    return false;
  }
  if (data_type_constraint.valid() && data_type_constraint != dtype) {
    source.Fail(absl::DataLossError(
        tensorstore::StrCat("Expected data type of ", data_type_constraint,
                            " but received: ", dtype)));
    return false;
  }
  DimensionIndex rank;
  if (!serialization::RankSerializer::Decode(source, rank)) return false;
  if (rank_constraint != dynamic_rank && rank != rank_constraint) {
    source.Fail(absl::DataLossError(tensorstore::StrCat(
        "Expected rank of ", rank_constraint, " but received: ", rank)));
    return false;
  }
  array.layout().set_rank(rank);
  if (!serialization::Decode(source, array.shape())) return false;
  if constexpr (OriginKind == offset_origin) {
    if (!serialization::Decode(source, array.origin())) return false;
  }
  uint32_t bits;
  if (!riegeli::ReadVarint32(source.reader(), bits)) return false;
  DimensionSet zero_byte_strides = DimensionSet::FromUint(bits);

  Index num_bytes = dtype.valid() ? dtype.size() : 0;
  for (DimensionIndex i = 0; i < rank; ++i) {
    if (zero_byte_strides[i]) {
      array.byte_strides()[i] = 0;
    } else {
      array.byte_strides()[i] = 1;
      if (internal::MulOverflow(num_bytes, array.shape()[i], &num_bytes)) {
        source.Fail(serialization::DecodeError(
            tensorstore::StrCat("Invalid array shape ", array.shape())));
        return false;
      }
    }
  }
  array.element_pointer() = tensorstore::AllocateArrayElementsLike<void>(
      array.layout(), array.byte_strides().data(),
      {c_order, skip_repeated_elements}, default_init, dtype);
  return internal::IterateOverArrays(
      {&internal::kUnalignedDataTypeFunctions[static_cast<size_t>(
                                                  array.dtype().id())]
            .read_native_endian,
       &source.reader()},
      /*arg=*/nullptr, {c_order, skip_repeated_elements}, array);
}

template struct DecodeArray<zero_origin>;
template struct DecodeArray<offset_origin>;

}  // namespace internal_array

}  // namespace tensorstore
