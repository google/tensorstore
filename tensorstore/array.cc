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

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "riegeli/varint/varint_reading.h"
#include "riegeli/varint/varint_writing.h"
#include "tensorstore/box.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/unaligned_data_type_functions.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/span.h"
#include "tensorstore/util/dimension_set.h"
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
  ABSL_CHECK_EQ(source.dtype(), dest.dtype());
  internal::IterateOverArrays({&source.dtype()->copy_assign, nullptr},
                              /*status=*/nullptr,
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
  return tensorstore::StrCat(
      "array with ", StaticCastTraits<DataType>::Describe(dtype), " and ",
      StaticCastTraits<DimensionIndex>::Describe(rank));
}

absl::Status ArrayOriginCastError(span<const Index> shape) {
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

namespace internal_array {
void UnbroadcastStridedLayout(StridedLayoutView<> layout,
                              span<Index> unbroadcast_shape,
                              span<Index> unbroadcast_byte_strides) {
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
}  // namespace internal_array

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
  internal_array::UnbroadcastStridedLayout(
      StridedLayoutView<>(
          new_rank, source.shape().data() + source.rank() - new_rank,
          source.byte_strides().data() + source.rank() - new_rank),
      new_array.shape(), new_array.byte_strides());
  new_array.element_pointer() =
      AddByteOffset(std::move(source.element_pointer()),
                    source.layout().origin_byte_offset());
  return new_array;
}

bool AreArraysSameValueEqual(const OffsetArrayView<const void>& a,
                             const OffsetArrayView<const void>& b) {
  if (a.dtype() != b.dtype()) return false;
  if (a.domain() != b.domain()) return false;
  return internal::IterateOverArrays({&a.dtype()->compare_same_value, nullptr},
                                     /*status=*/nullptr,
                                     /*constraints=*/skip_repeated_elements, a,
                                     b)
      .success;
}

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
  if (!riegeli::WriteVarint32(zero_byte_strides.bits(), sink.writer()))
    return false;

  return internal::IterateOverArrays(
             {&internal::kUnalignedDataTypeFunctions[static_cast<size_t>(
                                                         array.dtype().id())]
                   .write_native_endian,
              &sink.writer()},
             /*status=*/nullptr, {c_order, skip_repeated_elements}, array)
      .success;
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
  DimensionSet::Bits bits;
  if (!riegeli::ReadVarint32(source.reader(), bits)) return false;
  DimensionSet zero_byte_strides = DimensionSet::FromBits(bits);

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
             /*status=*/nullptr, {c_order, skip_repeated_elements}, array)
      .success;
}

template struct DecodeArray<zero_origin>;
template struct DecodeArray<offset_origin>;

}  // namespace internal_array

}  // namespace tensorstore
