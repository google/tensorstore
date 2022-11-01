// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/proto/array.h"

#include <limits>
#include <type_traits>

#include "absl/status/status.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/unaligned_data_type_functions.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace {

/// ProtoContext is a context object used for proto serialization and
/// deserialization via the tagged operator() overloads called by
/// WriteProtoDataLoopTemplate and ReadProtoDataLoopTemplate
struct WriteProtoContext {
  tensorstore::proto::Array& proto;

  /// Write a value into a proto field.
  void operator()(double item) { proto.add_double_data(item); }
  void operator()(float item) { proto.add_float_data(item); }
  void operator()(int16_t item) { proto.add_int_data(item); }
  void operator()(int32_t item) { proto.add_int_data(item); }
  void operator()(int64_t item) { proto.add_int_data(item); }
  void operator()(uint16_t item) { proto.add_uint_data(item); }
  void operator()(uint32_t item) { proto.add_uint_data(item); }
  void operator()(uint64_t item) { proto.add_uint_data(item); }
};

template <typename Element>
struct WriteProtoDataLoopTemplate {
  using ElementwiseFunctionType =
      internal::ElementwiseFunction<1, absl::Status*>;

  template <typename ArrayAccessor>
  static Index Loop(void* context, Index count,
                    internal::IterationBufferPointer source,
                    absl::Status* /*status*/) {
    auto& fn = *reinterpret_cast<WriteProtoContext*>(context);
    for (Index i = 0; i < count; ++i) {
      fn(*ArrayAccessor::template GetPointerAtOffset<Element>(source, i));
    }
    return count;
  }
};

struct ReadProtoContext {
  const tensorstore::proto::Array& proto;
  size_t index = 0;
  size_t error_count = 0;

  /// Read the next value from a proto field.
  void operator()(double& item) { item = proto.double_data(index++); }
  void operator()(float& item) { item = proto.float_data(index++); }
  void operator()(int16_t& item) {
    auto i = proto.int_data(index++);
    item = static_cast<int16_t>(i);
    if (i > std::numeric_limits<int16_t>::max() ||
        i < std::numeric_limits<int16_t>::min()) {
      error_count++;
    }
  }
  void operator()(int32_t& item) {
    auto i = proto.int_data(index++);
    item = static_cast<int32_t>(i);
    if (i > std::numeric_limits<int32_t>::max() ||
        i < std::numeric_limits<int32_t>::min()) {
      error_count++;
    }
  }
  void operator()(int64_t& item) { item = proto.int_data(index++); }

  void operator()(uint16_t& item) {
    auto i = proto.uint_data(index++);
    item = static_cast<uint16_t>(i);
    if (i > std::numeric_limits<uint16_t>::max()) {
      error_count++;
    }
  }
  void operator()(uint32_t& item) {
    auto i = proto.uint_data(index++);
    item = static_cast<uint32_t>(i);
    if (i > std::numeric_limits<uint32_t>::max()) {
      error_count++;
    }
  }
  void operator()(uint64_t& item) { item = proto.uint_data(index++); }
};

template <typename Element>
struct ReadProtoDataLoopTemplate {
  using ElementwiseFunctionType =
      internal::ElementwiseFunction<1, absl::Status*>;
  template <typename ArrayAccessor>
  static Index Loop(void* context, Index count,
                    internal::IterationBufferPointer source,
                    absl::Status* /*status*/) {
    auto& fn = *reinterpret_cast<ReadProtoContext*>(context);
    for (Index i = 0; i < count; ++i) {
      fn(*ArrayAccessor::template GetPointerAtOffset<Element>(source, i));
    }
    return count;
  }
};

/// kProtoFunctions is a mapping from DataTypeId to the appropriate
/// read/write function, sometimes null. When null, the fallback mechanism
/// is to use the serialization functions for that type. This fallback is
/// used for bytes, complex, json, etc.
struct ProtoArrayDataTypeFunctions {
  const internal::ElementwiseFunction<1, absl::Status*>* write_fn = nullptr;
  const internal::ElementwiseFunction<1, absl::Status*>* read_fn = nullptr;
};

const std::array<ProtoArrayDataTypeFunctions, kNumDataTypeIds> kProtoFunctions =
    MapCanonicalDataTypes([](auto dtype) {
      using T = typename decltype(dtype)::Element;
      ProtoArrayDataTypeFunctions functions;
      if constexpr (std::is_invocable_v<ReadProtoContext, T&>) {
        functions.write_fn =
            internal::GetElementwiseFunction<WriteProtoDataLoopTemplate<T>>();
        functions.read_fn =
            internal::GetElementwiseFunction<ReadProtoDataLoopTemplate<T>>();
      }
      return functions;
    });

}  // namespace

void EncodeToProtoImpl(::tensorstore::proto::Array& proto,
                       OffsetArrayView<const void> array) {
  const auto dtype = array.dtype();
  proto.set_dtype(std::string(dtype.name()));
  {
    bool all_zero = true;
    for (Index x : array.origin()) {
      proto.add_origin(x);
      all_zero &= (x == 0);
    }
    if (all_zero) proto.clear_origin();
  }
  for (Index x : array.shape()) {
    proto.add_shape(x);
  }

  /// Detect zero byte stride dimensions that are not c_order initialized; in
  /// in which case we write out the the zero-strides.
  {
    DimensionSet zero_byte_strides(false);
    for (DimensionIndex i = 0; i < array.rank(); i++) {
      zero_byte_strides[i] =
          (array.byte_strides()[i] == 0 && array.shape()[i] != 1);
    }
    if (zero_byte_strides) {
      proto.set_zero_byte_strides_bitset(zero_byte_strides.bits());
    }
  }

  const size_t index = static_cast<size_t>(dtype.id());
  if (kProtoFunctions[index].write_fn) {
    /// Reserve space in the protocol buffer output array.
    if (dtype.id() == DataTypeIdOf<int16_t> ||
        dtype.id() == DataTypeIdOf<int32_t> ||
        dtype.id() == DataTypeIdOf<int64_t>) {
      proto.mutable_int_data()->Reserve(array.num_elements());
    } else if (dtype.id() == DataTypeIdOf<uint16_t> ||
               dtype.id() == DataTypeIdOf<uint32_t> ||
               dtype.id() == DataTypeIdOf<uint64_t>) {
      proto.mutable_uint_data()->Reserve(array.num_elements());
    } else if (dtype.id() == DataTypeIdOf<double>) {
      proto.mutable_double_data()->Reserve(array.num_elements());
    } else if (dtype.id() == DataTypeIdOf<float>) {
      proto.mutable_float_data()->Reserve(array.num_elements());
    }

    // Use a discrete function.
    WriteProtoContext context{proto};
    internal::IterateOverArrays({kProtoFunctions[index].write_fn, &context},
                                /*status=*/nullptr,
                                {c_order, skip_repeated_elements}, array);
  } else {
    // Use the serialization function.
    proto.mutable_void_data()->reserve(dtype.size() * array.num_elements());
    riegeli::StringWriter writer(proto.mutable_void_data());
    internal::IterateOverArrays(
        {&internal::kUnalignedDataTypeFunctions[index].write_native_endian,
         &writer},
        /*status=*/nullptr, {c_order, skip_repeated_elements}, array);
    writer.Close();
  }
}

Result<SharedArray<void, dynamic_rank, offset_origin>> ParseArrayFromProto(
    const ::tensorstore::proto::Array& proto, ArrayOriginKind origin_kind,
    DimensionIndex rank_constraint) {
  SharedArray<void, dynamic_rank, offset_origin> array;

  DataType dtype = GetDataType(proto.dtype());
  if (!dtype.valid()) {
    return absl::DataLossError(
        "Cannot deserialize array with unspecified data type");
  }
  // Rank is always derived from the shape array.
  const size_t rank = proto.shape_size();
  if (rank_constraint != dynamic_rank && rank != rank_constraint) {
    return absl::InvalidArgumentError("Proto array rank mismatch");
  }
  if (rank > kMaxRank) {
    return absl::InvalidArgumentError("Proto rank exceeds maximum rank");
  }

  array.layout().set_rank(rank);
  std::copy(proto.shape().begin(), proto.shape().end(),
            array.layout().shape().begin());
  std::fill(array.layout().origin().begin(), array.layout().origin().end(),
            Index(0));
  if (proto.origin_size() > 0 &&
      std::any_of(proto.origin().begin(), proto.origin().end(),
                  [](auto x) { return x != 0; })) {
    if (origin_kind == zero_origin) {
      return absl::InvalidArgumentError(
          "Proto zero_origin array has non-zero origin");
    }
    if (proto.origin_size() != rank) {
      return absl::InvalidArgumentError("Proto origin/rank mismatch");
    }
    std::copy(proto.origin().begin(), proto.origin().end(),
              array.layout().origin().begin());
  }

  // Set the byte strides and check for valid intervals / data overflow.
  Index num_elements = 1;
  {
    DimensionSet zero_byte_strides =
        (proto.has_zero_byte_strides_bitset())
            ? DimensionSet::FromBits(proto.zero_byte_strides_bitset())
            : DimensionSet(false);

    for (DimensionIndex i = rank - 1; i >= 0; --i) {
      if (!IndexInterval::ValidSized(array.origin()[i], array.shape()[i])) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Proto origin and shape of {", array.origin()[i], ", ",
            array.shape()[i],
            "} do not specify a valid IndexInterval for rank ", i));
      }
      if (zero_byte_strides[i]) {
        array.layout().byte_strides()[i] = 0;
      } else {
        array.layout().byte_strides()[i] = 1;
        if (internal::MulOverflow(num_elements, array.shape()[i],
                                  &num_elements)) {
          return absl::DataLossError(
              tensorstore::StrCat("Invalid array shape ", array.shape()));
        }
      }
    }
  }

  array.element_pointer() = tensorstore::AllocateArrayElementsLike<void>(
      array.layout(), array.byte_strides().data(),
      {c_order, skip_repeated_elements}, default_init, dtype);

  const size_t index = static_cast<size_t>(dtype.id());

  if (kProtoFunctions[index].read_fn && proto.void_data().empty()) {
    // This branch encodes data into proto fields; the data is presently
    // always encoded in c_order and the zero_byte_strides is empty
    // in the default case.

    // Validate data lengths.
    if ((dtype.id() == DataTypeIdOf<int16_t> ||
         dtype.id() == DataTypeIdOf<int32_t> ||
         dtype.id() == DataTypeIdOf<int64_t>)&&proto.int_data_size() !=
        num_elements) {
      return absl::DataLossError("proto int_data incomplete");
    }
    if ((dtype.id() == DataTypeIdOf<uint16_t> ||
         dtype.id() == DataTypeIdOf<uint32_t> ||
         dtype.id() == DataTypeIdOf<uint64_t>)&&proto.uint_data_size() !=
        num_elements) {
      return absl::DataLossError("proto uint_data incomplete");
    }
    if (dtype.id() == DataTypeIdOf<double> &&
        proto.double_data_size() != num_elements) {
      return absl::DataLossError("proto double_data incomplete");
    }
    if (dtype.id() == DataTypeIdOf<float> &&
        proto.float_data_size() != num_elements) {
      return absl::DataLossError("proto float_data incomplete");
    }

    ReadProtoContext context{proto};
    internal::IterateOverArrays({kProtoFunctions[index].read_fn, &context},
                                /*status*/ nullptr,
                                {c_order, skip_repeated_elements}, array);
    if (context.error_count > 0) {
      return absl::DataLossError("Array element truncated");
    }
  } else {
    // Otherwise we use the serialization function to decode from the
    // void field.
    riegeli::StringReader reader(proto.void_data());
    internal::IterateOverArrays(
        {&internal::kUnalignedDataTypeFunctions[index].read_native_endian,
         &reader},
        /*status=*/nullptr, {c_order, skip_repeated_elements}, array);

    if (!reader.VerifyEndAndClose()) return reader.status();
  }

  return array;
}

}  // namespace tensorstore
