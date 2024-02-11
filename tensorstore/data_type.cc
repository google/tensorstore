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

#include "tensorstore/data_type.h"

#include <stdint.h>

#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/json/same.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/utf8.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

// On all common platforms, these assumptions holds.
//
// If TensorStore needs to support a platform where they do not hold, additional
// specializations of CanonicalElementType can be defined.
static_assert(std::is_same_v<signed char, ::tensorstore::dtypes::int8_t>);
static_assert(std::is_same_v<short, ::tensorstore::dtypes::int16_t>);  // NOLINT
static_assert(std::is_same_v<int, ::tensorstore::dtypes::int32_t>);
static_assert(std::is_same_v<unsigned char, ::tensorstore::dtypes::uint8_t>);
static_assert(
    std::is_same_v<unsigned short, ::tensorstore::dtypes::uint16_t>);  // NOLINT
static_assert(std::is_same_v<unsigned int, ::tensorstore::dtypes::uint32_t>);

std::ostream& operator<<(std::ostream& os, DataType r) {
  if (r.valid()) return os << r.name();
  return os << "<unspecified>";
}

void* AllocateAndConstruct(std::ptrdiff_t n,
                           ElementInitialization initialization, DataType r) {
  assert(n >= 0);
  assert(n < kInfSize);
  std::size_t alignment =
      RoundUpTo(static_cast<std::size_t>(r->alignment), sizeof(void*));
  std::size_t total_size =
      RoundUpTo(static_cast<std::size_t>(r->size * n), alignment);
  struct AlignedDeleter {
    std::align_val_t alignment;
    void operator()(void* p) const { ::operator delete(p, alignment); }
  };
  std::unique_ptr<void, AlignedDeleter> ptr(
      alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__
          ? ::operator new(total_size, std::align_val_t(alignment))
          : ::operator new(total_size),
      AlignedDeleter{std::align_val_t(alignment)});
  if (initialization == value_init) {
    // For simplicity, we just implement value initialization by zero
    // initializing prior to default construction.  For some types, like
    // std::string, this is unnecessary, but there is no C++ type trait that
    // allows us to detect whether there is a user-provided default
    // constructor.
    std::memset(ptr.get(), 0, total_size);
  }
  r->construct(n, ptr.get());
  return ptr.release();
}

void DestroyAndFree(std::ptrdiff_t n, DataType r, void* ptr) {
  r->destroy(n, ptr);
  if (r->alignment >
      static_cast<std::ptrdiff_t>(__STDCPP_DEFAULT_NEW_ALIGNMENT__)) {
    ::operator delete(ptr, std::align_val_t(r->alignment));
  } else {
    ::operator delete(ptr);
  }
}

template <>
std::shared_ptr<void> AllocateAndConstructShared<void>(
    std::ptrdiff_t n, ElementInitialization initialization, DataType r) {
  if (void* ptr = AllocateAndConstruct(n, initialization, r)) {
    return std::shared_ptr<void>(ptr,
                                 [r, n](void* x) { DestroyAndFree(n, r, x); });
  }
  return nullptr;
}

std::string StaticCastTraits<DataType>::Describe(DataType dtype) {
  if (!dtype.valid()) return "dynamic data type";
  return tensorstore::StrCat("data type of ", dtype);
}

namespace internal_data_type {
#define TENSORSTORE_INTERNAL_DO_INSTANTIATION(X, ...) \
  TENSORSTORE_DATA_TYPE_EXPLICIT_INSTANTIATION(X)     \
  /**/
TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_INSTANTIATION)
#undef TENSORSTORE_INTERNAL_DEFINE_ELEMENT_REPRESENTATION_INSTANTIATION
}  // namespace internal_data_type

DataType GetDataType(std::string_view id) {
#define TENSORSTORE_INTERNAL_MATCH_TYPE(X, ...)     \
  if (id == std::string_view(#X, sizeof(#X) - 3)) { \
    return dtype_v<::tensorstore::dtypes::X>;       \
  }                                                 \
  /**/
  TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_MATCH_TYPE)
#undef TENSORSTORE_INTERNAL_MATCH_TYPE
  return DataType();
}

/// Define conversions between canonical data types.

namespace {

template <typename T>
struct NumberToStringCanonicalType {
  using type = T;
};

template <>
struct NumberToStringCanonicalType<::tensorstore::dtypes::float8_e4m3fn_t> {
  using type = float;
};

template <>
struct NumberToStringCanonicalType<::tensorstore::dtypes::float8_e4m3fnuz_t> {
  using type = float;
};

template <>
struct NumberToStringCanonicalType<
    ::tensorstore::dtypes::float8_e4m3b11fnuz_t> {
  using type = float;
};

template <>
struct NumberToStringCanonicalType<::tensorstore::dtypes::float8_e5m2_t> {
  using type = float;
};

template <>
struct NumberToStringCanonicalType<::tensorstore::dtypes::float8_e5m2fnuz_t> {
  using type = float;
};

template <>
struct NumberToStringCanonicalType<::tensorstore::dtypes::float16_t> {
  using type = float;
};

template <>
struct NumberToStringCanonicalType<::tensorstore::dtypes::bfloat16_t> {
  using type = float;
};

template <>
struct NumberToStringCanonicalType<::tensorstore::dtypes::int4_t> {
  using type = int16_t;
};

template <>
struct NumberToStringCanonicalType<int8_t> {
  using type = int16_t;
};

template <>
struct NumberToStringCanonicalType<uint8_t> {
  using type = uint16_t;
};

template <typename T>
void NumberToString(T x, std::string* out) {
#if 0
  // std::to_chars not implemented for floating-point types in Clang and GCC
  constexpr size_t kBufferSize = 64;
  char buffer[kBufferSize];
  auto r = std::to_chars(buffer, buffer + kBufferSize, x);
  assert(r.ec == std::errc{});
  out->assign(buffer, r.ptr);
#else
  out->clear();
  absl::StrAppend(
      out, static_cast<typename NumberToStringCanonicalType<T>::type>(x));
#endif
}

template <typename T>
void ComplexToString(std::complex<T> x, std::string* out) {
#if 0
  constexpr size_t kBufferSize = 128;
  char buffer[kBufferSize];
  buffer[0] = '(';
  auto r = std::to_chars(buffer + 1, buffer + kBufferSize - 2, x.real());
  assert(r.ec == std::errc{});
  *r.ptr = ',';
  r = std::to_chars(r.ptr, buffer + kBufferSize - 1, x.imag());
  assert(r.ec == std::errc{});
  *r.ptr += ')';
  out->assign(buffer, r.ptr);
#else
  out->clear();
  absl::StrAppend(out, "(", x.real(), ",", x.imag(), ")");
#endif
}

}  // namespace

template <typename T>
struct ConvertDataType<std::complex<T>, ::tensorstore::dtypes::json_t> {
  void operator()(const std::complex<T>* from,
                  ::tensorstore::dtypes::json_t* to, void*) const {
    *to = ::tensorstore::dtypes::json_t::array_t{from->real(), from->imag()};
  }
};

template <typename T>
struct ConvertDataType<std::complex<T>, ::tensorstore::dtypes::string_t> {
  void operator()(const std::complex<T>* from,
                  ::tensorstore::dtypes::string_t* to, void*) const {
    ComplexToString(*from, to);
  }
};

template <typename T>
struct ConvertDataType<std::complex<T>, ::tensorstore::dtypes::ustring_t> {
  void operator()(const std::complex<T>* from,
                  ::tensorstore::dtypes::ustring_t* to, void*) const {
    ComplexToString(*from, &to->utf8);
  }
};

template <>
struct ConvertDataType<::tensorstore::dtypes::float16_t,
                       ::tensorstore::dtypes::json_t> {
  void operator()(const ::tensorstore::dtypes::float16_t* from,
                  ::tensorstore::dtypes::json_t* to, void*) const {
    *to = static_cast<double>(*from);
  }
};

namespace internal_data_type {

struct JsonIntegerConvertDataType {
  template <typename To>
  bool operator()(const ::tensorstore::dtypes::json_t* from, To* to,
                  void* arg) const {
    auto* status = static_cast<absl::Status*>(arg);
    auto s = internal_json::JsonRequireInteger(*from, to, /*strict=*/false);
    if (s.ok()) return true;
    *status = s;
    return false;
  }
};

struct JsonFloatConvertDataType {
  template <typename To>
  bool operator()(const ::tensorstore::dtypes::json_t* from, To* to,
                  void* arg) const {
    auto* status = static_cast<absl::Status*>(arg);
    double value;
    auto s = internal_json::JsonRequireValueAs(*from, &value, /*strict=*/false);
    if (s.ok()) {
      *to = static_cast<To>(value);
      return true;
    }
    *status = s;
    return false;
  }
};

struct ComplexNumericConvertDataType {
  template <typename T, typename To>
  void operator()(const std::complex<T>* from, To* to, void*) const {
    *to = static_cast<To>(from->real());
  }
};

struct NumericStringConvertDataType {
  template <typename From>
  void operator()(const From* from, ::tensorstore::dtypes::string_t* to,
                  void*) const {
    NumberToString(*from, to);
  }
};

struct NumericUstringConvertDataType {
  template <typename From>
  void operator()(const From* from, ::tensorstore::dtypes::ustring_t* to,
                  void*) const {
    NumberToString(*from, &to->utf8);
  }
};

}  // namespace internal_data_type

#define TENSORSTORE_INTERNAL_CONVERT_INT(T, ...)                     \
  template <>                                                        \
  struct ConvertDataType<::tensorstore::dtypes::T,                   \
                         ::tensorstore::dtypes::string_t>            \
      : public internal_data_type::NumericStringConvertDataType {};  \
  template <>                                                        \
  struct ConvertDataType<::tensorstore::dtypes::T,                   \
                         ::tensorstore::dtypes::ustring_t>           \
      : public internal_data_type::NumericUstringConvertDataType {}; \
  template <>                                                        \
  struct ConvertDataType<::tensorstore::dtypes::json_t,              \
                         ::tensorstore::dtypes::T>                   \
      : public internal_data_type::JsonIntegerConvertDataType {};    \
  /**/

#define TENSORSTORE_INTERNAL_CONVERT_FLOAT(T, ...)                   \
  template <>                                                        \
  struct ConvertDataType<::tensorstore::dtypes::T,                   \
                         ::tensorstore::dtypes::string_t>            \
      : public internal_data_type::NumericStringConvertDataType {};  \
  template <>                                                        \
  struct ConvertDataType<::tensorstore::dtypes::T,                   \
                         ::tensorstore::dtypes::ustring_t>           \
      : public internal_data_type::NumericUstringConvertDataType {}; \
  template <>                                                        \
  struct ConvertDataType<::tensorstore::dtypes::json_t,              \
                         ::tensorstore::dtypes::T>                   \
      : public internal_data_type::JsonFloatConvertDataType {};      \
  /**/

TENSORSTORE_FOR_EACH_INT_DATA_TYPE(TENSORSTORE_INTERNAL_CONVERT_INT)
TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE(TENSORSTORE_INTERNAL_CONVERT_FLOAT)

#undef TENSORSTORE_INTERNAL_CONVERT_INT
#undef TENSORSTORE_INTERNAL_CONVERT_FLOAT

#define TENSORSTORE_INTERNAL_INHERITED_CONVERT(FROM, TO, PARENT) \
  template <>                                                    \
  struct ConvertDataType<FROM, TO> : public PARENT {};           \
  /**/

// [BEGIN GENERATED: generate_data_type.py]

TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::int4_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::int8_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::uint8_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::int16_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::uint16_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::int32_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::uint32_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::int64_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::uint64_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::int4_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::int8_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::uint8_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::int16_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::uint16_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::int32_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::uint32_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::int64_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::uint64_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::float8_e4m3fn_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t,
    ::tensorstore::dtypes::float8_e4m3fnuz_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t,
    ::tensorstore::dtypes::float8_e4m3b11fnuz_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::float8_e5m2_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t,
    ::tensorstore::dtypes::float8_e5m2fnuz_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::float16_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::bfloat16_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::float32_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::float64_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::float8_e4m3fn_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t,
    ::tensorstore::dtypes::float8_e4m3fnuz_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t,
    ::tensorstore::dtypes::float8_e4m3b11fnuz_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::float8_e5m2_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t,
    ::tensorstore::dtypes::float8_e5m2fnuz_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::float16_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::bfloat16_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::float32_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::float64_t,
    internal_data_type::ComplexNumericConvertDataType)

// [END GENERATED: generate_data_type.py]

// TODO(jbms): implement json -> complex conversion
// TODO(jbms): implement string -> number conversion
// TODO(jbms): implement string -> complex conversion

template <>
struct ConvertDataType<::tensorstore::dtypes::ustring_t,
                       ::tensorstore::dtypes::json_t> {
  void operator()(const ::tensorstore::dtypes::ustring_t* from,
                  ::tensorstore::dtypes::json_t* to, void*) const {
    *to = from->utf8;
  }
};

template <>
struct ConvertDataType<::tensorstore::dtypes::json_t, bool> {
  bool operator()(const ::tensorstore::dtypes::json_t* from, bool* to,
                  void* arg) const {
    auto* status = static_cast<absl::Status*>(arg);
    auto s = internal_json::JsonRequireValueAs(*from, to, /*strict=*/false);
    if (s.ok()) return true;
    *status = s;
    return false;
  }
};

template <>
struct ConvertDataType<::tensorstore::dtypes::json_t,
                       ::tensorstore::dtypes::string_t> {
  bool operator()(const ::tensorstore::dtypes::json_t* from,
                  ::tensorstore::dtypes::string_t* to, void* arg) const {
    auto* status = static_cast<absl::Status*>(arg);
    auto s = internal_json::JsonRequireValueAs(*from, to, /*strict=*/false);
    if (s.ok()) return true;
    *status = s;
    return false;
  }
};

template <>
struct ConvertDataType<::tensorstore::dtypes::json_t,
                       ::tensorstore::dtypes::ustring_t> {
  bool operator()(const ::tensorstore::dtypes::json_t* from,
                  ::tensorstore::dtypes::ustring_t* to, void* arg) const {
    auto* status = static_cast<absl::Status*>(arg);
    auto s =
        internal_json::JsonRequireValueAs(*from, &to->utf8, /*strict=*/false);
    if (s.ok()) return true;
    *status = s;
    return false;
  }
};

template <>
struct ConvertDataType<::tensorstore::dtypes::string_t,
                       ::tensorstore::dtypes::ustring_t> {
  bool operator()(const ::tensorstore::dtypes::string_t* from,
                  ::tensorstore::dtypes::ustring_t* to, void* arg) const {
    auto* status = static_cast<absl::Status*>(arg);
    if (internal::IsValidUtf8(*from)) {
      to->utf8 = *from;
      return true;
    }
    *status = absl::InvalidArgumentError("Invalid UTF-8 sequence encountered");
    return false;
  }
};

template <>
struct ConvertDataType<::tensorstore::dtypes::string_t,
                       ::tensorstore::dtypes::json_t> {
  bool operator()(const ::tensorstore::dtypes::string_t* from,
                  ::tensorstore::dtypes::json_t* to, void* arg) const {
    auto* status = static_cast<absl::Status*>(arg);
    if (internal::IsValidUtf8(*from)) {
      *to = *from;
      return true;
    }
    *status = absl::InvalidArgumentError("Invalid UTF-8 sequence encountered");
    return false;
  }
};

namespace internal {
const std::array<DataTypeOperations::CanonicalConversionOperations,
                 kNumDataTypeIds>
    canonical_data_type_conversions = MapCanonicalDataTypes([](auto d) {
      using X = typename decltype(d)::Element;
      return internal_data_type::GetConvertToCanonicalOperations<X>();
    });

DataTypeConversionLookupResult GetDataTypeConverter(DataType from,
                                                    DataType to) {
  assert(from.valid());
  assert(to.valid());
  DataTypeConversionLookupResult lookup_result = {};
  if (from == to) {
    lookup_result.closure.function = &from->copy_assign;
    lookup_result.flags = DataTypeConversionFlags::kSupported |
                          DataTypeConversionFlags::kCanReinterpretCast |
                          DataTypeConversionFlags::kIdentity |
                          DataTypeConversionFlags::kSafeAndImplicit;
    return lookup_result;
  }
  const DataTypeId from_id = from->id;
  const DataTypeId to_id = to->id;
  if (from_id == DataTypeId::custom || to_id == DataTypeId::custom) {
    return lookup_result;
  }
  lookup_result.flags =
      canonical_data_type_conversions[static_cast<size_t>(from_id)]
          .flags[static_cast<size_t>(to_id)];
  if ((lookup_result.flags & DataTypeConversionFlags::kCanReinterpretCast) ==
      DataTypeConversionFlags::kCanReinterpretCast) {
    lookup_result.closure.function = &from->copy_assign;
  } else {
    lookup_result.closure.function =
        &canonical_data_type_conversions[static_cast<size_t>(from_id)]
             .convert[static_cast<size_t>(to_id)];
  }
  return lookup_result;
}

Result<DataTypeConversionLookupResult> GetDataTypeConverterOrError(
    DataType from, DataType to, DataTypeConversionFlags required_flags) {
  auto lookup_result = GetDataTypeConverter(from, to);
  required_flags = (required_flags | DataTypeConversionFlags::kSupported);
  if ((lookup_result.flags & required_flags) != required_flags) {
    if (!!(lookup_result.flags & DataTypeConversionFlags::kSupported)) {
      if (!!(required_flags & DataTypeConversionFlags::kSafeAndImplicit) &&
          !(lookup_result.flags & DataTypeConversionFlags::kSafeAndImplicit)) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Explicit data type conversion required to convert ", from, " -> ",
            to));
      }
    }
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Cannot convert ", from, " -> ", to));
  }
  return lookup_result;
}

absl::Status NonSerializableDataTypeError(DataType dtype) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Cannot serialize custom data type: ", dtype->type.name()));
}
}  // namespace internal

namespace internal_data_type {

template <>
bool CompareIdentical<::tensorstore::dtypes::json_t>(
    const ::tensorstore::dtypes::json_t& a,
    const ::tensorstore::dtypes::json_t& b) {
  return internal_json::JsonSame(a, b);
}

}  // namespace internal_data_type

namespace serialization {
bool Serializer<DataType>::Encode(EncodeSink& sink, const DataType& value) {
  if (!value.valid()) {
    return serialization::Encode(sink, std::string_view());
  }
  if (value.id() == DataTypeId::custom) {
    sink.Fail(internal::NonSerializableDataTypeError(value));
    return false;
  }
  return serialization::Encode(sink, value.name());
}

bool Serializer<DataType>::Decode(DecodeSource& source, DataType& value) {
  std::string_view name;
  if (!serialization::Decode(source, name)) return false;
  if (name.empty()) {
    value = DataType();
    return true;
  }
  value = GetDataType(name);
  if (!value.valid()) {
    source.Fail(absl::InvalidArgumentError(
        tensorstore::StrCat("Invalid data type: ", name)));
    return false;
  }
  return true;
}
}  // namespace serialization

}  // namespace tensorstore
