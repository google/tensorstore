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

#include <cstring>
#include <new>

#include <nlohmann/json.hpp>
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/internal/exception_macros.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/preprocessor.h"
#include "tensorstore/internal/utf8.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

// On all common platforms, these assumptions holds.
//
// If TensorStore needs to support a platform where they do not hold, additional
// specializations of CanonicalElementType can be defined.
static_assert(std::is_same_v<signed char, std::int8_t>);
static_assert(std::is_same_v<short, std::int16_t>);  // NOLINT
static_assert(std::is_same_v<int, std::int32_t>);
static_assert(std::is_same_v<unsigned char, std::uint8_t>);
static_assert(std::is_same_v<unsigned short, std::uint16_t>);  // NOLINT
static_assert(std::is_same_v<unsigned int, std::uint32_t>);

std::ostream& operator<<(std::ostream& os, DataType r) {
  if (r.valid()) return os << r.name();
  return os << "<unspecified>";
}

void* AllocateAndConstruct(std::ptrdiff_t n,
                           ElementInitialization initialization, DataType r) {
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

std::string StaticCastTraits<DataType>::Describe(DataType data_type) {
  if (!data_type.valid()) return "dynamic data type";
  return StrCat("data type of ", data_type);
}

namespace internal_data_type {
#define TENSORSTORE_INTERNAL_DO_INSTANTIATION(X, ...) \
  TENSORSTORE_DATA_TYPE_EXPLICIT_INSTANTIATION(X)     \
  /**/
TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_INSTANTIATION)
#undef TENSORSTORE_INTERNAL_DEFINE_ELEMENT_REPRESENTATION_INSTANTIATION
}  // namespace internal_data_type

DataType GetDataType(absl::string_view id) {
#define TENSORSTORE_INTERNAL_MATCH_TYPE(X, ...)      \
  if (id == absl::string_view(#X, sizeof(#X) - 3)) { \
    return DataTypeOf<X>();                          \
  }                                                  \
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
struct NumberToStringCanonicalType<float16_t> {
  using type = float;
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
struct ConvertDataType<std::complex<T>, json_t> {
  void operator()(const std::complex<T>* from, json_t* to, Status*) const {
    *to = json_t::array_t{from->real(), from->imag()};
  }
};

template <typename T>
struct ConvertDataType<std::complex<T>, string_t> {
  void operator()(const std::complex<T>* from, string_t* to, Status*) const {
    ComplexToString(*from, to);
  }
};

template <typename T>
struct ConvertDataType<std::complex<T>, ustring_t> {
  void operator()(const std::complex<T>* from, ustring_t* to, Status*) const {
    ComplexToString(*from, &to->utf8);
  }
};

template <>
struct ConvertDataType<float16_t, json_t> {
  void operator()(const float16_t* from, json_t* to, Status*) const {
    *to = static_cast<double>(*from);
  }
};

namespace internal_data_type {

struct JsonIntegerConvertDataType {
  template <typename To>
  bool operator()(const json_t* from, To* to, Status* status) const {
    auto s = internal::JsonRequireInteger(*from, to, /*strict=*/false);
    if (s.ok()) return true;
    *status = s;
    return false;
  }
};

struct JsonFloatConvertDataType {
  template <typename To>
  bool operator()(const json_t* from, To* to, Status* status) const {
    double value;
    auto s = internal::JsonRequireValueAs(*from, &value, /*strict=*/false);
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
  void operator()(const std::complex<T>* from, To* to, Status*) const {
    *to = static_cast<To>(from->real());
  }
};

struct NumericStringConvertDataType {
  template <typename From>
  void operator()(const From* from, string_t* to, Status*) const {
    NumberToString(*from, to);
  }
};

struct NumericUstringConvertDataType {
  template <typename From>
  void operator()(const From* from, ustring_t* to, Status*) const {
    NumberToString(*from, &to->utf8);
  }
};

}  // namespace internal_data_type

#define TENSORSTORE_INTERNAL_INHERITED_CONVERT1(TO, FROM, PARENT) \
  template <>                                                     \
  struct ConvertDataType<FROM, TO> : public PARENT {};            \
  /**/

#define TENSORSTORE_INTERNAL_INHERITED_CONVERT(...)                            \
  TENSORSTORE_PP_EXPAND1(TENSORSTORE_INTERNAL_INHERITED_CONVERT1(__VA_ARGS__)) \
  /**/

// TODO(jbms): implement json -> complex conversion

// TODO(jbms): implement string -> number conversion
// TODO(jbms): implement string -> complex conversion

template <>
struct ConvertDataType<ustring_t, json_t> {
  void operator()(const ustring_t* from, json_t* to, Status*) const {
    *to = from->utf8;
  }
};

#define TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_INT(X, ...)        \
  TENSORSTORE_INTERNAL_INHERITED_CONVERT(                               \
      string_t, X, internal_data_type::NumericStringConvertDataType);   \
  TENSORSTORE_INTERNAL_INHERITED_CONVERT(                               \
      ustring_t, X, internal_data_type::NumericUstringConvertDataType); \
  /**/

TENSORSTORE_PP_EXPAND(TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE(
    TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_INT))
#undef TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_INT

#define TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_FLOAT(X, ...)      \
  TENSORSTORE_INTERNAL_INHERITED_CONVERT(                               \
      string_t, X, internal_data_type::NumericStringConvertDataType);   \
  TENSORSTORE_INTERNAL_INHERITED_CONVERT(                               \
      ustring_t, X, internal_data_type::NumericUstringConvertDataType); \
  /**/
TENSORSTORE_PP_EXPAND(TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE(
    TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_FLOAT))
#undef TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_FLOAT

#define TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_COMPLEX(X, ...) \
  TENSORSTORE_PP_DEFER(TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE_ID)    \
  ()(TENSORSTORE_INTERNAL_INHERITED_CONVERT, X,                      \
     internal_data_type::ComplexNumericConvertDataType);             \
  TENSORSTORE_PP_DEFER(TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE_ID)      \
  ()(TENSORSTORE_INTERNAL_INHERITED_CONVERT, X,                      \
     internal_data_type::ComplexNumericConvertDataType);             \
  /**/
TENSORSTORE_PP_EXPAND(TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE(
    TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_COMPLEX))
#undef TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_COMPLEX

TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE(
    TENSORSTORE_INTERNAL_INHERITED_CONVERT, json_t,
    internal_data_type::JsonIntegerConvertDataType)

TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE(
    TENSORSTORE_INTERNAL_INHERITED_CONVERT, json_t,
    internal_data_type::JsonFloatConvertDataType)

template <>
struct ConvertDataType<json_t, bool> {
  bool operator()(const json_t* from, bool* to, Status* status) const {
    auto s = internal::JsonRequireValueAs(*from, to, /*strict=*/false);
    if (s.ok()) return true;
    *status = s;
    return false;
  }
};

template <>
struct ConvertDataType<json_t, string_t> {
  bool operator()(const json_t* from, string_t* to, Status* status) const {
    auto s = internal::JsonRequireValueAs(*from, to, /*strict=*/false);
    if (s.ok()) return true;
    *status = s;
    return false;
  }
};

template <>
struct ConvertDataType<json_t, ustring_t> {
  bool operator()(const json_t* from, ustring_t* to, Status* status) const {
    auto s = internal::JsonRequireValueAs(*from, &to->utf8, /*strict=*/false);
    if (s.ok()) return true;
    *status = s;
    return false;
  }
};

template <>
struct ConvertDataType<string_t, ustring_t> {
  bool operator()(const string_t* from, ustring_t* to, Status* status) const {
    if (internal::IsValidUtf8(*from)) {
      to->utf8 = *from;
      return true;
    }
    *status = absl::InvalidArgumentError("Invalid UTF-8 sequence encountered");
    return false;
  }
};

template <>
struct ConvertDataType<string_t, json_t> {
  bool operator()(const string_t* from, json_t* to, Status* status) const {
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
        return absl::InvalidArgumentError(
            StrCat("Explicit data type conversion required to convert ", from,
                   " -> ", to));
      }
    }
    return absl::InvalidArgumentError(
        StrCat("Cannot convert ", from, " -> ", to));
  }
  return lookup_result;
}

}  // namespace internal

}  // namespace tensorstore
