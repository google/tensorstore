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

#include "tensorstore/data_type_conversion.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/meta/attributes.h"
#include "tensorstore/internal/utf8.h"
#include "tensorstore/util/bfloat16.h"  // IWYU pragma: keep
#include "tensorstore/util/float8.h"    // IWYU pragma: keep
#include "tensorstore/util/int2.h"      // IWYU pragma: keep
#include "tensorstore/util/int4.h"      // IWYU pragma: keep
#include "tensorstore/util/mxfloat.h"   // IWYU pragma: keep
#include "tensorstore/util/result.h"
#include "tensorstore/util/utf8_string.h"

namespace tensorstore {

/// Define conversions between canonical data types.

namespace {

template <typename T>
struct NumberToStringCanonicalType {
  using type = T;
};

#define TENSORSTORE_INTERNAL_NUMBER_TO_STRING_CANONICAL_TYPE_FLOAT(T, ...) \
  template <>                                                              \
  struct NumberToStringCanonicalType<::tensorstore::dtypes::T> {           \
    using type = float;                                                    \
  };                                                                       \
  /**/
TENSORSTORE_FOR_EACH_LOW_PRECISION_FLOAT_DATA_TYPE(
    TENSORSTORE_INTERNAL_NUMBER_TO_STRING_CANONICAL_TYPE_FLOAT)
#undef TENSORSTORE_INTERNAL_NUMBER_TO_STRING_CANONICAL_TYPE_FLOAT

#define TENSORSTORE_INTERNAL_NUMBER_TO_STRING_CANONICAL_TYPE_INT(T, ...) \
  template <>                                                            \
  struct NumberToStringCanonicalType<::tensorstore::dtypes::T> {         \
    using type = int16_t;                                                \
  };                                                                     \
  /**/
TENSORSTORE_FOR_EACH_LOW_PRECISION_INT_DATA_TYPE(
    TENSORSTORE_INTERNAL_NUMBER_TO_STRING_CANONICAL_TYPE_INT)
#undef TENSORSTORE_INTERNAL_NUMBER_TO_STRING_CANONICAL_TYPE_INT

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

template <typename From, typename To>
struct ConvertDataType {
  static constexpr bool kUseStridedForContiguous = true;
  constexpr void operator()(const From* from, To* to, void*) const {
    *to = static_cast<To>(*from);
  }
};

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
  static constexpr bool kUseStridedForContiguous = true;
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
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::int2_t,
    internal_data_type::ComplexNumericConvertDataType)
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
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::int2_t,
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
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::float8_e3m4_t,
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
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::float8_e8m0fnu_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex64_t, ::tensorstore::dtypes::float4_e2m1fn_t,
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
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::float8_e3m4_t,
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
    ::tensorstore::dtypes::complex128_t,
    ::tensorstore::dtypes::float8_e8m0fnu_t,
    internal_data_type::ComplexNumericConvertDataType)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    ::tensorstore::dtypes::complex128_t, ::tensorstore::dtypes::float4_e2m1fn_t,
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

template <typename T>
constexpr bool IsOneByteFloat = false;
#define TENSORSTORE_INTERNAL_IS_ONE_BYTE_FLOAT(T, ...) \
  template <>                                          \
  constexpr bool IsOneByteFloat<::tensorstore::dtypes::T> = true;
TENSORSTORE_FOR_EACH_FLOAT8_DATA_TYPE(TENSORSTORE_INTERNAL_IS_ONE_BYTE_FLOAT)
TENSORSTORE_FOR_EACH_MXFLOAT_DATA_TYPE(TENSORSTORE_INTERNAL_IS_ONE_BYTE_FLOAT)
#undef TENSORSTORE_INTERNAL_IS_ONE_BYTE_FLOAT

template <typename T>
constexpr bool IsLowPrecisionInt = false;
#define TENSORSTORE_INTERNAL_IS_LOW_PRECISION_INT(T, ...) \
  template <>                                             \
  constexpr bool IsLowPrecisionInt<::tensorstore::dtypes::T> = true;
TENSORSTORE_FOR_EACH_LOW_PRECISION_INT_DATA_TYPE(
    TENSORSTORE_INTERNAL_IS_LOW_PRECISION_INT)
#undef TENSORSTORE_INTERNAL_IS_LOW_PRECISION_INT

template <typename T>
constexpr bool IsOneByteType =
    IsOneByteFloat<T> || IsLowPrecisionInt<T> ||
    std::is_same_v<T, ::tensorstore::dtypes::bool_t> ||
    std::is_same_v<T, ::tensorstore::dtypes::int8_t> ||
    std::is_same_v<T, ::tensorstore::dtypes::uint8_t>;

template <typename T>
constexpr T ByteFromUint8(uint8_t byte) {
  if constexpr (std::is_same_v<T, ::tensorstore::dtypes::bool_t>) {
    return byte != 0;
  } else if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(byte);
  } else {
    return absl::bit_cast<T>(byte);
  }
}

template <typename T>
constexpr uint8_t ByteToUint8(T val) {
  if constexpr (std::is_same_v<T, ::tensorstore::dtypes::bool_t>) {
    return static_cast<uint8_t>(val ? 1 : 0);
  } else if constexpr (std::is_integral_v<T>) {
    return static_cast<uint8_t>(val);
  } else {
    return absl::bit_cast<uint8_t>(val);
  }
}

// Regime A: 1-byte -> 1-byte conversion via 256-entry lookup table.
template <typename From, typename To>
constexpr std::array<uint8_t, 256> kUint8ToUint8Table = [] {
  std::array<uint8_t, 256> table{};
  for (int i = 0; i < 256; ++i) {
    const From from_val = ByteFromUint8<From>(static_cast<uint8_t>(i));
    To to_val{};
    if constexpr (IsOneByteFloat<From> && std::numeric_limits<To>::is_integer &&
                  !std::is_same_v<To, bool>) {
      to_val = internal::SaturatedFloatToInt<To>(from_val);
    } else {
      to_val = static_cast<To>(from_val);
    }
    table[i] = ByteToUint8<To>(to_val);
  }
  return table;
}();

static_assert(kUint8ToUint8Table<::tensorstore::dtypes::int8_t,
                                 ::tensorstore::dtypes::float8_e3m4_t>[255] ==
              0xB0);

struct Uint8TableLookupConvert {
  static constexpr bool kUseStridedForContiguous = true;
  void operator()(const std::array<uint8_t, 256>& table, const uint8_t* from,
                  uint8_t* to, void*) const {
    *to = table[*from];
  }
};

using Uint8TableLookupFn = internal::SimpleElementwiseFunction<
    internal_elementwise_function::Stateless<const std::array<uint8_t, 256>,
                                             Uint8TableLookupConvert>(
        const uint8_t, uint8_t),
    void*>;

// Regime B: 1-byte low-precision From -> multi-byte To via hub lookup tables.
// Every 1-byte float format has <= 4 mantissa bits and min exponent >= -127,
// so the lower 16 bits of its IEEE-754 float32 representation are identically
// zero and its upper 16 bits equal its BFloat16 bit representation.
template <typename From>
constexpr std::array<uint16_t, 256> kByteToBFloat16BitsTable = [] {
  std::array<uint16_t, 256> table{};
  for (int i = 0; i < 256; ++i) {
    const uint32_t f32_bits = absl::bit_cast<uint32_t>(
        static_cast<float>(ByteFromUint8<From>(static_cast<uint8_t>(i))));
    assert((f32_bits & 0xffffu) == 0);
    table[i] = static_cast<uint16_t>(f32_bits >> 16);
  }
  return table;
}();

template <typename From>
constexpr std::array<int8_t, 256> kByteToInt8Table = [] {
  std::array<int8_t, 256> table{};
  for (int i = 0; i < 256; ++i) {
    table[i] =
        static_cast<int8_t>(ByteFromUint8<From>(static_cast<uint8_t>(i)));
  }
  return table;
}();

static_assert(kByteToInt8Table<::tensorstore::dtypes::int8_t>[255] == -1);

template <typename TableElement, typename To>
struct HubTableConvertFromUint8 {
  static constexpr bool kUseStridedForContiguous = true;
  bool operator()(const std::array<TableElement, 256>& table,
                  const uint8_t* from, To* to, void* arg) const {
    if constexpr (std::is_same_v<TableElement, uint16_t>) {
      if constexpr (std::is_same_v<To, ::tensorstore::dtypes::bfloat16_t>) {
        *to = absl::bit_cast<::tensorstore::dtypes::bfloat16_t>(table[*from]);
        return true;
      } else {
        const float hub_val =
            absl::bit_cast<float>(static_cast<uint32_t>(table[*from]) << 16);
        if constexpr (std::is_integral_v<To> && !std::is_same_v<To, bool>) {
          *to = internal::SaturatedFloatToInt<To>(hub_val);
          return true;
        } else {
          return static_cast<bool>(internal::Void::CallAndWrap(
              ConvertDataType<float, To>{}, &hub_val, to, arg));
        }
      }
    } else {
      const int8_t hub_val = table[*from];
      return static_cast<bool>(internal::Void::CallAndWrap(
          ConvertDataType<int8_t, To>{}, &hub_val, to, arg));
    }
  }
};

template <typename TableElement, typename To>
using HubTableConvertFromUint8Fn = internal::SimpleElementwiseFunction<
    internal_elementwise_function::Stateless<
        const std::array<TableElement, 256>,
        HubTableConvertFromUint8<TableElement, To>>(const uint8_t, To),
    void*>;

// Regime C: Multi-byte From -> 1-byte low-precision To via shared hub columns.
template <typename From>
struct ConvertThroughInt8Table {
  static constexpr bool kUseStridedForContiguous = true;
  void operator()(const std::array<int8_t, 256>& table, const From* from,
                  int8_t* to, void*) const {
    int8_t v;
    if constexpr (std::is_same_v<From, ::tensorstore::dtypes::complex64_t> ||
                  std::is_same_v<From, ::tensorstore::dtypes::complex128_t>) {
      v = static_cast<int8_t>(from->real());
    } else {
      v = static_cast<int8_t>(*from);
    }
    *to = table[static_cast<uint8_t>(v)];
  }
};

template <typename From>
using ConvertThroughInt8TableFn = internal::SimpleElementwiseFunction<
    internal_elementwise_function::Stateless<const std::array<int8_t, 256>,
                                             ConvertThroughInt8Table<From>>(
        const From, int8_t),
    void*>;

template <typename Hub>
using HubToByteBatchFn = void (*)(const Hub* from, uint8_t* to, Index count);

template <typename Hub, typename To>
void HubToByteBatch(const Hub* from, uint8_t* to, Index count) {
  for (Index i = 0; i < count; ++i) {
    to[i] = absl::bit_cast<uint8_t>(static_cast<To>(from[i]));
  }
}

template <typename Hub, typename To>
constexpr HubToByteBatchFn<Hub> kHubToByteBatchFn = &HubToByteBatch<Hub, To>;

template <typename From, typename Hub>
struct TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN
    ConvertToByteThroughHubLoopTemplate {
  template <internal::IterationBufferKind Kind>
  TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN static bool Loop(
      void* context, internal::IterationBufferShape shape,
      internal::IterationBufferPointer from_ptr,
      internal::IterationBufferPointer to_ptr, [[maybe_unused]] void* status) {
    using Accessor = internal::IterationBufferAccessor<Kind>;
    const auto fn = *static_cast<const HubToByteBatchFn<Hub>*>(context);
    constexpr Index kBlockSize = 64;
    Hub hub_buf[kBlockSize];
    if constexpr (!std::is_same_v<From, ::tensorstore::dtypes::json_t> &&
                  Kind != internal::IterationBufferKind::kIndexed) {
      if (from_ptr.inner_byte_stride == static_cast<Index>(sizeof(From)) &&
          to_ptr.inner_byte_stride == 1) {
        for (Index outer = 0; outer < shape[0]; ++outer) {
          const From* from_row =
              Accessor::template GetPointerAtPosition<const From>(from_ptr,
                                                                  outer, 0);
          uint8_t* to_row = Accessor::template GetPointerAtPosition<uint8_t>(
              to_ptr, outer, 0);
          if constexpr (std::is_same_v<From, Hub>) {
            fn(from_row, to_row, shape[1]);
          } else {
            for (Index inner = 0; inner < shape[1]; inner += kBlockSize) {
              const Index block = std::min(kBlockSize, shape[1] - inner);
              for (Index i = 0; i < block; ++i) {
                if constexpr (std::is_same_v<
                                  From, ::tensorstore::dtypes::complex64_t> ||
                              std::is_same_v<
                                  From, ::tensorstore::dtypes::complex128_t>) {
                  hub_buf[i] = from_row[inner + i].real();
                } else {
                  hub_buf[i] = static_cast<Hub>(from_row[inner + i]);
                }
              }
              fn(hub_buf, to_row + inner, block);
            }
          }
        }
        return true;
      }
    }
    uint8_t byte_buf[kBlockSize];
    for (Index outer = 0; outer < shape[0]; ++outer) {
      for (Index inner = 0; inner < shape[1]; inner += kBlockSize) {
        const Index block = std::min(kBlockSize, shape[1] - inner);
        for (Index i = 0; i < block; ++i) {
          const From* src = Accessor::template GetPointerAtPosition<const From>(
              from_ptr, outer, inner + i);
          if constexpr (std::is_same_v<From, ::tensorstore::dtypes::json_t>) {
            auto s = internal_json::JsonRequireValueAs(*src, &hub_buf[i],
                                                       /*strict=*/false);
            if (!s.ok()) {
              *static_cast<absl::Status*>(status) = std::move(s);
              return false;
            }
          } else if constexpr (std::is_same_v<
                                   From, ::tensorstore::dtypes::complex64_t> ||
                               std::is_same_v<
                                   From, ::tensorstore::dtypes::complex128_t>) {
            hub_buf[i] = src->real();
          } else {
            hub_buf[i] = static_cast<Hub>(*src);
          }
        }
        fn(hub_buf, byte_buf, block);
        for (Index i = 0; i < block; ++i) {
          *Accessor::template GetPointerAtPosition<uint8_t>(
              to_ptr, outer, inner + i) = byte_buf[i];
        }
      }
    }
    return true;
  }

  template <typename ArrayAccessor>
  static constexpr internal::SpecializedElementwiseFunctionPointer<2, void*>
  GetLoopFn() {
    if constexpr (ArrayAccessor::buffer_kind ==
                  internal::IterationBufferKind::kContiguous) {
      return &Loop<internal::IterationBufferKind::kStrided>;
    } else {
      return &Loop<ArrayAccessor::buffer_kind>;
    }
  }
};

template <typename From, typename Hub>
constexpr internal::ElementwiseFunction<2, void*> kConvertToByteThroughHubFn(
    ConvertToByteThroughHubLoopTemplate<From, Hub>{});

template <typename From, typename To>
constexpr internal::ElementwiseClosure<2, void*> GetConvertClosure() {
  if constexpr ((DataTypeConversionTraits<From, To>::flags &
                 (DataTypeConversionFlags::kSupported |
                  DataTypeConversionFlags::kCanReinterpretCast)) !=
                    DataTypeConversionFlags::kSupported ||
                std::is_same_v<From, To>) {
    return {};
  } else if constexpr (IsOneByteType<From> && IsOneByteType<To>) {
    return {
        &Uint8TableLookupFn::function,
        const_cast<std::array<uint8_t, 256>*>(&kUint8ToUint8Table<From, To>)};
  } else if constexpr (IsOneByteFloat<From> && !IsOneByteType<To>) {
    return {&HubTableConvertFromUint8Fn<uint16_t, To>::function,
            const_cast<std::array<uint16_t, 256>*>(
                &kByteToBFloat16BitsTable<From>)};
  } else if constexpr ((IsLowPrecisionInt<From> ||
                        std::is_same_v<From, ::tensorstore::dtypes::int8_t> ||
                        (std::is_same_v<From, ::tensorstore::dtypes::bool_t> &&
                         !std::is_same_v<To, ::tensorstore::dtypes::json_t>)) &&
                       !IsOneByteType<To>) {
    return {&HubTableConvertFromUint8Fn<int8_t, To>::function,
            const_cast<std::array<int8_t, 256>*>(&kByteToInt8Table<From>)};
  } else if constexpr (!IsOneByteType<From> && IsLowPrecisionInt<To> &&
                       !std::is_same_v<From, ::tensorstore::dtypes::json_t>) {
    return {&ConvertThroughInt8TableFn<From>::function,
            const_cast<std::array<int8_t, 256>*>(&kByteToInt8Table<To>)};
  } else if constexpr (!IsOneByteType<From> && IsOneByteFloat<To>) {
    using Hub = std::conditional_t<
        std::is_same_v<From, ::tensorstore::dtypes::float64_t> ||
            std::is_same_v<From, ::tensorstore::dtypes::complex128_t> ||
            std::is_same_v<From, ::tensorstore::dtypes::json_t>,
        double, float>;
    return {&kConvertToByteThroughHubFn<From, Hub>,
            const_cast<HubToByteBatchFn<Hub>*>(&kHubToByteBatchFn<Hub, To>)};
  } else {
    return {&internal::SimpleElementwiseFunction<
                ConvertDataType<From, To>(const From, To), void*>::function,
            nullptr};
  }
}

struct CanonicalConversionOperations {
  // Closure for converting to/from canonical data type.
  //
  // The `void*` pointer is a pointer to an `absl::Status` that may be set to
  // an error status in the case of failure.
  std::array<internal::ElementwiseClosure<2, void*>, kNumDataTypeIds> convert;
  std::array<DataTypeConversionFlags, kNumDataTypeIds> flags;
};

template <typename From>
constexpr CanonicalConversionOperations GetConvertToCanonicalOperations() {
  return {
      /*.convert=*/MapCanonicalDataTypes([](auto dtype) {
        using X = typename decltype(dtype)::Element;
        return GetConvertClosure<From, X>();
      }),
      /*.flags=*/MapCanonicalDataTypes([](auto dtype) {
        using X = typename decltype(dtype)::Element;
        return DataTypeConversionTraits<From, X>::flags;
      }),
  };
}

/// Outer array is indexed by source type.  Inner array is indexed by target
/// type.
constexpr std::array<CanonicalConversionOperations, kNumDataTypeIds>
    canonical_data_type_conversions = MapCanonicalDataTypes([](auto d) {
      using X = typename decltype(d)::Element;
      return GetConvertToCanonicalOperations<X>();
    });

}  // namespace

namespace internal {

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
  const auto& ops =
      canonical_data_type_conversions[static_cast<size_t>(from_id)];
  lookup_result.flags = ops.flags[static_cast<size_t>(to_id)];
  if ((lookup_result.flags & DataTypeConversionFlags::kCanReinterpretCast) ==
      DataTypeConversionFlags::kCanReinterpretCast) {
    lookup_result.closure.function = &from->copy_assign;
  } else {
    lookup_result.closure = ops.convert[static_cast<size_t>(to_id)];
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
        return absl::InvalidArgumentError(absl::StrFormat(
            "Explicit data type conversion required to convert %v -> %v", from,
            to));
      }
    }
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot convert %v -> %v", from, to));
  }
  return lookup_result;
}

}  // namespace internal
}  // namespace tensorstore
