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

#ifndef TENSORSTORE_DATA_TYPE_CONVERSION_H_
#define TENSORSTORE_DATA_TYPE_CONVERSION_H_

#include <type_traits>

#include "absl/status/status.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/elementwise_function.h"

namespace tensorstore {

template <typename From, typename To>
struct ConvertDataType {
  void operator()(const From* from, To* to, absl::Status* status) const {
    *to = static_cast<To>(*from);
  }
};

template <typename From, typename To>
struct DataTypeConversionTraits {
  // By default, conversions are not supported.
  constexpr static DataTypeConversionFlags flags = DataTypeConversionFlags{};
};

/// `bool`-valued metafunction that specifies whether a conversion is allowed
/// from a compile-time data type of `From` to a compile-time data type of `To`.
///
/// .. warning::
///
///    If either `From` or `To` is `void`, the conversion is permitted at
///    compile time but may fail at run time.
///
/// \tparam From Unqualified element type, or `void` if unknown.
/// \tparam To Unqualified element type, or `void` if unknown.
/// \tparam AdditionalFlags Additional flags required,
///     e.g. `DataTypeConversionFlags::kSafeAndImplicit`.
/// \relates DataType
template <typename From, typename To,
          DataTypeConversionFlags AdditionalFlags = DataTypeConversionFlags{}>
constexpr inline bool IsDataTypeConversionSupported =
    ((DataTypeConversionTraits<From, To>::flags &
      (DataTypeConversionFlags::kSupported | AdditionalFlags)) ==
     (DataTypeConversionFlags::kSupported | AdditionalFlags));

template <typename From, DataTypeConversionFlags AdditionalFlags>
constexpr inline bool
    IsDataTypeConversionSupported<From, void, AdditionalFlags> = true;

template <typename To, DataTypeConversionFlags AdditionalFlags>
constexpr inline bool IsDataTypeConversionSupported<void, To, AdditionalFlags> =
    true;

template <typename T, DataTypeConversionFlags AdditionalFlags>
constexpr inline bool IsDataTypeConversionSupported<T, T, AdditionalFlags> =
    true;

template <DataTypeConversionFlags AdditionalFlags>
constexpr inline bool
    IsDataTypeConversionSupported<void, void, AdditionalFlags> = true;

namespace internal {

/// Outer array is indexed by source type.  Inner array is indexed by target
/// type.
extern const std::array<DataTypeOperations::CanonicalConversionOperations,
                        kNumDataTypeIds>
    canonical_data_type_conversions;

/// Returns the data type converter.
///
/// If `from == to`, `flags` is set to
/// `kSupported | kCanReinterpretCast | kIdentity | kSafeAndImplicit`, and
/// `closure` is set to `from->copy_assign` (with `nullptr` as context).
///
/// Otherwise, if both `from` and `to` are canonical data types, returns the
/// conversion function (if the conversion is supported) and flags.  The
/// conversion is supported if, and only if, the returned `flags` value includes
/// `kSupported`.  If the returned `flags` value includes `kCanReinterpretCast`,
/// the returned conversion function is set to `from->copy_assign`.
///
/// Otherwise, returns a result with `flags` set to 0 and an unspecified value
/// of `closure`.
DataTypeConversionLookupResult GetDataTypeConverter(DataType from, DataType to);

/// Same as above, but returns `absl::StatusCode::kInvalidArgument` if the
/// conversion is not supported with the specified `required_flags`.
///
/// \param from Source data type.
/// \param to Target data type.
/// \param required_flags Conversion flags to require,
///     e.g. `DataTypeConversionFlags::kSafeAndImplicit`.  Even if not
///     specified, `kSupported` is always assumed.
/// \dchecks `from.valid()`
/// \dchecks `to.valid()`
/// \returns `absl::OkStatus()` if the conversion is supported with the
///     specified `required_flags`.
/// \error `absl::StatusCode::kInvalidArgument` if the conversion is not
///     supported.
Result<DataTypeConversionLookupResult> GetDataTypeConverterOrError(
    DataType from, DataType to, DataTypeConversionFlags required_flags = {});

}  // namespace internal

namespace internal_data_type {

template <typename From, typename To>
std::enable_if_t<((DataTypeConversionTraits<From, To>::flags &
                   (DataTypeConversionFlags::kSupported |
                    DataTypeConversionFlags::kCanReinterpretCast)) ==
                      DataTypeConversionFlags::kSupported &&
                  !std::is_same_v<From, To>),
                 internal::ElementwiseFunction<2, absl::Status*>>
GetConvertFunction() {
  return internal::SimpleElementwiseFunction<
      ConvertDataType<From, To>(From, const To), absl::Status*>();
}

template <typename From, typename To>
std::enable_if_t<((DataTypeConversionTraits<From, To>::flags &
                   (DataTypeConversionFlags::kSupported |
                    DataTypeConversionFlags::kCanReinterpretCast)) !=
                      DataTypeConversionFlags::kSupported ||
                  std::is_same_v<From, To>),
                 internal::ElementwiseFunction<2, absl::Status*>>
GetConvertFunction() {
  return {};
}

template <typename From>
constexpr internal::DataTypeOperations::CanonicalConversionOperations
GetConvertToCanonicalOperations() {
  return {
      /*.convert=*/MapCanonicalDataTypes([](auto dtype) {
        using X = typename decltype(dtype)::Element;
        return GetConvertFunction<From, X>();
      }),
      /*.flags=*/MapCanonicalDataTypes([](auto dtype) {
        using X = typename decltype(dtype)::Element;
        return DataTypeConversionTraits<From, X>::flags;
      }),
  };
}

}  // namespace internal_data_type

// Define conversion traits between canonical data types.

namespace internal_data_type {

template <typename From, typename To>
struct IntegerIntegerDataTypeConversionTraits {
  constexpr static DataTypeConversionFlags flags =
      // integer -> integer conversions are always supported.
      DataTypeConversionFlags::kSupported |
      // `kSafeAndImplicit` if there is no reduction in precision or sign
      // support.
      ((std::numeric_limits<From>::digits <= std::numeric_limits<To>::digits &&
        std::numeric_limits<From>::is_signed <=
            std::numeric_limits<To>::is_signed)
           ? DataTypeConversionFlags::kSafeAndImplicit
           : DataTypeConversionFlags{}) |
      // `kCanReinterpretCast` if the size is the same.
      ((sizeof(To) == sizeof(From))
           ? DataTypeConversionFlags::kCanReinterpretCast
           : DataTypeConversionFlags{});
};

template <typename From, typename To>
struct IntegerFloatDataTypeConversionTraits {
  constexpr static DataTypeConversionFlags flags =
      // integer -> float conversions are always supported.
      DataTypeConversionFlags::kSupported |
      // `kSafeAndImplicit` if there is no reduction in precision.
      ((std::numeric_limits<From>::digits <= std::numeric_limits<To>::digits)
           ? DataTypeConversionFlags::kSafeAndImplicit
           : DataTypeConversionFlags{});
};

template <typename From, typename To>
struct FloatFloatDataTypeConversionTraits {
  constexpr static DataTypeConversionFlags flags =
      // float -> float conversions are always supported.
      DataTypeConversionFlags::kSupported |
      // `kSafeAndImplicit` if there is no reduction in mantissa or exponent
      // bits.
      ((std::numeric_limits<From>::digits <= std::numeric_limits<To>::digits &&
        std::numeric_limits<From>::min_exponent >=
            std::numeric_limits<To>::min_exponent &&
        std::numeric_limits<From>::max_exponent <=
            std::numeric_limits<To>::max_exponent)
           ? DataTypeConversionFlags::kSafeAndImplicit
           : DataTypeConversionFlags{});
};

template <typename From, typename To>
struct NumericComplexDataTypeConversionTraits {
  // integer/float -> complex conversion is always supported, and has the
  // `kSafeAndImplicit` flag if, and only if, the conversion from the
  // integer/float type to the `value_type` of the complex number does.
  constexpr static DataTypeConversionFlags flags =
      DataTypeConversionTraits<From, typename To::value_type>::flags &
      (DataTypeConversionFlags::kSupported |
       DataTypeConversionFlags::kSafeAndImplicit);
};

template <typename From, typename To>
struct ComplexComplexDataTypeConversionTraits
    // complex -> complex conversion has the same flags as the underlying
    // `value_type` -> `value_type`.
    : public DataTypeConversionTraits<typename From::value_type,
                                      typename To::value_type> {};

template <typename From, typename To>
struct IntegerJsonDataTypeConversionTraits {
  constexpr static DataTypeConversionFlags flags =
      // integer -> json conversion is always supported.
      DataTypeConversionFlags::kSupported |
      // `kSafeAndImplicit` if it the integer fits in 64 bits.
      ((std::numeric_limits<From>::digits <= 64)
           ? DataTypeConversionFlags::kSafeAndImplicit
           : DataTypeConversionFlags{});
};

template <typename From, typename To>
struct FloatJsonDataTypeConversionTraits {
  // float -> json conversion is always supported, and has `kSafeAndImplicit`
  // flag iff the conversion from `From` to `double` does.
  constexpr static DataTypeConversionFlags flags =
      DataTypeConversionTraits<From, double>::flags &
      (DataTypeConversionFlags::kSupported |
       DataTypeConversionFlags::kSafeAndImplicit);
};

}  // namespace internal_data_type

template <typename T>
struct DataTypeConversionTraits<std::complex<T>, json_t>
    : public DataTypeConversionTraits<T, json_t> {};

#define TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(FROM, TO, ...) \
  template <>                                                     \
  struct DataTypeConversionTraits<FROM, TO> {                     \
    using From = FROM;                                            \
    using To = TO;                                                \
    constexpr static DataTypeConversionFlags flags = __VA_ARGS__; \
  };                                                              \
  /**/

#define TENSORSTORE_INTERNAL_INHERITED_CONVERT(FROM, TO, PARENT)          \
  template <>                                                             \
  struct DataTypeConversionTraits<FROM, TO> : public PARENT<FROM, TO> {}; \
  /**/

// TODO(jbms): Define string_t and ustring_t -> number, complex, bool
// conversions

// [BEGIN GENERATED: generate_data_type.py]

TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    char_t, byte_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit |
        DataTypeConversionFlags::kCanReinterpretCast)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    ustring_t, string_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit |
        DataTypeConversionFlags::kCanReinterpretCast)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, int8_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, uint8_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, int16_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, uint16_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, int32_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, uint32_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, int64_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, uint64_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, float16_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, bfloat16_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, float32_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, float64_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, complex64_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, complex128_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bool_t, json_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    byte_t, char_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    ustring_t, json_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    int8_t, bool_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    int8_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    int8_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    uint8_t, bool_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    uint8_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    uint8_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    int16_t, bool_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    int16_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    int16_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    uint16_t, bool_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    uint16_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    uint16_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    int32_t, bool_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    int32_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    int32_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    uint32_t, bool_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    uint32_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    uint32_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    int64_t, bool_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    int64_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    int64_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    uint64_t, bool_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    uint64_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    uint64_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float16_t, bool_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float16_t, int8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float16_t, uint8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float16_t, int16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float16_t, uint16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float16_t, int32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float16_t, uint32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float16_t, int64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float16_t, uint64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float16_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float16_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bfloat16_t, bool_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bfloat16_t, int8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bfloat16_t, uint8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bfloat16_t, int16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bfloat16_t, uint16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bfloat16_t, int32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bfloat16_t, uint32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bfloat16_t, int64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bfloat16_t, uint64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bfloat16_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    bfloat16_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float32_t, bool_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float32_t, int8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float32_t, uint8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float32_t, int16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float32_t, uint16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float32_t, int32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float32_t, uint32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float32_t, int64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float32_t, uint64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float32_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float32_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float64_t, bool_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float64_t, int8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float64_t, uint8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float64_t, int16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float64_t, uint16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float64_t, int32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float64_t, uint32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float64_t, int64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float64_t, uint64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float64_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    float64_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, int8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, uint8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, int16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, uint16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, int32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, uint32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, int64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, uint64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, float16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, bfloat16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, float32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, float64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex64_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, int8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, uint8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, int16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, uint16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, int32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, uint32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, int64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, uint64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, float16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, bfloat16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, float32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, float64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    complex128_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, bool_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, int8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, uint8_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, int16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, uint16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, int32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, uint32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, int64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, uint64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, float16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, bfloat16_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, float32_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, float64_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, string_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    json_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    string_t, ustring_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(  //
    string_t, json_t, DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, int8_t, internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, uint8_t, internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, int16_t, internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, uint16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, int32_t, internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, uint32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, int64_t, internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, uint64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, int8_t, internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, uint8_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, int16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, uint16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, int32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, uint32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, int64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, uint64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, int8_t, internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, uint8_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, int16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, uint16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, int32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, uint32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, int64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, uint64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, int8_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, uint8_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, int16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, uint16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, int32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, uint32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, int64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, uint64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, int8_t, internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, uint8_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, int16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, uint16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, int32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, uint32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, int64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, uint64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, int8_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, uint8_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, int16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, uint16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, int32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, uint32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, int64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, uint64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, int8_t, internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, uint8_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, int16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, uint16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, int32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, uint32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, int64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, uint64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, int8_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, uint8_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, int16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, uint16_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, int32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, uint32_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, int64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, uint64_t,
    internal_data_type::IntegerIntegerDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, float16_t, internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, bfloat16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, float32_t, internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, float64_t, internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, float16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, bfloat16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, float32_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, float64_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, float16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, bfloat16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, float32_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, float64_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, float16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, bfloat16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, float32_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, float64_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, float16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, bfloat16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, float32_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, float64_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, float16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, bfloat16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, float32_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, float64_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, float16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, bfloat16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, float32_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, float64_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, float16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, bfloat16_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, float32_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, float64_t,
    internal_data_type::IntegerFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, complex64_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, complex128_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, complex64_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, complex128_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, complex64_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, complex128_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, complex64_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, complex128_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, complex64_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, complex128_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, complex64_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, complex128_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, complex64_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, complex128_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, complex64_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, complex128_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int8_t, json_t, internal_data_type::IntegerJsonDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint8_t, json_t, internal_data_type::IntegerJsonDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int16_t, json_t, internal_data_type::IntegerJsonDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint16_t, json_t, internal_data_type::IntegerJsonDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int32_t, json_t, internal_data_type::IntegerJsonDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint32_t, json_t, internal_data_type::IntegerJsonDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    int64_t, json_t, internal_data_type::IntegerJsonDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    uint64_t, json_t, internal_data_type::IntegerJsonDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float16_t, float16_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float16_t, bfloat16_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float16_t, float32_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float16_t, float64_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    bfloat16_t, float16_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    bfloat16_t, bfloat16_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    bfloat16_t, float32_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    bfloat16_t, float64_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float32_t, float16_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float32_t, bfloat16_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float32_t, float32_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float32_t, float64_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float64_t, float16_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float64_t, bfloat16_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float64_t, float32_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float64_t, float64_t,
    internal_data_type::FloatFloatDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float16_t, complex64_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float16_t, complex128_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    bfloat16_t, complex64_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    bfloat16_t, complex128_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float32_t, complex64_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float32_t, complex128_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float64_t, complex64_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float64_t, complex128_t,
    internal_data_type::NumericComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float16_t, json_t, internal_data_type::FloatJsonDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    bfloat16_t, json_t, internal_data_type::FloatJsonDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float32_t, json_t, internal_data_type::FloatJsonDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    float64_t, json_t, internal_data_type::FloatJsonDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    complex64_t, complex64_t,
    internal_data_type::ComplexComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    complex64_t, complex128_t,
    internal_data_type::ComplexComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    complex128_t, complex64_t,
    internal_data_type::ComplexComplexDataTypeConversionTraits)
TENSORSTORE_INTERNAL_INHERITED_CONVERT(  //
    complex128_t, complex128_t,
    internal_data_type::ComplexComplexDataTypeConversionTraits)

// [END GENERATED: generate_data_type.py]

#undef TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS
#undef TENSORSTORE_INTERNAL_INHERITED_CONVERT

}  // namespace tensorstore

#endif  // TENSORSTORE_DATA_TYPE_CONVERSION_H_
