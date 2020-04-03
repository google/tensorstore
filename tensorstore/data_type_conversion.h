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

#include "tensorstore/data_type.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/preprocessor.h"

namespace tensorstore {

template <typename From, typename To>
struct ConvertDataType {
  void operator()(const From* from, To* to, Status* status) const {
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
/// If either type is `void`, the conversion is permitted at compile time (but
/// may fail at run time).  Otherwise, the conversion is allowed if, and only
/// if, `DataTypeConversionTraits<From,To>::flags` includes `kSupported` and
/// `AdditionalFlags`.
///
/// \tparam From Unqualified element type, or `void` if unknown.
/// \tparam To Unqualified element type, or `void` if unknown.
/// \tparam AdditionalFlags Additional flags required,
///     e.g. `DataTypeConversionFlags::kSafeAndImplicit`.
template <typename From, typename To,
          DataTypeConversionFlags AdditionalFlags = DataTypeConversionFlags{}>
struct IsDataTypeConversionSupported
    : public std::integral_constant<
          bool, ((DataTypeConversionTraits<From, To>::flags &
                  (DataTypeConversionFlags::kSupported | AdditionalFlags)) ==
                 (DataTypeConversionFlags::kSupported | AdditionalFlags))> {};

template <typename From, DataTypeConversionFlags AdditionalFlags>
struct IsDataTypeConversionSupported<From, void, AdditionalFlags>
    : public std::true_type {};

template <typename To, DataTypeConversionFlags AdditionalFlags>
struct IsDataTypeConversionSupported<void, To, AdditionalFlags>
    : public std::true_type {};

template <typename T, DataTypeConversionFlags AdditionalFlags>
struct IsDataTypeConversionSupported<T, T, AdditionalFlags>
    : public std::true_type {};

template <DataTypeConversionFlags AdditionalFlags>
struct IsDataTypeConversionSupported<void, void, AdditionalFlags>
    : public std::true_type {};

namespace internal {

/// Outer array is indexed by source type.  Inner array is indexed by target
/// type.
extern const std::array<DataTypeOperations::CanonicalConversionOperations,
                        kNumDataTypeIds>
    canonical_data_type_conversions;

struct DataTypeConversionLookupResult {
  /// Valid only if the `flags` value includes `kSupported`.
  ElementwiseClosure<2, Status*> closure;
  DataTypeConversionFlags flags;
};

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
/// \returns `Status()` if the conversion is supported with the specified
///     `required_flags`.
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
                 internal::ElementwiseFunction<2, Status*>>
GetConvertFunction() {
  return internal::SimpleElementwiseFunction<
      ConvertDataType<From, To>(From, const To), Status*>();
}

template <typename From, typename To>
std::enable_if_t<((DataTypeConversionTraits<From, To>::flags &
                   (DataTypeConversionFlags::kSupported |
                    DataTypeConversionFlags::kCanReinterpretCast)) !=
                      DataTypeConversionFlags::kSupported ||
                  std::is_same_v<From, To>),
                 internal::ElementwiseFunction<2, Status*>>
GetConvertFunction() {
  return {};
}

template <typename From>
constexpr internal::DataTypeOperations::CanonicalConversionOperations
GetConvertToCanonicalOperations() {
  return {
      /*.convert=*/MapCanonicalDataTypes([](auto data_type) {
        using X = typename decltype(data_type)::Element;
        return GetConvertFunction<From, X>();
      }),
      /*.flags=*/MapCanonicalDataTypes([](auto data_type) {
        using X = typename decltype(data_type)::Element;
        return DataTypeConversionTraits<From, X>::flags;
      }),
  };
}

}  // namespace internal_data_type

/// Define conversion traits between canonical data types.

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

#define TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS1(TO, FROM, ...) \
  template <>                                                      \
  struct DataTypeConversionTraits<FROM, TO> {                      \
    using From = FROM;                                             \
    using To = TO;                                                 \
    constexpr static DataTypeConversionFlags flags = __VA_ARGS__;  \
  };                                                               \
  /**/

#define TENSORSTORE_INTERNAL_INHERITED_CONVERT1(TO, FROM, PARENT)         \
  template <>                                                             \
  struct DataTypeConversionTraits<FROM, TO> : public PARENT<FROM, TO> {}; \
  /**/

// The extra indirection for `TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS` and
// `TENSORSTORE_INTERNAL_INHERITED_CONVERT` is a workaround for preprocessor
// limitations in MSVC 2019 (not necessary on GCC/Clang or on MSVC with
// `/experimental:preprocessor` option).  Without this workaround, the
// invocations from `TENSORSTORE_FOR_EACH_*_DATA_TYPE` don't correctly map
// `__VA_ARGS__` to multiple arguments.

#define TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(...)         \
  TENSORSTORE_PP_EXPAND1(                                       \
      TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS1(__VA_ARGS__)) \
  /**/

#define TENSORSTORE_INTERNAL_INHERITED_CONVERT(...)                            \
  TENSORSTORE_PP_EXPAND1(TENSORSTORE_INTERNAL_INHERITED_CONVERT1(__VA_ARGS__)) \
  /**/

TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(
    char_t, byte_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kCanReinterpretCast);

TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(
    byte_t, char_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kCanReinterpretCast |
        DataTypeConversionFlags::kSafeAndImplicit);

/// Define conversion flags from bool to every other canonical type.
#define TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_BOOL(T, ...) \
  TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(                     \
      T, bool_t,                                                  \
      DataTypeConversionFlags::kSupported |                       \
          DataTypeConversionFlags::kSafeAndImplicit)              \
  /**/
TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE(
    TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_BOOL)
TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE(
    TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_BOOL)
TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE(
    TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_BOOL)
TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_BOOL(json_t)
#undef TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_BOOL

/// Define conversion flags from canonical integer types to every other
/// canonical type.
#define TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_INT(X, ...)           \
  TENSORSTORE_PP_DEFER(TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE_ID)          \
  ()(TENSORSTORE_PP_DEFER(TENSORSTORE_INTERNAL_INHERITED_CONVERT), X,      \
     internal_data_type::IntegerIntegerDataTypeConversionTraits);          \
  TENSORSTORE_PP_DEFER(TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE_ID)            \
  ()(TENSORSTORE_PP_DEFER(TENSORSTORE_INTERNAL_INHERITED_CONVERT), X,      \
     internal_data_type::IntegerFloatDataTypeConversionTraits);            \
  TENSORSTORE_PP_DEFER(TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE_ID)          \
  ()(TENSORSTORE_PP_DEFER(TENSORSTORE_INTERNAL_INHERITED_CONVERT), X,      \
     internal_data_type::NumericComplexDataTypeConversionTraits);          \
  TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(                              \
      bool_t, X, DataTypeConversionFlags::kSupported);                     \
  TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(                              \
      string_t, X, DataTypeConversionFlags::kSupported);                   \
  TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(                              \
      ustring_t, X, DataTypeConversionFlags::kSupported);                  \
  TENSORSTORE_INTERNAL_INHERITED_CONVERT(                                  \
      json_t, X, internal_data_type::IntegerJsonDataTypeConversionTraits); \
  /**/
TENSORSTORE_PP_EXPAND(TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE(
    TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_INT))
#undef TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_INT

/// Define conversion flags from canonical float types to every other canonical
/// type.
#define TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_FLOAT(X, ...)       \
  TENSORSTORE_PP_DEFER(TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE_ID)        \
  ()(TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS, X,                      \
     DataTypeConversionFlags::kSupported);                               \
  TENSORSTORE_PP_DEFER(TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE_ID)          \
  ()(TENSORSTORE_INTERNAL_INHERITED_CONVERT, X,                          \
     internal_data_type::FloatFloatDataTypeConversionTraits);            \
  TENSORSTORE_PP_DEFER(TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE_ID)        \
  ()(TENSORSTORE_INTERNAL_INHERITED_CONVERT, X,                          \
     internal_data_type::NumericComplexDataTypeConversionTraits);        \
  TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(                            \
      bool_t, X, DataTypeConversionFlags::kSupported);                   \
  TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(                            \
      string_t, X, DataTypeConversionFlags::kSupported);                 \
  TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(                            \
      ustring_t, X, DataTypeConversionFlags::kSupported);                \
  TENSORSTORE_INTERNAL_INHERITED_CONVERT(                                \
      json_t, X, internal_data_type::FloatJsonDataTypeConversionTraits); \
  /**/
TENSORSTORE_PP_EXPAND(TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE(
    TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_FLOAT))
#undef TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_FLOAT

/// Define conversion flags from canonical complex types to every other
/// canonical type.
#define TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_COMPLEX(X, ...) \
  TENSORSTORE_PP_DEFER(TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE_ID)    \
  ()(TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS, X,                  \
     DataTypeConversionFlags::kSupported);                           \
  TENSORSTORE_PP_DEFER(TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE_ID)      \
  ()(TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS, X,                  \
     DataTypeConversionFlags::kSupported);                           \
  TENSORSTORE_PP_DEFER(TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE_ID)    \
  ()(TENSORSTORE_INTERNAL_INHERITED_CONVERT, X,                      \
     internal_data_type::ComplexComplexDataTypeConversionTraits);    \
  TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(                        \
      string_t, X, DataTypeConversionFlags::kSupported);             \
  TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(                        \
      ustring_t, X, DataTypeConversionFlags::kSupported);            \
  /**/
TENSORSTORE_PP_EXPAND(TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE(
    TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_COMPLEX))
#undef TENSORSTORE_INTERNAL_DEFINE_CONVERSIONS_FROM_COMPLEX

template <typename T>
struct DataTypeConversionTraits<std::complex<T>, json_t>
    : public DataTypeConversionTraits<T, json_t> {};

/// Define conversion flags from json_t to every other canonical type.
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(bool, json_t,
                                           DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(string_t, json_t,
                                           DataTypeConversionFlags::kSupported)
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(ustring_t, json_t,
                                           DataTypeConversionFlags::kSupported)
TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE(  //
    TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS, json_t,
    DataTypeConversionFlags::kSupported)

TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE(  //
    TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS, json_t,
    DataTypeConversionFlags::kSupported)

// TODO(jbms): support JSON -> complex conversion

/// ustring_t -> string_t conversion: converts to UTF8 encoding.
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(
    string_t, ustring_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit |
        DataTypeConversionFlags::kCanReinterpretCast);

/// string_t -> ustring_t conversion: validates UTF-8 encoding.
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(ustring_t, string_t,
                                           DataTypeConversionFlags::kSupported);

/// string_t -> json_t conversion: validates UTF-8 encoding
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(json_t, string_t,
                                           DataTypeConversionFlags::kSupported);

// TODO(jbms): Define string_t and ustring_t -> number, complex, bool
// conversions

/// ustring_t -> json_t conversion always succeeds.
TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(
    json_t, ustring_t,
    DataTypeConversionFlags::kSupported |
        DataTypeConversionFlags::kSafeAndImplicit)

#undef TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS1
#undef TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS
#undef TENSORSTORE_INTERNAL_INHERITED_CONVERT1
#undef TENSORSTORE_INTERNAL_INHERITED_CONVERT

}  // namespace tensorstore

#endif  // TENSORSTORE_DATA_TYPE_CONVERSION_H_
