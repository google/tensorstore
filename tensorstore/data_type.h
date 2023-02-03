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

#ifndef TENSORSTORE_DATA_TYPE_H_
#define TENSORSTORE_DATA_TYPE_H_

/// \file
/// Defines data type aliases and supporting classes used by tensorstore.
/// Generic data type operations are available using the `DataType` class,
/// which may represent built-in or custom data types.
///
/// The type `void` in this use is synonymous with a dynamic type.
///
/// Each built-in data type has:
///
/// * An enum `id` field in `DataTypeId` named of `DataTypeId::x_t`, which
///   is returned by `DataTypeIdOf<T>`
///
/// * A `StaticDataType<T>` class with pre-defined operations which is
///   implicitly convertible to `DataType`, and where the type or value is
///   returned by `dtype_t<T>` and `dtype_v<T>`, respectively.
///
/// * An entry in the `kDataTypes` array corresponding to the enum value.
///
/// Dynamic named lookup of a `DataType` is available via `GetDataType(name)`.
///

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>

#include "absl/status/status.h"
#include <half.hpp>
#include "tensorstore/internal/bit_operations.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/integer_types.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/util/bfloat16.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/util/utf8_string.h"

#ifdef _MSC_VER
// On MSVC, if `MakeDataTypeOperations<T>::operations` is not declared
// constexpr, it is initialized dynamically, which can happen too late if
// `DataType` is used from a global dynamic initializer, e.g. in order to
// allocate an Array.
#define TENSORSTORE_DATA_TYPE_CONSTEXPR_OPERATIONS
#endif

#ifdef TENSORSTORE_DATA_TYPE_CONSTEXPR_OPERATIONS
// Required by constexpr definition of `MakeDataTypeOperations<T>::operations`.
#include <nlohmann/json.hpp>
#endif

namespace tensorstore {

/// Boolean value (always represented as 0 or 1).
///
/// \ingroup data types
using bool_t = bool;
/// Single ASCII/UTF-8 code unit.  Primarily intended to represent fixed-width
/// ASCII fields.
///
/// \ingroup data types
using char_t = char;
/// Opaque byte value.  Intended to represent opaque binary data.
///
/// \ingroup data types
using byte_t = std::byte;
/// Signed and unsigned integer types.
///
/// \ingroup data types
using int8_t = std::int8_t;
///
/// \ingroup data types
using uint8_t = std::uint8_t;
///
/// \ingroup data types
using int16_t = std::int16_t;
///
/// \ingroup data types
using uint16_t = std::uint16_t;
///
/// \ingroup data types
using int32_t = std::int32_t;
///
/// \ingroup data types
using uint32_t = std::uint32_t;
///
/// \ingroup data types
using int64_t = std::int64_t;
///
/// \ingroup data types
using uint64_t = std::uint64_t;
// TODO(jbms): consider adding 128-bit integer types
/// :wikipedia:`IEEE 754 binary16<Half-precision_floating-point_format>`
/// half-precision floating-point data type.
///
/// \ingroup data types
using float16_t = half_float::half;

/// :wikipedia:`IEEE 754 binary32<Single-precision_floating-point_format>`
/// single-precision floating-point data type.
///
/// \ingroup data types
using float32_t = float;
/// :wikipedia:`IEEE 754 binary64<Double-precision_floating-point_format>`
/// double-precision floating-point data type.
///
/// \ingroup data types
using float64_t = double;
/// Complex number based on `float32_t`.
///
/// \ingroup data types
using complex64_t = std::complex<float32_t>;
/// Complex number based on `float64_t`.
///
/// \ingroup data types
using complex128_t = std::complex<float64_t>;
/// Byte string.
///
/// \ingroup data types
using string_t = std::string;
/// Unicode string, represented in memory as UTF-8.
///
/// \ingroup data types
using ustring_t = Utf8String;
/// JSON value.
///
/// \ingroup data types
using json_t = ::nlohmann::json;

// Define a DataTypeId `x_t` corresponding to each C++ type `tensorstore::x_t`
// defined above.
enum class DataTypeId {
  custom = -1,
  bool_t,
  char_t,
  byte_t,
  int8_t,
  uint8_t,
  int16_t,
  uint16_t,
  int32_t,
  uint32_t,
  int64_t,
  uint64_t,
  float16_t,
  bfloat16_t,
  float32_t,
  float64_t,
  complex64_t,
  complex128_t,
  string_t,
  ustring_t,
  json_t,
  num_ids
};

inline constexpr size_t kNumDataTypeIds =
    static_cast<size_t>(DataTypeId::num_ids);

// Indicates whether an element type can be treated as trivial.
template <typename T>
constexpr inline bool IsTrivial =
    std::is_trivially_destructible_v<T> && std::is_trivially_copyable_v<T>;

// TENSORSTORE_FOR_EACH_DATA_TYPE(X, ...) macros will instantiate
// X(datatype, ...) for each tensorstore data type.
#define TENSORSTORE_FOR_EACH_BYTE_DATA_TYPE(X, ...) \
  X(char_t, ##__VA_ARGS__)                          \
  X(byte_t, ##__VA_ARGS__)                          \
  /**/

#define TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE(X, ...) \
  X(int8_t, ##__VA_ARGS__)                             \
  X(uint8_t, ##__VA_ARGS__)                            \
  X(int16_t, ##__VA_ARGS__)                            \
  X(uint16_t, ##__VA_ARGS__)                           \
  X(int32_t, ##__VA_ARGS__)                            \
  X(uint32_t, ##__VA_ARGS__)                           \
  X(int64_t, ##__VA_ARGS__)                            \
  X(uint64_t, ##__VA_ARGS__)                           \
  /**/

#define TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE(X, ...) \
  X(float16_t, ##__VA_ARGS__)                        \
  X(bfloat16_t, ##__VA_ARGS__)                       \
  X(float32_t, ##__VA_ARGS__)                        \
  X(float64_t, ##__VA_ARGS__)                        \
  /**/

#define TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE(X, ...) \
  X(complex64_t, ##__VA_ARGS__)                        \
  X(complex128_t, ##__VA_ARGS__)                       \
  /**/

#define TENSORSTORE_FOR_EACH_DATA_TYPE(X, ...)             \
  X(bool_t, ##__VA_ARGS__)                                 \
  TENSORSTORE_FOR_EACH_BYTE_DATA_TYPE(X, ##__VA_ARGS__)    \
  TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE(X, ##__VA_ARGS__) \
  TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE(X, ##__VA_ARGS__)   \
  TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE(X, ##__VA_ARGS__) \
  X(string_t, ##__VA_ARGS__)                               \
  X(ustring_t, ##__VA_ARGS__)                              \
  X(json_t, ##__VA_ARGS__)                                 \
  /**/

// Generates the indicated data_type name, permitting nested
// macro expansion. Used, for example, in generating the
// conversion specializations between data types.
// The _BYTE_ and other variants are unused.
#define TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE_ID() \
  TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE

#define TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE_ID() \
  TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE

#define TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE_ID() \
  TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE

namespace internal_data_type {

template <typename T>
struct CanonicalElementTypeImpl {
  using type = T;
};

template <>
struct CanonicalElementTypeImpl<long> {            // NOLINT
  using type = internal::int_t<sizeof(long) * 8>;  // NOLINT
};

template <>
struct CanonicalElementTypeImpl<unsigned long> {             // NOLINT
  using type = internal::uint_t<sizeof(unsigned long) * 8>;  // NOLINT
};

template <>
struct CanonicalElementTypeImpl<long long> {            // NOLINT
  using type = internal::int_t<sizeof(long long) * 8>;  // NOLINT
};

template <>
struct CanonicalElementTypeImpl<unsigned long long> {             // NOLINT
  using type = internal::uint_t<sizeof(unsigned long long) * 8>;  // NOLINT
};

template <typename T>
inline constexpr DataTypeId DataTypeIdOfHelper = DataTypeId::custom;

#define TENSORSTORE_INTERNAL_DO_DATA_TYPE_ID(T, ...)                 \
  template <>                                                        \
  inline constexpr DataTypeId DataTypeIdOfHelper<T> = DataTypeId::T; \
  /**/
TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_DATA_TYPE_ID)
#undef TENSORSTORE_INTERNAL_DO_DATA_TYPE_ID

}  // namespace internal_data_type

/// Metafunction that maps an unqualified type `T` to the equivalent canonical
/// element type, if there is one.
///
/// If `T` is an integer type of ``N`` equal to 8, 16, 32, or 64 bits, the
/// equivalent canonical element type is ``intN_t`` (if `T` is signed) or
/// ``uintN_t`` (if `T` is unsigned).  Otherwise, the canonical element type
/// is `T`.
///
/// On all common platforms::
///
///   signed char == int8_t
///   short == int16_t
///   int == int32_t
///
/// However, `long` may be 32-bit (all 32-bit platforms, and 64-bit Windows) or
/// 64-bit (Linux/Mac OS X 64-bit).
///
/// The `long long` type is always 64-bit.
///
/// Therefore, depending on the platform, either `int` and `long` are distinct
/// types of the same size, or `long` and `long long` are distinct types of the
/// same size.
///
/// TensorStore data types are defined by size, but also have a corresponding
/// canonical C++ type.  In order to allow `int`, `long`, `long long` to be used
/// with TensorStore, this metafunction is used to ensure that non-canonical
/// types (`long` if `long` is 32-bit, `long long` if `long` is 64-bit) map to
/// the same TensorStore `DataType` as the corresponding canonical type.
///
/// \relates DataType
template <typename T>
using CanonicalElementType =
    typename internal_data_type::CanonicalElementTypeImpl<T>::type;

// `DataTypeId` corresponding to `T`, or `DataTypeId::custom` if `T` is not a
// canonical data type.
template <typename T>
inline constexpr DataTypeId DataTypeIdOf =
    internal_data_type::DataTypeIdOfHelper<
        CanonicalElementType<std::remove_cv_t<T>>>;

/// An ElementType is any optionally ``const``-qualified fundamental type
/// (including `void`), pointer type, member pointer type, class/union type, or
/// enumeration type.  A type of `void` or `const void` indicates a type-erased
/// element type.
///
/// \relates DataType
template <typename T>
constexpr inline bool IsElementType =
    (!std::is_volatile_v<T> &&
     // This is defined in terms of these exclusions, rather than
     // inclusions based on e.g. `std::is_fundamental`, because
     // `std::is_fundamental` excludes certain special types like
     // `__int128_t` and `_Float16` that we wish to support.
     !std::is_reference_v<T> && !std::is_function_v<std::remove_const_t<T>> &&
     !std::is_array_v<std::remove_const_t<T>>);

/// Specifies traits for the conversion from one data type to another.
///
/// \relates DataType
enum class DataTypeConversionFlags : unsigned char {
  /// Conversion is possible.  If not set, no other flags should be specified.
  kSupported = 1,
  /// The conversion requires no change to the in-memory representation.
  /// (i.e. conversion from intN_t -> uintN_t and vice versa).
  kCanReinterpretCast = 2,
  /// The conversion is guaranteed to succeed without any loss of information.
  /// These conversions are performed implicitly when needed.  Examples include
  /// float64 -> float32, int32 -> int16, int32 -> uint16, float32 -> complex64.
  /// As a special case, conversions from numeric to string types are not marked
  /// safe even though there is no loss of information, in order to prevent this
  /// implicit conversion.
  kSafeAndImplicit = 4,
  /// Conversion is from a given data type to itself (i.e. no conversion).
  kIdentity = 8,
};

/// Checks if any flags are set.
///
/// \id DataTypeConversionFlags
/// \relates DataTypeConversionFlags
inline constexpr bool operator!(DataTypeConversionFlags x) {
  return !static_cast<bool>(x);
}

/// Computes the union of the flag sets.
///
/// \id DataTypeConversionFlags
/// \relates DataTypeConversionFlags
inline constexpr DataTypeConversionFlags operator|(DataTypeConversionFlags a,
                                                   DataTypeConversionFlags b) {
  return DataTypeConversionFlags(static_cast<unsigned char>(a) |
                                 static_cast<unsigned char>(b));
}

/// Computes the complement of the flag set.
///
/// \id DataTypeConversionFlags
/// \relates DataTypeConversionFlags
inline constexpr DataTypeConversionFlags operator~(DataTypeConversionFlags x) {
  return DataTypeConversionFlags(~static_cast<unsigned char>(x));
}

/// Computes the intersection of the flag sets.
///
/// \id DataTypeConversionFlags
/// \relates DataTypeConversionFlags
inline constexpr DataTypeConversionFlags operator&(DataTypeConversionFlags a,
                                                   DataTypeConversionFlags b) {
  return DataTypeConversionFlags(static_cast<unsigned char>(a) &
                                 static_cast<unsigned char>(b));
}

namespace internal {

#ifndef _MSC_VER
using TypeInfo = const std::type_info&;
template <typename T>
constexpr const std::type_info& GetTypeInfo() {
  return typeid(T);
}
#else
/// Wrapper that behaves like `const std::type_info&` but which is
/// constexpr-compatible on MSVC.
class TypeInfo {
 public:
  using Getter = const std::type_info& (*)();
  explicit constexpr TypeInfo(Getter getter) : getter_(getter) {}

  operator const std::type_info&() const { return getter_(); }
  const std::type_info& type() const { return getter_(); }
  const char* name() const noexcept { return getter_().name(); }
  friend bool operator==(TypeInfo a, TypeInfo b) {
    return a.type() == b.type();
  }
  friend bool operator==(TypeInfo a, const std::type_info& b) {
    return a.type() == b;
  }
  friend bool operator==(const std::type_info& a, TypeInfo b) {
    return a == b.type();
  }
  friend bool operator!=(TypeInfo a, TypeInfo b) { return !(a == b); }
  friend bool operator!=(TypeInfo a, const std::type_info& b) {
    return !(a == b);
  }
  friend bool operator!=(const std::type_info& a, TypeInfo b) {
    return !(a == b);
  }

  template <typename T>
  static const std::type_info& GetImpl() {
    return typeid(T);
  }

 private:
  Getter getter_;
};
template <typename T>
constexpr TypeInfo GetTypeInfo() {
  return TypeInfo(&TypeInfo::GetImpl<T>);
}
#endif

/// Type-specific operations needed for dynamically-typed multi-dimensional
/// arrays.
///
/// Instances of the struct should only be created by code within this module.
///
/// Use `DataType`, defined below, to refer to instances of this struct.
struct DataTypeOperations {
  DataTypeId id;

  std::string_view name;

  /// The type_info structure for this type.
  TypeInfo type;

  /// The size in bytes of this type.
  std::ptrdiff_t size;

  /// The alignment in bytes of this type.
  std::ptrdiff_t alignment;

  /// Default initialize an array of `count` elements.
  ///
  /// \note This does not initialize primitives types.
  using ConstructFunction = void (*)(Index count, void* ptr);
  ConstructFunction construct;

  /// Destroy an array of `count` elements.
  using DestroyFunction = void (*)(Index count, void* ptr);
  DestroyFunction destroy;

  /// Assign all elements of array to the result obtained by value
  /// initialization.
  ///
  /// \note For primitive types, this assigns to zero.
  using InitializeFunction = ElementwiseFunction<1, absl::Status*>;
  InitializeFunction initialize;

  /// Copy assign elements from one array to another.
  using CopyAssignFunction = ElementwiseFunction<2, absl::Status*>;
  CopyAssignFunction copy_assign;

  /// Move assign elements from one array to another.
  using MoveAssignFunction = ElementwiseFunction<2, absl::Status*>;
  MoveAssignFunction move_assign;

  /// Copy assign elements from one array to another where a third mask array is
  /// `false`.
  using CopyAssignUnmaskedFunction = ElementwiseFunction<3, absl::Status*>;
  CopyAssignUnmaskedFunction copy_assign_unmasked;

  /// Append a string representation of an element to `*result`.
  using AppendToStringFunction = void (*)(std::string* result, const void* ptr);
  AppendToStringFunction append_to_string;

  /// Compares two strided arrays for equality.
  using CompareEqualFunction = ElementwiseFunction<2, absl::Status*>;
  CompareEqualFunction compare_equal;

  /// Compares two strided arrays for equality, taking into account negative
  /// zero and NaN for floating point types (negative zero is not equal to
  /// positive zero, and NaN is equal to NaN).
  ///
  /// Note that this not the same as bit equality, because there are multiple
  /// possible bit representations of NaN, and this function considers all of
  /// them to be equal.
  CompareEqualFunction compare_same_value;

  struct CanonicalConversionOperations {
    // Function for converting to/from canonical data type.
    using ConvertFunction = ElementwiseFunction<2, absl::Status*>;
    std::array<ConvertFunction, kNumDataTypeIds> convert;
    std::array<DataTypeConversionFlags, kNumDataTypeIds> flags;
  };

  struct BidirectionalCanonicalConversionOperations {
    CanonicalConversionOperations to;
    CanonicalConversionOperations from;
  };

  const BidirectionalCanonicalConversionOperations* canonical_conversion;
};

/// Specifies a conversion between two data types.
///
/// This serves as the return type of `GetDataTypeConverter` declared in
/// `data_type_conversion.h`.
struct DataTypeConversionLookupResult {
  /// Valid only if the `flags` value includes `kSupported`.
  ElementwiseClosure<2, absl::Status*> closure;
  DataTypeConversionFlags flags;
};

}  // namespace internal

/// Run-time representation of a C++ type used as the element type for a
/// multi-dimensional array.
///
/// This is a Regular type that is inexpensive to copy (equivalent to a
/// pointer).
///
/// In generic code, StaticDataType can be used as a drop-in replacement when
/// the type is known at compile time.
///
/// This permits array operations, such as allocation, zero initialization,
/// copying/moving, printing to a string, and comparison to be performed on
/// arrays whose element type is specified at compile time.
///
/// Except when allocating new memory, an `DataType` value is typically paired
/// with a `void *` pointer to an element of the corresponding type (this
/// pairing is implemented by the ElementPointer class).
///
/// A `DataType` instance corresponding to a type known at compile time may be
/// obtained using `dtype_v`.
///
/// \ingroup data types
class DataType {
  using Ops = internal::DataTypeOperations;

 public:
  using Element = void;
  /// Initializes to an invalid data type.
  constexpr DataType() : operations_(nullptr) {}

  constexpr DataType(const internal::DataTypeOperations* operations)
      : operations_(operations) {}

  constexpr DataType(unchecked_t, DataType other) : DataType(other) {}

  /// Returns `true` if this represents a valid data type.
  constexpr bool valid() const { return operations_ != nullptr; }

  constexpr DataType dtype() const { return *this; }

  constexpr DataTypeId id() const { return operations_->id; }

  /// Returns the data type name, e.g. ``"bool"`` or ``"uint32"``.
  constexpr std::string_view name() const { return operations_->name; }

  /// Returns the size in bytes of the data type.
  constexpr std::ptrdiff_t size() const { return operations_->size; }

  /// Returns the alignment required by the data type.
  constexpr std::ptrdiff_t alignment() const { return operations_->alignment; }

  constexpr Ops::ConstructFunction construct_function() const {
    return operations_->construct;
  }

  constexpr Ops::DestroyFunction destroy_function() const {
    return operations_->destroy;
  }

  constexpr Ops::AppendToStringFunction append_to_string_function() const {
    return operations_->append_to_string;
  }

  constexpr const Ops::InitializeFunction& initialize_function() const {
    return operations_->initialize;
  }

  constexpr const Ops::CompareEqualFunction& compare_equal_function() const {
    return operations_->compare_equal;
  }

  constexpr const Ops::CompareEqualFunction& compare_same_value_function()
      const {
    return operations_->compare_same_value;
  }

  constexpr const Ops::CopyAssignFunction& copy_assign_function() const {
    return operations_->copy_assign;
  }

  constexpr const internal::DataTypeOperations* operator->() const {
    return operations_;
  }

  /// Abseil hash support.
  ///
  /// For consistency with the comparison operators, this simply forwards to the
  /// `std::type_index` hash code.
  template <typename H>
  friend H AbslHashValue(H h, DataType x) {
    return H::combine(std::move(h), std::type_index(x->type));
  }

  /// Comparison operators.
  friend constexpr bool operator==(DataType a, DataType b) {
    // These depend only on the `type` because there should only be a single
    // `DataTypeOperations` object per type.  To handle possible multiple
    // instances due to certain dynamic linking modes, however, we rely on the
    // `operator==` defined for `std::type_info` rather than comparing the
    // `operators_` pointers directly.
    return a.valid() == b.valid() &&
           (a.operations_ == b.operations_ || a->type == b->type);
  }
  friend constexpr bool operator!=(DataType a, DataType b) { return !(a == b); }
  friend constexpr bool operator==(DataType r, const std::type_info& type) {
    return r.valid() && r->type == type;
  }
  friend constexpr bool operator!=(DataType r, const std::type_info& type) {
    return !(r == type);
  }
  friend constexpr bool operator==(const std::type_info& type, DataType r) {
    return r.valid() && r->type == type;
  }
  friend constexpr bool operator!=(const std::type_info& type, DataType r) {
    return !(r == type);
  }

  /// Prints `name()` if `valid() == true`, otherwise prints `"<unspecified>"`.
  friend std::ostream& operator<<(std::ostream& os, DataType r);

 private:
  // \invariant operations_ != nullptr
  const internal::DataTypeOperations* operations_;
};

namespace internal_data_type {

/// Returns the name of the data type corresponding to the C++ type `T`.
///
/// For all of the standard data types, a specialization is defined below.  This
/// definition is only used for custom data types.
template <typename T>
constexpr std::string_view GetTypeName() {
  // While it would be nice to return a meaningful name, `typeid(T).name()` is
  // not constexpr (and includes mangling).
  return "unknown";
}

#define TENSORSTORE_INTERNAL_DO_DATA_TYPE_NAME(T, ...) \
  template <>                                          \
  constexpr std::string_view GetTypeName<T>() {        \
    return std::string_view(#T, sizeof(#T) - 3);       \
  }                                                    \
  /**/
TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_DATA_TYPE_NAME)
#undef TENSORSTORE_INTERNAL_DO_DATA_TYPE_NAME

/// Compares two values for equality. For comparable types, this
/// is the same as `operator==`.
template <typename T>
bool CompareEqual(const T& a, const T& b) {
  if constexpr (internal::IsEqualityComparable<T>) {
    return a == b;
  }
  return false;
}

/// Compares two values for "same value" equality.
///
/// For non-floating point types, this behaves the same as normal `operator==`.
/// For floating point types, this differs from normal `operator==` in that
/// negative zero is not equal to positive zero, and NaN is equal to NaN.
///
/// Note that this differs from bit equality, because there are multiple bit
/// representations of NaN, and this functions treats all of them as equal.
template <typename T>
bool CompareSameValue(const T& a, const T& b) {
  if constexpr (internal::IsEqualityComparable<T>) {
    return a == b;
  }
  return false;
}

#define TENSORSTORE_INTERNAL_DO_DEFINE_COMPARE_SAME_VALUE_FLOAT(T, ...) \
  template <>                                                           \
  inline bool CompareSameValue<T>(const T& a, const T& b) {             \
    using std::isnan;                                                   \
    if (isnan(a)) return isnan(b);                                      \
    using Int = internal::uint_t<sizeof(T) * 8>;                        \
    return internal::bit_cast<Int>(a) == internal::bit_cast<Int>(b);    \
  }                                                                     \
  /**/
TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE(
    TENSORSTORE_INTERNAL_DO_DEFINE_COMPARE_SAME_VALUE_FLOAT)
#undef TENSORSTORE_INTERNAL_DO_DEFINE_COMPARE_SAME_VALUE_FLOAT

#define TENSORSTORE_INTERNAL_DO_DEFINE_COMPARE_SAME_VALUE_COMPLEX(T, ...) \
  template <>                                                             \
  inline bool CompareSameValue<T>(const T& a, const T& b) {               \
    return CompareSameValue(a.real(), b.real()) &&                        \
           CompareSameValue(a.imag(), b.imag());                          \
  }                                                                       \
  /**/
TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE(
    TENSORSTORE_INTERNAL_DO_DEFINE_COMPARE_SAME_VALUE_COMPLEX)
#undef TENSORSTORE_INTERNAL_DO_DEFINE_COMPARE_SAME_VALUE_COMPLEX

/// Non-template functions referenced by `DataTypeOperations`.
///
/// These are defined separately so that they can be explicitly instantiated.
template <typename T>
struct DataTypeSimpleOperationsImpl {
  static void Construct(Index count, void* ptr) {
    std::uninitialized_default_construct(static_cast<T*>(ptr),
                                         static_cast<T*>(ptr) + count);
  }

  static void Destroy(Index count, void* ptr) {
    std::destroy(static_cast<T*>(ptr), static_cast<T*>(ptr) + count);
  }

  static void AppendToString(std::string* result, const void* ptr) {
    if constexpr (internal::IsOstreamable<T>) {
      tensorstore::StrAppend(result, *static_cast<const T*>(ptr));
    }
  }
};

/// Elementwise functions referenced by `DataTypeOperations`.
template <typename T>
struct DataTypeElementwiseOperationsImpl {
  struct InitializeImpl {
    void operator()(T* dest, absl::Status*) const { *dest = T(); }
  };

  struct CopyAssignImpl {
    void operator()(const T* source, T* dest, absl::Status*) const {
      *dest = *source;
    }
  };

  struct MoveAssignImpl {
    void operator()(T* source, T* dest, absl::Status*) const {
      *dest = std::move(*source);
    }
  };

  struct CopyAssignUnmaskedImpl {
    void operator()(const T* source, T* dest, const bool* mask,
                    absl::Status*) const {
      if (!*mask) *dest = *source;
    }
  };

  struct CompareEqualImpl {
    bool operator()(const T* source, const T* dest, absl::Status*) const {
      return internal_data_type::CompareEqual<T>(*source, *dest);
    }
  };

  struct CompareSameValueImpl {
    bool operator()(const T* source, const T* dest, absl::Status*) const {
      return internal_data_type::CompareSameValue<T>(*source, *dest);
    }
  };

  using Initialize =
      internal::SimpleElementwiseFunction<InitializeImpl(T), absl::Status*>;

  using CopyAssign =
      internal::SimpleElementwiseFunction<CopyAssignImpl(const T, T),
                                          absl::Status*>;
  using MoveAssign =
      internal::SimpleElementwiseFunction<MoveAssignImpl(T, T), absl::Status*>;

  using CopyAssignUnmasked = internal::SimpleElementwiseFunction<
      CopyAssignUnmaskedImpl(const T, T, const bool), absl::Status*>;
  using CompareEqual =
      internal::SimpleElementwiseFunction<CompareEqualImpl(const T, const T),
                                          absl::Status*>;
  using CompareSameValue = internal::SimpleElementwiseFunction<
      CompareSameValueImpl(const T, const T), absl::Status*>;
};

template <typename T>
constexpr internal::DataTypeOperations DataTypeOperationsImpl = {
    /*.id=*/DataTypeIdOf<T>,
    /*.name=*/GetTypeName<T>(),
    /*.type=*/internal::GetTypeInfo<T>(),
    /*.size=*/sizeof(T),
    /*.align=*/alignof(T),
    /*.construct=*/&DataTypeSimpleOperationsImpl<T>::Construct,
    /*.destroy=*/&DataTypeSimpleOperationsImpl<T>::Destroy,
    /*.initialize=*/
    typename DataTypeElementwiseOperationsImpl<T>::Initialize(),
    /*.copy_assign=*/
    typename DataTypeElementwiseOperationsImpl<T>::CopyAssign(),
    /*.move_assign=*/
    typename DataTypeElementwiseOperationsImpl<T>::MoveAssign(),
    /*.copy_assign_unmasked=*/
    typename DataTypeElementwiseOperationsImpl<T>::CopyAssignUnmasked(),
    /*.append_to_string=*/&DataTypeSimpleOperationsImpl<T>::AppendToString,
    /*.compare_equal=*/
    typename DataTypeElementwiseOperationsImpl<T>::CompareEqual(),
    /*.compare_equal=*/
    typename DataTypeElementwiseOperationsImpl<T>::CompareSameValue(),
    /*.canonical_conversion=*/nullptr,
};

template <typename T>
class MakeDataTypeOperations {
 public:
#ifdef TENSORSTORE_DATA_TYPE_CONSTEXPR_OPERATIONS
  static constexpr internal::DataTypeOperations operations =
      DataTypeOperationsImpl<T>;
#else
  static const internal::DataTypeOperations operations;
#endif
};

#ifndef TENSORSTORE_DATA_TYPE_CONSTEXPR_OPERATIONS
template <typename T>
const internal::DataTypeOperations MakeDataTypeOperations<T>::operations =
    DataTypeOperationsImpl<T>;
#endif

#define TENSORSTORE_DATA_TYPE_EXPLICIT_INSTANTIATION(T, ...)   \
  __VA_ARGS__ template class MakeDataTypeOperations<T>;        \
  __VA_ARGS__ template struct DataTypeSimpleOperationsImpl<T>; \
  /**/

// Declare explicit instantiations of MakeDataTypeOperations, which are defined
// in dtype.cc, in order to reduce compilation time and object file bloat.
TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_DATA_TYPE_EXPLICIT_INSTANTIATION,
                               extern)

}  // namespace internal_data_type

/// Empty/monostate type that represents a statically known element type.
///
/// In generic code, this can be used in place of an `DataType` when the element
/// type is statically known.
///
/// \relates DataType
template <typename T>
class StaticDataType {
 private:
  using Ops = internal::DataTypeOperations;
  using SimpleOps = internal_data_type::DataTypeSimpleOperationsImpl<T>;
  using ElementwiseOps =
      internal_data_type::DataTypeElementwiseOperationsImpl<T>;

 public:
  using Element = T;
  static_assert(std::is_same_v<T, std::decay_t<T>>,
                "T must be an unqualified type.");
  static_assert(IsElementType<T>, "T must satisfy IsElementType.");

  constexpr StaticDataType() = default;

  constexpr StaticDataType(unchecked_t, StaticDataType other) {}

  template <typename Other>
  constexpr StaticDataType(unchecked_t, StaticDataType<Other>) = delete;

  constexpr StaticDataType(unchecked_t, DataType other) {}

  static constexpr bool valid() { return true; }

  constexpr static StaticDataType dtype() { return {}; }

  constexpr static DataTypeId id() { return DataTypeIdOf<T>; }

  constexpr std::string_view name() const {
    return internal_data_type::GetTypeName<T>();
  }

  constexpr std::ptrdiff_t size() const { return sizeof(T); }

  constexpr std::ptrdiff_t alignment() const { return alignof(T); }

  constexpr Ops::ConstructFunction construct_function() const {
    return &SimpleOps::Construct;
  }

  constexpr Ops::DestroyFunction destroy_function() const {
    return &SimpleOps::Destroy;
  }

  constexpr Ops::AppendToStringFunction append_to_string_function() const {
    return &SimpleOps::AppendToString;
  }

  constexpr const Ops::InitializeFunction& initialize_function() const {
    return ElementwiseOps::Initialize::function;
  }

  constexpr const Ops::CompareEqualFunction& compare_equal_function() const {
    return ElementwiseOps::CompareEqual::function;
  }

  constexpr const Ops::CopyAssignFunction& copy_assign_function() const {
    return ElementwiseOps::CopyAssign::function;
  }

  constexpr const Ops* operator->() const {
    return &internal_data_type::MakeDataTypeOperations<T>::operations;
  }

  constexpr operator DataType() const {
    return DataType(&internal_data_type::MakeDataTypeOperations<T>::operations);
  }

  friend constexpr bool operator==(StaticDataType a, StaticDataType b) {
    return true;
  }

  friend constexpr bool operator==(StaticDataType a, const std::type_info& b) {
    return typeid(T) == b;
  }

  friend constexpr bool operator!=(StaticDataType a, const std::type_info& b) {
    return typeid(T) != b;
  }

  friend constexpr bool operator==(const std::type_info& b, StaticDataType a) {
    return typeid(T) == b;
  }

  friend constexpr bool operator!=(const std::type_info& b, StaticDataType a) {
    return typeid(T) != b;
  }

  friend constexpr bool operator==(StaticDataType a, DataType b) {
    return static_cast<DataType>(a) == b;
  }

  friend constexpr bool operator==(DataType b, StaticDataType a) {
    return static_cast<DataType>(a) == b;
  }

  friend constexpr bool operator!=(StaticDataType a, StaticDataType b) {
    return false;
  }

  friend constexpr bool operator!=(StaticDataType a, DataType b) {
    return static_cast<DataType>(a) != b;
  }

  friend constexpr bool operator!=(DataType b, StaticDataType a) {
    return static_cast<DataType>(a) != b;
  }

  template <typename U>
  friend constexpr bool operator==(StaticDataType a, StaticDataType<U> b) {
    return false;
  }

  template <typename U>
  friend constexpr bool operator!=(StaticDataType a, StaticDataType<U> b) {
    return true;
  }

  friend std::ostream& operator<<(std::ostream& os, StaticDataType r) {
    return os << DataType(r);
  }
};

// We declare but do not define specialization for void, as `void` is not a
// valid element representation type (it is only used to indicate that the type
// is not known at compile time).
template <>
class StaticDataType<void>;

/// Alias for the `StaticDataType` representing `T` if `T` is not `void`, or
/// `DataType` if `T` is `void`.
///
/// Qualifiers of `T` are ignored, and additionally `T` is "canonicalized"
/// according to `CanonicalElementType`.
///
/// \param T C++ element type for which to obtain the corresponding data type.
///     Any const/volatile qualifiers are ignored.
/// \relates DataType
template <typename T = void>
using dtype_t = std::conditional_t<
    std::is_void_v<T>, DataType,
    StaticDataType<CanonicalElementType<std::remove_cv_t<T>>>>;

/// Data type object representing the data type for `T`, convertible to
/// `DataType`.
///
/// If `T` is `void`, equal to `DataType()` (representing an unknown data type).
///
/// Otherwise, equal to ``StaticDataType<U>()``, where ``U`` is obtained
/// from `T` by striping cv-qualifiers and applying `CanonicalElementType`.
///
/// \param T C++ element type for which to obtain the corresponding data type.
///     Any const/volatile qualifiers are ignored.
/// \relates DataType
template <typename T>
inline constexpr auto dtype_v = dtype_t<T>();

// Returns `{ func(dtype_v<T>)... }` where `T` ranges over the canonical
// data types.
template <typename Func>
constexpr std::array<std::invoke_result_t<Func, dtype_t<bool>>, kNumDataTypeIds>
MapCanonicalDataTypes(Func func) {
  return {{
#define TENSORSTORE_INTERNAL_DO_DATA_TYPE(T, ...) func(dtype_v<T>),
      TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_DATA_TYPE)
#undef TENSORSTORE_INTERNAL_DO_DATA_TYPE
  }};
}

/// Specifies the form of initialization to use when allocating an array.
///
/// \relates DataType
/// \membergroup Allocation
enum class ElementInitialization {
  /// Specifies default initialization.  For primitive types, or class types for
  /// which the default constructor leaves some members uninitialized, this
  /// results in indeterminite values.
  default_init,

  /// Specifies value initialization.
  value_init
};

/// \relates ElementInitialization
constexpr ElementInitialization default_init =
    ElementInitialization::default_init;

/// \relates ElementInitialization
constexpr ElementInitialization value_init = ElementInitialization::value_init;

/// Allocates and initializes a contiguous 1-dimensional array of `n` elements
/// of type `r` specified at run time.
///
/// The memory must be freed by invoking `r->destroy` and then calling
/// `::operator delete` with an alignment of `r->alignment`.
///
/// On failure to allocate memory, throws `std::bad_alloc`, or terminates the
/// process if exceptions are disabled.
///
/// \param n The number of elements to allocate.
/// \param initialization The form of initialization to use.
/// \param r The element type.
/// \returns A pointer to the allocated array.
///
/// .. note::
///
///    For primitive types, default initialization leaves the elements
///    uninitialized.
///
/// \relates DataType
/// \membergroup Allocation
void* AllocateAndConstruct(std::ptrdiff_t n,
                           ElementInitialization initialization, DataType r);

/// Frees memory allocated by AllocateAndConsruct.
///
/// Equivalent to:
///
///     r->destroy(n, ptr);
///     ::operator delete(ptr, std::align_val_t(r->alignment));
///
/// \params n The number of elements that were allocated and constructed.
/// \params r The element type.
/// \params ptr Pointer to the allocated array of `n` elements.
/// \relates DataType
/// \membergroup Allocation
void DestroyAndFree(std::ptrdiff_t n, DataType r, void* ptr);

/// Returns a shared_ptr that manages the memory returned by
/// `AllocateAndConstruct`.
///
/// \tparam T Optional.  The element type.  If unspecified (or equal to `void`),
///     the element type must be specified at run time using the `r` parameter.
/// \param n The number of elements to allocate.
/// \param initialization Optional.  The form of initialization to use.
/// \param r The element type.  Optional if `T` is not `void`.
/// \relates DataType
/// \membergroup Allocation
template <typename T = void>
std::shared_ptr<T> AllocateAndConstructShared(
    std::ptrdiff_t n, ElementInitialization initialization = default_init,
    dtype_t<T> r = dtype_v<T>) {
  static_assert(std::is_same_v<std::remove_cv_t<T>, T>,
                "Element type T must not have cv qualifiers.");
  return std::static_pointer_cast<T>(
      AllocateAndConstructShared<void>(n, initialization, r));
}

template <>
std::shared_ptr<void> AllocateAndConstructShared<void>(
    std::ptrdiff_t n, ElementInitialization initialization, DataType r);

/// Checks if both data types are equal or at least one is unspecified.
///
/// \relates DataType
inline bool IsPossiblySameDataType(DataType a, DataType b) {
  return !b.valid() || !a.valid() || a == b;
}
template <typename T, typename U>
constexpr inline bool IsPossiblySameDataType(StaticDataType<T> a,
                                             StaticDataType<U> b) {
  return std::is_same_v<T, U>;
}

/// Evaluates to a type similar to `SourceRef` but with a static data type of
/// `TargetElement`.
///
/// Supported types include `ElementPointer`, `Array`, `TransformedArray`,
/// `TensorStore`.
///
/// \tparam SourceRef Optionally ``const``- and/or reference-qualified source
///     type.  Any qualifiers are ignored.
/// \tparam TargetElement Target element type.
/// \ingroup compile-time-constraints
template <typename SourceRef, typename TargetElement>
using RebindDataType = typename StaticCastTraitsType<
    SourceRef>::template RebindDataType<TargetElement>;

/// Casts `source` to have a static data type of `TargetElement`.
///
/// The source type must be supported by `RebindDataType` and define a nested
/// ``Element`` type, and both the source and target types must be supported
/// by `StaticCast`.
///
/// The semantics of the `Checking` parameter are the same as for `StaticCast`.
///
/// This cast cannot be used to cast away const qualification of the source
/// element type.  To do that, use `ConstDataTypeCast` instead.
///
/// Examples::
///
///     Array<void> array = ...;
///     Result<Array<int>> result = StaticDataTypeCast<int>(array);
///     Result<Array<const int>> result2 = StaticDataTypeCast<const int>(array);
///     Array<const int> unchecked_result =
///         StaticDataTypeCast<const int, unchecked>(array);
///
///     DataType d = ...;
///     Result<dtype_t<int>> d_static = StaticDataTypeCast<int>(d);
///
///     dtype_t<int> d_int;
///     DataType d_dynamic = StaticDataTypeCast<void>(d_int);
///
/// \tparam TargetElement Target element type.  Depending on the source type,
///     ``const``-qualified `TargetElement` types may or may not be
///     supported.
/// \tparam Checking Specifies whether the cast is checked or unchecked.
/// \param source Source value.
/// \ingroup compile-time-constraints
template <typename TargetElement, CastChecking Checking = CastChecking::checked,
          typename SourceRef>
StaticCastResultType<RebindDataType<SourceRef, TargetElement>, SourceRef,
                     Checking>
StaticDataTypeCast(SourceRef&& source) {
  using Source = internal::remove_cvref_t<SourceRef>;
  static_assert(IsElementTypeExplicitlyConvertible<typename Source::Element,
                                                   TargetElement>,
                "StaticDataTypeCast cannot cast away const qualification");
  return StaticCast<RebindDataType<SourceRef, TargetElement>, Checking>(
      std::forward<SourceRef>(source));
}

/// Casts `source` to a specified target element type which must differ from the
/// existing element type only in const qualification.
///
/// Supported types include `ElementPointer`, `Array`, `TransformedArray`,
/// `TensorStore`.
///
/// This cast is always unchecked.
///
/// Example::
///
///     Array<const int> const_array = ...;
///     Array<int> array = ConstDataTypeCast<int>(const_array);
///
/// \tparam TargetElement Target element type.
/// \ingroup compile-time-constraints
template <typename TargetElement, typename SourceRef>
inline StaticCastResultType<RebindDataType<SourceRef, TargetElement>, SourceRef>
ConstDataTypeCast(SourceRef&& source) {
  using Source = internal::remove_cvref_t<SourceRef>;
  static_assert(
      std::is_same_v<const typename Source::Element, const TargetElement>,
      "ConstDataTypeCast can only change const qualification");
  return StaticCast<RebindDataType<SourceRef, TargetElement>,
                    CastChecking::unchecked>(std::forward<SourceRef>(source));
}

// `StaticCastTraits` specialization for `DataType`.
template <>
struct StaticCastTraits<DataType> : public DefaultStaticCastTraits<DataType> {
  static std::string Describe() { return Describe(DataType{}); }
  static std::string Describe(DataType dtype);
  static constexpr bool IsCompatible(DataType other) { return true; }
  template <typename TargetElement>
  using RebindDataType = dtype_t<TargetElement>;
};

// `StaticCastTraits` specialization for `StaticDataType<T>`.
template <typename T>
struct StaticCastTraits<StaticDataType<T>>
    : public DefaultStaticCastTraits<StaticDataType<T>> {
  static std::string Describe() {
    return StaticCastTraits<DataType>::Describe(dtype_v<T>);
  }
  static std::string Describe(StaticDataType<T>) { return Describe(); }

  template <typename Other>
  static constexpr bool IsCompatible(Other other) {
    return !other.valid() || other == StaticDataType<T>();
  }
  template <typename TargetElement>
  using RebindDataType = dtype_t<TargetElement>;
};

/// Returns the `DataType` with `DataType::name` equal to `id`.
///
/// If `id` does not specify a supported data type name, returns the invalid
/// data type of `DataType()`.
///
/// Example::
///
///     EXPECT_EQ(dtype_v<std::int32_t>, GetDataType("int32"));
///     EXPECT_EQ(dtype_v<float>, GetDataType("float32"));
///
/// \relates DataType
DataType GetDataType(std::string_view id);

constexpr DataType kDataTypes[] = {
#define TENSORSTORE_INTERNAL_DO_DATA_TYPE(T, ...) dtype_v<T>,
    TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_DATA_TYPE)
#undef TENSORSTORE_INTERNAL_DO_DATA_TYPE
};

/// `StaticDataType` corresponding to the element type of `Pointer`, or
/// `DataType` if the element type is `void`.
///
/// \relates ElementPointer
template <typename Pointer>
using pointee_dtype_t = dtype_t<typename std::pointer_traits<
    internal::remove_cvref_t<Pointer>>::element_type>;

namespace internal {
absl::Status NonSerializableDataTypeError(DataType dtype);
}  // namespace internal
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::DataType)

#endif  //  TENSORSTORE_DATA_TYPE_H_
