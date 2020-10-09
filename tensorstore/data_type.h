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

#include <half.hpp>
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/integer_types.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/util/assert_macros.h"
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

namespace data_types {
/// Boolean value (always represented as 0 or 1).
using bool_t = bool;
/// Single ASCII/UTF-8 code unit.  Primarily intended to represent fixed-width
/// ASCII fields.
using char_t = char;
/// Opaque byte value.  Intended to represent opaque binary data.
using byte_t = std::byte;
/// Signed and unsigned integer types.
using int8_t = std::int8_t;
using uint8_t = std::uint8_t;
using int16_t = std::int16_t;
using uint16_t = std::uint16_t;
using int32_t = std::int32_t;
using uint32_t = std::uint32_t;
using int64_t = std::int64_t;
using uint64_t = std::uint64_t;
// TODO(jbms): consider adding 128-bit integer types
/// Floating-point types.
using float16_t = half_float::half;
using float32_t = float;
using float64_t = double;
/// Complex types.
using complex64_t = std::complex<float32_t>;
using complex128_t = std::complex<float64_t>;
/// Byte string.
using string_t = std::string;
/// Unicode string, represented in memory as UTF-8.
using ustring_t = Utf8String;
/// JSON value.
using json_t = ::nlohmann::json;
}  // namespace data_types
using namespace data_types;  // NOLINT

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

/// Permits nested expansions.
#define TENSORSTORE_FOR_EACH_BYTE_DATA_TYPE_ID() \
  TENSORSTORE_FOR_EACH_BYTE_DATA_TYPE

#define TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE_ID() \
  TENSORSTORE_FOR_EACH_INTEGER_DATA_TYPE

#define TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE_ID() \
  TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE

#define TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE_ID() \
  TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE

enum class DataTypeId {
  custom = -1,
// Define a DataTypeId `x_t` corresponding to each C++ type `tensorstore::x_t`
// defined above.
#define TENSORSTORE_INTERNAL_DO_DATA_TYPE_ID(T, ...) T,
  TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_DATA_TYPE_ID)
#undef TENSORSTORE_INTERNAL_DO_DATA_TYPE_ID
      num_ids,
};

inline constexpr size_t kNumDataTypeIds =
    static_cast<size_t>(DataTypeId::num_ids);

namespace internal_data_type {

/// Metafunction that maps an unqualified type `T` to the equivalent canonical
/// element type, if there is one.
///
/// If `T` is an integer type of `N` equal to 8, 16, 32, or 64 bits, the
/// equivalent canonical element type is `intN_t` (if `T` is signed) or
/// `uintN_t` (if `T` is unsigned).  Otherwise, the canonical element type is
/// `T`.
///
/// On all common platforms:
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
/// the same TensorStore DataType as the corresponding canonical type.
template <typename T>
struct CanonicalElementType {
  using type = T;
};

template <>
struct CanonicalElementType<long> {                // NOLINT
  using type = internal::int_t<sizeof(long) * 8>;  // NOLINT
};

template <>
struct CanonicalElementType<unsigned long> {                 // NOLINT
  using type = internal::uint_t<sizeof(unsigned long) * 8>;  // NOLINT
};

template <>
struct CanonicalElementType<long long> {                // NOLINT
  using type = internal::int_t<sizeof(long long) * 8>;  // NOLINT
};

template <>
struct CanonicalElementType<unsigned long long> {                 // NOLINT
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

/// `DataTypeId` corresponding to `T`, or `DataTypeId::custom` if `T` is not a
/// canonical data type.
template <typename T>
inline constexpr DataTypeId DataTypeIdOf =
    internal_data_type::DataTypeIdOfHelper<
        typename internal_data_type::CanonicalElementType<
            std::remove_cv_t<T>>::type>;

/// An ElementType is any optionally `const`-qualified fundamental type
/// (including `void`), pointer type, member pointer type, class/union type, or
/// enumeration type.  A type of `void` or `const void` indicates a type-erased
/// element type.
template <typename T>
struct IsElementType
    : public std::integral_constant<
          bool, (!std::is_volatile<T>::value &&
                 // This is defined in terms of these exclusions, rather than
                 // inclusions based on e.g. `std::is_fundamental`, because
                 // `std::is_fundamental` excludes certain special types like
                 // `__int128_t` and `_Float16` that we wish to support.
                 !std::is_reference<T>::value &&
                 !std::is_function<std::remove_const_t<T>>::value &&
                 !std::is_array<std::remove_const_t<T>>::value)> {};

/// Specifies traits for the conversion from one data type to another.
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

inline constexpr bool operator!(DataTypeConversionFlags x) {
  return !static_cast<bool>(x);
}

inline constexpr DataTypeConversionFlags operator|(DataTypeConversionFlags a,
                                                   DataTypeConversionFlags b) {
  return DataTypeConversionFlags(static_cast<unsigned char>(a) |
                                 static_cast<unsigned char>(b));
}

inline constexpr DataTypeConversionFlags operator~(DataTypeConversionFlags x) {
  return DataTypeConversionFlags(~static_cast<unsigned char>(x));
}

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

  absl::string_view name;

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
  using InitializeFunction = ElementwiseFunction<1, Status*>;
  InitializeFunction initialize;

  /// Copy assign elements from one array to another.
  using CopyAssignFunction = ElementwiseFunction<2, Status*>;
  CopyAssignFunction copy_assign;

  /// Move assign elements from one array to another.
  using MoveAssignFunction = ElementwiseFunction<2, Status*>;
  MoveAssignFunction move_assign;

  /// Copy assign elements from one array to another where a third mask array is
  /// `false`.
  using CopyAssignUnmaskedFunction = ElementwiseFunction<3, Status*>;
  CopyAssignUnmaskedFunction copy_assign_unmasked;

  /// Append a string representation of an element to `*result`.
  using AppendToStringFunction = void (*)(std::string* result, const void* ptr);
  AppendToStringFunction append_to_string;

  /// Compares two strided arrays for equality.
  using CompareEqualFunction = ElementwiseFunction<2, Status*>;
  CompareEqualFunction compare_equal;

  struct CanonicalConversionOperations {
    // Function for converting to/from canonical data type.
    using ConvertFunction = ElementwiseFunction<2, Status*>;
    std::array<ConvertFunction, kNumDataTypeIds> convert;
    std::array<DataTypeConversionFlags, kNumDataTypeIds> flags;
  };

  struct BidirectionalCanonicalConversionOperations {
    CanonicalConversionOperations to;
    CanonicalConversionOperations from;
  };

  const BidirectionalCanonicalConversionOperations* canonical_conversion;
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
/// An DataType instance corresponding to a type `T` known at compile time may
/// be obtained by implicit conversion from `StaticDataType<T>`.
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

  constexpr DataType data_type() const { return *this; }

  constexpr DataTypeId id() const { return operations_->id; }

  constexpr absl::string_view name() const { return operations_->name; }

  constexpr std::ptrdiff_t size() const { return operations_->size; }

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
  ///
  /// These depend only on the `type` because there should only be a single
  /// `DataTypeOperations` object per type.  To handle possible multiple
  /// instances due to certain dynamic linking modes, however, we rely on the
  /// `operator==` defined for `std::type_info` rather than comparing the
  /// `operators_` pointers directly.
  friend constexpr bool operator==(DataType a, DataType b) {
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
  /// \invariant operations_ != nullptr
  const internal::DataTypeOperations* operations_;
};

namespace internal_data_type {

/// Returns the name of the data type corresponding to the C++ type `T`.
///
/// For all of the standard data types, a specialization is defined below.  This
/// definition is only used for custom data types.
template <typename T>
constexpr absl::string_view GetTypeName() {
  // While it would be nice to return a meaningful name, `typeid(T).name()` is
  // not constexpr (and includes mangling).
  return "unknown";
}

#define TENSORSTORE_INTERNAL_DO_DATA_TYPE_NAME(T, ...) \
  template <>                                          \
  constexpr absl::string_view GetTypeName<T>() {       \
    return absl::string_view(#T, sizeof(#T) - 3);      \
  }                                                    \
  /**/
TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_DATA_TYPE_NAME)
#undef TENSORSTORE_INTERNAL_DO_DATA_TYPE_NAME

/// Compares two strided arrays for equality.  Handles the case where `T`
/// supports equality comparison.
template <typename T>
std::enable_if_t<internal::IsEqualityComparable<T>::value, bool> CompareEqual(
    const T* a, const T* b) {
  return *a == *b;
}

/// Handles the case where `T` does not support equality comparison.
/// \returns false
template <typename T>
std::enable_if_t<!internal::IsEqualityComparable<T>::value, bool> CompareEqual(
    const T*, const T*) {
  return false;
}

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
    tensorstore::StrAppend(result, *static_cast<const T*>(ptr));
  }
};

/// Elementwise functions referenced by `DataTypeOperations`.
template <typename T>
struct DataTypeElementwiseOperationsImpl {
  struct InitializeImpl {
    void operator()(T* dest, Status*) const { *dest = T(); }
  };

  struct CopyAssignImpl {
    void operator()(const T* source, T* dest, Status*) const {
      *dest = *source;
    }
  };

  struct MoveAssignImpl {
    void operator()(T* source, T* dest, Status*) const {
      *dest = std::move(*source);
    }
  };

  struct CopyAssignUnmaskedImpl {
    void operator()(const T* source, T* dest, const bool* mask, Status*) const {
      if (!*mask) *dest = *source;
    }
  };

  struct CompareEqualImpl {
    bool operator()(const T* source, const T* dest, Status*) const {
      return internal_data_type::CompareEqual<T>(source, dest);
    }
  };

  using Initialize =
      internal::SimpleElementwiseFunction<InitializeImpl(T), Status*>;

  using CopyAssign =
      internal::SimpleElementwiseFunction<CopyAssignImpl(const T, T), Status*>;
  using MoveAssign =
      internal::SimpleElementwiseFunction<MoveAssignImpl(T, T), Status*>;

  using CopyAssignUnmasked = internal::SimpleElementwiseFunction<
      CopyAssignUnmaskedImpl(const T, T, const bool), Status*>;
  using CompareEqual =
      internal::SimpleElementwiseFunction<CompareEqualImpl(const T, const T),
                                          Status*>;
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
// in data_type.cc, in order to reduce compilation time and object file bloat.
TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_DATA_TYPE_EXPLICIT_INSTANTIATION,
                               extern)

}  // namespace internal_data_type

/// Empty/monostate type that represents a statically known element type.  In
/// generic code, this can be used in place of an `DataType` when the element
/// type is statically known.
template <typename T>
class StaticDataType {
 private:
  using Ops = internal::DataTypeOperations;
  using SimpleOps = internal_data_type::DataTypeSimpleOperationsImpl<T>;
  using ElementwiseOps =
      internal_data_type::DataTypeElementwiseOperationsImpl<T>;

 public:
  using Element = T;
  static_assert(std::is_same<T, std::decay_t<T>>::value,
                "T must be an unqualified type.");
  static_assert(IsElementType<T>::value, "T must satisfy IsElementType.");

  constexpr StaticDataType() = default;

  constexpr StaticDataType(unchecked_t, StaticDataType other) {}

  template <typename Other>
  constexpr StaticDataType(unchecked_t, StaticDataType<Other>) = delete;

  constexpr StaticDataType(unchecked_t, DataType other) {}

  static constexpr bool valid() { return true; }

  constexpr static StaticDataType data_type() { return {}; }

  constexpr static DataTypeId id() { return DataTypeIdOf<T>; }

  constexpr absl::string_view name() const {
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

template <typename T>
using DataTypeOf =
    StaticDataType<typename internal_data_type::CanonicalElementType<
        std::remove_cv_t<T>>::type>;

/// Returns `{ func(DataTypeOf<T>())... }` where `T` ranges over the canonical
/// data types.
template <typename Func>
constexpr std::array<std::invoke_result_t<Func, DataTypeOf<bool>>,
                     kNumDataTypeIds>
MapCanonicalDataTypes(Func func) {
  return {{
#define TENSORSTORE_INTERNAL_DO_DATA_TYPE(T, ...) func(DataTypeOf<T>()),
      TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_DATA_TYPE)
#undef TENSORSTORE_INTERNAL_DO_DATA_TYPE
  }};
}

/// Specifies the form of initialization to use when allocating an array.
enum class ElementInitialization {
  /// Specifies default initialization.  For primitive types, or class types for
  /// which the default constructor leaves some members uninitialized, this
  /// results in indeterminite values.
  default_init,

  /// Specifies value initialization.
  value_init
};
constexpr ElementInitialization default_init =
    ElementInitialization::default_init;
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
/// \remark For primitive types, default initialization leaves the elements
///     uninitialized.
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
void DestroyAndFree(std::ptrdiff_t n, DataType r, void* ptr);

/// Metafunction that maps a given element type `T` to `DataType` if `T` is
/// `void`, and to `DataTypeOf<T>` otherwise.
///
/// \tparam T The element type, const and/or volatile qualification is permitted
///     and ignored.
template <typename T>
using StaticOrDynamicDataTypeOf =
    std::conditional_t<std::is_void<T>::value, DataType, DataTypeOf<T>>;

/// Returns a shared_ptr that manages the memory returned by
/// `AllocateAndConstruct`.
///
/// \tparam T Optional.  The element type.  If unspecified (or equal to `void`),
///     the element type must be specified at run time using the `r` parameter.
/// \param n The number of elements to allocate.
/// \param initialization Optional.  The form of initialization to use.
/// \param r The element type.  Optional if `T` is not `void`.
template <typename T = void>
std::shared_ptr<T> AllocateAndConstructShared(
    std::ptrdiff_t n, ElementInitialization initialization = default_init,
    StaticOrDynamicDataTypeOf<T> r = DataTypeOf<T>()) {
  static_assert(std::is_same<std::remove_cv_t<T>, T>::value,
                "Element type T must not have cv qualifiers.");
  return std::static_pointer_cast<T>(
      AllocateAndConstructShared<void>(n, initialization, r));
}

template <>
std::shared_ptr<void> AllocateAndConstructShared<void>(
    std::ptrdiff_t n, ElementInitialization initialization, DataType r);

inline bool IsPossiblySameDataType(DataType a, DataType b) {
  return !b.valid() || !a.valid() || a == b;
}

template <typename T, typename U>
constexpr inline bool IsPossiblySameDataType(StaticDataType<T> a,
                                             StaticDataType<U> b) {
  return std::is_same<T, U>::value;
}

/// Evaluates to a type similar to `SourceRef` but with a static data type of
/// `TargetElement`.
///
/// The actual type is determined by the `RebindDataType` template alias defined
/// by the `StaticCastTraits` specialization for `SourceRef`, which must be of
/// the following form:
///
///     template <typename TargetElement>
///     using RebindDataType = ...;
///
/// Supported types include `ElementPointer`, `Array`, `TransformedArray`,
/// `TensorStore`, `Spec`.
///
/// \tparam SourceRef Optionally `const`- and/or reference-qualified source
///     type.  Any qualifiers are ignored.
/// \tparam TargetElement Target element type.
template <typename SourceRef, typename TargetElement>
using RebindDataType =
    typename CastTraitsType<SourceRef>::template RebindDataType<TargetElement>;

/// Casts `source` to have a static data type of `TargetElement`.
///
/// The source type must be supported by `RebindDataType` and define a nested
/// `Element` type, and both the source and target types must be supported by
/// `StaticCast`.
///
/// The semantics of the `Checking` parameter are the same as for `StaticCast`.
///
/// This cast cannot be used to cast away const qualification of the source
/// element type.  To do that, use `ConstDataTypeCast` instead.
///
/// Examples:
///
///     Array<void> array = ...;
///     Result<Array<int>> result = StaticDataTypeCast<int>(array);
///     Result<Array<const int>> result2 = StaticDataTypeCast<const int>(array);
///     Array<const int> unchecked_result =
///         StaticDataTypeCast<const int, unchecked>(array);
///
///     DataType d = ...;
///     Result<DataTypeOf<int>> d_static = StaticDataTypeCast<int>(d);
///
///     DataTypeOf<int> d_int;
///     DataType d_dynamic = StaticDataTypeCast<void>(d_int);
///
/// \tparam TargetElement Target element type.  Depending on the source type,
///     `const`-qualified `TargetElement` types may or may not be supported.
/// \tparam Checking Specifies whether the cast is checked or unchecked.
/// \param source Source value.
/// \requires `typename remove_cvref_t<SourceRef>::Element` is compatible with
///     `Target` according to `IsElementExplicitlyConvertible`.
template <typename TargetElement, CastChecking Checking = CastChecking::checked,
          typename SourceRef>
SupportedCastResultType<RebindDataType<SourceRef, TargetElement>, SourceRef,
                        Checking>
StaticDataTypeCast(SourceRef&& source) {
  using Source = internal::remove_cvref_t<SourceRef>;
  static_assert(IsElementTypeExplicitlyConvertible<typename Source::Element,
                                                   TargetElement>::value,
                "StaticDataTypeCast cannot cast away const qualification");
  return StaticCast<RebindDataType<SourceRef, TargetElement>, Checking>(
      std::forward<SourceRef>(source));
}

/// Casts `source` to a specified target element type which must differ from the
/// existing element type only in const qualification.
///
/// The source type must be supported by `RebindDataType` and define a nested
/// `Element` type, and both the source and target types must be supported by
/// `StaticCast`.
///
/// This cast is always unchecked.
///
/// Example:
///
///     Array<const int> const_array = ...;
///     Array<int> array = ConstDataTypeCast<int>(const_array);
///
/// \tparam TargetElement Target element type.
/// \schecks `TargetElement` and `typename remove_cvref_t<SourceRef>::Element`
///     differ only in their `const` qualification.
template <typename TargetElement, typename SourceRef>
inline SupportedCastResultType<RebindDataType<SourceRef, TargetElement>,
                               SourceRef>
ConstDataTypeCast(SourceRef&& source) {
  using Source = internal::remove_cvref_t<SourceRef>;
  static_assert(
      std::is_same<const typename Source::Element, const TargetElement>::value,
      "ConstDataTypeCast can only change const qualification");
  return StaticCast<RebindDataType<SourceRef, TargetElement>,
                    CastChecking::unchecked>(std::forward<SourceRef>(source));
}

/// `StaticCastTraits` specialization for `DataType`.
template <>
struct StaticCastTraits<DataType> : public DefaultStaticCastTraits<DataType> {
  static std::string Describe() { return Describe(DataType{}); }
  static std::string Describe(DataType data_type);
  static constexpr bool IsCompatible(DataType other) { return true; }
  template <typename TargetElement>
  using RebindDataType = StaticOrDynamicDataTypeOf<TargetElement>;
};

/// `StaticCastTraits` specialization for `StaticDataType<T>`.
template <typename T>
struct StaticCastTraits<StaticDataType<T>>
    : public DefaultStaticCastTraits<StaticDataType<T>> {
  static std::string Describe() {
    return StaticCastTraits<DataType>::Describe(DataTypeOf<T>());
  }
  static std::string Describe(StaticDataType<T>) { return Describe(); }

  template <typename Other>
  static constexpr bool IsCompatible(Other other) {
    return !other.valid() || other == StaticDataType<T>();
  }
  template <typename TargetElement>
  using RebindDataType = StaticOrDynamicDataTypeOf<TargetElement>;
};

/// Returns the `DataType` with `name` equal to `id`.
///
/// If `id` does not specify a supported data type name, returns the invalid
/// data type of `DataType()`.
///
/// Example:
///
///     EXPECT_EQ(DataTypeOf<std::int32_t>(), GetDataType("int32"));
///     EXPECT_EQ(DataTypeOf<float>(), GetDataType("float32"));
DataType GetDataType(absl::string_view id);

constexpr DataType kDataTypes[] = {
#define TENSORSTORE_INTERNAL_DO_DATA_TYPE(T, ...) DataTypeOf<T>(),
    TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_DATA_TYPE)
#undef TENSORSTORE_INTERNAL_DO_DATA_TYPE
};

}  // namespace tensorstore

#endif  //  TENSORSTORE_DATA_TYPE_H_
