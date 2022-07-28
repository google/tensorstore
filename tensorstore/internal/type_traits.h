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

/// Type traits used in the implementation of TensorStore.

#ifndef TENSORSTORE_INTERNAL_TYPE_TRAITS_H_
#define TENSORSTORE_INTERNAL_TYPE_TRAITS_H_

#include <cstddef>
#include <initializer_list>
#include <iosfwd>
#include <type_traits>

#ifdef __has_builtin
#if __has_builtin(__type_pack_element)
#define TENSORSTORE_HAS_TYPE_PACK_ELEMENT
#endif
#endif

#ifndef TENSORSTORE_HAS_TYPE_PACK_ELEMENT
#include <tuple>
#endif

#include "tensorstore/index.h"

namespace tensorstore {
namespace internal {

/// Type alias that evaluates to `A` if `A` is not `void`, otherwise to `B` if
/// `B` is not `void`, otherwise results in a substitution failure.
template <typename A, typename B>
using FirstNonVoidType =
    typename std::conditional_t<!std::is_void_v<A>, std::common_type<A>,
                                std::enable_if<!std::is_void_v<B>, B>>::type;

/// Bool-valued metafunction that checks if `T == U` is convertible to `bool`.
template <typename T, typename U = T, typename = void>
constexpr inline bool IsEqualityComparable = false;

template <typename T, typename U>
constexpr inline bool IsEqualityComparable<
    T, U,
    std::enable_if_t<std::is_convertible_v<
        decltype(std::declval<T>() == std::declval<U>()), bool>>> = true;

template <typename To, typename, typename... From>
constexpr inline bool IsPackConvertibleWithoutNarrowingHelper = false;

template <typename To, typename... From>
constexpr inline bool IsPackConvertibleWithoutNarrowingHelper<
    To,
    std::void_t<decltype(std::initializer_list<To>{std::declval<From>()...})>,
    From...> = true;

/// `bool`-valued metafunction that evaluates to `true` if, and only if,
/// `Target` is constructible but not implicitly convertible from `Source`.
template <typename Source, typename Target>
constexpr inline bool IsOnlyExplicitlyConvertible =
    (std::is_constructible_v<Target, Source> &&
     !std::is_convertible_v<Source, Target>);

/// Bool-valued metafunction that evaluates to `true` if, and only if, all
/// `From...` are convertible to `To` without narrowing.
template <typename To, typename... From>
constexpr inline bool IsPackConvertibleWithoutNarrowing =
    IsPackConvertibleWithoutNarrowingHelper<To, void, From...>;

/// Bool-valued metafunction that evaluates to `true` if, and only if, all
/// `IndexType...` are convertible to `Index` without narrowing.
template <typename... IndexType>
constexpr inline bool IsIndexPack =
    IsPackConvertibleWithoutNarrowing<Index, IndexType...>;

/// Bool-valued metafunction that evaluates to `true` if, and only if, `ASource`
/// is implicitly convertible to `ADest` and `BSource` is implicitly convertible
/// to `BDest`.
template <typename ASource, typename BSource, typename ADest, typename BDest>
constexpr inline bool IsPairImplicitlyConvertible =
    std::is_convertible_v<ASource, ADest> &&
    std::is_convertible_v<BSource, BDest>;

/// Bool-valued metafunction that evaluates to `true` if, and only if, `ASource`
/// is explicitly convertible to `ADest` and `BSource` is explicitly convertible
/// to `BDest`.
template <typename ASource, typename BSource, typename ADest, typename BDest>
constexpr inline bool IsPairExplicitlyConvertible =
    std::is_constructible_v<ADest, ASource> &&
    std::is_constructible_v<BDest, BSource>;

/// Bool-valued metafunction equivalent to
/// `IsPairExplicitlyConvertible<ASource, BSource, ADest, BDest> &&
/// !IsPairImplicitlyConvertible<ASource, BSource, ADest, BDest>`.
template <typename ASource, typename BSource, typename ADest, typename BDest>
constexpr inline bool IsPairOnlyExplicitlyConvertible =
    IsPairExplicitlyConvertible<ASource, BSource, ADest, BDest> &&
    !IsPairImplicitlyConvertible<ASource, BSource, ADest, BDest>;

/// Bool-valued metafunction equivalent to
/// `std::is_assignable_v<ADest&, ASource> && std::is_assignable_v<BDest&,
/// BSource>`.
template <typename ASource, typename BSource, typename ADest, typename BDest>
constexpr inline bool IsPairAssignable =
    std::is_assignable_v<ADest&, ASource> &&
    std::is_assignable_v<BDest&, BSource>;

/// Bool-valued metafunction equivalent to:
/// `std::is_convertible_v<From, To> || std::is_void_v<To>`.
template <typename From, typename To>
constexpr inline bool IsConvertibleOrVoid = std::is_convertible_v<From, To>;

template <typename From>
constexpr inline bool IsConvertibleOrVoid<From, void> = true;

/// Bool-valued metafunction that evaluates to `true` if there is a suitable
/// `operator<<` overload for writing a value of type `const T &` to an
/// `std::ostream`.
template <typename T, typename = void>
constexpr inline bool IsOstreamable = false;

template <typename T>
constexpr inline bool
    IsOstreamable<T, std::void_t<decltype(std::declval<std::ostream&>()
                                          << std ::declval<const T&>())>> =
        true;

/// Type alias that removes both reference and const/volatile qualification.
///
/// Equivalent to C++20 `std::remove_cvref_t`.
template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename Qualified, typename T>
struct CopyQualifiersHelper {
  using type = T;
};

template <typename Qualified, typename T>
struct CopyQualifiersHelper<const Qualified, T> {
  using type = const typename CopyQualifiersHelper<Qualified, T>::type;
};

template <typename Qualified, typename T>
struct CopyQualifiersHelper<volatile Qualified, T> {
  using type = volatile typename CopyQualifiersHelper<Qualified, T>::type;
};

template <typename Qualified, typename T>
struct CopyQualifiersHelper<const volatile Qualified, T> {
  using type = const volatile typename CopyQualifiersHelper<Qualified, T>::type;
};

template <typename Qualified, typename T>
struct CopyQualifiersHelper<Qualified&, T> {
  using type = typename CopyQualifiersHelper<Qualified, T>::type&;
};

template <typename T, typename Qualified>
struct CopyQualifiersHelper<Qualified&&, T> {
  using type = typename CopyQualifiersHelper<Qualified, T>::type&&;
};

/// Returns `T` modified to have the same const, volatile, and reference
/// qualifications as `Qualified`.
///
/// The existing const/volatile and reference qualifications of `T` are ignored.
///
/// For example:
///
///     CopyQualifiers<const float&&, volatile int&> -> const int&&
template <typename Qualified, typename T>
using CopyQualifiers =
    typename CopyQualifiersHelper<Qualified, remove_cvref_t<T>>::type;

/// Converts an rvalue reference to an lvalue reference.  Among other things,
/// this permits taking the address of a temporary object, which can be useful
/// for specifying default argument values.
template <typename T>
inline T& GetLValue(T&& x) {
  return x;
}

/// Type alias that evaluates to its first argument.  Additional type arguments,
/// that do not affect the result, can also be supplied, which can be useful
/// with parameter pack expansion and for SFINAE.
template <typename T, typename... U>
using FirstType = T;

/// Bool-valued metafunction that evaluates to `true` if, and only if, `Source`
/// is the same as `Dest` or `const Source` is the same as `Dest`.
template <typename Source, typename Dest>
constexpr inline bool IsConstConvertible =
    (std::is_same_v<Source, Dest> || std::is_same_v<const Source, Dest>);

/// Bool-valued metafunction that evaluates to `true`, if, and only if`,
/// `IsConstConvertible<Source, Dest> || std::is_void_v<Dest>`.
template <typename Source, typename Dest>
constexpr inline bool IsConstConvertibleOrVoid =
    (std::is_same_v<Source, Dest> || std::is_same_v<const Source, Dest> ||
     std::is_void_v<Dest>);

#ifdef TENSORSTORE_HAS_TYPE_PACK_ELEMENT
template <std::size_t I, typename... Ts>
using TypePackElement = __type_pack_element<I, Ts...>;
#else
template <std::size_t I, typename... Ts>
using TypePackElement = typename std::tuple_element<I, std::tuple<Ts...>>::type;
#endif

/// This class can be used to obtain an instance of an empty object (such as a
/// lambda without any captures) even if it is not default constructible or
/// movable.
///
/// This uses the trick by Louis Dionne described here:
///
/// https://github.com/ldionne/dyno/blob/03eaeded898225660787f03655edb89642a72e7c/include/dyno/detail/empty_object.hpp#L13
///
/// to achieve this without relying on undefined behavior.
template <typename T>
class EmptyObject {
  static_assert(std::is_empty_v<T>, "T must be an empty type.");
  static_assert(std::is_standard_layout_v<T>, "T must be standard layout.");
  // Define two structs, T1 and T2, that have a single member `c` as a common
  // initial sequence of non-static data members and bit-fields, and are
  // layout-compatible because this common initial sequence consists of every
  // non-static data member and bit field.
  //
  //     Two standard-layout struct (Clause [class]) types are layout-compatible
  //     if they have the same number of non-static data members and
  //     corresponding non-static data members (in declaration order) have
  //     layout-compatible types ([basic.types]).
  //
  //     [class.mem] https://timsong-cpp.github.io/cppwp/n3337/class#mem-17
  struct T1 {
    char c;
  };
  struct T2 : T {
    char c;
  };
  // This union allows us to initialize the `T1` member, then because `T1` and
  // `T2` are layout compatible, legally access the `T2` member and then cast it
  // to its base class `T`.
  //
  //     If a standard-layout union contains two or more standard-layout structs
  //     that share a common initial sequence, and if the standard-layout union
  //     object currently contains one of these standard-layout structs, it is
  //     permitted to inspect the common initial part of any of them. Two
  //     standard-layout structs share a common initial sequence if
  //     corresponding members have layout-compatible types and either neither
  //     member is a bit-field or both are bit-fields with the same width for a
  //     sequence of one or more initial members.
  //
  //     [class.mem] https://timsong-cpp.github.io/cppwp/n3337/class#mem-19
  //
  //
  //     A pointer to a standard-layout struct object, suitably converted using
  //     a reinterpret_cast, points to its initial member (or if that member is
  //     a bit-field, then to the unit in which it resides) and vice versa. [
  //     Note: There might therefore be unnamed padding within a standard-layout
  //     struct object, but not at its beginning, as necessary to achieve
  //     appropriate alignment.  — end note ]
  //
  //     https://timsong-cpp.github.io/cppwp/n3337/class#mem-20
  union Storage {
    constexpr Storage() : t1{} {}
    T1 t1;
    T2 t2;
  };
  Storage storage{};

 public:
  /// Returns a reference to an object of type T.
  ///
  /// The argument is for use with PossiblyEmptyObject and is ignored.
  T& get(T* = nullptr) {
    char* c = &storage.t2.c;
    T2* t2 = reinterpret_cast<T2*>(c);
    return *static_cast<T*>(t2);
  }
};

/// Class for use by PossiblyEmptyObjectGetter that handles the case of a
/// non-empty type.
class NonEmptyObjectGetter {
 public:
  template <typename T>
  static T& get(T* pointer) {
    return *pointer;
  }
};

/// Alias that evaluates to a default constructible class type that defines a
/// member:
///
///     T& get(T* pointer);
///
/// If `is_empty_v<T>`, `pointer` is ignored (and may be `nullptr`), and
/// `get` returns a reference to a valid instance of `T` (with the same lifetime
/// as the object upon which `get` was invoked).
///
/// Otherwise, `pointer` must not be `nullptr` and `get` returns `*pointer`.
///
/// This is useful when an interface requires the user to supply a pointer to a
/// possibly-empty type (e.g. a function object), and wishes to permit `nullptr`
/// to be specified if the type is in fact empty.
template <typename T>
using PossiblyEmptyObjectGetter =
    std::conditional_t<std::is_empty_v<T>, EmptyObject<T>,
                       NonEmptyObjectGetter>;

template <typename T>
struct DefaultConstructibleFunction {
  constexpr DefaultConstructibleFunction() = default;
  constexpr DefaultConstructibleFunction(const T&) {}
  template <typename... Arg>
  constexpr std::invoke_result_t<T&, Arg...> operator()(Arg&&... arg) const {
    EmptyObject<T> obj;
    return obj.get()(static_cast<Arg&&>(arg)...);
  }
};

template <typename T>
using DefaultConstructibleFunctionIfEmpty =
    std::conditional_t<(std::is_empty_v<T> &&
                        !std::is_default_constructible_v<T>),
                       DefaultConstructibleFunction<T>, T>;

/// Identity metafunction for types, as added in C++20.
///
/// See https://en.cppreference.com/w/cpp/types/type_identity
///
/// This is useful for preventing deduction of a function parameter.
template <typename T>
struct type_identity {
  using type = T;
};

template <typename T>
using type_identity_t = typename type_identity<T>::type;

/// Implementation of C++20 `std::identity`.
struct identity {
  template <typename T>
  constexpr T&& operator()(T&& t) const noexcept {
    return static_cast<T&&>(t);
  }
};

template <typename Base, typename Derived>
Base* BaseCast(Derived* derived) {
  return derived;
}

template <typename Base, typename Derived>
const Base* BaseCast(const Derived* derived) {
  return derived;
}

template <typename Base, typename Derived>
Base& BaseCast(Derived& derived) {
  return derived;
}

template <typename Base, typename Derived>
const Base& BaseCast(const Derived& derived) {
  return derived;
}

/// Hides a given type in the documentation.
template <typename T>
using Undocumented = T;

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_TYPE_TRAITS_H_
