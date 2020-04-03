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

#include "absl/meta/type_traits.h"
#include "tensorstore/index.h"

namespace tensorstore {
namespace internal {

/// Type alias that evaluates to `A` if `A` is not `void`, otherwise to `B` if
/// `B` is not `void`, otherwise results in a substitution failure.
template <typename A, typename B>
using FirstNonVoidType = typename absl::conditional_t<
    !std::is_void<A>::value, std::common_type<A>,
    std::enable_if<!std::is_void<B>::value, B>>::type;

/// Bool-valued metafunction that checks if `T == U` is convertible to `bool`.
template <typename T, typename U = T, typename = void>
struct IsEqualityComparable : public std::false_type {};

template <typename T, typename U>
struct IsEqualityComparable<
    T, U,
    absl::enable_if_t<std::is_convertible<
        decltype(std::declval<T>() == std::declval<U>()), bool>::value>>
    : public std::true_type {};

template <typename To, typename, typename... From>
struct IsPackConvertibleWithoutNarrowingHelper : public std::false_type {};

template <typename To, typename... From>
struct IsPackConvertibleWithoutNarrowingHelper<
    To,
    absl::void_t<decltype(std::initializer_list<To>{std::declval<From>()...})>,
    From...> : public std::true_type {};

/// `bool`-valued metafunction that evaluates to `true` if, and only if,
/// `Target` is constructible but not implicitly convertible from `Source`.
template <typename Source, typename Target>
struct IsOnlyExplicitlyConvertible
    : public std::integral_constant<
          bool, (std::is_constructible<Target, Source>::value &&
                 !std::is_convertible<Source, Target>::value)> {};

/// Bool-valued metafunction that evaluates to `true` if, and only if, all
/// `From...` are convertible to `To` without narrowing.
template <typename To, typename... From>
using IsPackConvertibleWithoutNarrowing =
    IsPackConvertibleWithoutNarrowingHelper<To, void, From...>;

/// Bool-valued metafunction that evaluates to `true` if, and only if, all
/// `IndexType...` are convertible to `Index` without narrowing.
template <typename... IndexType>
using IsIndexPack = IsPackConvertibleWithoutNarrowing<Index, IndexType...>;

/// Bool-valued metafunction that evaluates to `true` if, and only if, `ASource`
/// is implicitly convertible to `ADest` and `BSource` is implicitly convertible
/// to `BDest`.
template <typename ASource, typename BSource, typename ADest, typename BDest>
struct IsPairImplicitlyConvertible
    : public std::integral_constant<
          bool, std::is_convertible<ASource, ADest>::value &&
                    std::is_convertible<BSource, BDest>::value> {};

/// Bool-valued metafunction that evaluates to `true` if, and only if, `ASource`
/// is explicitly convertible to `ADest` and `BSource` is explicitly convertible
/// to `BDest`.
template <typename ASource, typename BSource, typename ADest, typename BDest>
struct IsPairExplicitlyConvertible
    : public std::integral_constant<
          bool, std::is_constructible<ADest, ASource>::value &&
                    std::is_constructible<BDest, BSource>::value> {};

/// Bool-valued metafunction equivalent to `IsPairExplicitlyConvertible<ASource,
/// BSource, ADest, BDest>::value && !IsPairImplicitlyConvertible<ASource,
/// BSource, ADest, BDest>::value`.
template <typename ASource, typename BSource, typename ADest, typename BDest>
struct IsPairOnlyExplicitlyConvertible
    : public std::integral_constant<
          bool,
          IsPairExplicitlyConvertible<ASource, BSource, ADest, BDest>::value &&
              !IsPairImplicitlyConvertible<ASource, BSource, ADest,
                                           BDest>::value> {};

/// Bool-valued metafunction equivalent to `std::is_assignable<ADest&,
/// ASource>::value && std::is_assignable<BDest&, BSource>::value`.
template <typename ASource, typename BSource, typename ADest, typename BDest>
struct IsPairAssignable
    : public std::integral_constant<
          bool, std::is_assignable<ADest&, ASource>::value &&
                    std::is_assignable<BDest&, BSource>::value> {};

/// Bool-valued metafunction equivalent to:
/// `std::is_convertible<From, To>::value || std::is_void<To>::value`.
template <typename From, typename To>
struct IsConvertibleOrVoid : public std::is_convertible<From, To> {};

template <typename From>
struct IsConvertibleOrVoid<From, void> : public std::true_type {};

/// Bool-valued metafunction that evaluates to `true` if there is a suitable
/// `operator<<` overload for writing a value of type `const T &` to an
/// `std::ostream`.
template <typename T, typename = void>
struct IsOstreamable : public std::false_type {};

template <typename T>
struct IsOstreamable<T, absl::void_t<decltype(std::declval<std::ostream&>()
                                              << std ::declval<const T&>())>>
    : public std::true_type {};

/// Type alias that removes both reference and const/volatile qualification.
///
/// Equivalent to C++20 `std::remove_cvref_t`.
template <typename T>
using remove_cvref_t = absl::remove_cv_t<absl::remove_reference_t<T>>;

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
struct IsConstConvertible
    : public std::integral_constant<bool,
                                    (std::is_same<Source, Dest>::value ||
                                     std::is_same<const Source, Dest>::value)> {
};

/// Bool-valued metafunction that evaluates to `true`, if, and only if`,
/// `IsConstConvertible<Source, Dest>::value || std::is_void<Dest>::value`.
template <typename Source, typename Dest>
struct IsConstConvertibleOrVoid
    : public std::integral_constant<bool,
                                    (std::is_same<Source, Dest>::value ||
                                     std::is_same<const Source, Dest>::value ||
                                     std::is_void<Dest>::value)> {};

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
  static_assert(std::is_empty<T>{}, "T must be an empty type.");
  static_assert(std::is_standard_layout<T>{}, "T must be standard layout.");
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
/// If `is_empty<T>::value`, `pointer` is ignored (and may be `nullptr`), and
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
    absl::conditional_t<std::is_empty<T>::value, EmptyObject<T>,
                        NonEmptyObjectGetter>;

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

}  // namespace internal
}  // namespace tensorstore

/// Defines a `bool`-valued metafunction `HasMethodNAME<R, T, Arg...>::value` in
/// the enclosing namespace that evaluates to `true` if a method named `NAME`
/// can be called with arguments `Arg&&...` on an object of type `T&&` and
/// returns a value convertible to `R` (if `R` is `void`, the return type is
/// unconstrained).
///
/// For example:
///
///     TENSORSTORE_INTERNAL_DEFINE_HAS_METHOD(Foo)
///
///     struct X {
///       int* Foo(int);
///       float* Foo(float);
///       int Foo(const char *) &;
///     };
///
///     static_assert(HasMethodFoo<int*, X, int>::value, "");
///     static_assert(HasMethodFoo<int, X&, const char*>::value, "");
///     static_assert(!HasMethodFoo<int, X, const char*>::value, "");
///     static_assert(!HasMethodFoo<int*, const X, int>::value, "");
///     static_assert(HasMethodFoo<void, X, int>::value, "");
///     static_assert(!HasMethodFoo<float*, X, int>::value, "");
///     static_assert(HasMethodFoo<float*, X, float>::value, "");
///     static_assert(HasMethodFoo<void, X, float>::value, "");
///     static_assert(!HasMethodFoo<void, X, const char*>::value, "");
#define TENSORSTORE_INTERNAL_DEFINE_HAS_METHOD(NAME)                          \
  template <typename, typename R, typename T, typename... Arg>                \
  struct HasMethod0##NAME : std::false_type {};                               \
  template <typename R, typename T, typename... Arg>                          \
  struct HasMethod0##NAME<                                                    \
      typename ::tensorstore::internal::IsConvertibleOrVoid<                  \
          decltype(std::declval<T>().NAME(std::declval<Arg>()...)), R>::type, \
      R, T, Arg...> : std::true_type {};                                      \
  template <typename R, typename T, typename... Arg>                          \
  using HasMethod##NAME = HasMethod0##NAME<std::true_type, R, T, Arg...>;     \
  /**/

/// Defines a `bool`-valued metafunction `HasAdlFunctionNAME<R, Arg...>::value`
/// in the enclosing namespace that evaluates to `true` if the unqualified
/// function call `NAME(std::declval<Arg>()...)` is valid and returns a value
/// convertible to `R` (if `R` is `void`, the return type is unconstrained).
///
/// For example:
///
///     TENSORSTORE_INTERNAL_DEFINE_HAS_ADL_FUNCTION(Foo)
///
///     struct X {
///       friend int* Foo(X, int);
///       friend float* Foo(X, float);
///     };
///
///     static_assert(HasAdlFunctionFoo<int*, X, int>::value, "");
///     static_assert(HasAdlFunctionFoo<void, X, int>::value, "");
///     static_assert(!HasAdlFunctionFoo<float *, X, int>::value, "");
///     static_assert(HasAdlFunctionFoo<float*, X, float>::value, "");
///     static_assert(HasAdlFunctionFoo<void, X, float>::value, "");
///     static_assert(!HasAdlFunctionFoo<void, X, const char*>::value, "");
#define TENSORSTORE_INTERNAL_DEFINE_HAS_ADL_FUNCTION(NAME)   \
  template <typename, typename R, typename... Arg>           \
  struct HasAdlFunction0##NAME : std::false_type {};         \
  template <typename R, typename... Arg>                     \
  struct HasAdlFunction0##NAME<                              \
      typename ::tensorstore::internal::IsConvertibleOrVoid< \
          decltype(NAME(std::declval<Arg>()...)), R>::type,  \
      R, Arg...> : std::true_type {};                        \
  template <typename R, typename... Arg>                     \
  using HasAdlFunction##NAME =                               \
      HasAdlFunction0##NAME<std::true_type, R, Arg...>;      \
  /**/

#endif  // TENSORSTORE_INTERNAL_TYPE_TRAITS_H_
