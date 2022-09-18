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

#ifndef TENSORSTORE_INTERNAL_POLY_POLY_IMPL_H_
#define TENSORSTORE_INTERNAL_POLY_POLY_IMPL_H_

/// \file
/// Implementation details for poly.h
///
/// The `Storage` class defined below contains a `VTableBase*` pointer to a
/// static constant vtable object, and a buffer that either stores the contained
/// object directly or stores a pointer to heap-allocated memory that stores the
/// contained object.
///
/// The vtable of type `VTableType<Signature...>`, which inherits from
/// `VTableBase`, stores:
///
///   1. A `std::type_info` reference for the contained object, the
///      `inline_size` required to store the object inline;
///
///   2. Pointers to `destroy`, `move_destroy`, and `copy` functions used to
///      manage the lifetime of the contained object;
///
///   3. One function pointer corresponding to each `Signature` specified for
///      the `Poly` object.
///
/// `Poly` contains a `Storage` object that and for each `Signature` recursively
/// inherits from an instance the `PolyImpl` class template, an empty type that
/// defines an `operator()` that forwards to the corresponding function pointer
/// stored in the vtable.
///
/// For a type `Self` from which a `Poly<InlineSize, Copyable, Signature...>` is
/// constructed, there is a static constant vtable
///
///     `VTableInstance<
///          GetInlineSize<Self>() <= InlineSize,
///          Copyable,
///          Signature...
///      >::vtable`.
///
/// This is similar to the way that C++ compilers generate a single vtable for
/// each class that defines at least one virtual method.  Because the actual
/// vtables are static constants, there is no need to manage the lifetime of
/// vtables at run-time.  One downside to using static constant vtables is that
/// conversion from a `Poly<InlineSize, Copyable, S1...>` to another
/// `Poly<InlineSize, Copyable, S2...>`, where `S2...` is an arbitrary subset of
/// `S1...`, is not supported without double wrapping, even though it could be
/// done by constructing a vtable at run time.

// IWYU pragma: private, include "third_party/tensorstore/internal/poly/poly.h"

#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "tensorstore/internal/poly/storage.h"

namespace tensorstore {
namespace internal_poly {

/// Type alias that removes both reference and const/volatile qualification.
///
/// Equivalent to C++20 `std::remove_cvref_t`.
template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

/// Alias that evaluates to the return type of an unqualified `PolyApply` call
/// (found via ADL) with the specified arguments.
template <typename... Arg>
using PolyApplyResult = decltype(PolyApply(std::declval<Arg>()...));

template <typename, typename... Arg>
struct HasPolyApplyHelper : public std::false_type {};

template <typename... Arg>
struct HasPolyApplyHelper<std::void_t<PolyApplyResult<Arg...>>, Arg...>
    : public std::true_type {};

/// `bool`-valued metafunction that evaluates to `true` if an unqualified call
/// to `PolyApply` (found via ADL) with the specified arguments is valid.
template <typename... Arg>
constexpr inline bool HasPolyApply = HasPolyApplyHelper<void, Arg...>::value;

/// Forwards to unqualified `PolyApply` (found via ADL).
///
/// \requires A matching overload of `PolyApply` exists..
template <typename Self, typename... Arg>
std::enable_if_t<HasPolyApply<Self&&, Arg&&...>,
                 PolyApplyResult<Self&&, Arg&&...>>
CallPolyApply(Self&& self, Arg&&... arg) {
  return PolyApply(std::forward<Self>(self), std::forward<Arg>(arg)...);
}

/// Forwards `arg...` to `self`.
/// \requires A matching overload of `operator()` exists.
/// \requires A matching overload of `PolyApply` does not exist.
template <typename Self, typename... Arg>
std::enable_if_t<!HasPolyApply<Self&&, Arg&&...>,
                 std::invoke_result_t<Self, Arg...>>
CallPolyApply(Self&& self, Arg&&... arg) {
  return std::forward<Self>(self)(std::forward<Arg>(arg)...);
}

/// Bool-valued metafunction equivalent to:
/// `std::is_convertible_v<From, To> || std::is_void_v<To>`.
template <typename From, typename To>
constexpr inline bool IsConvertibleOrVoid = std::is_convertible_v<From, To>;

template <typename From>
constexpr inline bool IsConvertibleOrVoid<From, void> = true;

/// Alias that evaluates to the return type of `CallPolyApply` when invoked with
/// `Arg...`.
template <typename... Arg>
using CallPolyApplyResult =
    decltype(internal_poly::CallPolyApply(std::declval<Arg>()...));

template <typename, typename Self, typename R, typename... Arg>
struct IsCallPolyApplyResultConvertibleHelper : public std::false_type {};

template <typename Self, typename R, typename... Arg>
struct IsCallPolyApplyResultConvertibleHelper<
    std::void_t<CallPolyApplyResult<Self, Arg...>>, Self, R, Arg...>
    : std::integral_constant<
          bool, IsConvertibleOrVoid<CallPolyApplyResult<Self, Arg...>, R>> {};

/// `bool`-valued metafunction that evaluates to `true` if
/// `CallPolyApplyResult<Self, Arg...>` is valid and convertible to `R` (if `R`
/// is `void`, all types are considered convertible).
template <typename Self, typename R, typename... Arg>
using IsCallPolyApplyResultConvertible =
    IsCallPolyApplyResultConvertibleHelper<void, Self, R, Arg...>;

/// CRTP empty base class of `Derived` (which must be an instance of `Poly`)
/// that defines an `operator()` overload corresponding to each `Signature` that
/// simply forwards to the corresponding function pointer in the vtable.
///
/// For C++11 compatibility, this has to be implemented via recursive
/// inheritance, as a `using X::operator()` declaration can't be used with a
/// pack expansion until C++17.
template <typename Derived, typename... Signature>
class PolyImpl;

/// Base case for the recursive inheritance.
template <typename Derived>
class PolyImpl<Derived> {
 public:
  /// Dummy `operator()` definition to allow the using declaration in PolyImpl
  /// specializations defined below to work.
  template <int&>
  void operator()();
};

template <typename Signature>
struct SignatureTraits;

/// Stores a function pointer in the vtable corresponding to an `R (Arg...)`
/// signature, and inherits from `Base`, which is either `VTableBase` or another
/// `VTableImpl` instance.
///
/// This is used as an implementation detail of the `VTableType` alias defined
/// below.
template <typename Base, typename R, typename... Arg>
struct VTableImpl : public Base {
  using Entry = R (*)(const void*, Arg...);
  template <typename... Other>
  constexpr VTableImpl(Entry entry, Other... other)
      : Base(other...), entry(entry) {}
  Entry entry;
};

/// Recursive class template used to implement the `VTableType` alias defined
/// below.
///
/// \tparam Signature An unqualified function call signature of the form
///     `R (Arg...)`.
template <typename... Signature>
struct VTableTypeHelper;

template <>
struct VTableTypeHelper<> {
  using type = internal_poly_storage::VTableBase;
};

template <typename R, typename... Arg, typename... Signature>
struct VTableTypeHelper<R(Arg...), Signature...> {
  using type =
      VTableImpl<typename VTableTypeHelper<Signature...>::type, R, Arg...>;
};

/// Alias that evaluates to the derived VTable type (that inherits from
/// `VTableBase`) for the specified sequence of call signatures.
///
/// It is guaranteed that `VTableType<S1..., S2...>` inherits from
/// `VTableType<S2...>`, which permits conversion without double-wrapping from a
/// `Poly` type with signatures `S1..., S2...` to a `Poly` type with signatures
/// `S2...`.
///
/// \tparam Signature A function call signature of the form
///     `R (Arg...) QUALIFIERS`.  The `QUALIFIERS` are ignored.
template <typename... Signature>
using VTableType = typename VTableTypeHelper<
    typename SignatureTraits<Signature>::Unqualified...>::type;

/// Forwards to `CallPolyApply` on an object of type `Self` obtained by calling
/// `Getter::Get` on `const_cast<void*>(obj)`.
///
/// The vtable entry for each call signature points to an instance of this
/// function template.
template <typename Ops, typename Self, typename R, typename... Arg>
R CallImpl(const void* obj, Arg... arg) {
  return static_cast<R>(internal_poly::CallPolyApply(
      static_cast<Self>(Ops::Get(const_cast<void*>(obj))),
      static_cast<Arg&&>(arg)...));
}

/// Defines specializations of `SignatureTraits` for the
/// specified `DECLARED_QUALIFIERS`, which are treated as equivalent to the
/// specified `EFFECTIVE_QUALIFIERS`.
///
/// This has to be done with a macro to avoid repetition of the body for each
/// set of qualifiers, due to limitations of the C++ template mechanism.
///
/// \param DECLARED_QUALIFIERS An optional `const` qualifier followed by an
///     optional `&`/`&&` qualifier.
/// \param EFFECTIVE_QUALIFIERS An optional `const` qualifier followed by a
///     required `&` or `&&` qualifier.
#define TENSORSTORE_INTERNAL_POLY_SIG_TRAITS(DECLARED_QUALIFIERS,      \
                                             EFFECTIVE_QUALIFIERS)     \
  template <typename R, typename... Arg>                               \
  struct SignatureTraits<R(Arg...) DECLARED_QUALIFIERS> {              \
    using Unqualified = R(Arg...);                                     \
    using VTableEntry = R (*)(const void*, Arg...);                    \
    template <typename Ops, typename Self>                             \
    constexpr static VTableEntry GetVTableEntry() {                    \
      return &CallImpl<Ops, Self EFFECTIVE_QUALIFIERS, R, Arg...>;     \
    }                                                                  \
    template <typename Self>                                           \
    using IsSupportedBy =                                              \
        IsCallPolyApplyResultConvertible<Self EFFECTIVE_QUALIFIERS, R, \
                                         Arg...>;                      \
  }

TENSORSTORE_INTERNAL_POLY_SIG_TRAITS(, &);
TENSORSTORE_INTERNAL_POLY_SIG_TRAITS(const, const&);
TENSORSTORE_INTERNAL_POLY_SIG_TRAITS(&, &);
TENSORSTORE_INTERNAL_POLY_SIG_TRAITS(&&, &&);
TENSORSTORE_INTERNAL_POLY_SIG_TRAITS(const&, const&);

#undef TENSORSTORE_INTERNAL_POLY_SIG_TRAITS

/// Defines specializations of `PolyImpl` for the specified `QUALIFIERS`.
///
/// This has to be done with a macro to avoid repetition of the body for each
/// set of qualifiers, due to limitations of the C++ template mechanism.
/// NOTE: Uses __VA_ARGS__ to avoid MSVC warning C4003
///
/// \param ... An optional C++ qualifier, one of `const, const &, &, &&`.
#define TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL(...)                            \
  template <typename Derived, typename R, typename... Arg,                    \
            typename... Signature>                                            \
  class PolyImpl<Derived, R(Arg...) __VA_ARGS__, Signature...>                \
      : public PolyImpl<Derived, Signature...> {                              \
   public:                                                                    \
    using PolyImpl<Derived, Signature...>::operator();                        \
    R operator()(Arg... arg) __VA_ARGS__ {                                    \
      using VTable = VTableType<R(Arg...), Signature...>;                     \
      auto& impl = static_cast<const Derived&>(*this).storage_;               \
      return static_cast<const VTable*>(                                      \
                 static_cast<const typename Derived::VTable*>(impl.vtable())) \
          ->entry(impl.storage(), static_cast<Arg&&>(arg)...);                \
    }                                                                         \
  }

TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL();
TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL(&);
TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL(&&);
TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL(const);
TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL(const&);

#undef TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL

/// Defines a static constant VTable representing an type `Self` contained in a
/// `Poly<InlineSize, Copyable, Signature...>`, where
/// `Inline == GetInlineSize<Self>() <= InlineSize`.
///
/// \tparam Self Contained object type.
/// \tparam Copyable Specifies whether the container is copyable.  If `true`, a
///     pointer to a copy operation is stored.  If `false`, a `nullptr` is
///     stored in the `copy` slot.
/// \tparam Inline Specifies whether the contained object is stored inline in
///     the container (no heap allocation).
/// \tparam Signature Supported signature.
template <typename Ops, bool Copyable, typename... Signature>
struct VTableInstance {
  using Self = typename Ops::Type;
  constexpr static VTableType<Signature...> vtable{
      SignatureTraits<Signature>::template GetVTableEntry<Ops, Self>()...,
      internal_poly_storage::GetVTableBase<Ops, Copyable>(),
  };
};

template <typename Ops, bool Copyable, typename... Signature>
constexpr VTableType<Signature...>
    VTableInstance<Ops, Copyable, Signature...>::vtable;

}  // namespace internal_poly
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_POLY_POLY_IMPL_H_
