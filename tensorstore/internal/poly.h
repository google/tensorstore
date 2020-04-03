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

/// Defines Poly, a generalization of `std::function` that supports multiple
/// overloaded call signatures and inline object storage.
///
/// By using tag types to distinguish between signatures, Poly can also be used
/// as a general single-dispatch runtime polymorphism/type erasure facility.
///
/// For example:
///
///     struct GetWidth {};
///     struct GetHeight {};
///     struct Scale {};
///     using PolyRectangle = Poly<sizeof(double), /*Copyable=*/true,
///                                double(GetWidth) const,
///                                double(GetHeight) const,
///                                void(Scale, double scalar)>;
///
///     struct Rectangle {
///       double width;
///       double height;
///       double operator()(GetWidth) const { return width; }
///       double operator()(GetHeight) const { return height; }
///       void operator()(Scale, double scalar) {
///         width *= scalar;
///         height *= scalar;
///       }
///     };
///     struct Square {
///       double size;
///       double operator()(GetWidth) const { return size; }
///       double operator()(GetHeight) const { return size; }
///     };
///     // Define Scale operation on Square non-intrusively via PolyApply.
///     void PolyApply(Square &self, Scale, double scalar) {
///       self.size *= scalar;
///     }
///
///     // No heap allocation because `sizeof(Square) <= sizeof(double)`.
///     PolyRectangle square = Square{5};
///     EXPECT_EQ(5, square(GetWidth{}));
///     EXPECT_EQ(5, square(GetHeight{}));
///     square(Scale{}, 2);
///     EXPECT_EQ(10, square(GetWidth{}));
///     EXPECT_EQ(10, square(GetHeight{}));
///
///     // Heap-allocated because `sizeof(Rectangle) > sizeof(double)`.
///     PolyRectangle rect = Rectangle{6, 7};
///     EXPECT_EQ(6, rect(GetWidth{}));
///     EXPECT_EQ(7, rect(GetHeight{}));
///     rect(Scale{}, 2);
///     EXPECT_EQ(12, rect(GetWidth{}));
///     EXPECT_EQ(14, rect(GetHeight{}));
///
/// A key reason to use Poly rather than the builtin dynamic dispatch mechanism
/// in C++ based on virtual methods is that Poly has value semantics and avoids
/// heap allocation.
///
/// Similar to `std::function` in C++14 and newer, the `Poly` type erasing
/// constructor does not participate in overload resolution unless the provided
/// type satisfies `SupportsPolySignatures`.  For example:
///
///     std::string Foo(Poly<0, true, std::string ()> poly) {
///       return "Text: " + poly();
///     }
///     int Foo(Poly<0, true, int ()> poly) {
///       return 3 + poly();
///     }
///     EXPECT_EQ(6, Foo([] { return 3; }));
///     EXPECT_EQ("Text: Message", Foo([] { return "Message"; }));
///
///
/// This is most directly inspired by: https://github.com/potswa/cxx_function
///
/// Other related libraries include:
///   Boost.TypeErasure:
///   https://www.boost.org/doc/libs/1_66_0/doc/html/boost_typeerasure.html
///   Dyno: https://github.com/ldionne/dyno
///   Adobe Poly: http://stlab.adobe.com/group__poly__related.html
///   Folly Poly:
///   https://github.com/facebook/folly/blob/master/folly/docs/Poly.md

#ifndef TENSORSTORE_INTERNAL_POLY_H_
#define TENSORSTORE_INTERNAL_POLY_H_

#include <cstddef>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "tensorstore/internal/poly_impl.h"
#include "tensorstore/internal/type_traits.h"

namespace tensorstore {
namespace internal {

/// Evaluates to `true` if `T` is compatible with the specified function call or
/// PolyApply signatures.
///
/// A type `T` is compatible with a given `Signature`, which must be of the form
/// `R (Arg...) QUALIFIERS`, where `QUALIFIERS` includes an optional `const`
/// qualifier followed by an optional `&`/`&&` qualifier, if
///
///     `std::declval<T QUALIFIERS>()(std::declval<Arg>()...)`
///
/// is well-formed and has a return value convertible to `R`, or:
///
///     `PolyApply(std::declval<T QUALIFIERS>(), std::declval<Arg>()...)`
///
/// is well-formed and has a return value convertible to `R`.
///
/// If `R` is `void`, there are no constraints on the return type.
///
/// The behavior of `Poly` for a given type `T` and `Signature` can be
/// overridden by defining and overload of `PolyApply` that is found by argument
/// dependent lookup (ADL).
///
/// \remark `volatile` is not supported since it is not likely to be useful.
template <typename T, typename... Signature>
using SupportsPolySignatures =
    std::conjunction<typename internal_poly::SignatureTraits<
        Signature>::template IsSupportedBy<T>...>;

template <std::size_t InlineSize, bool Copyable, typename... Signature>
class Poly;

/// `bool`-valued metafunction that evaluates to `true` if `T` is an instance of
/// `Poly`.
template <typename T>
struct IsPoly : public std::false_type {};

template <std::size_t InlineSize, bool Copyable, typename... Signature>
struct IsPoly<Poly<InlineSize, Copyable, Signature...>>
    : public std::true_type {};

template <typename T, bool Copyable, typename... Signature>
struct IsCompatibleWithPoly : public SupportsPolySignatures<T, Signature...> {};

template <typename T, typename... Signature>
struct IsCompatibleWithPoly<T, true, Signature...>
    : public std::integral_constant<
          bool, (std::is_copy_constructible<T>::value &&
                 SupportsPolySignatures<T, Signature...>::value)> {};

/// Type-erased container of an overloaded function object.
///
/// \tparam InlineSize If the contained object type `T` has `alignof(T) <=
///     alignof(void*)`, `sizeof(T) <= max(sizeof(void*), InlineSize)`, and
///     satisfies `std::is_nothrow_move_constructible<T>`, then heap allocation
///     is avoided.
/// \tparam Signature Function signature of the form:
///         `R (Arg...)`
///         `R (Arg...) const`
///         `R (Arg...) &`,
///         `R (Arg...) const &`,
///         `R (Arg...) &&`.
///
///     For each `Signature`, a corresponding `operator()` overload is defined.
///     The reference-unqualified `R (Arg...)` and R (Arg...) const` overloads
///     always forward to the lvalue reference overload of the contained object.
///
///     If the contained object does not provide an `operator()` overload but a
///     `PolyApply` free function can be found by ADL, it is used instead.
/// \remark All `Signature` types must include an lvalue or rvalue reference
///     qualification.  To obtain the equivalent behavior of a signature without
///     reference qualification, specify both an lvalue and rvalue-qualified
///     signature.
template <std::size_t InlineSize, bool Copyable, typename... Signature>
class Poly
    : private internal_poly::PolyImpl<Poly<InlineSize, Copyable, Signature...>,
                                      Signature...> {
  template <typename, typename...>
  friend class internal_poly::PolyImpl;

  template <std::size_t, bool, typename...>
  friend class Poly;

  using Storage = internal_poly::Storage<InlineSize, Copyable>;
  using Base = internal_poly::PolyImpl<Poly, Signature...>;
  using VTable = internal_poly::VTableType<Signature...>;

  template <typename Self>
  using VTInstance = internal_poly::VTableInstance<
      Self, Copyable, Storage::template UsesInline<Self>(), Signature...>;

  template <typename... S>
  using HasConvertibleVTable =
      std::is_convertible<internal_poly::VTableType<S...>, VTable>;

 public:
  /// `bool`-valued metafunction that evaluates to `true` if this Poly type can
  /// contain an object of type `T`.
  template <typename T>
  using IsCompatible =
      // Prevent instantiation of IsCompatibleWithPoly with `T = Poly`, as that
      // may lead to a "SFINAE loop", namely `std::is_copy_constructible` being
      // instantiated before `Poly` is complete.
      std::disjunction<std::is_same<Poly, T>,
                       IsCompatibleWithPoly<T, Copyable, Signature...>>;

  /// `bool`-valued metafunction that evaluates to `true` if
  /// `IsCompatible<remove_cvref_t<T>>` and
  /// `std::is_constructible<remove_cvref_t<T>, T&&>` are both `true`.
  template <typename T>
  using IsCompatibleAndConstructible =
      // Prevent instantiation of `IsCompatibleWithPoly` and `is_constructible`
      // with `T = Poly`, as that may lead to a "SFINAE loop", namely
      // `std::is_copy_constructible` being instantiated before `Poly` is
      // complete.
      std::disjunction<
          std::is_same<Poly, remove_cvref_t<T>>,
          std::conjunction<
              IsCompatibleWithPoly<remove_cvref_t<T>, Copyable, Signature...>,
              std::is_constructible<remove_cvref_t<T>, T&&>>>;

  /// Constructs in a null state.
  ///
  /// \post `bool(*this) == false`
  Poly() = default;

  /// Constructs in a null state.
  ///
  /// \post `bool(*this) == false`
  Poly(std::nullptr_t) noexcept {}

  /// Constructs an object of type `remove_cvref_t<T>` from `obj`.
  ///
  /// \requires `IsCompatible<remove_cvref_t<T>>`.
  /// \requires `remove_cvref_t<T>` is constructible from `T&&`.
  /// \post `bool(*this) == true`
  template <typename T,
            std::enable_if_t<IsCompatibleAndConstructible<T>::value>* = nullptr>
  Poly(T&& obj)
      : Poly(std::in_place_type_t<remove_cvref_t<T>>{}, std::forward<T>(obj)) {}

  /// Constructs an object of type `T` from `arg...`.
  ///
  /// When `T` is a Poly instance, double wrapping is avoided when possible.
  ///
  /// \requires `IsCompatible<T>`.
  /// \post `bool(*this) == true`
  template <
      typename T, typename... U,
      std::enable_if_t<(IsCompatible<T>::value &&
                        std::is_constructible<T, U&&...>::value)>* = nullptr>
  Poly(std::in_place_type_t<T> in_place, U&&... arg) {
    Construct(in_place, std::forward<U>(arg)...);
  }

  Poly(const Poly&) = default;
  Poly(Poly&&) = default;
  Poly& operator=(const Poly&) = default;
  Poly& operator=(Poly&&) noexcept = default;

  /// Constructs an object of type `remove_cvref_t<T>` from `obj`.
  ///
  /// If this is in a non-null state, the existing object is destroyed first.
  ///
  /// \post `bool(*this) == true`.
  template <typename T,
            std::enable_if_t<IsCompatibleAndConstructible<T>::value>* = nullptr>
  Poly& operator=(T&& obj) {
    emplace(std::forward<T>(obj));
    return *this;
  }

  /// Sets this to a null state.
  ///
  /// \post `bool(*this) == false`.
  Poly& operator=(std::nullptr_t) noexcept {
    storage_.Destroy();
    return *this;
  }

  /// Constructs an object of type `T` from `arg...`.
  ///
  /// If this is in a non-null state, the existing object is destroyed first.
  ///
  /// \returns A reference to the contained object.
  /// \post `bool(*this) == true`.
  template <
      typename T, typename... U,
      std::enable_if_t<(IsCompatible<T>::value &&
                        std::is_constructible<T, U&&...>::value)>* = nullptr>
  void emplace(U&&... arg) {
    storage_.Destroy();
    Construct(std::in_place_type_t<T>{}, std::forward<U>(arg)...);
  }

  /// Constructs an object of type `remove_cvref_t<T>` from `obj`.
  ///
  /// If this is in a non-null state, the existing object is destroyed first.
  ///
  /// \returns A reference to the contained object.
  /// \post `bool(*this) == true`.
  template <typename T,
            std::enable_if_t<IsCompatibleAndConstructible<T>::value>* = nullptr>
  void emplace(T&& obj) {
    storage_.Destroy();
    Construct(std::in_place_type_t<remove_cvref_t<T>>{}, std::forward<T>(obj));
  }

  /// For each `Signature...`, a corresponding `operator()` overload is defined
  /// that forwards to the contained object.
  ///
  /// \dchecks `bool(*this)`
  using Base::operator();

  /// Returns `true` if this is bound to a valid object.
  explicit operator bool() const { return static_cast<bool>(storage_); }

  /// Returns `true` if the contained object is stored inline.
  bool is_inline() const { return storage_.is_inline(); }

  /// Returns a pointer to the contained object, or `nullptr` if in null state.
  void* target() { return storage_ ? storage_.target() : nullptr; }

  /// Returns a pointer to the contained object, or `nullptr` if in null state.
  const void* target() const { return const_cast<Poly*>(this)->target(); }

  /// Returns a pointer to the contained object if it is of type `T`, or
  /// `nullptr` otherwise.
  template <typename T>
  T* target() {
    return storage_ && storage_.vtable()->type == internal_poly::GetTypeId<T>
               ? storage_.template target<T>()
               : nullptr;
  }

  /// Returns a pointer to the contained object if it is of type `T`, or
  /// `nullptr` otherwise.
  template <typename T>
  const T* target() const {
    return const_cast<Poly*>(this)->target<T>();
  }

  friend bool operator==(std::nullptr_t, const Poly& poly) {
    return static_cast<bool>(poly) == false;
  }
  friend bool operator!=(std::nullptr_t, const Poly& poly) {
    return static_cast<bool>(poly) == true;
  }
  friend bool operator==(const Poly& poly, std::nullptr_t) {
    return static_cast<bool>(poly) == false;
  }
  friend bool operator!=(const Poly& poly, std::nullptr_t) {
    return static_cast<bool>(poly) == true;
  }

 private:
  /// Constructs from a non-Poly-like object.
  template <typename T, typename... U>
  std::enable_if_t<!IsPoly<T>::value> Construct(std::in_place_type_t<T>,
                                                U&&... arg) {
    return storage_.template Construct<T>(VTInstance<T>::vtable,
                                          static_cast<U&&>(arg)...);
  }

  /// Copy/move constructs from a Poly.
  template <std::size_t ISize, bool C, typename... S, typename T>
  void Construct(std::in_place_type_t<Poly<ISize, C, S...>>, T&& poly) {
    if constexpr (ISize <= InlineSize && HasConvertibleVTable<S...>::value) {
      storage_.Construct(static_cast<T&&>(poly).storage_);
    } else {
      storage_.template Construct<Poly<ISize, C, S...>>(
          VTInstance<Poly<ISize, C, S...>>::vtable, static_cast<T&&>(poly));
    }
  }

  Storage storage_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_POLY_H_
