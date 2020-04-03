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

#ifndef TENSORSTORE_INTERNAL_POLY_IMPL_H_
#define TENSORSTORE_INTERNAL_POLY_IMPL_H_

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

// IWYU pragma: private, include "third_party/tensorstore/internal/poly.h"

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "tensorstore/internal/type_traits.h"

namespace tensorstore {
namespace internal_poly {

using internal::remove_cvref_t;

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
using HasPolyApply = HasPolyApplyHelper<void, Arg...>;

/// Forwards to unqualified `PolyApply` (found via ADL).
///
/// \requires A matching overload of `PolyApply` exists..
template <typename Self, typename... Arg>
std::enable_if_t<HasPolyApply<Self&&, Arg&&...>::value,
                 PolyApplyResult<Self&&, Arg&&...>>
CallPolyApply(Self&& self, Arg&&... arg) {
  return PolyApply(std::forward<Self>(self), std::forward<Arg>(arg)...);
}

/// Forwards `arg...` to `self`.
/// \requires A matching overload of `operator()` exists.
/// \requires A matching overload of `PolyApply` does not exist.
template <typename Self, typename... Arg>
std::enable_if_t<!HasPolyApply<Self&&, Arg&&...>::value,
                 std::invoke_result_t<Self, Arg...>>
CallPolyApply(Self&& self, Arg&&... arg) {
  return std::forward<Self>(self)(std::forward<Arg>(arg)...);
}

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
    : public internal::IsConvertibleOrVoid<CallPolyApplyResult<Self, Arg...>,
                                           R> {};

/// `bool`-valued metafunction that evaluates to `true` if
/// `CallPolyApplyResult<Self, Arg...>` is valid and convertible to `R` (if `R`
/// is `void`, all types are considered convertible).
template <typename Self, typename R, typename... Arg>
using IsCallPolyApplyResultConvertible =
    IsCallPolyApplyResultConvertibleHelper<void, Self, R, Arg...>;

// MSVC 2019 does not permit `typeid` to be used in constexpr contexts.
// https://developercommunity.visualstudio.com/content/problem/462846/address-of-typeid-is-not-constexpr.html
#ifdef _MSC_VER
using TypeId = const char*;
template <typename T>
inline constexpr char type_id_impl = 0;
template <typename T>
inline constexpr TypeId GetTypeId = &type_id_impl<T>;
#else
using TypeId = const std::type_info&;
template <typename T>
inline constexpr TypeId GetTypeId = typeid(T);
#endif

/// Base class for vtables.
///
/// This defines operations common to all vtables.  The vtable always includes a
/// pointer to a `copy` function, which is `nullptr` (and never used) if copying
/// is not supported.  This allows a `Poly<InlineSize, false, Signature>` to be
/// constructed from a `Poly<InlineSize, true, Signature...>` without double
/// wrapping.
///
/// The function pointers stored in the vtable are invoked with a `void*` or
/// `const void*` "object" pointer that either points directly to the contained
/// object stored inline in the Poly object, or points to a pointer to the
/// contained object which is stored in heap-allocated memory.  There is no need
/// to explicitly indicate to the operation the storage mechanism, because the
/// operation itself has already been instantiated specifically for the storage
/// mechanism (see the `Inline` template parameter of the `VTableInstances`
/// class template defined below).
struct VTableBase {
  /// Destroys the contained object (deallocating the heap memory if stored on
  /// the heap).
  using Destroy = void (*)(void* obj);

  /// Move constructs `dest` from `source`, and destroys `source`.  If the
  /// contained object is stored inline, this simply calls the move constructor
  /// followed by the destructor.  If the contained object is heap allocated,
  /// this just adjusts the stored pointers.
  using MoveDestroy = void (*)(void* source, void* dest);

  /// Copy constructs `dest` from `source`.
  using Copy = void (*)(const void* source, void* dest);

  /// Equals `GetInlineSize<Self>()`, or 0 if this is the null vtable.
  std::size_t inline_size;

  /// Equal to `GetTypeId<Self>`, or `GetTypeId<void>` if this is the null
  /// vtable.
  std::add_const_t<TypeId> type;

  Destroy destroy;
  MoveDestroy move_destroy;
  Copy copy;
};

/// Defines construct, destroy, move_destroy, and copy operations for an object
/// type `Self` contained in a `Poly`.
///
/// \tparam Self Unqualified object type.
/// \tparam Inline Specifies whether the object is stored inline.  If `true`,
///     the storage pointer points directly to the object.  If `false`, the
///     storage pointer points to a pointer to the heap allocated object.
template <typename Self, bool Inline>
struct ObjectOps;

/// Specialization for `Inline==true`.
template <typename Self>
struct ObjectOps<Self, true> {
  template <typename... U>
  static void Construct(void* storage, U&&... arg) {
    new (storage) Self(std::forward<U>(arg)...);
  }
  static void Destroy(void* obj) { static_cast<Self*>(obj)->~Self(); }
  static void Copy(const void* source, void* dest) {
    new (dest) Self(*static_cast<const Self*>(source));
  }
  static void MoveDestroy(void* source, void* dest) {
    Self* source_obj = static_cast<Self*>(source);
    new (dest) Self(std::move(*source_obj));
    source_obj->~Self();
  }
  static Self& Get(void* obj) { return *static_cast<Self*>(obj); }
};

/// Specialization for `Inline==false`.
template <typename Self>
struct ObjectOps<Self, false> {
  template <typename... U>
  static void Construct(void* storage, U&&... arg) {
    new (storage) Self*(new Self(std::forward<U>(arg)...));
  }
  static void Destroy(void* obj) { delete *static_cast<Self**>(obj); }
  static void Copy(const void* source, void* dest) {
    new (dest) Self*(new Self(**static_cast<const Self* const*>(source)));
  }
  static void MoveDestroy(void* source, void* dest) noexcept {
    new (dest) Self*(*static_cast<Self**>(source));
  }
  static Self& Get(void* obj) { return **static_cast<Self**>(obj); }
};

/// VTable used by Poly objects in a "null" state.
struct NullVTable {
  static void Destroy(void*) {}
  static void MoveDestroy(void*, void*) {}
  static void Copy(const void*, void*) {}

  // Defined as static data member to ensure there is a single instance.
  constexpr static VTableBase vtable = {
      /*.inline_size=*/0,
      /*.type=*/GetTypeId<void>,
      /*.destroy=*/&Destroy,
      /*.move_destroy=*/&MoveDestroy,
      /*.copy=*/&Copy,
  };
};

/// Returns `sizeof(Self)` if `Self` satisfies `is_nothrow_move_constructible`,
/// otherwise returns `std::numeric_limits<std::size_t>::max()`.  The
/// `is_nothrow_move_constructible` constraint ensures that `Poly` is
/// unconditionally nothrow-movable (even if it stores an object that is not
/// itself nothrow-movable).
///
/// \tparam Self Unqualified object type.
template <typename Self>
constexpr std::size_t GetInlineSize() {
  return (alignof(Self) <= alignof(void*) &&
          std::is_nothrow_move_constructible<Self>::value)
             ? sizeof(Self)
             : std::numeric_limits<std::size_t>::max();
}

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
  using type = VTableBase;
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
template <typename Getter, typename Self, typename R, typename... Arg>
R CallImpl(const void* obj, Arg... arg) {
  return static_cast<R>(internal_poly::CallPolyApply(
      static_cast<Self>(Getter::Get(const_cast<void*>(obj))),
      static_cast<Arg&&>(arg)...));
}

/// Defines specializations of `SignatureTraits` and `PolyImpl` for the
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
#define TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL(DECLARED_QUALIFIERS,     \
                                              EFFECTIVE_QUALIFIERS)    \
  template <typename R, typename... Arg>                               \
  struct SignatureTraits<R(Arg...) DECLARED_QUALIFIERS> {              \
    using Unqualified = R(Arg...);                                     \
    using VTableEntry = R (*)(const void*, Arg...);                    \
    template <typename Getter, typename Self>                          \
    constexpr static VTableEntry GetVTableEntry() {                    \
      return &CallImpl<Getter, Self EFFECTIVE_QUALIFIERS, R, Arg...>;  \
    }                                                                  \
    template <typename Self>                                           \
    using IsSupportedBy =                                              \
        IsCallPolyApplyResultConvertible<Self EFFECTIVE_QUALIFIERS, R, \
                                         Arg...>;                      \
  };                                                                   \
  template <typename Derived, typename R, typename... Arg,             \
            typename... Signature>                                     \
  class PolyImpl<Derived, R(Arg...) DECLARED_QUALIFIERS, Signature...> \
      : public PolyImpl<Derived, Signature...> {                       \
   public:                                                             \
    using PolyImpl<Derived, Signature...>::operator();                 \
    R operator()(Arg... arg) DECLARED_QUALIFIERS {                     \
      using VTable = VTableType<R(Arg...), Signature...>;              \
      auto& storage = static_cast<const Derived&>(*this).storage_;     \
      assert(storage);                                                 \
      return static_cast<const VTable*>(                               \
                 static_cast<const typename Derived::VTable*>(         \
                     storage.vtable()))                                \
          ->entry(storage.data(), static_cast<Arg&&>(arg)...);         \
    }                                                                  \
  };                                                                   \
  /**/

TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL(, &)
TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL(const, const&)
TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL(&, &)
TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL(&&, &&)
TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL(const&, const&)

#undef TENSORSTORE_INTERNAL_POLY_DEFINE_IMPL

/// Returns the `copy` pointer to be stored in the vtable.
///
/// \tparam Ops An instance of `ObjectOps`.
/// \param copyable `std::true_type` if the vtable is for a copyable Poly,
///     otherwise `std::false_type`.  This overload handles the `std::true_type`
///     case.
/// \returns `&Ops::Copy` if `copyable` is `std::true_type`, or `nullptr`
///     otherwise.
template <typename Ops>
constexpr VTableBase::Copy GetCopyImpl(std::true_type copyable) {
  return &Ops::Copy;
}

/// Overload that handles the non-copyable case, and avoids attempting to
/// instantiate the copy constructor of the object type.
template <typename Ops>
constexpr VTableBase::Copy GetCopyImpl(std::false_type copyable) {
  return nullptr;
}

template <typename Self, bool Copyable, bool Inline>
constexpr VTableBase GetVTableBase() {
  using ObjectOps = internal_poly::ObjectOps<Self, Inline>;
  return {
      GetInlineSize<Self>(),
      GetTypeId<Self>,
      &ObjectOps::Destroy,
      &ObjectOps::MoveDestroy,
      GetCopyImpl<ObjectOps>(std::integral_constant<bool, Copyable>{}),
  };
}

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
template <typename Self, bool Copyable, bool Inline, typename... Signature>
struct VTableInstance {
  constexpr static VTableType<Signature...> vtable{
      SignatureTraits<Signature>::template GetVTableEntry<
          ObjectOps<Self, Inline>, Self>()...,
      GetVTableBase<Self, Copyable, Inline>(),
  };
};

template <typename Self, bool Copyable, bool Inline, typename... Signature>
constexpr VTableType<Signature...>
    VTableInstance<Self, Copyable, Inline, Signature...>::vtable;

template <std::size_t InlineSize, bool Copyable>
class Storage {
  constexpr static std::size_t ActualInlineSize =
      (InlineSize < sizeof(void*) ? sizeof(void*) : InlineSize);
  friend class Storage<InlineSize, true>;

 public:
  template <typename Self>
  constexpr static bool UsesInline() {
    return GetInlineSize<Self>() <= ActualInlineSize;
  }

  template <typename Self>
  using ObjectOps = internal_poly::ObjectOps<Self, UsesInline<Self>()>;

  /// Constructs in a null state.
  Storage() = default;

  Storage(Storage&& other) noexcept { Construct(std::move(other)); }

  Storage& operator=(Storage&& other) noexcept {
    vtable_->destroy(data());
    Construct(std::move(other));
    return *this;
  }

  ~Storage() { vtable_->destroy(data()); }

  explicit operator bool() const { return vtable_->inline_size != 0; }

  const VTableBase* vtable() const { return vtable_; }

  /// Returns either a pointer to the contained object (if `is_inline()` is
  /// `true`), or a pointer to a pointer to the contained object (if
  /// `is_inline()` is `false`).
  void* data() const { return const_cast<char*>(data_); }

  bool is_inline() const { return vtable_->inline_size <= InlineSize; }

  /// Returns a pointer to the contained object.
  void* target() {
    return is_inline() ? data() : *reinterpret_cast<void**>(data());
  }

  template <typename T>
  T* target() {
    return &ObjectOps<T>::Get(data());
  }

  template <typename T, typename... U>
  void Construct(const VTableBase& vtable, U&&... arg) {
    vtable_ = &vtable;
    ObjectOps<T>::Construct(data_, std::forward<U>(arg)...);
  }

  template <std::size_t ISize, bool C>
  void Construct(Storage<ISize, C>&& other) {
    vtable_ = std::exchange(other.vtable_, &NullVTable::vtable);
    static_assert(
        ISize <= InlineSize,
        "Cannot move construct from Storage object with larger inline size.");
    vtable_->move_destroy(other.data(), data());
  }

  // Make `C` a template argument even though it must always be `true` in order
  // to ensure MSVC 2019 does not consider this a better match for rvalues as
  // well.
  template <std::size_t ISize, bool C>
  void Construct(const Storage<ISize, C>& other) {
    static_assert(C, "Source storage type must be copyable");
    static_assert(
        ISize <= InlineSize,
        "Cannot copy construct from Storage object with larger inline size.");
    other.vtable_->copy(other.data(), this->data());
    this->vtable_ = other.vtable();
  }

  void Destroy() {
    std::exchange(vtable_, &NullVTable::vtable)->destroy(data());
  }

 private:
  const VTableBase* vtable_ = &NullVTable::vtable;
  alignas(void*) char data_[ActualInlineSize];
};

/// Specialization for `Copyable==true`.
template <std::size_t InlineSize>
class Storage<InlineSize, true> : public Storage<InlineSize, false> {
 public:
  using Storage<InlineSize, false>::Storage;

  Storage(const Storage& other) { this->Construct(other); }
  Storage(Storage&&) = default;

  Storage& operator=(Storage&& other) = default;
  Storage& operator=(const Storage& other) {
    this->Destroy();
    this->Construct(other);
    return *this;
  }
};

}  // namespace internal_poly
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_POLY_IMPL_H_
