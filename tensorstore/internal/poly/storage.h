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

#ifndef TENSORSTORE_INTERNAL_POLY_STORAGE_H_
#define TENSORSTORE_INTERNAL_POLY_STORAGE_H_

/// \file
/// Implementation details for poly.h
///
/// The `Storage` class defined below contains a `VTableBase*` pointer to a
/// static constant vtable object, and a buffer that either stores the contained
/// object directly or stores a pointer to heap-allocated memory that stores the
/// contained object.
///
/// `VTableBase`, stores:
///
///   1. A `std::type_info` reference for the contained object.
///
///   2. Pointers to `destroy`, `relocate`, and `copy` functions used to
///      manage the lifetime of the contained object;
///
///   3. Derived VTableBase functions include function pointers for other
///      poly operations.

// IWYU pragma: private, include "third_party/tensorstore/internal/poly/poly.h"

#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace tensorstore {
namespace internal_poly_storage {

constexpr static inline std::size_t kAlignment = alignof(std::max_align_t);

/// Indicates whether a type T is eligible to be stored inline based on
/// alignment and other requirements.
template <class T>
static inline constexpr bool CanBeStoredInline =
    std::is_nothrow_move_constructible_v<T> && alignof(T) <= kAlignment &&
    (kAlignment % alignof(T) == 0);

/// ActualInlineSize(InlineSize) returns the number of bytes needed by
/// Storage. Within this file, references to InlineSize have had this function
/// applied to the original input.
static inline constexpr std::size_t ActualInlineSize(std::size_t InlineSize) {
  return InlineSize <= sizeof(void*)
             ? sizeof(void*)
             : ((InlineSize + sizeof(void*) - 1) / sizeof(void*)) *
                   sizeof(void*);
}

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

template <typename T>
T& Launder(void* storage) {
  return *std::launder(reinterpret_cast<T*>(storage));
}

template <typename T>
const T& Launder(const void* storage) {
  return *std::launder(reinterpret_cast<const T*>(storage));
}

/// Defines construct, destroy, relocate, and copy operations for an object
/// type `Self` stored inline.
///
/// \tparam Self Unqualified object type.
template <typename Self>
struct InlineStorageOps {
  static_assert(
      std::is_same_v<Self, std::remove_cv_t<std::remove_reference_t<Self>>>);

  using Type = Self;

  static constexpr bool UsesInlineStorage() { return true; }

  static Self& Get(void* storage) { return Launder<Self>(storage); }

  template <typename... Arg>
  static void Construct(void* storage, Arg&&... arg) {
    new (storage) Self(std::forward<Arg>(arg)...);
  }

  static void Destroy(void* storage) { Launder<Self>(storage).~Self(); }

  static void Copy(void* dest, const void* source) {
    new (dest) Self(Launder<Self>(source));
  }

  static void Relocate(void* dest, void* source) {
    Self& s = Launder<Self>(source);
    new (dest) Self(std::move(s));
    s.~Self();
  }
};

/// Defines construct, destroy, relocate, and copy operations for an object
/// type `Self` stored in a PointerAndSize structure.
///
/// \tparam Self Unqualified object type.
template <typename Self>
struct HeapStorageOps {
  static_assert(
      std::is_same_v<Self, std::remove_cv_t<std::remove_reference_t<Self>>>);

  using Type = Self;

  static constexpr bool UsesInlineStorage() { return false; }

  static Self& Get(void* storage) { return *Launder<Self*>(storage); }

  template <typename... Arg>
  static void Construct(void* storage, Arg&&... arg) {
    Launder<Self*>(storage) = new Self(std::forward<Arg>(arg)...);
  }

  static void Destroy(void* storage) { delete Launder<Self*>(storage); }

  static void Copy(void* dest, const void* source) {
    Launder<Self*>(dest) = new Self(*Launder<Self*>(source));
  }

  static void Relocate(void* dest, void* source) {
    Self*& s = Launder<Self*>(source);
    Launder<Self*>(dest) = s;
    s = nullptr;
  }
};

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
  using Relocate = void (*)(void* dest, void* source);

  /// Copy constructs `dest` from `source`.
  using Copy = void (*)(void* dest, const void* source);

  /// Equal to `GetTypeId<Self>`, or `GetTypeId<void>` if this is the null
  /// vtable.
  std::add_const_t<TypeId> type;

  Destroy destroy;
  Relocate relocate;
  Copy copy;
};

/// VTable used by Poly objects in a "null" state.
struct NullVTable {
  static void Destroy(void*) {}
  static void Relocate(void*, void*) {}
  static void Copy(void*, const void*) {}

  // Defined as static data member to ensure there is a single instance.
  constexpr static VTableBase vtable = {
      /*.type=*/GetTypeId<void>,
      /*.destroy=*/&Destroy,
      /*.relocate=*/&Relocate,
      /*.copy=*/&Copy,
  };
};

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

template <typename Ops, bool Copyable>
constexpr VTableBase GetVTableBase() {
  return {
      /*.type=*/GetTypeId<typename Ops::Type>,
      /*.destroy=*/&Ops::Destroy,
      /*.relocate=*/&Ops::Relocate,
      /*.copy=*/GetCopyImpl<Ops>(std::integral_constant<bool, Copyable>{}),
  };
}

/// Type-erased Storage container. The StorageConfig type owns inline or
/// out-of-line storage, and is specialized by StorageConfig.
template <size_t InlineSize, bool Copyable>
class StorageImpl {
  friend class StorageImpl<InlineSize, true>;
  static_assert(InlineSize == ActualInlineSize(InlineSize));

 public:
  /// Selects the approriate operations for Self based on inline size, etc.
  template <typename T>
  using Ops =
      std::conditional_t<(sizeof(T) <= InlineSize && CanBeStoredInline<T>),
                         InlineStorageOps<T>, HeapStorageOps<T>>;

  using VTable = VTableBase;

  StorageImpl() = default;

  StorageImpl(StorageImpl&& other) noexcept { Construct(std::move(other)); }

  StorageImpl& operator=(StorageImpl&& other) noexcept {
    vtable_->destroy(&storage_);
    Construct(std::move(other));
    return *this;
  }

  ~StorageImpl() { vtable_->destroy(storage()); }

  bool null() const { return vtable_->type == GetTypeId<void>; }

  void* storage() const { return const_cast<char*>(&storage_[0]); }
  const VTable* vtable() const { return vtable_; }

  template <typename T>
  T* get_if() {
    return (GetTypeId<T> != vtable_->type) ? nullptr : &Ops<T>::Get(storage());
  }
  template <typename T>
  const T* get_if() const {
    return (GetTypeId<T> != vtable_->type) ? nullptr : &Ops<T>::Get(storage());
  }

  template <typename T, typename... U>
  void ConstructT(const VTable* vtable, U&&... arg) {
    vtable_ = vtable;
    Ops<T>::Construct(storage(), std::forward<U>(arg)...);
  }

  void Construct(StorageImpl&& other) {
    vtable_ = std::exchange(other.vtable_, &NullVTable::vtable);
    vtable_->relocate(storage(), other.storage());
  }

  void Destroy() {
    std::exchange(vtable_, &NullVTable::vtable)->destroy(storage());
  }

 private:
  // Local-storage for the type-erased object when small and trivial enough
  alignas(kAlignment) char storage_[InlineSize];
  const VTable* vtable_ = &NullVTable::vtable;
};

/// Specialization for `Copyable==true`.
template <size_t InlineSize>
class StorageImpl<InlineSize, true> : public StorageImpl<InlineSize, false> {
  using Base = StorageImpl<InlineSize, false>;

 public:
  using Base::Base;

  StorageImpl(const StorageImpl& other) { this->CopyConstruct(other); }
  StorageImpl(StorageImpl&&) = default;

  StorageImpl& operator=(StorageImpl&& other) = default;
  StorageImpl& operator=(const StorageImpl& other) {
    this->Destroy();
    this->CopyConstruct(other);
    return *this;
  }

  // using Base::null;
  // using Base::storage;
  // using Base::vtable;
  // using Base::get_if;
  // using Base::Destroy;
  // using Base::Construct;

  void CopyConstruct(const StorageImpl& other) {
    this->vtable_ = other.vtable_;
    this->vtable_->copy(this->storage(), other.storage());
  }
};

template <size_t TargetInlineSize, bool Copyable>
using Storage = StorageImpl<ActualInlineSize(TargetInlineSize), Copyable>;

}  // namespace internal_poly_storage
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_POLY_STORAGE_H_
