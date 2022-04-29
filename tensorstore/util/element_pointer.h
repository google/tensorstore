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

#ifndef TENSORSTORE_UTIL_ELEMENT_POINTER_H_
#define TENSORSTORE_UTIL_ELEMENT_POINTER_H_

/// \file
/// Defines the `ElementPointer` and `SharedElementPointer` classes that combine
/// a raw pointer or shared_ptr to an `Element` with a
/// `dtype_t<Element>` value.

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/compressed_pair.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/internal/unowned_to_shared.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/element_traits.h"

namespace tensorstore {

/// Tag type that represents an `Element` type to be held via an
/// `std::shared_ptr` rather than a raw, unowned pointer.
///
/// \relates ElementPointer
template <typename Element>
class Shared;  // Intentionally left undefined.

/// `bool`-valued metafunction that evaluates to `true` if `T` is an instance of
/// `Shared`.
///
/// \relates Shared
template <typename T>
constexpr inline bool IsShared = false;

template <typename T>
constexpr inline bool IsShared<Shared<T>> = true;
template <typename T>
constexpr inline bool IsShared<const Shared<T>> = true;

/// An ElementTag type is a type `T` or `Shared<T>` where `T` satisfies
/// `IsElementType<T>`.  It specifies a pointer type of `T*` or
/// `std::shared_ptr<T>`, respectively.
///
/// \relates ElementPointer
template <typename T>
constexpr inline bool IsElementTag = (IsElementType<T> &&
                                      // Detect accidental `const`-qualification
                                      // of `Shared<T>`.
                                      !IsShared<T>);

template <typename T>
constexpr inline bool IsElementTag<Shared<T>> = (IsElementType<T> &&
                                                 // Detect accidental nested
                                                 // `Shared<Shared<T>>`.
                                                 !IsShared<T>);

/// Traits for element tag types.
///
/// Defines a nested `Pointer` type alias that specifies the pointer type
/// corresponding to the element tag, and a `rebind` template for obtaining an
/// element tag with the same ownership semantics for a different element type.
///
/// \tparam T Type satisfying `IsElementTag<T>`.
/// \relates ElementPointer
template <typename T>
struct ElementTagTraits {
  static_assert(IsElementTag<T>, "T must be an ElementTag type.");
  /// Element type of `Pointer`.
  using Element = T;

  /// Pointer type.
  using Pointer = T*;

  /// Same kind of tag type, but for another element type `U`.
  template <typename U>
  using rebind = U;
};

template <typename T>
struct ElementTagTraits<Shared<T>> {
  static_assert(IsElementTag<Shared<T>>,
                "Shared<T> must be an ElementTag type.");
  using Element = T;
  using Pointer = std::shared_ptr<T>;
  template <typename U>
  using rebind = Shared<U>;
};

namespace internal_element_pointer {
template <typename T>
struct PointerElementTagType;

template <typename T>
struct PointerElementTagType<T*> {
  using type = T;
};

template <typename T>
struct PointerElementTagType<std::shared_ptr<T>> {
  using type = Shared<T>;
};
}  // namespace internal_element_pointer

/// Element tag type corresponding to a given pointer type.
///
/// This is the inverse metafunction of `ElementTagTraits::Pointer`.
///
/// \tparam T Pointer type corresponding to an element tag.  Must be either
///     ``U*`` or ``std::shared_ptr<U>``.
///
/// \relates ElementPointer
template <typename T>
using PointerElementTag =
    typename internal_element_pointer::PointerElementTagType<T>::type;

/// `bool`-valued metafunction that evaluates to `true` if `SourcePointer` may
/// be converted to `TargetPointer` when used as an array base pointer.
///
/// Specifically, the following conversions are permitted:
///
/// =========================    =========================
/// `SourcePointer`              `TargetPointer`
/// =========================    =========================
/// ``T*``                       ``U*``
/// ``ByteStridedPointer<T>``    ``U*``
/// ``std::shared_ptr<T>``       ``U*``
/// ``std::shared_ptr<T>``       ``std::shared_ptr<U>``
/// =========================    =========================
///
/// where ``IsElementTypeImplicitlyConvertible<T,U>`` is `true`.
///
/// \relates ElementPointer
template <typename SourcePointer, typename TargetPointer>
constexpr inline bool IsArrayBasePointerConvertible = false;

template <typename SourceElement, typename TargetElement>
constexpr inline bool
    IsArrayBasePointerConvertible<SourceElement*, TargetElement*> =
        IsElementTypeImplicitlyConvertible<SourceElement, TargetElement>;

template <typename SourceElement, typename TargetElement>
constexpr inline bool IsArrayBasePointerConvertible<
    ByteStridedPointer<SourceElement>, TargetElement*> =
    IsElementTypeImplicitlyConvertible<SourceElement, TargetElement>;

template <typename SourceElement, typename TargetElement>
constexpr inline bool IsArrayBasePointerConvertible<
    std::shared_ptr<SourceElement>, TargetElement*> =
    IsElementTypeImplicitlyConvertible<SourceElement, TargetElement>;

template <typename SourceElement, typename TargetElement>
constexpr inline bool IsArrayBasePointerConvertible<
    std::shared_ptr<SourceElement>, std::shared_ptr<TargetElement>> =
    IsElementTypeImplicitlyConvertible<SourceElement, TargetElement>;

/// Determine if `Source` and `Target` are instances of `ElementPointer` and
/// `Source` is (explicitly) convertible to `Target`.
///
/// \relates ElementPointer
template <typename Source, typename Target>
constexpr inline bool IsElementPointerCastConvertible = false;

template <typename ElementTag>
class ElementPointer;

template <typename SourceTag, typename TargetTag>
constexpr inline bool IsElementPointerCastConvertible<
    ElementPointer<SourceTag>, ElementPointer<TargetTag>> =
    ((IsShared<SourceTag> >= IsShared<TargetTag>)&&  //
     IsElementTypeExplicitlyConvertible<
         std::remove_const_t<typename ElementTagTraits<SourceTag>::Element>,
         std::remove_const_t<typename ElementTagTraits<TargetTag>::Element>>);

/// `bool`-valued metafunction that evaluates to `true` if `T` is
/// ``Element*`` or ``std::shared_ptr<Element>``, where ``Element`` is
/// non-`void`.
///
/// \relates ElementPointer
template <typename T>
constexpr inline bool IsNonVoidArrayBasePointer = false;

template <typename T>
constexpr inline bool IsNonVoidArrayBasePointer<T*> = !std::is_void_v<T>;

template <typename T>
constexpr inline bool IsNonVoidArrayBasePointer<std::shared_ptr<T>> =
    !std::is_void_v<T>;

namespace internal_element_pointer {
template <typename Target, typename Source>
inline std::enable_if_t<(std::is_pointer_v<Target> ==
                         std::is_pointer_v<internal::remove_cvref_t<Source>>),
                        Target>
ConvertPointer(Source&& x) {
  return internal::StaticConstPointerCast<
      typename std::pointer_traits<Target>::element_type>(
      static_cast<Source&&>(x));
}
template <typename Target, typename Source>
inline std::enable_if_t<(std::is_pointer_v<Target> >
                         std::is_pointer_v<internal::remove_cvref_t<Source>>),
                        Target>
ConvertPointer(Source&& x) {
  return internal::StaticConstPointerCast<
      typename std::pointer_traits<Target>::element_type>(x.get());
}

std::string DescribeForCast(DataType dtype);
}  // namespace internal_element_pointer

template <typename ElementTagType>
class ElementPointer;

/// `bool`-valued metafunction that evaluates to `true` if `T` is an instance of
/// `ElementPointer`.
///
/// \relates ElementPointer
template <typename T>
constexpr inline bool IsElementPointer = false;

template <typename PointerType>
constexpr inline bool IsElementPointer<ElementPointer<PointerType>> = true;

/// Pairs an array base pointer (either an `Element*` or an
/// `std::shared_ptr<Element>`) with a `DataType` in order to support a
/// dynamic element type determined at run time.
///
/// If `Element` is non-`void`, the `DataType` is not actually stored,
/// making this object the same size as `Pointer`.
///
/// Instances of this type are used by `Array` to point to the actual array
/// data.
///
/// \tparam ElementTagType An ElementTag type that is either `Element` (to
///     represent unowned array data) or `Shared<Element>` (to represent
///     array data with shared ownership).
/// \ingroup array
template <typename ElementTagType>
class ElementPointer {
  using Traits = ElementTagTraits<ElementTagType>;

 public:
  static_assert(IsElementTag<ElementTagType>,
                "ElementTagType must be an ElementTag type.");

  /// Element tag type.
  using ElementTag = ElementTagType;

  /// Underlying data pointer type, either `Element*` or
  /// `std::shared_ptr<Element>`.
  using Pointer = typename Traits::Pointer;

  /// Underlying element type.
  using element_type = typename std::pointer_traits<Pointer>::
      element_type;  // NONITPICK: std::pointer_traits<Pointer>::element_type
  using Element = element_type;

  /// Static or dynamic data type representation.
  using DataType = dtype_t<Element>;

  // For compatibility with `std::pointer_traits`.
  template <typename OtherElement>
  using rebind = ElementPointer<typename Traits::template rebind<OtherElement>>;

  /// Initializes to a null pointer.
  ///
  /// \id default
  ElementPointer() = default;
  ElementPointer(std::nullptr_t) {}

  /// Constructs from a compatible `ElementPointer` type.
  ///
  /// \id element_pointer
  template <
      typename Source,
      std::enable_if_t<(IsElementPointer<internal::remove_cvref_t<Source>> &&
                        IsArrayBasePointerConvertible<
                            typename internal::remove_cvref_t<Source>::Pointer,
                            Pointer>)>* = nullptr>
  // NONITPICK: std::remove_cvref_t<Source>::Pointer
  ElementPointer(Source&& source)
      : storage_(source.dtype(),
                 internal_element_pointer::ConvertPointer<Pointer>(
                     std::forward<Source>(source).pointer())) {}

  /// Unchecked conversion.
  ///
  /// \id unchecked
  template <typename Source,
            std::enable_if_t<IsElementPointerCastConvertible<
                internal::remove_cvref_t<Source>, ElementPointer>>* = nullptr>
  explicit ElementPointer(unchecked_t, Source&& source)
      : storage_(DataType(unchecked, source.dtype()),
                 internal_element_pointer::ConvertPointer<Pointer>(
                     std::forward<Source>(source).pointer())) {}

  /// Constructs from another compatible pointer and optional data type.
  ///
  /// \param pointer The element pointer.
  /// \param dtype The data type, must be specified if `SourcePointer` has an
  ///     element type of `void`.
  /// \id pointer
  template <
      typename SourcePointer,
      std::enable_if_t<
          IsNonVoidArrayBasePointer<internal::remove_cvref_t<SourcePointer>> &&
          IsArrayBasePointerConvertible<internal::remove_cvref_t<SourcePointer>,
                                        Pointer>>* = nullptr>
  ElementPointer(SourcePointer&& pointer)
      : storage_(pointee_dtype_t<SourcePointer>(),
                 internal::static_pointer_cast<Element>(
                     internal_element_pointer::ConvertPointer<Pointer>(
                         std::forward<SourcePointer>(pointer)))) {}
  template <typename SourcePointer,
            std::enable_if_t<IsArrayBasePointerConvertible<
                internal::remove_cvref_t<SourcePointer>, Pointer>>* = nullptr>
  ElementPointer(SourcePointer&& pointer, pointee_dtype_t<SourcePointer> dtype)
      : storage_(dtype, internal::static_pointer_cast<Element>(
                            internal_element_pointer::ConvertPointer<Pointer>(
                                std::forward<SourcePointer>(pointer)))) {}

  /// Assigns from a `nullptr`, pointer type, or `ElementPointer` type.
  template <typename Source>
  std::enable_if_t<std::is_constructible_v<ElementPointer, Source&&>,
                   ElementPointer&>
  operator=(Source&& source) {
    return *this = ElementPointer(static_cast<Source&&>(source));
  }

  /// Returns the data type.
  ///
  /// \membergroup Accessors
  constexpr DataType dtype() const { return storage_.first(); }

  /// Returns the raw pointer value.
  ///
  /// \membergroup Accessors
  Element* data() const { return internal::to_address(pointer()); }

  /// Returns the raw pointer value as a `ByteStridedPointer`.
  ///
  /// \membergroup Accessors
  ByteStridedPointer<Element> byte_strided_pointer() const { return data(); }

  /// Returns a reference to the stored pointer.
  ///
  /// \membergroup Accessors
  const Pointer& pointer() const& { return storage_.second(); }
  Pointer& pointer() & { return storage_.second(); }
  Pointer&& pointer() && { return static_cast<Pointer&&>(storage_.second()); }

  /// Returns `data() != nullptr`.
  ///
  /// \membergroup Accessors
  explicit operator bool() const { return data() != nullptr; }

  /// Compares the data pointers and data types.
  ///
  /// \id element_pointer
  /// \membergroup Comparison
  template <typename B>
  friend bool operator==(const ElementPointer& a, const ElementPointer<B>& b) {
    return a.data() == b.data() && a.dtype() == b.dtype();
  }
  template <typename B>
  friend bool operator!=(const ElementPointer& a, const ElementPointer<B>& b) {
    return !(a == b);
  }

  /// Checks if the data pointer is null.
  ///
  /// \id nullptr
  /// \membergroup Comparison
  friend bool operator==(const ElementPointer& p, std::nullptr_t) {
    return p.data() == nullptr;
  }

  friend bool operator==(std::nullptr_t, const ElementPointer& p) {
    return p.data() == nullptr;
  }
  friend bool operator!=(const ElementPointer& p, std::nullptr_t) {
    return p.data() != nullptr;
  }
  friend bool operator!=(std::nullptr_t, const ElementPointer& p) {
    return p.data() != nullptr;
  }

 private:
  using Storage = internal::CompressedPair<DataType, Pointer>;
  Storage storage_{DataType(), nullptr};
};

/// Represents a pointer to array data with shared ownership.
///
/// \relates ElementPointer
template <typename Element>
using SharedElementPointer = ElementPointer<Shared<Element>>;

// Specialization of `StaticCastTraits` for `ElementPointer`, which enables
// `StaticCast`, `StaticDataTypeCast`, and `ConstDataTypeCast`.
template <typename ElementTag>
struct StaticCastTraits<ElementPointer<ElementTag>>
    : public DefaultStaticCastTraits<ElementPointer<ElementTag>> {
  using type = ElementPointer<ElementTag>;
  template <typename TargetElement>
  using RebindDataType =
      typename ElementPointer<ElementTag>::template rebind<TargetElement>;

  template <typename OtherElementTag>
  static bool IsCompatible(const ElementPointer<OtherElementTag>& other) {
    return IsPossiblySameDataType(typename type::DataType(), other.dtype());
  }

  static std::string Describe() {
    return internal_element_pointer::DescribeForCast(typename type::DataType());
  }
  static std::string Describe(const type& x) {
    return internal_element_pointer::DescribeForCast(x.dtype());
  }
};

/// Converts a non-Shared ElementPointer to a SharedElementPointer that does not
/// manage ownership.
///
/// The caller is responsible for ensuring that the returned
/// SharedElementPointer is not used after the element data to which it points
/// becomes invalid.
///
/// The returned SharedElementPointer can be copied more efficiently than a
/// SharedElementPointer that does manage ownership, because it does not perform
/// any atomic reference count operations.
///
/// \relates ElementPointer
/// \membergroup Ownership conversion
/// \id element_pointer
template <typename Element>
std::enable_if_t<!IsShared<Element>, ElementPointer<Shared<Element>>>
UnownedToShared(ElementPointer<Element> element_pointer) {
  return {internal::UnownedToShared(element_pointer.pointer()),
          element_pointer.dtype()};
}

/// Converts a non-`Shared` `ElementPointer` to a `SharedElementPointer` that
/// shares ownership of the specified `owned` pointer, in the same way as the
/// `std::shared_ptr` aliasing constructor.
///
/// The caller is responsible for ensuring that the returned
/// `SharedElementPointer` is not used after the element data to which it points
/// becomes invalid.
///
/// \relates ElementPointer
/// \membergroup Ownership conversion
/// \id owned, element_pointer
template <typename T, typename Element>
std::enable_if_t<!IsShared<Element>, ElementPointer<Shared<Element>>>
UnownedToShared(const std::shared_ptr<T>& owned,
                ElementPointer<Element> element_pointer) {
  return {std::shared_ptr<Element>(owned, element_pointer.pointer()),
          element_pointer.dtype()};
}

/// Adds a byte offset to a raw pointer.
///
/// \relates ElementPointer
/// \membergroup Arithmetic operations
/// \id raw
template <typename T>
inline T* AddByteOffset(T* x, Index byte_offset) {
  return (ByteStridedPointer<T>(x) + byte_offset).get();
}

/// Adds a byte offset to a shared_ptr.
///
/// The returned pointer shares ownership with the original.
///
/// \relates ElementPointer
/// \membergroup Arithmetic operations
/// \id shared_ptr
template <typename T>
inline std::shared_ptr<T> AddByteOffset(const std::shared_ptr<T>& x,
                                        Index byte_offset) {
  return std::shared_ptr<T>(x, AddByteOffset(x.get(), byte_offset));
}

/// Adds a byte offset to an `ElementPointer` type.
///
/// If `IsShared<ElementTag>`, the returned pointer shares ownership with the
/// original.
///
/// \relates ElementPointer
/// \membergroup Arithmetic operations
/// \id element_pointer
template <typename ElementTag>
inline ElementPointer<ElementTag> AddByteOffset(
    const ElementPointer<ElementTag>& x, Index byte_offset) {
  return {AddByteOffset(x.pointer(), byte_offset), x.dtype()};
}
template <typename ElementTag>
inline ElementPointer<ElementTag> AddByteOffset(ElementPointer<ElementTag>&& x,
                                                Index byte_offset) {
  return {AddByteOffset(std::move(x.pointer()), byte_offset), x.dtype()};
}

namespace internal_element_pointer {

template <typename Pointer>
struct DeducedElementTagHelper {};

template <typename T>
struct DeducedElementTagHelper<ElementPointer<T>> {
  using type = T;
};

template <typename T>
struct DeducedElementTagHelper<std::shared_ptr<T>>
    : public std::enable_if<!std::is_void_v<T>, Shared<T>> {};

template <typename T>
struct DeducedElementTagHelper<T*>
    : public std::enable_if<!std::is_void_v<T>, T> {};
}  // namespace internal_element_pointer

template <typename T>
using DeducedElementTag =
    typename internal_element_pointer::DeducedElementTagHelper<
        internal::remove_cvref_t<T>>::type;

template <typename Pointer>
ElementPointer(Pointer pointer) -> ElementPointer<DeducedElementTag<Pointer>>;

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_ELEMENT_POINTER_H_
