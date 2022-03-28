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

/// Tag type that represents an Element type to be held via a std::shared_ptr
/// rather than a raw, unowned pointer.
template <typename Element>
class Shared;  // Intentionally left undefined.

/// `bool`-valued metafunction that evaluates to `true` if `T` is `Shared<U>` or
/// `const Shared<U>`.
template <typename T>
constexpr inline bool IsShared = false;
template <typename T>
constexpr inline bool IsShared<Shared<T>> = true;
template <typename T>
constexpr inline bool IsShared<const Shared<T>> = true;

/// An ElementTag type is a type `T` or `Shared<T>` where `T` satisfies
/// `IsElementType<T>`.  It specifies a pointer type of `T*` or
/// `std::shared_ptr<T>`, respectively.
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
template <typename T>
struct ElementTagTraits {
  static_assert(IsElementTag<T>, "T must be an ElementTag type.");
  using Element = T;
  using Pointer = T*;
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

/// Metafunction with a nested `type` alias that specifies the element tag type
/// corresponding to a given pointer type.
///
/// This is the inverse metafunction of `ElementTagPointerType`.
///
/// \tparam T Pointer type corresponding to an element tag.  Must be either `U*`
///     or `std::shared_ptr<U>`.
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

template <typename T>
using PointerElementTag = typename PointerElementTagType<T>::type;

/// `bool`-valued metafunction that evaluates to `true` if `SourcePointer` may
/// be converted to `TargetPointer` when used as an array base pointer.
///
/// Specifically, the following conversions are permitted:
///
///     T*                    -> U*
///     ByteStridedPointer<T> -> U*
///     shared_ptr<T>         -> U*
///     shared_ptr<T>         -> shared_ptr<U>
///
/// where `IsElementTypeImplicitlyConvertible<T,U>` is `true`.
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

/// `bool`-valued metafunction that evaluates to `true` if `T` is `Element*` or
/// `std::shared_ptr<Element>`, where `Element` is non-`void`.
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
template <typename T>
constexpr inline bool IsElementPointer = false;

template <typename PointerType>
constexpr inline bool IsElementPointer<ElementPointer<PointerType>> = true;

/// Pairs an array base pointer (either an `Element*` or an
/// `std::shared_ptr<Element>`) with an `DataType` in order to support a dynamic
/// element type determined at run time.
///
/// If `Element` is non-`void`, the `DataType` is not actually stored, making
/// this object the same size as `PointerType`.
///
/// Instances of this type are used by the array types defined in array.h to
/// pointer to the actual array data.
///
/// \tparam ElementTagType An ElementTag type that is either `Element` (to
///     represent unowned array data) or `Shared<Element>` (to represent array
///     data with shared ownership).
template <typename ElementTagType>
class ElementPointer {
  using Traits = ElementTagTraits<ElementTagType>;

 public:
  static_assert(IsElementTag<ElementTagType>,
                "ElementTagType must be an ElementTag type.");
  using ElementTag = ElementTagType;
  using Pointer = typename Traits::Pointer;
  using element_type = typename std::pointer_traits<Pointer>::element_type;
  using Element = element_type;
  using DataType = dtype_t<Element>;

  /// For compatibility with `std::pointer_traits`.
  template <typename OtherElement>
  using rebind = ElementPointer<typename Traits::template rebind<OtherElement>>;

  /// Initializes to a null pointer.
  /// \post data() == nullptr
  ElementPointer() = default;

  /// Initializes to a null pointer.
  /// \post data() == nullptr
  ElementPointer(std::nullptr_t) {}

  /// Constructs from a compatible ElementPointer type.
  ///
  /// \post this->pointer() == source.pointer()
  /// \post this->dtype() == source.dtype()
  template <
      typename Source,
      std::enable_if_t<(IsElementPointer<internal::remove_cvref_t<Source>> &&
                        IsArrayBasePointerConvertible<
                            typename internal::remove_cvref_t<Source>::Pointer,
                            Pointer>)>* = nullptr>
  ElementPointer(Source&& source)
      : storage_(source.dtype(),
                 internal_element_pointer::ConvertPointer<Pointer>(
                     std::forward<Source>(source).pointer())) {}

  /// Unchecked conversion.
  template <typename Source,
            std::enable_if_t<IsElementPointerCastConvertible<
                internal::remove_cvref_t<Source>, ElementPointer>>* = nullptr>
  explicit ElementPointer(unchecked_t, Source&& source)
      : storage_(DataType(unchecked, source.dtype()),
                 internal_element_pointer::ConvertPointer<Pointer>(
                     std::forward<Source>(source).pointer())) {}

  /// Constructs from another compatible pointer with a static `Element` type.
  ///
  /// \param pointer The element pointer.
  /// \post `this->pointer() == pointer`
  /// \post `this->dtype() == dtype_v<SourcePointer::element_type>`
  template <
      typename SourcePointer,
      std::enable_if_t<
          IsNonVoidArrayBasePointer<internal::remove_cvref_t<SourcePointer>> &&
          IsArrayBasePointerConvertible<internal::remove_cvref_t<SourcePointer>,
                                        Pointer>>* = nullptr>
  ElementPointer(SourcePointer&& pointer)
      : storage_(dtype_v<typename std::pointer_traits<
                     internal::remove_cvref_t<SourcePointer>>::element_type>,
                 internal::static_pointer_cast<Element>(
                     internal_element_pointer::ConvertPointer<Pointer>(
                         std::forward<SourcePointer>(pointer)))) {}

  /// Constructs from another compatible pointer paired with an `DataType`.
  ///
  /// \param pointer The element pointer.
  /// \param dtype The element representation type.
  /// \post this->pointer() == pointer
  /// \post this->dtype() == dtype
  template <typename SourcePointer,
            std::enable_if_t<IsArrayBasePointerConvertible<
                internal::remove_cvref_t<SourcePointer>, Pointer>>* = nullptr>
  ElementPointer(SourcePointer&& pointer,
                 dtype_t<typename std::pointer_traits<
                     internal::remove_cvref_t<SourcePointer>>::element_type>
                     dtype)
      : storage_(dtype, internal::static_pointer_cast<Element>(
                            internal_element_pointer::ConvertPointer<Pointer>(
                                std::forward<SourcePointer>(pointer)))) {}

  /// Assigns from a `nullptr`, pointer type, or ElementPointer type.
  template <typename Source>
  std::enable_if_t<std::is_constructible_v<ElementPointer, Source&&>,
                   ElementPointer&>
  operator=(Source&& source) {
    return *this = ElementPointer(static_cast<Source&&>(source));
  }

  constexpr DataType dtype() const { return storage_.first(); }

  /// Returns the raw pointer value.
  Element* data() const { return internal::to_address(pointer()); }

  /// Returns the raw pointer value as a ByteStridedPointer.
  ByteStridedPointer<Element> byte_strided_pointer() const { return data(); }

  /// Returns a const reference to the stored pointer.
  const Pointer& pointer() const& { return storage_.second(); }

  /// Returns a non-const reference to the stored pointer.
  Pointer& pointer() & { return storage_.second(); }

  /// Returns an rvalue reference to the stored pointer.
  Pointer&& pointer() && { return static_cast<Pointer&&>(storage_.second()); }

  /// Returns `data() != nullptr`.
  explicit operator bool() const { return data() != nullptr; }

  /// Compares an element pointer against `nullptr`.
  ///
  /// \returns p.data() == nullptr
  friend bool operator==(const ElementPointer& p, std::nullptr_t) {
    return p.data() == nullptr;
  }

  /// Compares an element pointer against `nullptr`.
  ///
  /// \returns p.data() == nullptr
  friend bool operator==(std::nullptr_t, const ElementPointer& p) {
    return p.data() == nullptr;
  }

  /// Compares an element pointer against `nullptr`.
  ///
  /// \returns p.data() != nullptr
  friend bool operator!=(const ElementPointer& p, std::nullptr_t) {
    return p.data() != nullptr;
  }

  /// Compares an element pointer against `nullptr`.
  ///
  /// \returns p.data() != nullptr
  friend bool operator!=(std::nullptr_t, const ElementPointer& p) {
    return p.data() != nullptr;
  }

 private:
  using Storage = internal::CompressedPair<DataType, Pointer>;
  Storage storage_{DataType(), nullptr};
};

/// Compares two element pointers for equality.
///
/// \returns `a.data() == b.data() && a.dtype() == b.dtype()`
template <typename A, typename B>
bool operator==(const ElementPointer<A>& a, const ElementPointer<B>& b) {
  return a.data() == b.data() && a.dtype() == b.dtype();
}

/// Compares two element pointers for inequality.
///
/// \returns !(a == b)
template <typename A, typename B>
bool operator!=(const ElementPointer<A>& a, const ElementPointer<B>& b) {
  return !(a == b);
}

/// Represents a pointer to array data with shared ownership.
template <typename Element>
using SharedElementPointer = ElementPointer<Shared<Element>>;

/// Specialization of `StaticCastTraits` for `ElementPointer`, which enables
/// `StaticCast`, `StaticDataTypeCast`, and `ConstDataTypeCast`.
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
template <typename T, typename Element>
std::enable_if_t<!IsShared<Element>, ElementPointer<Shared<Element>>>
UnownedToShared(const std::shared_ptr<T>& owned,
                ElementPointer<Element> element_pointer) {
  return {std::shared_ptr<Element>(owned, element_pointer.pointer()),
          element_pointer.dtype()};
}

/// Adds a byte offset to a raw pointer.
template <typename T>
inline T* AddByteOffset(T* x, Index byte_offset) {
  return (ByteStridedPointer<T>(x) + byte_offset).get();
}

/// Adds a byte offset to a shared_ptr.
///
/// The returned pointer shares ownership with the original.
template <typename T>
inline std::shared_ptr<T> AddByteOffset(const std::shared_ptr<T>& x,
                                        Index byte_offset) {
  return std::shared_ptr<T>(x, AddByteOffset(x.get(), byte_offset));
}

/// Adds a byte offset to an ElementPointer type.
///
/// If `Pointer` is an instance of `std::shared_ptr`, the returned pointer
/// shares ownership with the original.
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
