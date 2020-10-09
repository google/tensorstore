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

#ifndef TENSORSTORE_INTERNAL_TAGGED_PTR_H_
#define TENSORSTORE_INTERNAL_TAGGED_PTR_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "absl/base/macros.h"
#include "absl/meta/type_traits.h"

namespace tensorstore {
namespace internal {

/// Class that supports storing extra tag bits within the unused least
/// significant bits of a pointer.
///
/// This class behaves like a raw pointer to `T` without any ownership
/// management.
///
/// Using an `alignas(1 << TagBits)` annotation on a type ensures that it is
/// compatible with TaggedPtr for a given number of `TagBits`.
///
/// Example:
///
///     struct alignas(8) X {
///       int value;
///     };
///     X x;
///     TaggedPtr<X, 3> ptr(&x, 5);
///     ptr->value = 5;
///     (*x).value = 4;
///     X *x_ptr = ptr;
///     assert(ptr.get() == &x);
///     assert(ptr.tag() == 5);
///     assert(ptr.tag<0>() == true);
///     assert(ptr.tag<1>() == false);
///
template <typename T, int TagBits>
class TaggedPtr {
  constexpr static std::uintptr_t kTagMask =
      (static_cast<std::uintptr_t>(1) << TagBits) - 1;
  constexpr static std::uintptr_t kPointerMask = ~kTagMask;

 public:
  using element_type = T;

  // Define `rebind` for compatibility with `std::pointer_traits`.
  template <typename U>
  using rebind = TaggedPtr<U, TagBits>;

  constexpr TaggedPtr() noexcept : value_(0) {}

  /// Constructs from a nullptr and tag.
  /// \dchecks `(tag >> TagBits) == 0`.
  /// \post `this->get() == nullptr`.
  /// \post `this->tag() == tag`.
  constexpr TaggedPtr(std::nullptr_t, std::uintptr_t tag = 0) noexcept
      : value_((ABSL_ASSERT((tag & kPointerMask) == 0), tag)) {}

  /// Constructs from a raw pointer and tag.
  /// \requires `U*` is convertible to `T*`.
  /// \dchecks `(tag >> TagBits) == 0`.
  /// \post `this->get() == ptr`.
  /// \post `this->tag() == tag`.
  template <typename U,
            absl::enable_if_t<std::is_convertible<U*, T*>::value>* = nullptr>
  TaggedPtr(U* ptr, std::uintptr_t tag = 0) noexcept {
    ABSL_ASSERT((reinterpret_cast<std::uintptr_t>(static_cast<T*>(ptr)) &
                 kTagMask) == 0 &&
                (tag & kPointerMask) == 0);
    value_ = reinterpret_cast<std::uintptr_t>(static_cast<T*>(ptr)) | tag;
  }

  /// Implicitly constructs from another tagged pointer.
  /// \requires `U*` is convertible to `T*`.
  /// \post `this->get() == other.get()`.
  /// \post `this->tag() == other.tag()`.
  template <typename U,
            absl::enable_if_t<std::is_convertible<U*, T*>::value>* = nullptr>
  TaggedPtr(TaggedPtr<U, TagBits> other) noexcept
      : TaggedPtr(other.get(), other.tag()) {}

  /// Sets the pointer to `nullptr` and the tag to `0`.
  /// \post `this->get() == nullptr`.
  /// \post `this->tag() == 0`.
  TaggedPtr& operator=(std::nullptr_t) noexcept {
    value_ = 0;
    return *this;
  }

  /// Assigns from a raw pointer.
  ///
  /// We define this as a template, rather than a non-templated assigned from `T
  /// *`, in order to prevent implicit conversion of the argument to `T *`.
  /// \requires `U*` is convertible to `T*`.
  /// \post `this->get() == ptr`.
  /// \post `this->tag() == 0`.
  template <typename U>
  absl::enable_if_t<std::is_convertible<U*, T*>::value, TaggedPtr&> operator=(
      U* ptr) noexcept {
    *this = TaggedPtr(ptr);
    return *this;
  }

  /// Returns `true` if, and only if, the pointer is not null.
  explicit operator bool() const noexcept { return get() != nullptr; }

  /// Returns the pointer.
  T* get() const noexcept {
    // Check alignment here, rather than at the class level, to allow `T` to be
    // incomplete when the class is instantiated.
    static_assert(alignof(T) >= (1 << TagBits),
                  "Number of TagBits is incompatible with alignment of T.");
    return reinterpret_cast<T*>(value_ & kPointerMask);
  }

  /// Returns the pointer.
  operator T*() const noexcept { return get(); }

  /// Returns the tag value.
  std::uintptr_t tag() const noexcept { return value_ & kTagMask; }

  /// Returns the specified bit of the tag value.
  template <int Bit>
  absl::enable_if_t<(Bit >= 0 && Bit < TagBits), bool> tag() const noexcept {
    return static_cast<bool>((value_ >> Bit) & 1);
  }

  /// Sets the specified bit of the tag value.
  template <int Bit>
  absl::enable_if_t<(Bit >= 0 && Bit < TagBits), void> set_tag(
      bool value) noexcept {
    constexpr std::uintptr_t mask = (static_cast<std::uintptr_t>(1) << Bit);
    value_ = (value_ & ~mask) | (static_cast<std::uintptr_t>(value) << Bit);
  }

  /// Sets the tag value to the specified value.
  void set_tag(std::uintptr_t tag) noexcept {
    ABSL_ASSERT((tag & kPointerMask) == 0);
    value_ = (value_ & kPointerMask) | tag;
  }

  T* operator->() const noexcept {
    T* ptr = get();
    ABSL_ASSERT(ptr != nullptr);
    return ptr;
  }

  T& operator*() const noexcept {
    T* ptr = get();
    ABSL_ASSERT(ptr != nullptr);
    return *ptr;
  }

  /// Checks for equality the pointers and tags of two tagged pointers.
  friend bool operator==(TaggedPtr x, TaggedPtr y) {
    return x.get() == y.get() && x.tag() == y.tag();
  }

  /// Checks for inequality the pointers and tags of two tagged pointers.
  friend bool operator!=(TaggedPtr x, TaggedPtr y) { return !(x == y); }

  /// Abseil hash support.
  template <typename H>
  friend H AbslHashValue(H h, TaggedPtr x) {
    return H::combine(std::move(h), x.value_);
  }

 private:
  std::uintptr_t value_;
};

/// Converts a `TaggedPtr` to a raw pointer.
template <typename T, int TagBits>
inline T* to_address(TaggedPtr<T, TagBits> p) {
  return p.get();
}

/// Returns a tagged pointer with the pointer value converted to `T*` using
/// `static_cast`, and the same tag value.
template <typename T, typename U, int TagBits>
TaggedPtr<T, TagBits> static_pointer_cast(TaggedPtr<U, TagBits> p) {
  return TaggedPtr<T, TagBits>(static_cast<T*>(p.get()), p.tag());
}

/// Returns a tagged pointer with the pointer value converted to `T*` using
/// `const_cast`, and the same tag value.
template <typename T, typename U, int TagBits>
TaggedPtr<T, TagBits> const_pointer_cast(TaggedPtr<U, TagBits> p) {
  return TaggedPtr<T, TagBits>(const_cast<T*>(p.get()), p.tag());
}

/// Returns a tagged pointer with the pointer value converted to `T*` using
/// `dynamic_cast`, and the same tag value.
template <typename T, typename U, int TagBits>
TaggedPtr<T, TagBits> dynamic_pointer_cast(TaggedPtr<U, TagBits> p) {
  return TaggedPtr<T, TagBits>(dynamic_cast<T*>(p.get()), p.tag());
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_TAGGED_PTR_H_
