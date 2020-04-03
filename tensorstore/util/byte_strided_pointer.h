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

#ifndef TENSORSTORE_UTIL_BYTE_STRIDED_POINTER_H_
#define TENSORSTORE_UTIL_BYTE_STRIDED_POINTER_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/util/element_traits.h"

namespace tensorstore {

/// Wrapper type for a raw pointer for which pointer arithmetic operates with a
/// stride of 1 byte, rather than a stride of sizeof(T) bytes.
///
/// If the run-time type of the pointee is `U`, all byte offsets used for
/// arithmetic MUST be a multiple of `alignof(U)`.
///
/// \tparam T The (possibly const-qualified) type of the pointee.  It may be
///     `void` or `const void` to indicate that the type is not known at compile
///     time.
template <typename T>
class ByteStridedPointer {
 public:
  using element_type = T;
  using difference_type = std::ptrdiff_t;

  constexpr static size_t alignment =
      alignof(std::conditional_t<std::is_void<T>::value, char, T>);

  /// Default initialization, leaves the wrapped raw pointer in an uninitialized
  /// state.
  ByteStridedPointer() = default;

  template <typename U,
            std::enable_if_t<IsElementTypeImplicitlyConvertible<U, T>::value>* =
                nullptr>
  ByteStridedPointer(U* value)
      : value_(reinterpret_cast<std::uintptr_t>(value)) {
    assert(value_ % alignment == 0);
  }

  template <typename U, std::enable_if_t<IsElementTypeOnlyExplicitlyConvertible<
                            U, T>::value>* = nullptr>
  explicit ByteStridedPointer(U* value)
      : value_(reinterpret_cast<std::uintptr_t>(value)) {
    assert(value_ % alignment == 0);
  }

  template <typename U,
            std::enable_if_t<IsElementTypeImplicitlyConvertible<U, T>::value>* =
                nullptr>
  ByteStridedPointer(ByteStridedPointer<U> value)
      : value_(reinterpret_cast<std::uintptr_t>(value.get())) {
    assert(value_ % alignment == 0);
  }

  template <typename U, std::enable_if_t<IsElementTypeOnlyExplicitlyConvertible<
                            U, T>::value>* = nullptr>
  explicit ByteStridedPointer(ByteStridedPointer<U> value)
      : value_(reinterpret_cast<std::uintptr_t>(value.get())) {
    assert(value_ % alignment == 0);
  }

  T* get() const {
    assert(value_ % alignment == 0);
    return reinterpret_cast<T*>(value_);
  }
  T* operator->() const { return get(); }

  /// Dereferences the raw pointer.
  ///
  /// This is a template solely to avoid compilation errors when T=void.
  template <typename U = T>
  U& operator*() const {
    return *static_cast<U*>(get());
  }

  operator T*() const { return get(); }

  template <typename U, std::enable_if_t<IsElementTypeOnlyExplicitlyConvertible<
                            T, U>::value>* = nullptr>
  explicit operator U*() const {
    return static_cast<U*>(get());
  }

  /// Increments the raw pointer by `byte_offset` bytes.
  template <typename Integer>
  std::enable_if_t<std::is_integral<Integer>::value, ByteStridedPointer&>
  operator+=(Integer byte_offset) {
    value_ = internal::wrap_on_overflow::Add(
        value_, static_cast<std::uintptr_t>(byte_offset));
    assert(value_ % alignment == 0);
    return *this;
  }

  /// Decrements the raw pointer by `byte_offset` bytes.
  template <typename Integer>
  std::enable_if_t<std::is_integral<Integer>::value, ByteStridedPointer&>
  operator-=(Integer byte_offset) {
    value_ = internal::wrap_on_overflow::Subtract(
        value_, static_cast<std::uintptr_t>(byte_offset));
    assert(value_ % alignment == 0);
    return *this;
  }

  /// Returns a reference to the element starting at the specified `byte_offset`
  /// relative to `get()`.
  template <typename Integer>
  std::enable_if_t<std::is_integral<Integer>::value, T>& operator[](
      Integer byte_offset) const {
    ByteStridedPointer x = *this;
    x += byte_offset;
    assert(x.value_ % alignment == 0);
    return *x;
  }

 private:
  std::uintptr_t value_;
};

template <typename T, typename U>
std::ptrdiff_t operator-(ByteStridedPointer<T> a, ByteStridedPointer<U> b) {
  return reinterpret_cast<const char*>(a.get()) -
         reinterpret_cast<const char*>(b.get());
}

template <typename T, typename Integer>
inline std::enable_if_t<std::is_integral<Integer>::value, ByteStridedPointer<T>>
operator+(ByteStridedPointer<T> ptr, Integer byte_offset) {
  ptr += static_cast<std::uintptr_t>(byte_offset);
  return ptr;
}

template <typename T, typename Integer>
inline std::enable_if_t<std::is_integral<Integer>::value, ByteStridedPointer<T>>
operator+(Integer byte_offset, ByteStridedPointer<T> ptr) {
  ptr += static_cast<std::uintptr_t>(byte_offset);
  return ptr;
}

template <typename T, typename Integer>
inline std::enable_if_t<std::is_integral<Integer>::value, ByteStridedPointer<T>>
operator-(ByteStridedPointer<T> ptr, Integer byte_offset) {
  ptr -= static_cast<std::uintptr_t>(byte_offset);
  return ptr;
}

// We don't explicitly define comparison operators, because comparison
// operations work without any explicit definition based on the conversion to a
// raw pointer.

}  // namespace tensorstore

#endif  //  TENSORSTORE_UTIL_BYTE_STRIDED_POINTER_H_
