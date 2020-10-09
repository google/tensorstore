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

#ifndef TENSORSTORE_INTERNAL_MEMORY_H_
#define TENSORSTORE_INTERNAL_MEMORY_H_

#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "absl/meta/type_traits.h"
#include "tensorstore/internal/exception_macros.h"

namespace tensorstore {
namespace internal {

/// Returns the address represented by `x` without forming a reference to the
/// pointee.
///
/// This is a C++11 implementation of the function with the same name specified
/// by the C++20 standard.
///
/// See https://en.cppreference.com/w/cpp/memory/to_address.
template <typename T>
inline T* to_address(const std::shared_ptr<T>& x) {
  return x.get();
}

template <typename T, typename D>
inline T* to_address(const std::unique_ptr<T, D>& x) {
  return x.get();
}

template <typename T, typename D>
inline T* to_address(const std::unique_ptr<T[], D>& x) {
  return x.get();
}

template <typename T>
inline T* to_address(T* x) {
  return x;
}

inline std::nullptr_t to_address(std::nullptr_t) { return nullptr; }

// MSVC 2019 already defines an rvalue static_pointer_cast
#ifndef _MSC_VER
/// Additional overload of static_pointer_cast for the no-op case that avoids an
/// extra copy.
template <typename T>
inline std::shared_ptr<T> static_pointer_cast(std::shared_ptr<T>&& other) {
  return std::move(other);
}
#endif

using std::const_pointer_cast;    // NOLINT
using std::dynamic_pointer_cast;  // NOLINT
using std::static_pointer_cast;   // NOLINT

/// Overload of static_pointer_cast for raw pointers.
///
/// In conjunction with the overloads defined/imported above, this is useful in
/// generic code that operates on both raw pointers and smart pointers.
template <typename T, typename U>
inline T* static_pointer_cast(U* other) {
  return static_cast<T*>(other);
}

/// Overload of const_pointer_cast for raw pointers.
///
/// In conjunction with the overloads defined/imported above, this is useful in
/// generic code that operates on both raw pointers and smart pointers.
template <typename T, typename U>
inline T* const_pointer_cast(U* other) {
  return const_cast<T*>(other);
}

/// Overload of dynamic_pointer_cast for raw pointers.
///
/// In conjunction with the overloads defined/imported above, this is useful in
/// generic code that operates on both raw pointers and smart pointers.
template <typename T, typename U>
inline T* dynamic_pointer_cast(U* other) {
  return dynamic_cast<T*>(other);
}

/// Performs any pointer cast that can be performed with a combination of
/// `static_pointer_cast` and `const_pointer_cast`.
///
/// This is useful when implementing unchecked conversion for `ElementPointer`.
template <typename T, typename U>
inline T* StaticConstPointerCast(U* other) {
  return static_cast<T*>(const_cast<absl::remove_const_t<U>*>(other));
}

template <typename T, typename U>
inline absl::enable_if_t<std::is_convertible<U*, T*>::value, std::shared_ptr<T>>
StaticConstPointerCast(const std::shared_ptr<U>& other) {
  return other;
}

template <typename T, typename U>
inline absl::enable_if_t<std::is_convertible<U*, T*>::value, std::shared_ptr<T>>
StaticConstPointerCast(std::shared_ptr<U>&& other) {
  return std::move(other);
}

template <typename T, typename U>
inline absl::enable_if_t<!std::is_convertible<U*, T*>::value,
                         std::shared_ptr<T>>
StaticConstPointerCast(const std::shared_ptr<U>& other) {
  return std::shared_ptr<T>(other, StaticConstPointerCast<T>(other.get()));
}

struct FreeDeleter {
  void operator()(void* ptr) const { std::free(ptr); }
};

template <typename T>
struct ArrayDeleter {
  void operator()(T* ptr) const { delete[] ptr; }
};

/// Allocates an array of `n` objects of type `U`, where `T = U[]`.
///
/// The objects are default initialized, meaning that primitive types remain
/// uninitialized.
///
/// This matches the behavior of the C++20 standard library function of the same
/// name, though unlike the C++20 standard library function, it still requires
/// an allocation for the control block separate from the array allocation.
/// Also, it returns `std::shared_ptr<U>` rather than `std::shared_ptr<U[]>` for
/// compatibility with Apple Clang 12.0.0 and MSVC // 19.24.
template <typename T>
std::enable_if_t<std::is_array_v<T>, std::shared_ptr<std::remove_extent_t<T>>>
make_shared_for_overwrite(size_t n) {
  using U = std::remove_extent_t<T>;
  return std::shared_ptr<U>(new U[n], ArrayDeleter<U>{});
}

}  // namespace internal
}  // namespace tensorstore

#endif  //  TENSORSTORE_INTERNAL_MEMORY_H_
