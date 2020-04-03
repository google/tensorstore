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

#ifndef TENSORSTORE_INTERNAL_VOID_WRAPPER_H_
#define TENSORSTORE_INTERNAL_VOID_WRAPPER_H_

#include <type_traits>
#include <utility>

#include "absl/meta/type_traits.h"

namespace tensorstore {
namespace internal {

/// Helper type used as a return type in place of void, to enable uniform
/// handling of `void` and non-`void` returns from iteration functions.
///
/// Various iteration facilities in this library support an element-wise
/// callback function that may return a value that is either `void` or
/// convertible to `bool`.  A value convertible to `false` causes iteration to
/// stop and the returned value (which may be a `Status` type, for example) is
/// returned back to the caller of the iteration facility.
///
/// `void` can only be used as a type in very limited ways in C++, which makes
/// generic programming involving `void` cumbersome.  To simplify the
/// implementation of these iteration facilities, we can internally represent a
/// return type of `void` using the empty wrapper type `Void`, which is
/// convertible to `true`.
///
/// Example usage:
///
/// \snippet tensorstore/internal/void_wrapper_test.cc Repeat example
struct Void {
  explicit operator bool() const { return true; }

  /// Maps a `Void` value back to `void`.  All other values are identity mapped.
  static void Unwrap(Void) {}
  template <typename T>
  static T Unwrap(T x) {
    return x;
  }

  /// Type alias that maps `void` -> `Void`, and otherwise `T` -> `T`.
  template <typename T>
  using WrappedType = absl::conditional_t<std::is_void<T>::value, Void, T>;

  /// Type alias that maps `Void` -> `void`, and otherwise `T` -> `T`.
  template <typename T>
  using UnwrappedType =
      absl::conditional_t<std::is_same<T, Void>::value, void, T>;

  /// Invokes `func` with `args` (using perfect forwarding), and returns the
  /// result if it is not `void`, or otherwise returns `Void{}`.
  ///
  /// This overload is used in the case that the return type of `func` is
  /// `void`.
  template <typename Func, typename... Args>
  static absl::enable_if_t<std::is_void<decltype(std::declval<Func>()(
                               std::declval<Args>()...))>::value,
                           Void>
  CallAndWrap(Func&& func, Args&&... args) {
    std::forward<Func>(func)(std::forward<Args>(args)...);
    return {};
  }

  /// Overload for the case that the return type of `func` is not `void`.
  template <typename Func, typename... Args>
  static absl::enable_if_t<
      !std::is_void<
          decltype(std::declval<Func>()(std::declval<Args>()...))>::value,
      decltype(std::declval<Func>()(std::declval<Args>()...))>
  CallAndWrap(Func&& func, Args&&... args) {
    return std::forward<Func>(func)(std::forward<Args>(args)...);
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_VOID_WRAPPER_H_
