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

#ifndef TENSORSTORE_UTIL_FUNCTION_VIEW_H_
#define TENSORSTORE_UTIL_FUNCTION_VIEW_H_

/// \file
/// `FunctionView` is a lightweight type-erased view of a function object.
///
/// It should normally be passed by value.
///
/// It behaves nearly the same as constructing an `std::function` from an
/// `std::reference_wrapper`, but is more efficient because it is the size of
/// two pointers and trivially copyable.
///
/// A `FunctionView` does not own the function object it references; the user
/// must ensure that the `FunctionView` is not called after the lifetime of the
/// referenced object ends.
///
/// This is inspired by:
///
/// LLVM function_ref
/// (https://github.com/llvm-mirror/llvm/blob/19a56211e133d6981fca86913ca6b97a701cee52/include/llvm/ADT/STLExtras.h#L119)
///
/// https://vittorioromeo.info/index/blog/passing_functions_to_functions.html
///
/// Example:
///
///     void ForEach(const std::vector<int> &v,
///                  FunctionView<void(int)> callback) {
///       for (int x : v) callback(x);
///     }
///
///     int sum = 0;
///     ForEach({1, 2, 3}, [&] (int x) { sum += x; });
///     EXPECT_EQ(6, sum);
///

#include <cstddef>
#include <type_traits>
#include <utility>

#include "tensorstore/internal/type_traits.h"

namespace tensorstore {

template <typename Signature>
class FunctionView;

template <typename R, typename... Arg>
class FunctionView<R(Arg...)> {
 public:
  /// Creates a null function view.
  constexpr FunctionView() = default;
  constexpr FunctionView(std::nullptr_t) {}

  /// Creates a view that references an existing function.
  ///
  /// The constructed view is only valid for the lifetime of `f`.
  ///
  /// \requires `F&` has a call signature compatible with `R (Arg...)`.
  template <typename F,
            std::enable_if_t<
                (!std::is_same_v<internal::remove_cvref_t<F>, FunctionView> &&
                 internal::IsConvertibleOrVoid<std::invoke_result_t<F&, Arg...>,
                                               R>::value)>* = nullptr>
  constexpr FunctionView(F&& f) noexcept
      : erased_fn_(&Wrapper<std::remove_reference_t<F>>), erased_obj_(&f) {}

  /// Calls the referenced function.
  R operator()(Arg... arg) const {
    return erased_fn_(const_cast<void*>(erased_obj_),
                      std::forward<Arg>(arg)...);
  }

  /// Returns `true` if this is a null function view.
  explicit operator bool() const { return erased_fn_ != nullptr; }

 private:
  template <typename F>
  static R Wrapper(void* obj, Arg... arg) {
    return (*static_cast<std::add_pointer_t<F>>(obj))(
        std::forward<Arg>(arg)...);
  }

  using ErasedFunctionPointer = R (*)(void*, Arg...);
  ErasedFunctionPointer erased_fn_ = nullptr;
  const void* erased_obj_ = nullptr;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_FUNCTION_VIEW_H_
