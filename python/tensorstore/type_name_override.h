// Copyright 2021 The TensorStore Authors
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

#ifndef THIRD_PARTY_PY_TENSORSTORE_TYPE_NAME_OVERRIDE_H_
#define THIRD_PARTY_PY_TENSORSTORE_TYPE_NAME_OVERRIDE_H_

/// \file Defines a mechanism for overriding the type name shown in
///     pybind11-generated signatures.
///
/// To use this mechanism, define a class/struct with a public `value` data
/// member that holds the contained value, and a
/// `static constexpr auto tensorstore_pybind11_type_name_override` member of
/// type `pybind11::detail::descr<...>`.
///
/// Example usage:
///
///     struct MyType {
///       bool value;
///       constexpr static auto tensorstore_pybind11_type_name_override =
///           pybind11::detail::_("MyType");
///     };
///
///     template <typename T>
///     struct MyType2 {
///       T value;
///       constexpr static auto tensorstore_pybind11_type_name_override =
///           pybind11::detail::_("MyType[") +
///           pybind11::detail::_make_caster<T>::name +
///           pybind11::detail::__("]");
///     };

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <type_traits>

namespace tensorstore {
namespace internal_python {

/// Wrapper for a `pybind11::handle` that displays as a `Callable` type.
template <typename R, typename... Arg>
struct Callable {
  pybind11::handle value;
  constexpr static auto tensorstore_pybind11_type_name_override =
      pybind11::detail::_("Callable[[") +
      pybind11::detail::concat(pybind11::detail::make_caster<Arg>::name...) +
      pybind11::detail::_("], ") +
      pybind11::detail::make_caster<std::conditional_t<
          std::is_void_v<R>, pybind11::detail::void_type, R>>::name +
      pybind11::detail::_("]");
};

}  // namespace internal_python
}  // namespace tensorstore

/// Type caster for wrapper types that define a
/// `tensorstore_pybind11_type_name_override` member.
template <typename T>
struct pybind11::detail::type_caster<
    T, std::void_t<decltype(T::tensorstore_pybind11_type_name_override)>> {
  using InnerType = decltype(std::declval<T>().value);
  using value_conv = pybind11::detail::make_caster<InnerType>;

  template <typename U>
  static pybind11::handle cast(U&& src, pybind11::return_value_policy policy,
                               pybind11::handle parent) {
    return value_conv::cast(std::forward<U>(src).value, policy, parent);
  }

  bool load(pybind11::handle src, bool convert) {
    value_conv inner_caster;
    if (!inner_caster.load(src, convert)) return false;
    value.value =
        pybind11::detail::cast_op<InnerType&&>(std::move(inner_caster));
    return true;
  }

  PYBIND11_TYPE_CASTER(T, T::tensorstore_pybind11_type_name_override);
};

#endif  // THIRD_PARTY_PY_TENSORSTORE_TYPE_NAME_OVERRIDE_H_
