// Copyright 2025 The TensorStore Authors
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

#ifndef PYTHON_TENSORSTORE_WITH_HANDLE_H_
#define PYTHON_TENSORSTORE_WITH_HANDLE_H_

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <optional>
#include <type_traits>

namespace tensorstore {
namespace internal_python {

// with_handle<T> is a C++ object that holds a Python handle and a C++ value,
// and uses the Python handle as the borrowed reference to the C++ value.
template <typename T>
struct with_handle {
  static_assert(std::is_reference_v<T>);

  pybind11::handle handle;
  T value;
};

template <typename T>
struct with_handle_caster {
 private:
  static_assert(std::is_reference_v<T>);
  using base_caster = pybind11::detail::make_caster<T>;

 public:
  static constexpr auto name = base_caster::name;

  // load converts from Python -> C++
  bool load(pybind11::handle src, bool convert) {
    if (base.load(src, convert)) {
      value.emplace(with_handle<T>{src, pybind11::detail::cast_op<T>(base)});
      return true;
    }
    return false;
  }

  explicit operator with_handle<T>&() { return *value; }
  explicit operator with_handle<T>() { return *value; }

  template <typename _T>
  using cast_op_type = pybind11::detail::cast_op_type<_T>;

 private:
  base_caster base;
  std::optional<with_handle<T>> value;
};

template <typename T>
constexpr inline bool is_with_handle_v = false;

template <typename T>
constexpr inline bool is_with_handle_v<with_handle<T>> = true;

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

// Specialization of pybind11::detail::type_caster<T> for with_handle<T> types.
template <typename T>
struct type_caster<tensorstore::internal_python::with_handle<T>>
    : public tensorstore::internal_python::with_handle_caster<T> {};

}  // namespace detail
}  // namespace pybind11

#endif  // PYTHON_TENSORSTORE_WITH_HANDLE_H_
