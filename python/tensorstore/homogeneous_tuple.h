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

#ifndef THIRD_PARTY_PY_TENSORSTORE_HOMOGENEOUS_TUPLE_H_
#define THIRD_PARTY_PY_TENSORSTORE_HOMOGENEOUS_TUPLE_H_

#include <cstddef>

#include "pybind11/pybind11.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_python {

/// Wrapper around a Python tuple that displays as `Tuple[T, ...]`.
///
/// This is intended as a pybind11 function return type, in order to indicate
/// that the return type is a homogeneous tuple.  For example, this is used as
/// the return type for `IndexDomain.shape`.
template <typename T>
struct HomogeneousTuple {
  pybind11::tuple obj;
};

template <typename T>
HomogeneousTuple<T> SpanToHomogeneousTuple(span<const T> vec) {
  pybind11::tuple t(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    t[i] = pybind11::cast(vec[i]);
  }
  return HomogeneousTuple<T>{std::move(t)};
}

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion of `HomogeneousTuple<T>` to a Python object.
///
/// Since `HomogeneousTuple` is only intended as a function return type, we do
/// not define the reverse conversion from a Python object.
template <typename T>
struct type_caster<tensorstore::internal_python::HomogeneousTuple<T>> {
  using Value = tensorstore::internal_python::HomogeneousTuple<T>;
  using base_value_conv = make_caster<T>;
  PYBIND11_TYPE_CASTER(Value,
                       _("Tuple[") + base_value_conv::name + _(", ...]"));
  static handle cast(Value value, return_value_policy policy, handle parent) {
    return value.obj.release();
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_HOMOGENEOUS_TUPLE_H_
