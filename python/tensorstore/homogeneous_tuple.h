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

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <cstddef>

#include "python/tensorstore/type_name_override.h"
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
  pybind11::object value;

  constexpr static auto tensorstore_pybind11_type_name_override =
      pybind11::detail::_("Tuple[") + pybind11::detail::make_caster<T>::name +
      pybind11::detail::_(", ...]");
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

#endif  // THIRD_PARTY_PY_TENSORSTORE_HOMOGENEOUS_TUPLE_H_
