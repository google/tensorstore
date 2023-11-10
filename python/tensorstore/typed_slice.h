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

#ifndef PYTHON_TENSORSTORE_TYPED_SLICE_H_
#define PYTHON_TENSORSTORE_TYPED_SLICE_H_

/// \file Defines a `TypedSlice` class and a pybind11 caster to/from `slice`
///     objects.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

namespace tensorstore {
namespace internal_python {

template <typename Start, typename Stop, typename Step>
struct TypedSlice {
  Start start;
  Stop stop;
  Step step;
};

}  // namespace internal_python
}  // namespace tensorstore

template <typename Start, typename Stop, typename Step>
struct pybind11::detail::type_caster<
    tensorstore::internal_python::TypedSlice<Start, Stop, Step>> {
  using T = tensorstore::internal_python::TypedSlice<Start, Stop, Step>;
  template <typename U>
  static pybind11::handle cast(U&& src, pybind11::return_value_policy policy,
                               pybind11::handle parent) {
    handle h(PySlice_New(pybind11::cast(std::forward<U>(src).start),
                         pybind11::cast(std::forward<U>(src).stop),
                         pybind11::cast(std::forward<U>(src).step)));
    if (!h.ptr()) throw error_already_set();
    return h;
  }

  bool load(pybind11::handle src, bool convert) {
    pybind11::detail::make_caster<Start> start_caster;
    pybind11::detail::make_caster<Stop> stop_caster;
    pybind11::detail::make_caster<Step> step_caster;
    if (!PySlice_Check(src.ptr())) return false;
    auto* slice_obj = reinterpret_cast<PySliceObject*>(src.ptr());
    if (!start_caster.load(slice_obj->start, convert) ||
        !stop_caster.load(slice_obj->stop, convert) ||
        !step_caster.load(slice_obj->step, convert)) {
      return false;
    }
    value.start = pybind11::detail::cast_op<Start&&>(std::move(start_caster));
    value.stop = pybind11::detail::cast_op<Stop&&>(std::move(stop_caster));
    value.step = pybind11::detail::cast_op<Step&&>(std::move(step_caster));
    return true;
  }

  // Python unfortunately does not support type annotations for `slice`
  // components
  //
  // https://github.com/python/typeshed/issues/8647
  PYBIND11_TYPE_CASTER(T, _("slice"));
};

#endif  // PYTHON_TENSORSTORE_TYPED_SLICE_H_
