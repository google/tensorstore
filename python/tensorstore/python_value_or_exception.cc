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

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "python/tensorstore/python_value_or_exception.h"

namespace tensorstore {
namespace internal_python {

PythonValueOrException PythonValueOrException::FromErrorIndicator() {
  PythonValueOrException v;
  PyErr_Fetch(&v.error_type.ptr(), &v.error_value.ptr(),
              &v.error_traceback.ptr());
  PyErr_NormalizeException(&v.error_type.ptr(), &v.error_value.ptr(),
                           &v.error_traceback.ptr());
  assert(v.error_type.ptr());
  return v;
}

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

handle
type_caster<tensorstore::internal_python::PythonValueOrExceptionWeakRef>::cast(
    const tensorstore::internal_python::PythonValueOrExceptionWeakRef& obj,
    return_value_policy policy, handle parent) {
  if (obj.value) {
    return obj.value.get_value_or_none().inc_ref();
  }
  auto error_type =
      reinterpret_borrow<object>(obj.error_type.get_value_or_null());
  auto error_value =
      reinterpret_borrow<object>(obj.error_value.get_value_or_null());
  auto error_traceback =
      reinterpret_borrow<object>(obj.error_traceback.get_value_or_null());
  if (!error_type || !error_value) {
    throw value_error("internal error: weak referent destroyed");
  }
  ::PyErr_Restore(error_type.release().ptr(), error_value.release().ptr(),
                  error_traceback.release().ptr());
  throw error_already_set();
}

}  // namespace detail
}  // namespace pybind11
