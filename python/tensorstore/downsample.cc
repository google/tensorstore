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

#include "python/tensorstore/downsample.h"
#include "python/tensorstore/index.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/spec.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "python/tensorstore/tensorstore_class.h"
#include "tensorstore/downsample.h"
#include "tensorstore/downsample_method.h"
#include "tensorstore/driver/downsample/downsample_method_json_binder.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/spec.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {
namespace {

namespace py = ::pybind11;

void RegisterDownsampleBindings(pybind11::module m, Executor defer) {
  defer([m]() mutable {
    m.def(
        "downsample",
        [](PythonTensorStoreObject& base, std::vector<Index> downsample_factors,
           DownsampleMethod method) -> PythonTensorStore {
          return TensorStore<>(ValueOrThrow(
              tensorstore::Downsample(base.value, downsample_factors, method)));
        },
        R"(
Returns a virtual :ref:`downsampled view<driver/downsample>` of a :py:obj:`TensorStore`.

Group:
  Views

Overload:
  store
)",
        py::arg("base"), py::arg("downsample_factors"), py::arg("method"));

    m.def(
        "downsample",
        [](PythonSpecObject& base, std::vector<Index> downsample_factors,
           DownsampleMethod method) -> PythonSpec {
          return ValueOrThrow(
              tensorstore::Downsample(base.value, downsample_factors, method));
        },
        R"(
Returns a virtual :ref:`downsampled view<driver/downsample>` view of a :py:obj:`Spec`.

Group:
  Views

Overload:
  spec
)",
        py::arg("base"), py::arg("downsample_factors"), py::arg("method"));
  });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterDownsampleBindings, /*priority=*/-350);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

bool type_caster<tensorstore::DownsampleMethod>::load(handle src,
                                                      bool convert) {
  std::string_view s;
  if (!PyUnicode_Check(src.ptr())) return false;
  if (PyUnicode_READY(src.ptr()) != 0) throw error_already_set();
  Py_ssize_t size;
  const char* data = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
  if (!data) throw error_already_set();
  s = std::string_view(data, size);
  tensorstore::internal_python::ThrowStatusException(
      tensorstore::internal_downsample::DownsampleMethodJsonBinder(
          std::true_type{}, tensorstore::internal_json_binding::NoOptions{},
          &value, &s));
  return true;
}

handle type_caster<tensorstore::DownsampleMethod>::cast(
    tensorstore::DownsampleMethod value, return_value_policy /*policy */,
    handle /*parent*/) {
  std::string_view s;
  tensorstore::internal_python::ThrowStatusException(
      tensorstore::internal_downsample::DownsampleMethodJsonBinder(
          std::false_type{}, tensorstore::internal_json_binding::NoOptions{},
          &value, &s));
  return pybind11::cast(s).release();
}

}  // namespace detail
}  // namespace pybind11
