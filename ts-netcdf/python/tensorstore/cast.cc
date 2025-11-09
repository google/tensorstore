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

#include "python/tensorstore/data_type.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/spec.h"
#include "python/tensorstore/tensorstore_class.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/cast.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {
namespace {

namespace py = ::pybind11;

void RegisterCastBindings(pybind11::module m, Executor defer) {
  m.def(
      "cast",
      [](PythonTensorStoreObject& base, DataTypeLike target_dtype) {
        return ValueOrThrow(tensorstore::Cast(base.value, target_dtype.value));
      },
      R"(
Returns a read/write view with the data type converted.

Example:

    >>> array = ts.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=ts.float32)
    >>> view = ts.cast(array, ts.uint32)
    >>> view
    TensorStore({
      'base': {
        'array': [1.5, 2.5, 3.5, 4.5, 5.5],
        'driver': 'array',
        'dtype': 'float32',
      },
      'context': {'data_copy_concurrency': {}},
      'driver': 'cast',
      'dtype': 'uint32',
      'transform': {'input_exclusive_max': [5], 'input_inclusive_min': [0]},
    })
    >>> await view.read()
    array([1, 2, 3, 4, 5], dtype=uint32)

Overload:
  store

Group:
  Views
)",
      py::arg("base"), py::arg("dtype"));

  m.def(
      "cast",
      [](PythonSpecObject& base, DataTypeLike target_dtype) {
        return ValueOrThrow(tensorstore::Cast(base.value, target_dtype.value));
      },
      R"(
Returns a view with the data type converted.

Example:

    >>> base = ts.Spec({"driver": "zarr", "kvstore": "memory://"})
    >>> view = ts.cast(base, ts.uint32)
    >>> view
    Spec({
      'base': {'driver': 'zarr', 'kvstore': {'driver': 'memory'}},
      'driver': 'cast',
      'dtype': 'uint32',
    })

Overload:
  spec

Group:
  Views
)",
      py::arg("base"), py::arg("dtype"));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterCastBindings, /*priority=*/-950);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore
