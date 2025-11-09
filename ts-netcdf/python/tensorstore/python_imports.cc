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

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "python/tensorstore/python_imports.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

PythonImports python_imports;

void InitializePythonImports() {
  auto& i = python_imports;

  i.asyncio_module = py::module_::import("asyncio").release();
  i.asyncio_cancelled_error_class =
      py::object(i.asyncio_module.attr("CancelledError")).release();
  i.asyncio_get_event_loop_function =
      py::object(i.asyncio_module.attr("get_event_loop")).release();
  i.asyncio__get_running_loop_function =
      py::object(i.asyncio_module.attr("_get_running_loop")).release();
  i.asyncio_iscoroutine_function =
      py::object(i.asyncio_module.attr("iscoroutine")).release();
  i.asyncio_run_coroutine_threadsafe_function =
      py::object(i.asyncio_module.attr("run_coroutine_threadsafe")).release();

  i.atexit_module = py::module_::import("atexit").release();
  i.atexit_register_function =
      py::object(i.atexit_module.attr("register")).release();

  i.builtins_module = py::module_::import("builtins").release();
  i.builtins_range_function =
      py::object(i.builtins_module.attr("range")).release();
  i.builtins_timeout_error_class =
      py::object(i.builtins_module.attr("TimeoutError")).release();

  i.pickle_module = py::module_::import("pickle").release();
  i.pickle_dumps_function = py::object(i.pickle_module.attr("dumps")).release();
  i.pickle_loads_function = py::object(i.pickle_module.attr("loads")).release();
}

}  // namespace internal_python
}  // namespace tensorstore
