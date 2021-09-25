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

#ifndef THIRD_PARTY_PY_TENSORSTORE_PYTHON_IMPORTS_H_
#define THIRD_PARTY_PY_TENSORSTORE_PYTHON_IMPORTS_H_

/// \file
///
/// Imports of Python builtin and standard library modules/functions that are
/// required.
///
/// These imports are resolved once during module initialization for efficiency.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

namespace tensorstore {
namespace internal_python {

struct PythonImports {
  pybind11::handle asyncio_module;
  pybind11::handle asyncio_cancelled_error_class;
  pybind11::handle asyncio_get_event_loop_function;
  pybind11::handle asyncio__get_running_loop_function;
  pybind11::handle asyncio_iscoroutine_function;
  pybind11::handle asyncio_run_coroutine_threadsafe_function;

  pybind11::handle atexit_module;
  pybind11::handle atexit_register_function;

  pybind11::handle builtins_module;
  pybind11::handle builtins_range_function;
  pybind11::handle builtins_timeout_error_class;

  pybind11::handle pickle_module;
  pybind11::handle pickle_dumps_function;
  pybind11::handle pickle_loads_function;
};

extern PythonImports python_imports;

/// Must be called exactly once, at the start of the tensorstore module
/// initialization, with the GIL held.
void InitializePythonImports();

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_PYTHON_IMPORTS_H_
