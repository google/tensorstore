// Copyright 2023 The TensorStore Authors
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

#ifndef THIRD_PARTY_PY_TENSORSTORE_TENSORSTORE_MODULE_COMPONENTS_H_
#define THIRD_PARTY_PY_TENSORSTORE_TENSORSTORE_MODULE_COMPONENTS_H_

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <functional>

#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {

using PythonComponentRegistrationCallback =
    std::function<void(pybind11::module_ m, Executor defer)>;

// Registers a callback that defines bindings within the `tensorstore` Python
// module.
//
// Args:
//   callback: Function to be invoked during the ``tensorstore`` Python module
//     initialization.
//   priority: Determine the relative order in which to invoke all registered
//     callbacks.  Callbacks are invoked in order of increasing priority number.
//     Priority is mostly useful for influencing the documentation order.
void RegisterPythonComponent(PythonComponentRegistrationCallback callback,
                             int priority = 0);

// Initializes all Python components.
//
// Should be called once during ``tensorstore._tensorstore`` module
// initialization.
void InitializePythonComponents(pybind11::module_ m);

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_TENSORSTORE_MODULE_COMPONENTS_H_
