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

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <algorithm>
#include <utility>
#include <vector>

#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {

namespace {
using CallbackWithPriority =
    std::pair<PythonComponentRegistrationCallback, int>;

std::vector<CallbackWithPriority>& GetRegisteredPythonComponents() {
  static internal::NoDestructor<std::vector<CallbackWithPriority>> x;
  return *x;
}
}  // namespace

void RegisterPythonComponent(PythonComponentRegistrationCallback callback,
                             int priority) {
  GetRegisteredPythonComponents().emplace_back(std::move(callback), priority);
}

void InitializePythonComponents(pybind11::module_ m) {
  std::vector<ExecutorTask> deferred_registration_tasks;

  // Executor used to defer definition of functions and methods until after all
  // classes have been registered.
  //
  // pybind11 requires that any class parameter types are registered before the
  // function/method in order to include the correct type name in the generated
  // signature.
  auto defer = [&](ExecutorTask task) {
    deferred_registration_tasks.push_back(std::move(task));
  };

  auto components = GetRegisteredPythonComponents();
  std::sort(components.begin(), components.end(),
            [&](const auto& a, const auto& b) { return a.second < b.second; });

  for (const auto& component : components) {
    component.first(m, defer);
  }

  for (auto& task : deferred_registration_tasks) {
    task();
  }
}

}  // namespace internal_python
}  // namespace tensorstore
