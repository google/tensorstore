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

#include "python/tensorstore/define_heap_type.h"

namespace tensorstore {
namespace internal_python {

void DisallowInstantiationFromPython(pybind11::handle type_object) {
  auto* python_type = reinterpret_cast<PyTypeObject*>(type_object.ptr());
  python_type->tp_new = nullptr;
}

}  // namespace internal_python
}  // namespace tensorstore
