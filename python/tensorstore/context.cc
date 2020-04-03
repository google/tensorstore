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

#include "python/tensorstore/context.h"

#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/result_type_caster.h"
#include "pybind11/stl.h"
#include "tensorstore/context.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

void RegisterContextBindings(pybind11::module m) {
  py::class_<Context::Spec> cls_context_spec(m, "ContextSpec");
  cls_context_spec
      .def(py::init([](const ::nlohmann::json& json, bool allow_unregistered) {
             return ValueOrThrow(Context::Spec::FromJson(
                 json, AllowUnregistered{allow_unregistered}));
           }),
           "Creates a ContextSpec from a JSON object.", py::arg("json"),
           py::arg("allow_unregistered") = false)
      .def(
          "json",
          [](const Context::Spec& self, bool include_defaults) {
            return self.ToJson(IncludeDefaults{include_defaults});
          },
          py::arg("include_defaults") = false);

  py::class_<Context> cls_context(m, "Context",
                                  R"(
Manages shared TensorStore resources, such as caches and credentials.
)");
  cls_context
      .def(py::init([] { return Context::Default(); }),
           "Returns a default context")
      .def(py::init(
               [](const Context::Spec& spec, std::optional<Context> parent) {
                 return Context(spec, parent.value_or(Context()));
               }),
           "Constructs a context.", py::arg("spec"),
           py::arg("parent") = std::nullopt);
}

}  // namespace internal_python
}  // namespace tensorstore
