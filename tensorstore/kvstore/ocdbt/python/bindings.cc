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

#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/kvstore/ocdbt/distributed/coordinator_server.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {
namespace {

namespace py = ::pybind11;

using ::tensorstore::ocdbt::CoordinatorServer;
using CoordinatorServerClass = py::class_<CoordinatorServer>;

CoordinatorServerClass MakeCoordinatorServerClass(py::module m) {
  return CoordinatorServerClass(m, "DistributedCoordinatorServer", R"(
Distributed coordinator server for the OCDBT (Optionally-Cooperative Distributed
B+Tree) database.

Example:

    >> server = ts.ocdbt.DistributedCoordinatorServer()

)");
}

void DefineCoordinatorServerAttributes(CoordinatorServerClass& cls) {
  cls.def(py::init([](::nlohmann::json json_spec) -> CoordinatorServer {
            CoordinatorServer::Options options;
            options.spec = ValueOrThrow(
                CoordinatorServer::Spec::FromJson(std::move(json_spec)));
            return ValueOrThrow(CoordinatorServer::Start(std::move(options)));
          }),
          py::arg("json") = ::nlohmann::json(::nlohmann::json::object_t()));

  cls.def_property_readonly(
      "port", [](CoordinatorServer& self) -> int { return self.port(); });
}

void RegisterOcdbtBindings(py::module m, Executor defer) {
  auto ocdbt_m = m.def_submodule("ocdbt");

  defer([cls = MakeCoordinatorServerClass(ocdbt_m)]() mutable {
    DefineCoordinatorServerAttributes(cls);
  });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterOcdbtBindings, /*priority=*/100);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore
