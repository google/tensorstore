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
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <stdint.h>

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/cord.h"
#include "python/tensorstore/context.h"
#include "python/tensorstore/future.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/kvstore.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/kvstore/ocdbt/distributed/coordinator_server.h"
#include "tensorstore/kvstore/ocdbt/dump_util.h"
#include "tensorstore/kvstore/ocdbt/format/dump.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/json.h"

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

Group:
  OCDBT

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

  defer([cls = MakeCoordinatorServerClass(ocdbt_m), ocdbt_m]() mutable {
    DefineCoordinatorServerAttributes(cls);

    ocdbt_m.def(
        "dump",
        [](PythonKvStoreObject& base, std::optional<std::string> node,
           std::optional<internal_context::ContextImplPtr> context)
            -> PythonFutureWrapper<::nlohmann::json> {
          using internal_ocdbt::LabeledIndirectDataReference;
          std::optional<LabeledIndirectDataReference> node_identifier;
          if (node) {
            node_identifier =
                ValueOrThrow(LabeledIndirectDataReference::Parse(*node));
          }
          auto read_and_dump_future = internal_ocdbt::ReadAndDump(
              base.value, node_identifier,
              context ? WrapImpl(std::move(*context)) : Context());
          return PythonFutureWrapper<::nlohmann::json>(
              MapFutureValue(
                  InlineExecutor{},
                  [](auto& value) {
                    if (auto* cord = std::get_if<absl::Cord>(&value)) {
                      auto cord_copy = *cord;
                      auto s = cord_copy.Flatten();
                      return ::nlohmann::json::binary(
                          std::vector<uint8_t>(s.begin(), s.end()));
                    } else {
                      return std::move(std::get<::nlohmann::json>(value));
                    }
                  },
                  std::move(read_and_dump_future)),
              base.reference_manager());
        },
        py::arg("base"), py::arg("node") = std::nullopt, py::kw_only(),
        py::arg("context") = std::nullopt,
        R"(
Dumps the internal representation of an OCDBT database.

Args:
  base: Base kvstore containing the OCDBT database.

  node: Reference to the node or value to dump, of the form
    ``"<type>:<file-id>:<offset>:<length>"`` where ``<type>`` is one of
    ``"value"``, ``"btreenode"``, or ``"versionnode"``, as specified in a
    ``"location"`` field within the manifest, a B+tree node, or a version node.
    If not specified, the manifest is dumped.

  context: Context from which the :json:schema:`Context.cache_pool` and
    :json:schema:`Context.data_copy_concurrency` resources will be used.  If not
    specified, a new default context is used.

Returns:
  The manifest or node representation as JSON (augmented to include byte
  strings), or the value as a byte string.

Group:
  OCDBT

Examples:
---------

  >>> store = ts.KvStore.open({
  ...     "driver": "ocdbt",
  ...     "config": {
  ...         "max_inline_value_bytes": 1
  ...     },
  ...     "base": "memory://"
  ... }).result()
  >>> store["a"] = b"b"
  >>> store["b"] = b"ce"
  >>> manifest = ts.ocdbt.dump(store.base).result()
  >>> manifest
  {'config': {'compression': {'id': 'zstd'},
              'max_decoded_node_bytes': 8388608,
              'max_inline_value_bytes': 1,
              'uuid': '...',
              'version_tree_arity_log2': 4},
   'version_tree_nodes': [],
   'versions': [{'commit_time': ...,
                 'generation_number': 1,
                 'root': {'statistics': {'num_indirect_value_bytes': 0,
                                         'num_keys': 0,
                                         'num_tree_bytes': 0}},
                 'root_height': 0},
                {'commit_time': ...,
                 'generation_number': 2,
                 'root': {'location': 'btreenode::d/...:0:35',
                          'statistics': {'num_indirect_value_bytes': 0,
                                         'num_keys': 1,
                                         'num_tree_bytes': 35}},
                 'root_height': 0},
                {'commit_time': ...,
                 'generation_number': 3,
                 'root': {'location': 'btreenode::d/...:2:78',
                          'statistics': {'num_indirect_value_bytes': 2,
                                         'num_keys': 2,
                                         'num_tree_bytes': 78}},
                 'root_height': 0}]}
  >>> btree = ts.ocdbt.dump(
  ...     store.base, manifest["versions"][-1]["root"]["location"]).result()
  >>> btree
  {'entries': [{'inline_value': b'b', 'key': b'a'},
               {'indirect_value': 'value::d/...:0:2',
                'key': b'b'}],
   'height': 0}
  >>> ts.ocdbt.dump(store.base,
  ...               btree["entries"][1]["indirect_value"]).result()
  b'ce'

)");
  });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterOcdbtBindings, /*priority=*/100);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore
