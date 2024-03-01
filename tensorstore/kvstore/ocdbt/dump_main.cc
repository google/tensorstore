// Copyright 2022 The TensorStore Authors
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

#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include <nlohmann/json.hpp>
#include "absl/flags/parse.h"
#include "tensorstore/internal/json/pprint_python.h"
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/dump_util.h"
#include "tensorstore/kvstore/ocdbt/format/dump.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"

ABSL_FLAG(tensorstore::JsonAbslFlag<std::optional<tensorstore::kvstore::Spec>>,
          kvstore, std::nullopt, "Underlying kvstore");
ABSL_FLAG(std::string, node, "", "Node/value to dump");

namespace tensorstore {
namespace internal_ocdbt {

namespace {
absl::Status RunDumpCommand() {
  auto kvs_spec = absl::GetFlag(FLAGS_kvstore).value;
  if (!kvs_spec) {
    return absl::InvalidArgumentError("Must specify --kvstore");
  }
  internal::EnsureDirectoryPath(kvs_spec->path);
  TENSORSTORE_ASSIGN_OR_RETURN(auto kvs, kvstore::Open(*kvs_spec).result());

  std::optional<LabeledIndirectDataReference> node_identifier;
  if (auto node_identifier_s = absl::GetFlag(FLAGS_node);
      !node_identifier_s.empty()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        node_identifier,
        LabeledIndirectDataReference::Parse(node_identifier_s));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto dump_result,
                               ReadAndDump(kvs, node_identifier).result());
  if (auto* raw = std::get_if<absl::Cord>(&dump_result)) {
    std::cout << *raw;
  } else {
    std::cout << internal_python::PrettyPrintJsonAsPython(
                     std::get<::nlohmann::json>(dump_result))
              << std::endl;
  }
  return absl::OkStatus();
}
}  // namespace
}  // namespace internal_ocdbt
}  // namespace tensorstore

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);  // InitTensorstore
  auto status = tensorstore::internal_ocdbt::RunDumpCommand();
  if (!status.ok()) {
    std::cerr << status << std::endl;

    if (absl::IsInvalidArgument(status)) {
      std::cerr << "Usage: " << argv[0]
                << " --kvstore <kvstore-json-spec> [--node "
                   "<type>:<file-id>:<offset>:<length> ]"
                << "\n";
      std::cerr << R"(
The kvstore must refer to a prefix/directory containing an OCDBT database.

If `--node` is omitted, the manifest is printed to stdout.

Otherwise, the node <type> must be one of `value`, `btreenode`, or
`versionnode`, as specified in a `location` field within the manifest, a
btreenode, or a versionnode, and the corresponding node data is printed to
stdout.

When printing a manifest, a btreenode, or a versionnode, the decoded value is
printed as a Python literal, rather than JSON, to accommodate byte strings (such
as keys), which cannot be directly represented as JSON (as JSON only supports
Unicode strings).

For a node type of `value`, the raw data is just printed directly to stdout.

Example usage:

rm -rf /tmp/source
rm -rf /tmp/ocdbt
mkdir -p /tmp/source
echo "Hello" > /tmp/source/a
echo "World" > /tmp/source/b

bazel run //tensorstore/kvstore:copy -- --source '"file:///tmp/source/"' --target '{"driver":"ocdbt","base":"file:///tmp/ocdbt/"}'

bazel run //tensorstore/kvstore/ocdbt:dump -- --kvstore '"file:///tmp/ocdbt/"'

echo "Test" > /tmp/source/c
bazel run //tensorstore/kvstore:copy -- --source '"file:///tmp/source/c"' --target '{"driver":"ocdbt","base":"file:///tmp/ocdbt/","path":"c"}'

bazel run //tensorstore/kvstore/ocdbt:dump -- --kvstore '"file:///tmp/ocdbt/"'

)";
      std::cerr << std::flush;
    }
    return 1;
  }
  return 0;
}
