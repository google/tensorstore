// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/tscli/ocdbt_dump_command.h"

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/tscli/command.h"
#include "tensorstore/tscli/lib/ocdbt_dump.h"
#include "tensorstore/util/json_absl_flag.h"

/*
Example usage:

rm -rf /tmp/ocdbt
rm -rf /tmp/source
mkdir -p /tmp/source
echo "Hello" > /tmp/source/a
echo "World" > /tmp/source/b

bazel run //tensorstore/tscli -- copy --source file:///tmp/source/
--target '{"driver":"ocdbt","base":"file:///tmp/ocdbt/"}' bazel run
//tensorstore/tscli -- ls 'file:///tmp/ocdbt/|ocdbt:' bazel run
//tensorstore/tscli -- ocdbt_dump --source file:///tmp/ocdbt/

echo "Test" > /tmp/source/c

bazel run //tensorstore/tscli -- copy --source file:///tmp/source/c
--target '{"driver":"ocdbt","base":"file:///tmp/ocdbt/","path":"c"}' bazel run
//tensorstore/tscli -- ocdbt_dump --source file:///tmp/ocdbt/
*/

namespace tensorstore {
namespace cli {
namespace {

static constexpr const char kCommand[] =
    R"(Dump ocdbt nodes for a tensorstore kvstore


When printing a manifest, a btreenode, or a versionnode, the decoded value is
printed as a Python literal, rather than JSON, to accommodate byte strings (such
as keys), which cannot be directly represented as JSON (as JSON only supports
Unicode strings).

For a node type of `value`, the raw data is just printed directly to stdout.
)";

static constexpr const char kSource[] = R"(Source kvstore spec. Required.

kvstore spec must refer to a prefix/directory containing an OCDBT database.
)";

static constexpr const char kNode[] = R"(OCDBT nodes to dump. Optional.

If no node is provided, the manifest is printed.

Otherwise, the node <type> must be one of `value`, `btreenode`, or
`versionnode`, as specified in a `location` field within the manifest, a
btreenode, or a versionnode, and the corresponding node data is printed.
)";

}  // namespace

OcdbtDumpCommand::OcdbtDumpCommand() : Command("ocdbt_dump", kCommand) {
  parser().AddLongOption("--source", kSource, [this](std::string_view value) {
    tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec> spec;
    std::string error;
    if (!AbslParseFlag(value, &spec, &error)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid spec: ", value, " ", error));
    }
    specs_.push_back(spec.value);
    return absl::OkStatus();
  });

  parser().AddPositionalArgs("node", kNode, [this](std::string_view value) {
    nodes_.push_back(std::string(value));
    return absl::OkStatus();
  });
}

absl::Status OcdbtDumpCommand::Run(Context::Spec context_spec) {
  tensorstore::Context context(context_spec);

  if (specs_.empty()) {
    return absl::InvalidArgumentError("Must specify --source");
  }
  std::vector<std::string_view> nodes(nodes_.begin(), nodes_.end());

  absl::Status status;
  for (const auto& spec : specs_) {
    status.Update(OcdbtDump(context, spec, nodes, std::cout));
  }
  return status;
}

}  // namespace cli
}  // namespace tensorstore
