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

#include "tensorstore/tscli/copy_command.h"

#include <iostream>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/tscli/command.h"
#include "tensorstore/tscli/lib/kvstore_copy.h"
#include "tensorstore/util/json_absl_flag.h"

/*
Example usage:

rm -rf /tmp/dest
rm -rf /tmp/source
mkdir -p /tmp/source
echo "Hello" > /tmp/source/a
echo "World" > /tmp/source/b

bazel run //tensorstore/tscli -- copy --source file:///tmp/source/
--target file:///tmp/dest/ bazel run //tensorstore/tscli -- ls
file:///tmp/dest/
*/

namespace tensorstore {
namespace cli {
namespace {

static constexpr const char kCommand[] = R"(Copy a kvstore to another kvstore

All values are copied from the --source to the --target kvstore.
)";

static constexpr const char kSource[] = R"(Source kvstore spec. Required.)";

static constexpr const char kTarget[] = R"(Target kvstore spec. Required.)";

}  // namespace

CopyCommand::CopyCommand() : Command("copy", kCommand) {
  parser().AddLongOption("--source", kSource, [this](std::string_view value) {
    tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec> spec;
    std::string error;
    if (!AbslParseFlag(value, &spec, &error)) {
      return absl::InvalidArgumentError(error);
    }
    source_ = spec.value;
    return absl::OkStatus();
  });
  parser().AddLongOption("--target", kTarget, [this](std::string_view value) {
    tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec> spec;
    std::string error;
    if (!AbslParseFlag(value, &spec, &error)) {
      return absl::InvalidArgumentError(error);
    }
    target_ = spec.value;
    return absl::OkStatus();
  });
}

absl::Status CopyCommand::Run(Context::Spec context_spec) {
  if (!source_.valid()) {
    return absl::InvalidArgumentError("Must specify --source");
  }
  if (!target_.valid()) {
    return absl::InvalidArgumentError("Must specify --target");
  }

  tensorstore::Context context(context_spec);

  // TODO: Use positional args as optional keys.
  return KvstoreCopy(context, source_, target_, std::cout);
}

}  // namespace cli
}  // namespace tensorstore
