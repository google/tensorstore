// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/tscli/ocdbt_check_command.h"

#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/tscli/command.h"
#include "tensorstore/tscli/lib/ocdbt_check.h"
#include "tensorstore/util/json_absl_flag.h"

namespace tensorstore {
namespace cli {
namespace {

static constexpr const char kCommand[] =
    R"(Check structural integrity of an OCDBT database

By default, this command traverses and validates all versions present in the database.
If --version is specified, only that specific version is validated.
)";

static constexpr const char kSource[] = R"(Source kvstore spec. Required.

kvstore spec must refer to a prefix/directory containing an OCDBT database.
)";

static constexpr const char kVersion[] = R"(Optional version to check.

Can be a generation number (e.g., `v1`) or a timestamp (e.g., `2023-01-01T00:00:00Z`).
)";

}  // namespace

OcdbtCheckCommand::OcdbtCheckCommand() : Command("ocdbt_check", kCommand) {
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

  parser().AddLongOption("--version", kVersion, [this](std::string_view value) {
    options_.version = std::string(value);
    return absl::OkStatus();
  });

  parser().AddBoolOption(
      "--detailed",
      "Output detailed lists of orphaned files and unused ranges.",
      [this]() { options_.detailed = true; });

  parser().AddLongOption(
      "--alignment",
      "Byte alignment used when calculating unused ranges. Defaults to 4096. "
      "Set to 1 to find raw gaps.",
      [this](std::string_view value) {
        if (!absl::SimpleAtoi(value, &options_.alignment) ||
            options_.alignment == 0) {
          return absl::InvalidArgumentError("Invalid alignment value");
        }
        return absl::OkStatus();
      });

  parser().AddLongOption(
      "--read-concurrency", "Limit on concurrent node reads during check.",
      [this](std::string_view value) {
        if (!absl::SimpleAtoi(value, &options_.concurrency) ||
            options_.concurrency == 0) {
          return absl::InvalidArgumentError("Invalid concurrency value");
        }
        return absl::OkStatus();
      });
}

absl::Status OcdbtCheckCommand::Run(Context::Spec context_spec) {
  tensorstore::Context context(context_spec);

  if (specs_.empty()) {
    return absl::InvalidArgumentError("Must specify --source");
  }

  absl::Status status;
  for (const auto& spec : specs_) {
    status.Update(OcdbtCheck(context, spec, std::cout, options_));
  }
  return status;
}

}  // namespace cli
}  // namespace tensorstore
