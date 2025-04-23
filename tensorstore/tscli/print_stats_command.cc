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

#include "tensorstore/tscli/print_stats_command.h"

#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/box.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/box.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/json_binding.h"  // IWYU pragma: keep
#include "tensorstore/spec.h"
#include "tensorstore/tscli/command.h"
#include "tensorstore/tscli/command_parser.h"
#include "tensorstore/tscli/lib/ts_print_stats.h"
#include "tensorstore/util/json_absl_flag.h"

namespace tensorstore {
namespace cli {

PrintStatsCommand::PrintStatsCommand()
    : Command("print_stats", "Print storage statistics for a TensorStore.") {
  parser().AddBoolOption("--brief", "Brief", [this]() { brief_ = true; });
  parser().AddBoolOption("-b", "Brief", [this]() { brief_ = true; });
  parser().AddBoolOption("--full", "Full", [this]() { brief_ = false; });
  parser().AddBoolOption("-f", "Full", [this]() { brief_ = false; });

  parser().AddLongOption(
      "--spec", "Tensorstore spec", [this](std::string_view value) {
        tensorstore::JsonAbslFlag<std::optional<tensorstore::Spec>> spec;
        std::string error;
        if (!AbslParseFlag(value, &spec, &error)) {
          return absl::InvalidArgumentError(error);
        }
        if (!spec.value) {
          return absl::InvalidArgumentError("Must specify --spec");
        }
        has_spec_ = true;
        specs_.push_back(*spec.value);
        return absl::OkStatus();
      });

  parser().AddPositionalArgs("spec/box", [this](std::string_view value) {
    if (has_spec_) {
      tensorstore::JsonAbslFlag<Box<>> box_flag;
      std::string error;
      if (!AbslParseFlag(value, &box_flag, &error)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid box: ", value, " ", error));
      }
      boxes_.push_back(std::move(box_flag).value);
      return absl::OkStatus();
    }
    tensorstore::JsonAbslFlag<tensorstore::Spec> arg_spec;
    std::string error;
    if (!AbslParseFlag(value, &arg_spec, &error)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid spec: ", value, " ", error));
    }
    specs_.push_back(arg_spec.value);
    return absl::OkStatus();
  });
}

absl::Status PrintStatsCommand::Run(Context::Spec context_spec) {
  if (specs_.empty()) {
    return absl::InvalidArgumentError(
        "print_stats: Must include --spec or a sequence of specs");
  }

  tensorstore::Context context(context_spec);

  absl::Status status;
  for (const auto& spec : specs_) {
    if (boxes_.empty()) {
      status.Update(TsPrintStoredChunks(context, spec, brief_, std::cout));
    } else {
      status.Update(
          TsPrintStorageStatistics(context, spec, boxes_, brief_, std::cout));
    }
  }

  return status;
}

}  // namespace cli
}  // namespace tensorstore
