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

#include "tensorstore/tscli/print_spec_command.h"

#include <iostream>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/context.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/spec.h"
#include "tensorstore/tscli/command.h"
#include "tensorstore/tscli/lib/ts_print_spec.h"
#include "tensorstore/util/json_absl_flag.h"

namespace tensorstore {
namespace cli {

PrintSpecCommand::PrintSpecCommand()
    : Command("print_spec", "Print the spec of a TensorStore") {
  parser().AddBoolOption("--include_defaults", "Include defaults", [this]() {
    include_defaults_ = IncludeDefaults(true);
  });
  parser().AddBoolOption("--noinclude_defaults", "Include defaults", [this]() {
    include_defaults_ = IncludeDefaults(false);
  });

  auto parse_spec = [this](std::string_view value) {
    tensorstore::JsonAbslFlag<tensorstore::Spec> arg_spec;
    std::string error;
    if (!AbslParseFlag(value, &arg_spec, &error)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid spec: ", value, " ", error));
    }
    specs_.push_back(arg_spec.value);
    return absl::OkStatus();
  };

  parser().AddLongOption("--spec", "Tensorstore spec", parse_spec);
  parser().AddPositionalArgs("tensorstore spec", parse_spec);
}

absl::Status PrintSpecCommand::Run(Context::Spec context_spec) {
  if (specs_.empty()) {
    return absl::InvalidArgumentError("Must specify --spec");
  }

  tensorstore::Context context(context_spec);

  absl::Status status;
  for (const auto& spec : specs_) {
    status.Update(TsPrintSpec(context, spec, include_defaults_, std::cout));
  }
  return status;
}

}  // namespace cli
}  // namespace tensorstore
