// Copyright 2024 The TensorStore Authors
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
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/json_binding/json_binding.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/tscli/args.h"
#include "tensorstore/tscli/cli.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace cli {

absl::Status TsPrintSpec(Context context, tensorstore::Spec spec,
                         IncludeDefaults include_defaults) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto ts,
      tensorstore::Open(spec, context, tensorstore::ReadWriteMode::read,
                        tensorstore::OpenMode::open)
          .result());
  TENSORSTORE_ASSIGN_OR_RETURN(auto actual_spec, ts.spec());
  TENSORSTORE_ASSIGN_OR_RETURN(auto json_spec,
                               actual_spec.ToJson(include_defaults));
  std::cout << json_spec.dump() << std::endl;
  return absl::OkStatus();
}

absl::Status RunTsPrintSpec(Context::Spec context_spec, CommandFlags flags) {
  tensorstore::JsonAbslFlag<std::optional<tensorstore::Spec>> spec;
  IncludeDefaults include_defaults(false);

  std::vector<LongOption> long_options({
      LongOption{"--spec",
                 [&](std::string_view value) {
                   std::string error;
                   if (!AbslParseFlag(value, &spec, &error)) {
                     return absl::InvalidArgumentError(error);
                   }
                   return absl::OkStatus();
                 }},
  });
  std::vector<BoolOption> bool_options({
      BoolOption{"--include_defaults",
                 [&]() { include_defaults = IncludeDefaults(true); }},
  });

  TENSORSTORE_RETURN_IF_ERROR(
      TryParseOptions(flags, long_options, bool_options));

  tensorstore::Context context(context_spec);

  if (spec.value) {
    return TsPrintSpec(context, *spec.value, include_defaults);
  }
  if (flags.positional_args.empty()) {
    return absl::InvalidArgumentError(
        "print_spec: Must include --spec or a sequence of specs");
  }

  absl::Status status;
  for (const std::string_view spec : flags.positional_args) {
    tensorstore::JsonAbslFlag<tensorstore::Spec> arg_spec;
    std::string error;
    if (AbslParseFlag(spec, &arg_spec, &error)) {
      status.Update(TsPrintSpec(context, arg_spec.value, include_defaults));
      continue;
    }
    std::cerr << "Invalid spec: " << spec << ": " << error << std::endl;
  }
  return status;
}

}  // namespace cli
}  // namespace tensorstore
