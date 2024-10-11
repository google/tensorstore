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
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/box.h"
#include "tensorstore/context.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/json_binding/box.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/json_binding.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep
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

absl::Status TsPrintStorageStatistics(
    Context::Spec context_spec, tensorstore::Spec spec,
    tensorstore::span<std::string_view> args) {
  tensorstore::Context context(context_spec);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto ts,
      tensorstore::Open(spec, context, tensorstore::ReadWriteMode::read,
                        tensorstore::OpenMode::open)
          .result());

  for (const auto& arg : args) {
    tensorstore::JsonAbslFlag<Box<>> box_flag;
    std::string error;
    if (!AbslParseFlag(arg, &box_flag, &error)) {
      std::cerr << "Invalid box: " << arg << ": " << error << std::endl;
      continue;
    }
    auto future = GetStorageStatistics(
        ts | tensorstore::AllDims().BoxSlice(box_flag.value),
        tensorstore::ArrayStorageStatistics::query_not_stored,
        tensorstore::ArrayStorageStatistics::query_fully_stored);
    future.Wait();
    if (!future.status().ok()) {
      std::cerr << "Error getting storage statistics for " << arg << ": "
                << future.status() << std::endl;
      continue;
    }
    const char* txt = "partial";
    if (future.value().not_stored) {
      txt = "not_stored";
    } else if (future.value().fully_stored) {
      txt = "fully_stored";
    }
    std::cout << "Box: " << arg << " " << txt << std::endl;
  }

  return absl::OkStatus();
}

absl::Status RunTsPrintStorageStatistics(Context::Spec context_spec,
                                         CommandFlags flags) {
  tensorstore::JsonAbslFlag<std::optional<tensorstore::Spec>> spec;
  std::vector<Option> options({
      Option{"--spec",
             [&](std::string_view value) {
               std::string error;
               if (!AbslParseFlag(value, &spec, &error)) {
                 return absl::InvalidArgumentError(error);
               }
               return absl::OkStatus();
             }},
  });

  TENSORSTORE_RETURN_IF_ERROR(TryParseOptions(flags, options));

  if (!spec.value) {
    return absl::InvalidArgumentError("print_spec: Must include --spec");
  }

  return TsPrintStorageStatistics(context_spec, *spec.value,
                                  flags.positional_args);
}

}  // namespace cli
}  // namespace tensorstore
