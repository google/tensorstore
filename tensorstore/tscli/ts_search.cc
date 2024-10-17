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
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include <nlohmann/json_fwd.hpp>
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/json_binding/box.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/json_binding.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/tscli/args.h"
#include "tensorstore/tscli/cli.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace cli {
namespace {

absl::Status TryOpenTensorstore(Context context, kvstore::Spec base_spec,
                                std::string_view driver_name,
                                std::string_view key) {
  auto [dirname, basename] = internal::PathDirnameBasename(key);
  if (!dirname.empty()) {
    base_spec.AppendPathComponent(dirname);
  }
  if (!absl::EndsWith(base_spec.path, "/")) {
    base_spec.AppendSuffix("/");
  }

  // Adjust the base_spec to be an ocdbt spec.
  TENSORSTORE_ASSIGN_OR_RETURN(auto base_json, base_spec.ToJson());
  ::nlohmann::json json{
      {"driver", std::string(driver_name)},
      {"kvstore", std::move(base_json)},
  };
  TENSORSTORE_ASSIGN_OR_RETURN(auto parsed_spec,
                               tensorstore::Spec::FromJson(json));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto ts,
      tensorstore::Open(parsed_spec, context, tensorstore::ReadWriteMode::read,
                        tensorstore::OpenMode::open)
          .result());

  std::cout << json.dump() << std::endl;

  return absl::OkStatus();
}

}  // namespace

absl::Status TsSearch(Context::Spec context_spec,
                      tensorstore::kvstore::Spec source_spec) {
  tensorstore::Context context(context_spec);

  TENSORSTORE_ASSIGN_OR_RETURN(auto source,
                               kvstore::Open(source_spec, context).result());

  TENSORSTORE_ASSIGN_OR_RETURN(auto list_entries,
                               kvstore::ListFuture(source, {}).result());

  for (const auto& entry : list_entries) {
    auto [dirname, basename] = internal::PathDirnameBasename(entry.key);

    if (basename == "info.json") {
      TryOpenTensorstore(context, source_spec, "neuroglancer_precomputed",
                         entry.key)
          .IgnoreError();
    } else if (basename == "metadata.json") {
      TryOpenTensorstore(context, source_spec, "n5", entry.key).IgnoreError();
    } else if (basename == "zarr.json") {
      TryOpenTensorstore(context, source_spec, "zarr3", entry.key)
          .IgnoreError();
    } else if (absl::EndsWith(basename, "zarray")) {
      TryOpenTensorstore(context, source_spec, "zarr", entry.key).IgnoreError();
    } else if (absl::StrContains(basename, "manifest.ocdbt")) {
      // Found an ocdbt manifest file. try to recurse into it.
      kvstore::Spec base_spec = source_spec;
      base_spec.AppendPathComponent(dirname);
      if (!absl::EndsWith(base_spec.path, "/")) {
        base_spec.AppendSuffix("/");
      }
      auto base_json = base_spec.ToJson();
      if (!base_json.ok()) continue;
      if (auto* obj = base_json->get_ptr<nlohmann::json::object_t*>();
          obj != nullptr) {
        if (auto it = obj->find("driver");
            it != obj->end() && it->second.get<std::string>() == "ocdbt") {
          // Already an ocdbt driver;
          continue;
        }
      }
      auto ocdbt_spec = kvstore::Spec::FromJson({
          {"driver", "ocdbt"},
          {"base", std::move(base_json.value())},
      });
      if (!ocdbt_spec.ok()) continue;
      TsSearch(context_spec, ocdbt_spec.value()).IgnoreError();
    }
  }

  return absl::OkStatus();
}

absl::Status RunTsSearch(Context::Spec context_spec, CommandFlags flags) {
  tensorstore::JsonAbslFlag<std::optional<tensorstore::kvstore::Spec>> source;
  std::vector<LongOption> options({
      LongOption{"--source",
                 [&](std::string_view value) {
                   std::string error;
                   if (!AbslParseFlag(value, &source, &error)) {
                     return absl::InvalidArgumentError(error);
                   }
                   return absl::OkStatus();
                 }},
  });

  TENSORSTORE_RETURN_IF_ERROR(TryParseOptions(flags, options));

  if (source.value) {
    return TsSearch(context_spec, *source.value);
  }
  if (flags.positional_args.empty()) {
    return absl::InvalidArgumentError(
        "search: Must include --source or a sequence of specs");
  }
  absl::Status status;
  for (const std::string_view spec : flags.positional_args) {
    auto from_url = kvstore::Spec::FromUrl(spec);
    if (from_url.ok()) {
      status.Update(TsSearch(context_spec, from_url.value()));
      continue;
    }
    tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec> arg_spec;
    std::string error;
    if (AbslParseFlag(spec, &arg_spec, &error)) {
      status.Update(TsSearch(context_spec, arg_spec.value));
      continue;
    }
    std::cerr << "Invalid spec: " << spec << ": " << error << std::endl;
  }
  return status;
}

}  // namespace cli
}  // namespace tensorstore
