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

#include "tensorstore/tscli/lib/ts_search.h"

#include <iostream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace cli {
namespace {

absl::Status TryOpenTensorstore(Context context, kvstore::Spec base_spec,
                                std::string_view driver_name,
                                std::string_view key, bool brief,
                                std::ostream& output) {
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

  if (brief) {
    output << parsed_spec.ToJson()->dump() << std::endl;
  } else {
    output << ts.spec()->ToJson()->dump() << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status TsSearch(Context context, tensorstore::kvstore::Spec source_spec,
                      bool brief, std::ostream& output) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto source,
                               kvstore::Open(source_spec, context).result());

  TENSORSTORE_ASSIGN_OR_RETURN(auto list_entries,
                               kvstore::ListFuture(source, {}).result());

  for (const auto& entry : list_entries) {
    auto [dirname, basename] = internal::PathDirnameBasename(entry.key);

    if (basename == "info.json") {
      TryOpenTensorstore(context, source_spec, "neuroglancer_precomputed",
                         entry.key, brief, output)
          .IgnoreError();
    } else if (basename == "attributes.json") {
      TryOpenTensorstore(context, source_spec, "n5", entry.key, brief, output)
          .IgnoreError();
    } else if (basename == "zarr.json") {
      TryOpenTensorstore(context, source_spec, "zarr3", entry.key, brief,
                         output)
          .IgnoreError();
    } else if (basename == ".zarray") {
      TryOpenTensorstore(context, source_spec, "zarr", entry.key, brief, output)
          .IgnoreError();
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
      TsSearch(context, ocdbt_spec.value(), brief, output).IgnoreError();
    }
  }

  return absl::OkStatus();
}

}  // namespace cli
}  // namespace tensorstore
