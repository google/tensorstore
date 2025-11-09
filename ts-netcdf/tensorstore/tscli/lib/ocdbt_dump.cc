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

#include "tensorstore/tscli/lib/ocdbt_dump.h"

#include <optional>
#include <ostream>
#include <string_view>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/json/pprint_python.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/dump_util.h"
#include "tensorstore/kvstore/ocdbt/format/dump.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

using ::tensorstore::internal_ocdbt::LabeledIndirectDataReference;
using ::tensorstore::internal_ocdbt::ReadAndDump;

namespace tensorstore {
namespace cli {

absl::Status OcdbtDump(Context context, tensorstore::kvstore::Spec source_spec,
                       tensorstore::span<std::string_view> nodes,
                       std::ostream& output) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto kvs, kvstore::Open(source_spec).result());

  if (!nodes.empty()) {
    absl::Status final_status;
    for (std::string_view node : nodes) {
      auto parsed_node = LabeledIndirectDataReference::Parse(node);
      if (!parsed_node.ok()) {
        final_status.Update(parsed_node.status());
        continue;
      }
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto dump_result,
          ReadAndDump(kvs, parsed_node.value(), context).result());
      if (auto* raw = std::get_if<absl::Cord>(&dump_result)) {
        output << *raw;
      } else {
        output << internal_python::PrettyPrintJsonAsPython(
                      std::get<::nlohmann::json>(dump_result))
               << std::endl;
      }
    }
    return final_status;
  }

  // No nodes specified, dump the root metadata.
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto dump_result, ReadAndDump(kvs, std::nullopt, context).result());
  if (auto* raw = std::get_if<absl::Cord>(&dump_result)) {
    output << *raw;
  } else {
    output << internal_python::PrettyPrintJsonAsPython(
                  std::get<::nlohmann::json>(dump_result))
           << std::endl;
  }
  return absl::OkStatus();
}

}  // namespace cli
}  // namespace tensorstore
