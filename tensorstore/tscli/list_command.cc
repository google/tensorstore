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

#include "tensorstore/tscli/list_command.h"

#include <stddef.h>
#include <stdint.h>

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/tscli/command.h"
#include "tensorstore/tscli/lib/kvstore_list.h"
#include "tensorstore/util/json_absl_flag.h"

namespace tensorstore {
namespace cli {
namespace {

std::string FormatBytesHumanReadable(int64_t num_bytes) {
  if (num_bytes < 0) return "";

  // Special case for bytes.
  if (num_bytes < 1024) {
    return absl::StrFormat("%dB", num_bytes);
  }

  // Discard the least significant bits when finding SI prefix.
  static const char units[] = "KMGTPE";
  const char* unit = units;
  while (num_bytes >= 1024 * 1024) {
    num_bytes /= static_cast<int64_t>(1024);
    ++unit;
  }

  std::string result = absl::StrFormat("%3f", num_bytes / 1024.0);
  std::string_view result_view = result;
  while (absl::EndsWith(result_view, "0")) result_view.remove_suffix(1);
  if (absl::EndsWith(result_view, ".")) result_view.remove_suffix(1);
  return absl::StrFormat("%s%ciB", result_view, *unit);
}

// Formats a list entry for printing according to the brief/human_readable
// flags.
struct Formatter {
  bool brief = false;
  bool human_readable = false;

  std::string operator()(size_t width, const kvstore::ListEntry& entry) const {
    if (brief || entry.size < 0) {
      return absl::StrFormat("%s", entry.key);
    }
    if (!human_readable) {
      return absl::StrFormat("%-*s%d", width + 2, entry.key, entry.size);
    }
    return absl::StrFormat("%-*s%s", width + 2, entry.key,
                           FormatBytesHumanReadable(entry.size));
  }
};

}  // namespace

ListCommand::ListCommand() : Command("list", "List the contents of a kvstore") {
  AddAlias("ls");

  auto set_brief = [this]() {
    this->brief = true;
    this->human_readable = false;
  };
  auto set_human_readable = [this]() {
    this->human_readable = true;
    this->brief = false;
  };

  parser().AddBoolOption("-h", "Human readable", set_human_readable);
  parser().AddBoolOption("--human", "Human readable", set_human_readable);
  parser().AddBoolOption("-b", "Brief", set_brief);
  parser().AddBoolOption("--brief", "Brief", set_brief);

  parser().AddLongOption(
      "--source", "Source kvstore spec", [this](std::string_view value) {
        tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec> spec;
        std::string error;
        if (!AbslParseFlag(value, &spec, &error)) {
          return absl::InvalidArgumentError(
              absl::StrCat("Invalid spec: ", value, " ", error));
        }
        has_source_ = true;
        sources_.push_back(spec.value);
        return absl::OkStatus();
      });

  parser().AddLongOption("--match", "Glob matching patterns",
                         [this](std::string_view value) {
                           match_args_.push_back(value);
                           return absl::OkStatus();
                         });

  parser().AddPositionalArgs("spec/match", [this](std::string_view value) {
    if (has_source_) {
      match_args_.push_back(value);
      return absl::OkStatus();
    }
    tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec> spec;
    std::string error;
    if (!AbslParseFlag(value, &spec, &error)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid spec: ", value, " ", error));
    }
    sources_.push_back(spec.value);
    return absl::OkStatus();
  });
}

absl::Status ListCommand::Run(Context::Spec context_spec) {
  if (sources_.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat(name(), ": Must include --source or a sequence of specs"));
  }

  tensorstore::Context context(context_spec);

  if (match_args_.empty()) {
    match_args_.push_back("");
  }

  absl::Status status;
  for (const auto& spec : sources_) {
    status.Update(KvstoreList(context, spec, match_args_,
                              Formatter{brief, human_readable}, std::cout));
  }
  return status;
}

}  // namespace cli
}  // namespace tensorstore
