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

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "re2/re2.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/tscli/args.h"
#include "tensorstore/tscli/cli.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace cli {
namespace {

std::string_view NonGlobPrefix(std::string_view glob) {
  auto first_glob_char = glob.find_first_of("*?[]");
  if (first_glob_char == std::string_view::npos) {
    return glob;
  }
  return glob.substr(0, first_glob_char);
}

}  // namespace
absl::Status KvstoreList(Context::Spec context_spec,
                         tensorstore::kvstore::Spec source_spec,
                         tensorstore::span<std::string_view> args) {
  tensorstore::Context context(context_spec);
  TENSORSTORE_ASSIGN_OR_RETURN(auto source,
                               kvstore::Open(source_spec, context).result());

  // Derive the common prefix and a regex that matches all the args
  // to minimize the number of list operations on the kvstore.
  std::string common_prefix;
  std::string re_string;
  for (const std::string_view glob : args) {
    if (glob.empty()) {
      common_prefix.clear();
      re_string.clear();
      break;
    }

    auto non_glob_prefix = NonGlobPrefix(glob);
    auto glob_range = non_glob_prefix.empty()
                          ? KeyRange()
                          : KeyRange::Prefix(std::string(non_glob_prefix));

    if (re_string.empty()) {
      common_prefix = std::string(non_glob_prefix);
      absl::StrAppend(&re_string, "(", GlobToRegex(glob), ")");
      continue;
    }
    if (!common_prefix.empty()) {
      common_prefix = std::string(
          absl::FindLongestCommonPrefix(common_prefix, non_glob_prefix));
    }
    absl::StrAppend(&re_string, "|(", GlobToRegex(glob), ")");
  }

  kvstore::ListOptions options;
  options.range =
      common_prefix.empty() ? KeyRange() : KeyRange::Prefix(common_prefix);
  TENSORSTORE_ASSIGN_OR_RETURN(auto list_entries,
                               kvstore::ListFuture(source, options).result());

  RE2 re2(re_string);
  for (const auto& entry : list_entries) {
    if (re_string.empty() || RE2::FullMatch(entry.key, re2)) {
      std::cout << entry.key << std::endl;
    }
  }
  return absl::OkStatus();
}

absl::Status RunKvstoreList(Context::Spec context_spec, CommandFlags flags) {
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
    if (flags.positional_args.empty()) flags.positional_args.push_back("");
    return KvstoreList(context_spec, *source.value, flags.positional_args);
  } else if (flags.positional_args.empty()) {
    return absl::InvalidArgumentError(
        "list: Must include --source or a sequence of specs");
  }

  absl::Status status;
  std::vector<std::string_view> args({""});
  for (const auto spec : flags.positional_args) {
    auto from_url = kvstore::Spec::FromUrl(spec);
    if (from_url.ok()) {
      status.Update(KvstoreList(context_spec, from_url.value(), args));
      continue;
    }
    tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec> arg_spec;
    std::string error;
    if (AbslParseFlag(spec, &arg_spec, &error)) {
      status.Update(KvstoreList(context_spec, arg_spec.value, args));
      continue;
    }
    std::cerr << "Invalid spec: " << spec << ": " << error << std::endl;
  }
  return status;
}

}  // namespace cli
}  // namespace tensorstore
