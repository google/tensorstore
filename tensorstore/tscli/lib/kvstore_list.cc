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

#include "tensorstore/tscli/lib/kvstore_list.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "re2/re2.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/tscli/lib/glob_to_regex.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

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

absl::Status KvstoreList(
    Context context, tensorstore::kvstore::Spec source_spec,
    tensorstore::span<std::string_view> match_args,
    absl::FunctionRef<std::string(size_t, const kvstore::ListEntry&)> formatter,
    std::ostream& output) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto source,
                               kvstore::Open(source_spec, context).result());

  // Derive the common prefix and a regex that matches all the args
  // to minimize the number of list operations on the kvstore.
  std::string common_prefix;
  std::string re_string;
  for (const std::string_view glob : match_args) {
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

  size_t max_width = 0;
  RE2 re2(re_string);
  for (const auto& entry : list_entries) {
    if (re_string.empty() || RE2::FullMatch(entry.key, re2)) {
      max_width = std::max(max_width, entry.key.size());
    }
  }
  for (const auto& entry : list_entries) {
    if (re_string.empty() || RE2::FullMatch(entry.key, re2)) {
      output << formatter(max_width, entry);
    }
  }

  return absl::OkStatus();
}

}  // namespace cli
}  // namespace tensorstore
