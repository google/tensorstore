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

#include "tensorstore/tscli/command_parser.h"

#include <stdint.h>

#include <cassert>
#include <functional>
#include <ostream>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/strip.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace cli {

void CommandParser::AddLongOption(std::string_view name,
                                  std::string_view description,
                                  ParseLongOption fn) {
  ABSL_CHECK(absl::StartsWith(name, "--"));
  long_options_.push_back(LongOption{name, description, fn});
}

void CommandParser::AddBoolOption(std::string_view name,
                                  std::string_view description,
                                  ParseBoolOption fn) {
  // A boolean option may start with -- or -; when using - it must be a
  // single letter.
  if (absl::StartsWith(name, "--")) {
    ABSL_CHECK(name.size() > 2);
  } else {
    ABSL_CHECK(absl::StartsWith(name, "-"));
    ABSL_CHECK(name.size() == 2);
  }
  bool_options_.push_back(BoolOption{name, description, fn});
}

void CommandParser::AddPositionalArgs(std::string_view name,
                                      ParseLongOption fn) {
  positional_name_ = name;
  positional_fn_ = std::move(fn);
}

void CommandParser::PrintHelp(std::ostream& out) {
  // Group options by description.
  absl::flat_hash_map<std::string_view, std::vector<std::string_view>>
      desc_to_option;
  for (const auto& opt : bool_options_) {
    desc_to_option[opt.description].push_back(opt.boolname);
  }
  for (const auto& opt : long_options_) {
    desc_to_option[opt.description].push_back(opt.longname);
  }

  out << "  " << name();
  for (const auto& desc_and_options : desc_to_option) {
    out << " [" << absl::StrJoin(desc_and_options.second, "/") << "]";
  }
  if (!positional_name_.empty()) {
    out << " <" << positional_name_ << "...>";
  }
  out << "\n";
  out << "    " << short_description() << "\n\n";

  for (const auto& desc_and_options : desc_to_option) {
    out << "    [" << absl::StrJoin(desc_and_options.second, "/") << "] "
        << desc_and_options.first << "\n";
  }

  // TODO: Print verbose description.

  out << "\n";
}

absl::Status CommandParser::TryParse(tensorstore::span<char*> argv,
                                     tensorstore::span<char*> positional_args) {
  // Create a mapping from a single-character short option to presence function.
  absl::flat_hash_map<char, std::function<void()>> shortmapping;
  for (auto& opt : bool_options_) {
    if (absl::StartsWith(opt.boolname, "--")) continue;
    shortmapping[opt.boolname[1]] = opt.found;
  }

  absl::flat_hash_set<uintptr_t> handled;
  auto it = argv.begin() + 1;
  while (it != argv.end()) {
    bool parsed = false;
    std::string_view it_str = *it;

    if (absl::StartsWith(it_str, "--")) {
      // Try to parse as a long option.
      for (auto& opt : long_options_) {
        if (opt.longname.empty()) continue;
        std::string_view arg = it_str;
        if (!absl::ConsumePrefix(&arg, opt.longname)) continue;
        if (arg.empty()) {
          parsed = true;
          std::string_view value;
          if (it + 1 != argv.end()) {
            value = *(it + 1);
            handled.insert(reinterpret_cast<uintptr_t>(*it));
            it++;
          }
          handled.insert(reinterpret_cast<uintptr_t>(*it));
          TENSORSTORE_RETURN_IF_ERROR(opt.parse(value));
        } else if (absl::ConsumePrefix(&arg, "=")) {
          parsed = true;
          TENSORSTORE_RETURN_IF_ERROR(opt.parse(arg));
          handled.insert(reinterpret_cast<uintptr_t>(*it));
        }
      }
      if (parsed) break;
      for (auto& opt : bool_options_) {
        if (opt.boolname.empty()) continue;
        std::string_view arg = it_str;
        if (!absl::ConsumePrefix(&arg, opt.boolname)) continue;
        if (arg.empty()) {
          parsed = true;
          handled.insert(reinterpret_cast<uintptr_t>(*it));
          opt.found();
          break;
        }
      }
      if (parsed) break;
    } else if (absl::StartsWith(it_str, "-")) {
      // Try to parse as a short option, which may be combined.
      handled.insert(reinterpret_cast<uintptr_t>(*it));
      parsed = true;
      for (int i = 1; i < it_str.size(); ++i) {
        char c = it_str[i];
        assert(c != '-');
        auto it = shortmapping.find(c);
        if (it == shortmapping.end()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Unknown short option: ", it_str));
        }
        it->second();
      }
    }
    it++;
  }

  // Handle any remaining positional arguments.
  if (positional_fn_) {
    for (auto j = positional_args.begin(); j != positional_args.end(); ++j) {
      if (handled.contains(reinterpret_cast<uintptr_t>(j))) {
        continue;
      }
      TENSORSTORE_RETURN_IF_ERROR(positional_fn_(*j));
    }
  }
  return absl::OkStatus();
}

}  // namespace cli
}  // namespace tensorstore
