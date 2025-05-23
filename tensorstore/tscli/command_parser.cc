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

#include <stddef.h>
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
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace cli {
namespace {

std::pair<std::string_view, std::string_view> SplitDescription(
    std::string_view description) {
  std::pair<std::string_view, std::string_view> desc_parts =
      absl::StrSplit(description, absl::MaxSplits("\n\n", 1));
  desc_parts.first = absl::StripAsciiWhitespace(desc_parts.first);
  desc_parts.second = absl::StripAsciiWhitespace(desc_parts.second);
  return desc_parts;
}

}  // namespace

void CommandParser::AddLongOption(std::string_view name,
                                  std::string_view description,
                                  ParseLongOption fn) {
  ABSL_CHECK(absl::StartsWith(name, "--"));
  ABSL_CHECK(name.size() > 2);
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
                                      std::string_view description,
                                      ParseLongOption fn) {
  ABSL_CHECK(!positional_.has_value());
  positional_ = PositionalArg{name, description, fn};
}

void CommandParser::PrintHelp(std::ostream& out) {
  // Group options by description.
  absl::flat_hash_map<std::string_view, std::vector<std::string_view>>
      shortdesc_to_name;
  ;
  absl::flat_hash_set<std::string_view> has_longdesc;

  auto build_maps = [&](std::string_view name, std::string_view desc) {
    auto desc_parts = SplitDescription(desc);
    shortdesc_to_name[desc_parts.first].push_back(name);
    if (!desc_parts.second.empty()) {
      has_longdesc.insert(name);
    }
  };
  for (const auto& opt : long_options_) {
    build_maps(opt.longname, opt.description);
  }
  for (const auto& opt : bool_options_) {
    build_maps(opt.boolname, opt.description);
  }

  out << "  " << name();
  for (const auto& desc_and_options : shortdesc_to_name) {
    out << " [" << absl::StrJoin(desc_and_options.second, "/") << "]";
  }
  if (positional_) {
    out << " <" << positional_->name << "...>";
  }

  auto desc_parts = SplitDescription(description());
  out << "\n";
  out << "    " << desc_parts.first << "\n";

  // output options grouped by description.
  if (!shortdesc_to_name.empty()) {
    out << "\n";
  }
  for (const auto& desc_and_options : shortdesc_to_name) {
    out << "    [" << absl::StrJoin(desc_and_options.second, "/") << "] "
        << desc_and_options.first << "\n";
  }
  if (positional_) {
    auto desc_parts = SplitDescription(positional_->description);
    if (!desc_parts.first.empty()) {
      out << "    <" << positional_->name << "...> " << desc_parts.first
          << "\n";
    }
  }

  bool sep_output = false;
  auto maybe_output_sep = [&]() {
    if (sep_output) return;
    out << "\n    Detailed options:\n";
    sep_output = true;
  };

  // Output long argument descriptions.
  auto output_option_long = [&](std::string_view name,
                                std::string_view description) {
    if (!has_longdesc.contains(name)) return;
    auto desc_parts = SplitDescription(description);
    maybe_output_sep();
    out << "    [" << name << "] " << desc_parts.first << "\n";
    if (!desc_parts.second.empty()) {
      for (std::string_view part : absl::StrSplit(desc_parts.second, '\n')) {
        out << "      " << absl::StripAsciiWhitespace(part) << "\n";
      }
      out << "\n";
    }
  };

  // Output long options.
  for (const auto& opt : long_options_) {
    output_option_long(opt.longname, opt.description);
  }
  for (const auto& opt : bool_options_) {
    output_option_long(opt.boolname, opt.description);
  }
  if (positional_) {
    auto desc_parts = SplitDescription(positional_->description);
    if (!desc_parts.second.empty()) {
      maybe_output_sep();
      out << "    <" << positional_->name << "...> " << desc_parts.first
          << "\n";
      for (std::string_view part : absl::StrSplit(desc_parts.second, '\n')) {
        out << "      " << absl::StripAsciiWhitespace(part) << "\n";
      }
      out << "\n";
    }
  }

  // Output long description.
  if (!desc_parts.second.empty()) {
    maybe_output_sep();
    for (std::string_view part : absl::StrSplit(desc_parts.second, '\n'))
      out << "    " << absl::StripAsciiWhitespace(part) << "\n";
    out << "\n";
  }
}

absl::Status CommandParser::TryParse(tensorstore::span<char*> argv,
                                     tensorstore::span<char*> positional_args) {
  // Create a mapping from a single-character short option to presence function.
  absl::flat_hash_map<char, std::function<void()>> shortmapping;
  for (auto& opt : bool_options_) {
    if (absl::StartsWith(opt.boolname, "--")) continue;
    shortmapping[opt.boolname[1]] = opt.found;
  }

  // Stop processing at the first "--" argument.
  const size_t end = [argv]() {
    size_t end = argv.size();
    for (size_t i = 1; i < end; ++i) {
      if (std::string_view(argv[i]) == "--") {
        return i - 1;
      }
    }
    return end;
  }();

  // Keep track of which arguments have been handled.  The positional arguments
  // include values from the argv array, so when a long option is split by a
  // a separator we need to keep track of both.
  absl::flat_hash_set<uintptr_t> handled;
  auto try_parse_long_option = [&](size_t i) -> absl::Status {
    std::string_view it_str = argv[i];

    // Try to parse as a long option.
    if (!absl::StartsWith(it_str, "--")) return absl::OkStatus();

    for (auto& opt : long_options_) {
      if (opt.longname.empty()) continue;
      std::string_view arg = it_str;
      if (!absl::ConsumePrefix(&arg, opt.longname)) continue;

      if (absl::ConsumePrefix(&arg, "=")) {
        // Long option with --name=value.
        handled.insert(reinterpret_cast<uintptr_t>(argv[i]));
        return opt.parse(arg);
      }
      if (!arg.empty()) continue;
      // Long option with --name value.
      if (i + 1 >= end) {
        return absl::InvalidArgumentError(
            absl::StrCat("Expected value for option: ", opt.longname));
      }
      handled.insert(reinterpret_cast<uintptr_t>(argv[i]));
      handled.insert(reinterpret_cast<uintptr_t>(argv[i + 1]));
      std::string_view value = argv[i + 1];
      TENSORSTORE_RETURN_IF_ERROR(opt.parse(value));
      return absl::OkStatus();
    }
    for (auto& opt : bool_options_) {
      if (opt.boolname.empty()) continue;
      std::string_view arg = it_str;
      if (!absl::ConsumePrefix(&arg, opt.boolname)) continue;
      if (arg.empty()) {
        handled.insert(reinterpret_cast<uintptr_t>(argv[i]));
        opt.found();
        return absl::OkStatus();
      }
    }

    return absl::OkStatus();
  };

  // Parse all the arguments.
  for (size_t i = 1; i < end; ++i) {
    if (handled.count(reinterpret_cast<uintptr_t>(argv[i]))) {
      continue;
    }
    std::string_view it_str = argv[i];
    if (absl::StartsWith(it_str, "--")) {
      TENSORSTORE_RETURN_IF_ERROR(try_parse_long_option(i));
    } else if (absl::StartsWith(it_str, "-")) {
      handled.insert(reinterpret_cast<uintptr_t>(argv[i]));
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
  }

  // Handle any remaining positional arguments.
  if (positional_) {
    for (char* x : positional_args) {
      if (handled.count(reinterpret_cast<uintptr_t>(x))) {
        continue;
      }
      TENSORSTORE_RETURN_IF_ERROR(positional_->parse(x));
    }
  }
  return absl::OkStatus();
}

}  // namespace cli
}  // namespace tensorstore
