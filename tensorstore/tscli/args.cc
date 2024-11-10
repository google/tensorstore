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

#include "tensorstore/tscli/args.h"

#include <stdint.h>

#include <cassert>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace cli {

absl::Status TryParseOptions(CommandFlags& flags,
                             tensorstore::span<LongOption> long_options,
                             tensorstore::span<BoolOption> bool_options) {
  absl::flat_hash_set<uintptr_t> handled;
  absl::flat_hash_map<char, std::function<void()>> shortmapping;
  for (auto& opt : bool_options) {
    if (absl::StartsWith(opt.boolname, "--")) continue;
    // A Short option may start with -- or -; when using - it must be a single
    // letter.
    assert(absl::StartsWith(opt.boolname, "-"));
    assert(opt.boolname.size() == 2);
    shortmapping[opt.boolname[1]] = opt.found;
  }

  auto it = flags.argv.begin() + 1;
  while (it != flags.argv.end()) {
    bool parsed = false;
    std::string_view it_str = *it;

    if (absl::StartsWith(it_str, "--")) {
      // Try to parse as a long option.
      for (auto& opt : long_options) {
        if (opt.longname.empty()) continue;
        std::string_view arg = it_str;
        if (!absl::ConsumePrefix(&arg, opt.longname)) continue;
        if (arg.empty()) {
          parsed = true;
          std::string_view value;
          if (it + 1 != flags.argv.end()) {
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
      for (auto& opt : bool_options) {
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

  // Erase any additionally used values from positional args.
  auto i = flags.positional_args.begin();
  for (auto j = flags.positional_args.begin(); j != flags.positional_args.end();
       ++j) {
    if (handled.contains(reinterpret_cast<uintptr_t>(j->data()))) {
      continue;
    }
    *i++ = *j;
  }
  flags.positional_args.erase(i, flags.positional_args.end());
  return absl::OkStatus();
}

std::string GlobToRegex(std::string_view glob) {
  std::string re;
  re.reserve(glob.size() * 2);
  re.append("^");

  while (!glob.empty()) {
    char c = glob[0];
    glob.remove_prefix(1);
    switch (c) {
      case '*': {
        bool is_star_star = false;
        while (!glob.empty() && glob[0] == '*') {
          glob.remove_prefix(1);
          is_star_star = true;
        }
        // TODO: Handle **? / **/, etc.
        if (is_star_star) {
          absl::StrAppend(&re, ".*");
        } else {
          absl::StrAppend(&re, "[^/]*");
        }
        break;
      }
      case '?':
        absl::StrAppend(&re, "[^/]");
        break;
      case '[': {
        if (glob.size() < 2 || glob.find(']', 1) == std::string_view::npos) {
          // Literal [
          absl::StrAppend(&re, "\\[");
          break;
        }
        re.push_back('[');
        bool is_exclude = false;
        if (glob[0] == '!' || glob[0] == '^') {
          is_exclude = true;
          re.push_back('^');
          re.push_back('/');
          glob.remove_prefix(1);
        }
        // Copy the characters.
        while (glob[0] != ']') {
          if (glob[0] == '[' && glob[1] == ':') {
            // Escape '[' to avoid character classes.
            absl::StrAppend(&re, "\\[");
            glob.remove_prefix(1);
          } else if (glob[1] != '-' || glob[2] == ']') {
            // Not a range, so copy the character unless it is '/'.
            if (glob[0] != '/') re.push_back(glob[0]);
            glob.remove_prefix(1);
          } else if (!is_exclude && glob[0] <= '/' && '/' <= glob[2]) {
            // Make sure that the included range does not contain '/'.
            //
            // NOTE: "/-/" is dropped entirely, which it should,
            // because by definition there is no matching pathname.
            if (glob[0] < '/') {
              re.push_back(glob[0]);
              re.push_back('-');
              re.push_back('/' - 1);
            }
            if ('/' < glob[2]) {
              re.push_back('/' + 1);
              re.push_back('-');
              re.push_back(glob[2]);
            }
            glob.remove_prefix(3);
          } else {
            // Range will not match '/', so copy it blindly
            re.push_back(glob[0]);
            re.push_back('-');
            re.push_back(glob[2]);
            glob.remove_prefix(3);
          }
        }
        re.push_back(']');
        glob.remove_prefix(1);
        break;
      }
      case '.':
      case '+':
      case '{':
      case '}':
      case '(':
      case ')':
      case '|':
      case '^':
      case '$': {
        // Escape special characters.
        re.push_back('\\');
        re.push_back(c);
        break;
      }
      case '\\':
        if (glob.empty()) {
          re.push_back('\\');
          re.push_back('\\');
        } else if (!absl::ascii_isalnum(glob[0])) {
          re.push_back('\\');
          re.push_back(glob[0]);
          glob.remove_prefix(1);
        } else {
          // ignore.
        }
        break;

      default:
        re.push_back(c);
        break;
    }
  }
  re.push_back('$');
  return re;
}

}  // namespace cli
}  // namespace tensorstore
