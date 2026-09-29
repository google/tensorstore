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

#include "tensorstore/tscli/lib/glob_to_regex.h"

#include <string>
#include <string_view>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tensorstore {
namespace cli {
namespace {

void AppendBracketChar(std::string* out, char c) {
  if (c == '\\') {
    absl::StrAppend(out, "\\\\");
  } else {
    out->push_back(c);
  }
}

}  // namespace

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
        bool is_exclude = false;
        if (glob[0] == '!' || glob[0] == '^') {
          is_exclude = true;
          glob.remove_prefix(1);
        }
        std::string bracket_body;
        if (glob[0] == ']' && glob.find(']', 1) != std::string_view::npos) {
          absl::StrAppend(&bracket_body, "\\]");
          glob.remove_prefix(1);
        }
        // Copy the characters.
        while (glob[0] != ']') {
          if (glob[0] == '[' && glob[1] == ':') {
            // Escape '[' to avoid character classes.
            absl::StrAppend(&bracket_body, "\\[");
            glob.remove_prefix(1);
          } else if (glob[1] != '-' || glob[2] == ']') {
            // Not a range, so copy the character unless it is '/'.
            if (glob[0] != '/') AppendBracketChar(&bracket_body, glob[0]);
            glob.remove_prefix(1);
          } else if (!is_exclude && glob[0] <= '/' && '/' <= glob[2]) {
            // Make sure that the included range does not contain '/'.
            //
            // NOTE: "/-/" is dropped entirely, which it should,
            // because by definition there is no matching pathname.
            if (glob[0] < '/') {
              AppendBracketChar(&bracket_body, glob[0]);
              bracket_body.push_back('-');
              AppendBracketChar(&bracket_body, '/' - 1);
            }
            if ('/' < glob[2]) {
              AppendBracketChar(&bracket_body, '/' + 1);
              bracket_body.push_back('-');
              AppendBracketChar(&bracket_body, glob[2]);
            }
            glob.remove_prefix(3);
          } else {
            // Range will not match '/', so copy it blindly
            AppendBracketChar(&bracket_body, glob[0]);
            bracket_body.push_back('-');
            AppendBracketChar(&bracket_body, glob[2]);
            glob.remove_prefix(3);
          }
        }
        glob.remove_prefix(1);
        if (is_exclude) {
          absl::StrAppend(&re, "[^/", bracket_body, "]");
        } else if (bracket_body.empty()) {
          absl::StrAppend(&re, "[^\\x00-\\xff]");
        } else if (bracket_body[0] == '^') {
          absl::StrAppend(&re, "[\\", bracket_body, "]");
        } else {
          absl::StrAppend(&re, "[", bracket_body, "]");
        }
        break;
      }
      case '{':
      case '}': {
        // TODO: Support {} grouping.
        re.push_back('\\');
        re.push_back(c);
        break;
      }
      case '.':
      case '+':
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
        } else if (absl::ascii_isascii(static_cast<unsigned char>(glob[0])) &&
                   !absl::ascii_isalnum(static_cast<unsigned char>(glob[0]))) {
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
