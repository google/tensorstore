// Copyright 2020 The TensorStore Authors
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

#include "tensorstore/internal/path.h"

#include <initializer_list>
#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace {
inline bool IsValidSchemeChar(char ch) {
  return absl::ascii_isalpha(ch) || absl::ascii_isdigit(ch) || ch == '.' ||
         ch == '+' || ch == '-';
}

}  // namespace
namespace tensorstore {
namespace internal_path {

std::string JoinPathImpl(std::initializer_list<absl::string_view> paths) {
  size_t s = 0;
  for (absl::string_view path : paths) {
    s += path.size() + 1;
  }

  std::string result;
  result.reserve(s);
  for (absl::string_view path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      absl::StrAppend(&result, path);
      continue;
    }

    const bool begins_with_slash = (path[0] == '/');
    if (result[result.size() - 1] == '/') {
      if (begins_with_slash) {
        absl::StrAppend(&result, path.substr(1));
      } else {
        absl::StrAppend(&result, path);
      }
    } else {
      if (begins_with_slash) {
        absl::StrAppend(&result, path);
      } else {
        absl::StrAppend(&result, "/", path);
      }
    }
  }
  return result;
}

}  // namespace internal_path
namespace internal {

void ParseURI(absl::string_view uri, absl::string_view* scheme,
              absl::string_view* host, absl::string_view* path) {
  static const absl::string_view kSep("://");

  if (scheme) *scheme = absl::string_view(uri.data(), 0);
  if (host) *host = absl::string_view(uri.data(), 0);
  if (path) *path = uri;  // By default, everything is a path.
  if (uri.empty()) {
    return;
  }

  // 0. Attempt to parse scheme.
  if (!absl::ascii_isalpha(uri[0])) {
    return;
  }
  absl::string_view::size_type scheme_loc = 1;
  absl::string_view remaining;
  for (;;) {
    if (scheme_loc + kSep.size() > uri.size()) {
      // No scheme. Everything is a path.
      return;
    }
    if (uri.substr(scheme_loc, kSep.size()) == kSep) {
      // Scheme found.
      if (scheme) *scheme = uri.substr(0, scheme_loc);
      remaining = uri.substr(scheme_loc + 3);
      break;
    }
    if (!IsValidSchemeChar(uri[scheme_loc++])) {
      // Illegal Scheme found. Everything is a path.
      return;
    }
  }

  // 1. Parse host
  auto path_loc = remaining.find("/");
  if (path_loc == absl::string_view::npos) {
    // No path, everything is the host.
    if (host) *host = remaining;
    if (path) *path = absl::string_view(remaining.data() + remaining.size(), 0);
    return;
  }
  // 2. There is a host and a path.
  if (host) *host = remaining.substr(0, path_loc);
  if (path) *path = remaining.substr(path_loc);
}

std::string CreateURI(absl::string_view scheme, absl::string_view host,
                      absl::string_view path) {
  if (scheme.empty()) {
    return std::string(path);
  }
  return absl::StrCat(scheme, "://", host, path);
}

}  // namespace internal
}  // namespace tensorstore
