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
#include <string_view>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"

namespace {
inline bool IsValidSchemeChar(char ch) {
  return absl::ascii_isalpha(ch) || absl::ascii_isdigit(ch) || ch == '.' ||
         ch == '+' || ch == '-';
}

#ifdef _WIN32
constexpr inline bool IsDirSeparator(char c) { return c == '\\' || c == '/'; }
#else
constexpr inline bool IsDirSeparator(char c) { return c == '/'; }
#endif

}  // namespace
namespace tensorstore {
namespace internal_path {

std::string JoinPathImpl(std::initializer_list<std::string_view> paths) {
  size_t s = 0;
  for (std::string_view path : paths) {
    s += path.size() + 1;
  }

  std::string result;
  result.reserve(s);
  for (std::string_view path : paths) {
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

// Splits a path into the pair {dirname, basename}
std::pair<std::string_view, std::string_view> PathDirnameBasename(
    std::string_view path) {
  size_t pos = path.size();
  while (pos != 0 && !IsDirSeparator(path[pos - 1])) {
    --pos;
  }
  size_t basename = pos;
  --pos;
  if (pos == std::string_view::npos) {
    return {"", path};
  }
  while (pos != 0 && IsDirSeparator(path[pos - 1])) {
    --pos;
  }
  if (pos == 0) {
    return {"/", path.substr(basename)};
  }
  return {path.substr(0, pos), path.substr(basename)};
}

void ParseURI(std::string_view uri, std::string_view* scheme,
              std::string_view* host, std::string_view* path) {
  static const std::string_view kSep("://");

  if (scheme) *scheme = std::string_view(uri.data(), 0);
  if (host) *host = std::string_view(uri.data(), 0);
  if (path) *path = uri;  // By default, everything is a path.
  if (uri.empty()) {
    return;
  }

  // 0. Attempt to parse scheme.
  if (!absl::ascii_isalpha(uri[0])) {
    return;
  }
  std::string_view::size_type scheme_loc = 1;
  std::string_view remaining;
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
  auto path_loc = remaining.find('/');
  if (path_loc == std::string_view::npos) {
    // No path, everything is the host.
    if (host) *host = remaining;
    if (path) *path = std::string_view(remaining.data() + remaining.size(), 0);
    return;
  }
  // 2. There is a host and a path.
  if (host) *host = remaining.substr(0, path_loc);
  if (path) *path = remaining.substr(path_loc);
}

std::string CreateURI(std::string_view scheme, std::string_view host,
                      std::string_view path) {
  if (scheme.empty()) {
    return std::string(path);
  }
  return absl::StrCat(scheme, "://", host, path);
}

}  // namespace internal
}  // namespace tensorstore
