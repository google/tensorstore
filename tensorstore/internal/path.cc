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

#include "tensorstore/internal/ascii_utils.h"
#include "tensorstore/internal/path.h"

#include <initializer_list>
#include <string>
#include <string_view>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"

using ::tensorstore::internal_ascii_utils::AsciiSet;
using ::tensorstore::internal_ascii_utils::HexDigitToInt;
using ::tensorstore::internal_ascii_utils::IntToHexDigit;


namespace {

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
    internal::AppendPathComponent(result, path);
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

void EnsureDirectoryPath(std::string& path) {
  if (path.size() == 1 && path[0] == '/') {
    path.clear();
  } else if (!path.empty() && path.back() != '/') {
    path += '/';
  }
}

void EnsureNonDirectoryPath(std::string& path) {
  size_t size = path.size();
  while (size > 0 && path[size - 1] == '/') {
    --size;
  }
  path.resize(size);
}

void AppendPathComponent(std::string& path, std::string_view component) {
  if (!path.empty() && path.back() != '/' && !component.empty() &&
      component.front() != '/') {
    absl::StrAppend(&path, "/", component);
  } else {
    path += component;
  }
}

namespace {

constexpr AsciiSet kUriUnreservedChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_.!~*'()"};

constexpr AsciiSet kUriPathUnreservedChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_.!~*'():@&=+$,;/"};

}  // namespace

void PercentDecode(std::string_view src, std::string& dest) {
  dest.clear();
  dest.reserve(src.size());
  for (size_t i = 0; i < src.size();) {
    char c = src[i];
    char x, y;
    if (c != '%' || i + 2 >= src.size() ||
        !absl::ascii_isxdigit((x = src[i + 1])) ||
        !absl::ascii_isxdigit((y = src[i + 2]))) {
      dest += c;
      ++i;
      continue;
    }
    dest += static_cast<char>(HexDigitToInt(x) * 16 + HexDigitToInt(y));
    i += 3;
  }
}

std::string PercentDecode(std::string_view src) {
  std::string dest;
  PercentDecode(src, dest);
  return dest;
}

void PercentEncodeUriPath(std::string_view src, std::string& dest) {
  return PercentEncodeReserved(src, dest, kUriPathUnreservedChars);
}

std::string PercentEncodeUriPath(std::string_view src) {
  std::string dest;
  PercentEncodeUriPath(src, dest);
  return dest;
}

void PercentEncodeUriComponent(std::string_view src, std::string& dest) {
  return PercentEncodeReserved(src, dest, kUriUnreservedChars);
}

std::string PercentEncodeUriComponent(std::string_view src) {
  std::string dest;
  PercentEncodeUriComponent(src, dest);
  return dest;
}

ParsedGenericUri ParseGenericUri(std::string_view uri) {
  static constexpr std::string_view kSchemeSep("://");
  ParsedGenericUri result;
  const size_t scheme_start = uri.find(kSchemeSep);
  std::string_view uri_suffix;
  if (scheme_start == std::string_view::npos) {
    // No scheme
    uri_suffix = uri;
  } else {
    result.scheme = uri.substr(0, scheme_start);
    uri_suffix = uri.substr(scheme_start + kSchemeSep.size());
  }
  const size_t fragment_start = uri_suffix.find('#');
  const size_t query_start = uri_suffix.substr(0, fragment_start).find('?');
  const size_t path_end = std::min(query_start, fragment_start);
  // Note: Since substr clips out-of-range count, this works even if
  // `path_end == npos`.
  result.authority_and_path = uri_suffix.substr(0, path_end);
  if (query_start != std::string_view::npos) {
    result.query =
        uri_suffix.substr(query_start + 1, fragment_start - query_start - 1);
  }
  if (fragment_start != std::string_view::npos) {
    result.fragment = uri_suffix.substr(fragment_start + 1);
  }
  return result;
}

}  // namespace internal
}  // namespace tensorstore
