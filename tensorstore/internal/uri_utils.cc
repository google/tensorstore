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

#include "tensorstore/internal/uri_utils.h"

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <string>
#include <string_view>

#include "absl/strings/ascii.h"

namespace tensorstore {
namespace internal {
namespace {

inline int HexDigitToInt(char c) {
  assert(absl::ascii_isxdigit(c));
  int x = static_cast<unsigned char>(c);
  if (x > '9') {
    x += 9;
  }
  return x & 0xf;
}

inline char IntToHexDigit(int x) {
  assert(x >= 0 && x < 16);
  return "0123456789ABCDEF"[x];
}

}  // namespace

void PercentEncodeReserved(std::string_view src, std::string& dest,
                           AsciiSet unreserved) {
  size_t num_escaped = 0;
  for (char c : src) {
    if (!unreserved.Test(c)) ++num_escaped;
  }
  if (num_escaped == 0) {
    dest = src;
    return;
  }
  dest.clear();
  dest.reserve(src.size() + 2 * num_escaped);
  for (char c : src) {
    if (unreserved.Test(c)) {
      dest += c;
    } else {
      dest += '%';
      dest += IntToHexDigit(static_cast<unsigned char>(c) / 16);
      dest += IntToHexDigit(static_cast<unsigned char>(c) % 16);
    }
  }
}

void PercentDecodeAppend(std::string_view src, std::string& dest) {
  dest.reserve(dest.size() + src.size());
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

ParsedGenericUri ParseGenericUri(std::string_view uri) {
  static constexpr std::string_view kSchemeSep("://");
  ParsedGenericUri result;
  const auto scheme_start = uri.find(kSchemeSep);
  std::string_view uri_suffix;
  if (scheme_start == std::string_view::npos) {
    // No scheme
    uri_suffix = uri;
  } else {
    result.scheme = uri.substr(0, scheme_start);
    uri_suffix = uri.substr(scheme_start + kSchemeSep.size());
  }
  const auto fragment_start = uri_suffix.find('#');
  const auto query_start = uri_suffix.substr(0, fragment_start).find('?');
  const auto path_end = std::min(query_start, fragment_start);
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

/// Parses the hostname from "authority_and_path".
std::string_view ParseHostname(std::string_view authority_and_path) {
  // Does not include authority.
  const auto path_start = authority_and_path.find('/');
  if (path_start == 0 || authority_and_path.empty()) {
    return {};
  }
  std::string_view authority = authority_and_path.substr(0, path_start);
  if (authority[0] == '[') {
    // IPv6 literal host.
    auto close = authority.rfind(']');
    if (close == std::string_view::npos) {
      return {};
    }
    return authority.substr(1, close - 1);
  }

  const auto colon = authority.rfind(':');
  if (colon != std::string_view::npos) {
    return authority.substr(0, colon);
  }
  return authority;
}

}  // namespace internal
}  // namespace tensorstore
