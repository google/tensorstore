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
#include <stdint.h>

#include <cassert>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/ascii_set.h"

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

static inline constexpr AsciiSet kSchemeChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "+-._"};  // NOTE: rfc3986 scheme do not allow '_'

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
  ParsedGenericUri result;
  std::string_view unparsed_uri = uri;

  if (uri.empty()) return result;

  // Parse the scheme.
  for (size_t i = 0; i < uri.size(); ++i) {
    if (i == 0 && !absl::ascii_isalpha(uri[i])) break;
    if (kSchemeChars.Test(uri[i])) continue;
    if (uri[i] == ':') {
      result.scheme = uri.substr(0, i);
      unparsed_uri = uri.substr(i + 1);
    }
    break;
  }

  // Parse the query parts and fragment.
  const auto fragment_start = unparsed_uri.find('#');
  if (fragment_start != std::string_view::npos) {
    result.fragment = unparsed_uri.substr(fragment_start + 1);
    unparsed_uri = unparsed_uri.substr(0, fragment_start);
  }
  const auto query_start = unparsed_uri.substr(0, fragment_start).find('?');
  if (query_start != std::string_view::npos) {
    result.query = unparsed_uri.substr(query_start + 1);
    unparsed_uri = unparsed_uri.substr(0, query_start);
  }

  // Maybe parse the authority from the hier-part.
  if (absl::StartsWith(unparsed_uri, "//")) {
    result.has_authority_delimiter = true;
    unparsed_uri = unparsed_uri.substr(2);
    result.authority_and_path = unparsed_uri;

    // TODO: Actually parse the authority and path
    if (const auto path_start = unparsed_uri.find('/');
        path_start != std::string_view::npos) {
      result.authority = unparsed_uri.substr(0, path_start);
      result.path = unparsed_uri.substr(path_start);
    } else {
      result.authority = unparsed_uri;
      result.path = {};
    }
  } else {
    result.authority_and_path = unparsed_uri;
    result.authority = {};
    result.path = unparsed_uri;
  }
  return result;
}

absl::Status EnsureSchema(const ParsedGenericUri& parsed_uri,
                          std::string_view scheme) {
  if (parsed_uri.scheme != scheme) {
    return absl::InvalidArgumentError(
        absl::StrCat("Scheme \"", scheme, ":\" not present in url"));
  }
  return absl::OkStatus();
}

absl::Status EnsureSchemaWithAuthorityDelimiter(
    const ParsedGenericUri& parsed_uri, std::string_view scheme) {
  if (parsed_uri.scheme != scheme || !parsed_uri.has_authority_delimiter) {
    return absl::InvalidArgumentError(
        absl::StrCat("Scheme \"", scheme, "://\" not present in url"));
  }
  return absl::OkStatus();
}

absl::Status EnsureNoQueryOrFragment(const ParsedGenericUri& parsed_uri) {
  if (!parsed_uri.query.empty()) {
    return absl::InvalidArgumentError("Query string not supported");
  }
  if (!parsed_uri.fragment.empty()) {
    return absl::InvalidArgumentError("Fragment identifier not supported");
  }
  return absl::OkStatus();
}

absl::Status EnsureNoPathOrQueryOrFragment(const ParsedGenericUri& parsed_uri) {
  if (!parsed_uri.path.empty()) {
    return absl::InvalidArgumentError("Path not supported");
  }
  if (!parsed_uri.authority.empty()) {
    return absl::InvalidArgumentError("Query string not supported");
  }
  return EnsureNoQueryOrFragment(parsed_uri);
}

std::optional<HostPort> SplitHostPort(std::string_view host_port) {
  if (host_port.empty()) return std::nullopt;
  if (host_port[0] == '[') {
    // Parse a bracketed host, typically an IPv6 literal.
    const size_t rbracket = host_port.find(']', 1);
    if (rbracket == std::string_view::npos) {
      // Invalid: Unmatched [
      return std::nullopt;
    }
    if (!absl::StrContains(host_port.substr(1, rbracket - 1), ':')) {
      // Invalid: No colons in IPv6 literal
      return std::nullopt;
    }
    if (rbracket == host_port.size() - 1) {
      // [...]
      return HostPort{host_port, {}};
    }
    if (host_port[rbracket + 1] == ':') {
      if (host_port.rfind(':') != rbracket + 1) {
        // Invalid: multiple colons
        return std::nullopt;
      }
      // [...]:port
      return HostPort{host_port.substr(0, rbracket + 1),
                      host_port.substr(rbracket + 2)};
    }
    return std::nullopt;
  }

  // IPv4 or bare hostname.
  size_t colon = host_port.find(':');
  if (colon == std::string_view::npos ||
      host_port.find(':', colon + 1) != std::string_view::npos) {
    // 0 or 2 colons, assume a hostname.
    return HostPort{host_port, {}};
  }

  return HostPort{host_port.substr(0, colon), host_port.substr(colon + 1)};
}

}  // namespace internal
}  // namespace tensorstore
