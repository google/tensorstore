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

#include "tensorstore/internal/uri/parse.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "tensorstore/internal/uri/ascii_set.h"

namespace tensorstore {
namespace internal_uri {
namespace {

static inline constexpr AsciiSet kSchemeChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "+-._"};  // NOTE: RFC 3986 scheme does not allow '_' but we do.

static inline constexpr AsciiSet kDecimalDigitChars{"0123456789"};
static inline constexpr AsciiSet kHexDigitChars{"0123456789abcdefABCDEF"};

bool IsValidRegNameChar(unsigned char c) {
  if (c <= 0x20 || c >= 0x7f) return false;
  switch (c) {
    case '/':
    case ':':
    case '?':
    case '#':
    case '[':
    case ']':
    case '@':
    case '\\':
    case '"':
    case '<':
    case '>':
    case '^':
    case '`':
    case '{':
    case '|':
    case '}':
      return false;
    default:
      return true;
  }
}

bool IsValidIpv4DottedQuad(std::string_view s) {
  int octets = 0;
  for (std::string_view part : absl::StrSplit(s, '.')) {
    if (++octets > 4) return false;
    if (part.empty() || part.size() > 3 ||
        (part.size() > 1 && part.front() == '0') ||
        !kDecimalDigitChars.ContainsAll(part)) {
      return false;
    }
    uint32_t val = 0;
    if (!absl::SimpleAtoi(part, &val) || val > 255) return false;
  }
  return octets == 4;
}

}  // namespace

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
  const auto query_start = unparsed_uri.find('?');
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

std::string_view StripUserinfo(std::string_view authority) {
  if (size_t at_pos = authority.rfind('@'); at_pos != std::string_view::npos) {
    authority.remove_prefix(at_pos + 1);
  }
  return authority;
}

std::string_view StripIpv6Brackets(std::string_view host) {
  if (absl::StartsWith(host, "[") && absl::EndsWith(host, "]")) {
    return host.substr(1, host.size() - 2);
  }
  return host;
}

std::string FormatHostPort(std::string_view host, std::string_view port) {
  return absl::StrContains(host, ':') ? absl::StrCat("[", host, "]:", port)
                                      : absl::StrCat(host, ":", port);
}

bool IsValidIpv6Literal(std::string_view host) {
  if (host.empty() || host.size() >= 46) return false;
  int groups = 0;
  if (absl::StrContains(host, '.')) {
    const size_t last_colon = host.rfind(':');
    if (last_colon == std::string_view::npos ||
        !IsValidIpv4DottedQuad(host.substr(last_colon + 1))) {
      return false;
    }
    groups = 2;
    host = (last_colon > 0 && host[last_colon - 1] == ':')
               ? host.substr(0, last_colon + 1)
               : host.substr(0, last_colon);
  }

  bool has_double_colon = false;
  if (absl::StartsWith(host, ":")) {
    if (!absl::StartsWith(host, "::")) return false;
    has_double_colon = true;
    host.remove_prefix(2);
    if (host.empty()) return groups < 8;
  }
  if (absl::EndsWith(host, ":")) {
    if (has_double_colon || !absl::EndsWith(host, "::")) return false;
    has_double_colon = true;
    host.remove_suffix(2);
    if (host.empty()) return groups < 8;
  }

  for (std::string_view part : absl::StrSplit(host, ':')) {
    if (part.empty()) {
      if (has_double_colon) return false;
      has_double_colon = true;
      continue;
    }
    if (part.size() > 4 || !kHexDigitChars.ContainsAll(part)) return false;
    ++groups;
  }
  return has_double_colon ? (groups < 8) : (groups == 8);
}

bool IsValidHttpHost(std::string_view host) {
  if (host.empty() || host.size() > 255) return false;
  if (absl::StrContains(host, ':')) {
    return IsValidIpv6Literal(host);
  }
  for (unsigned char c : host) {
    if (!IsValidRegNameChar(c)) return false;
  }
  return true;
}

bool IsValidHttpPort(std::string_view port) {
  uint64_t port_num = 0;
  return kDecimalDigitChars.ContainsAll(port) &&
         absl::SimpleAtoi(port, &port_num) && port_num >= 1 &&
         port_num <= 65535;
}

bool IsValidHttpAuthority(std::string_view authority) {
  if (authority.empty()) return false;
  if (authority.front() == '[') {
    auto close_bracket = authority.find(']');
    if (close_bracket == std::string_view::npos) return false;
    std::string_view ipv6 = authority.substr(1, close_bracket - 1);
    if (!IsValidIpv6Literal(ipv6)) return false;
    std::string_view rest = authority.substr(close_bracket + 1);
    if (rest.empty()) return true;
    if (rest.front() != ':') return false;
    return IsValidHttpPort(rest.substr(1));
  }
  if (authority.find_first_of("[]") != std::string_view::npos) return false;
  auto colon = authority.find(':');
  if (colon == std::string_view::npos) {
    return IsValidHttpHost(authority);
  }
  if (authority.find(':', colon + 1) != std::string_view::npos) {
    return false;  // Unbracketed IPv6 or multiple colons are invalid in URLs.
  }
  return IsValidHttpHost(authority.substr(0, colon)) &&
         IsValidHttpPort(authority.substr(colon + 1));
}

std::string RedactUrlForError(std::string_view raw_url) {
  const size_t q_or_f = raw_url.find_first_of("?#");
  const bool has_query_or_fragment = (q_or_f != std::string_view::npos);
  std::string_view base = raw_url.substr(0, q_or_f);

  size_t auth_start = 0;
  if (size_t scheme_sep = base.find("://");
      scheme_sep != std::string_view::npos) {
    auth_start = scheme_sep + 3;
  } else if (absl::StartsWith(base, "//")) {
    auth_start = 2;
  } else if (size_t colon = base.find(':'); colon != std::string_view::npos &&
                                            colon + 1 < base.size() &&
                                            base[colon + 1] == '/') {
    auth_start = colon + 1;
  }
  while (auth_start < base.size() && base[auth_start] == '/') {
    ++auth_start;
  }

  const size_t path_start = std::min(base.find('/', auth_start), base.size());
  std::string_view authority_part =
      base.substr(auth_start, path_start - auth_start);

  std::string redacted;
  if (size_t at_pos = authority_part.rfind('@');
      at_pos != std::string_view::npos) {
    redacted = absl::StrCat(base.substr(0, auth_start), "[REDACTED]@",
                            authority_part.substr(at_pos + 1),
                            base.substr(path_start));
  } else {
    redacted = std::string(base);
  }
  if (has_query_or_fragment) {
    absl::StrAppend(&redacted, "?[REDACTED]");
  }
  return absl::CEscape(redacted);
}

absl::Status EnsureSchema(const ParsedGenericUri& parsed_uri,
                          std::string_view scheme) {
  if (parsed_uri.scheme != scheme) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Scheme \"%s:\" not present in url", scheme));
  }
  return absl::OkStatus();
}

absl::Status EnsureSchemaWithAuthorityDelimiter(
    const ParsedGenericUri& parsed_uri, std::string_view scheme) {
  if (parsed_uri.scheme != scheme || !parsed_uri.has_authority_delimiter) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Scheme \"%s://\" not present in url", scheme));
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

}  // namespace internal_uri
}  // namespace tensorstore
