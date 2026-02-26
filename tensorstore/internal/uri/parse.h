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

#ifndef TENSORSTORE_INTERNAL_URI_PARSE_H_
#define TENSORSTORE_INTERNAL_URI_PARSE_H_

#include <stdint.h>

#include <optional>
#include <string_view>

#include "absl/status/status.h"

namespace tensorstore {
namespace internal_uri {

struct ParsedGenericUri {
  /// Portion of URI before the initial ":", or empty if there is no ":".
  std::string_view scheme;
  /// Portion of URI after the initial ":" or "://" (if present) and before the
  /// first `?` or `#`.  Not percent decoded.
  std::string_view authority_and_path;
  /// Authority portion of uri; empty when there is no authority.
  std::string_view authority;
  /// Path portion of uri.
  std::string_view path;
  /// Portion of URI after the first `?` but before the first `#`.
  /// Not percent decoded.
  std::string_view query;
  /// Portion of URI after the first `#`.  Not percent decoded.
  std::string_view fragment;
  /// Whether the URI has a "://" authority delimiter.
  bool has_authority_delimiter = false;
};

/// Parses a "generic" URI of the form
/// `<scheme>:<//<authority>><path>?<query>#<fragment>`
/// where the `?<query>` and `#<fragment>` portions are optional.
ParsedGenericUri ParseGenericUri(std::string_view uri);

struct HostPort {
  std::string_view host;
  std::string_view port;

  friend bool operator==(const HostPort& a, const HostPort& b) {
    return a.host == b.host && a.port == b.port;
  }
};

/// Splits an authority, or host:port string into host and port.
/// Only minimal validation is performed.
std::optional<HostPort> SplitHostPort(std::string_view host_port);

/// Returns an error if the schema doesn't match.
absl::Status EnsureSchema(const ParsedGenericUri& parsed_uri,
                          std::string_view scheme);
absl::Status EnsureSchemaWithAuthorityDelimiter(
    const ParsedGenericUri& parsed_uri, std::string_view scheme);

// Returns an error if there is a query or fragment.
absl::Status EnsureNoQueryOrFragment(const ParsedGenericUri& parsed_uri);

// Returns an error if there is a path, query or fragment.
absl::Status EnsureNoPathOrQueryOrFragment(const ParsedGenericUri& parsed_uri);

}  // namespace internal_uri
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_URI_PARSE_H_
