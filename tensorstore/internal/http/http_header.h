// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_HTTP_HTTP_HEADER_H_
#define TENSORSTORE_INTERNAL_HTTP_HTTP_HEADER_H_

#include <stddef.h>

#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"

namespace tensorstore {
namespace internal_http {

/// `strptime`-compatible format string for the HTTP date header.
///
/// https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Date
///
/// Note that the time zone is always UTC and is specified as "GMT".
constexpr const char kHttpTimeFormat[] = "%a, %d %b %E4Y %H:%M:%S GMT";

/// Validates that `header` is a valid HTTP header line (excluding the final
/// "\r\n").
///
/// The syntax is defined by:
/// https://datatracker.ietf.org/doc/html/rfc7230#section-3.2
///
///   header-field   = field-name ":" OWS field-value OWS
///   field-name     = token
///   field-value    = *( field-content / obs-fold )
///   field-content  = field-vchar [ 1*( SP / HTAB ) field-vchar ]
///   field-vchar    = VCHAR / obs-text
///   OWS            = *( SP / HTAB )
///   VCHAR          = any visible ascii character
///   obs-text       = %x80-FF
///   obs-fold = CRLF 1*( SP / HTAB )
///
/// This function differs from the spec in that it does not accept `obs-fold`
/// (obsolete line folding) syntax.
absl::Status ValidateHttpHeader(std::string_view header);

/// AppendHeaderData parses `data` as a header and append to the set of
/// `headers`.
size_t AppendHeaderData(absl::btree_multimap<std::string, std::string>& headers,
                        std::string_view data);

/// Parses the "content-range" header, which can be used to determine the
/// portion of an object returned by an HTTP request (with status code 206).
/// Returned tuple fields are {start, end, total_length}
std::optional<std::tuple<size_t, size_t, size_t>> TryParseContentRangeHeader(
    const absl::btree_multimap<std::string, std::string>& headers);

/// Attempts to parse a header using SimpleAtoi.
template <typename T>
std::optional<T> TryParseIntHeader(
    const absl::btree_multimap<std::string, std::string>& headers,
    std::string_view header) {
  auto it = headers.find(header);
  T result;
  if (it != headers.end() && absl::SimpleAtoi(it->second, &result)) {
    return result;
  }
  return std::nullopt;
}

/// Attempts to parse a header using SimpleAtob.
std::optional<bool> TryParseBoolHeader(
    const absl::btree_multimap<std::string, std::string>& headers,
    std::string_view header);

/// Try to get the content length from the headers.
std::optional<size_t> TryGetContentLength(
    const absl::btree_multimap<std::string, std::string>& headers);

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_HTTP_HEADER_H_
