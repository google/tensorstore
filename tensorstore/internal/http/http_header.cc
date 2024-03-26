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

#include "tensorstore/internal/http/http_header.h"

#include <stddef.h>

#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "re2/re2.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_http {
namespace {

// Parse the header field. Per RFC 7230:
//  header-field   = field-name ":" OWS field-value OWS
//
//  field-name     = token
//  field-value    = *( field-content / obs-fold )
//  field-content  = field-vchar [ 1*( SP / HTAB ) field-vchar ]
//  field-vchar    = VCHAR / obs-text
//
//  OWS            = *( SP / HTAB )
//  tchar          = "!" / "#" / "$" / "%" / "&" / "'" / "*"  /
//                   "+" / "-" / "." / "^" / "_" / "`" / "|" / "~" /
//                   DIGIT / ALPHA
//  token          = 1*tchar
//
static inline constexpr internal::AsciiSet kTChar{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    R"(!#$%&'*+-.)"};

inline bool IsTchar(char ch) { return kTChar.Test(ch); }

inline bool IsOWS(char ch) { return ch == ' ' || ch == '\t'; }
}  // namespace

absl::Status ValidateHttpHeader(std::string_view header) {
  static LazyRE2 kHeaderPattern = {// field-name
                                   "[!#\\$%&'*+\\-\\.\\^_`|~0-9a-zA-Z]+"
                                   ":"
                                   // OWS field-content OWS
                                   "[\t\x20-\x7e\x80-\xff]*",
                                   // Use Latin1 because the pattern applies to
                                   // raw bytes, not UTF-8.
                                   RE2::Latin1};

  if (!RE2::FullMatch(header, *kHeaderPattern)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid HTTP header: ", tensorstore::QuoteString(header)));
  }
  return absl::OkStatus();
}

size_t AppendHeaderData(absl::btree_multimap<std::string, std::string>& headers,
                        std::string_view data) {
  // Header fields must be separated in CRLF; thus data must end in LF,
  // and the individual fields are split on LF.
  if (data.empty() || *data.rbegin() != '\n') return data.size();
  for (std::string_view field : absl::StrSplit(data, '\n', absl::SkipEmpty())) {
    // Ensure the header field ends in CRLF by testing the CR part.
    if (field.empty() || *field.rbegin() != '\r') break;
    field.remove_suffix(1);
    // Also remove OWS.
    while (!field.empty() && IsOWS(*field.rbegin())) field.remove_suffix(1);
    if (field.empty()) continue;

    // Parse field-name.
    auto it = field.begin();
    for (; it != field.end() && IsTchar(*it); ++it) {
      /**/
    }
    if (it == field.begin() || it == field.end() || *it != ':') {
      // Invalid header: empty token, not split by :, or no :
      continue;
    }

    std::string field_name = absl::AsciiStrToLower(
        std::string_view(field.data(), std::distance(field.begin(), it)));

    // Transform the value by dropping OWS in the field value prefix.
    field.remove_prefix(field_name.size() + 1);
    while (!field.empty() && IsOWS(*field.begin())) field.remove_prefix(1);
    headers.emplace(std::move(field_name), std::string(field));
  }
  return data.size();
}

std::optional<std::tuple<size_t, size_t, size_t>> TryParseContentRangeHeader(
    const absl::btree_multimap<std::string, std::string>& headers) {
  auto it = headers.find("content-range");
  if (it == headers.end()) {
    return std::nullopt;
  }
  // Expected header format:
  // "bytes <inclusive_start>-<inclusive_end>/<total_size>"
  static LazyRE2 kContentRange1 = {R"(^bytes (\d+)-(\d+)/(\d+))"};
  static LazyRE2 kContentRange2 = {R"(^bytes (\d+)-(\d+)(/[*])?)"};

  std::tuple<size_t, size_t, size_t> result(0, 0, 0);
  if (RE2::FullMatch(it->second, *kContentRange1, &std::get<0>(result),
                     &std::get<1>(result), &std::get<2>(result))) {
    return result;
  }
  if (RE2::FullMatch(it->second, *kContentRange2, &std::get<0>(result),
                     &std::get<1>(result))) {
    return result;
  }
  // Unexpected content-range header format; return nothing.
  return std::nullopt;
}

std::optional<bool> TryParseBoolHeader(
    const absl::btree_multimap<std::string, std::string>& headers,
    std::string_view header) {
  auto it = headers.find(header);
  bool result;
  if (it != headers.end() && absl::SimpleAtob(it->second, &result)) {
    return result;
  }
  return std::nullopt;
}

std::optional<size_t> TryGetContentLength(
    const absl::btree_multimap<std::string, std::string>& headers) {
  std::optional<size_t> content_length;
  // Extract the size of the returned content.
  if (headers.find("transfer-encoding") == headers.end() &&
      headers.find("content-encoding") == headers.end()) {
    content_length = TryParseIntHeader<size_t>(headers, "content-length");
  }
  if (!content_length) {
    auto content_range = TryParseContentRangeHeader(headers);
    if (content_range) {
      content_length =
          1 + std::get<1>(*content_range) - std::get<0>(*content_range);
    }
  }
  return content_length;
}

}  // namespace internal_http
}  // namespace tensorstore
