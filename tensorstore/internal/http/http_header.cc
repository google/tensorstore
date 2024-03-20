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
  size_t size = data.size();
  if (size <= 2) {
    // Invalid header (too short), ignore.
    return size;
  }
  if ('\r' != data[size - 2] || '\n' != data[size - 1]) {
    // Invalid header (should end in CRLF), ignore.
    return size;
  }
  data.remove_suffix(2);
  if (data.empty()) {
    // Empty header, ignore.
    return size;
  }

  // Parse field-name.
  auto it = data.begin();
  for (; it != data.end() && IsTchar(*it); ++it) {
    /**/
  }
  if (it == data.begin() || it == data.end() || *it != ':') {
    // Invalid header: empty token, not split by :, or no :
    return size;
  }
  std::string field_name = absl::AsciiStrToLower(
      std::string_view(data.data(), std::distance(data.begin(), it)));

  // Transform the value by dropping OWS in the field value.
  data.remove_prefix(field_name.size() + 1);
  while (!data.empty() && IsOWS(*data.begin())) data.remove_prefix(1);
  while (!data.empty() && IsOWS(*data.rbegin())) data.remove_suffix(1);

  std::string value(data);
  headers.emplace(std::move(field_name), std::move(value));
  return size;
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

}  // namespace internal_http
}  // namespace tensorstore
