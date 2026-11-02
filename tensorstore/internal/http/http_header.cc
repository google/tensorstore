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
#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "re2/re2.h"
#include "tensorstore/internal/uri/ascii_set.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_http {
namespace {

bool IsLowercase(std::string_view s) {
  for (char c : s) {
    if (c >= 'A' && c <= 'Z') return false;
  }
  return true;
}

}  // namespace

bool IsSensitiveHttpHeader(std::string_view name) {
  std::string_view stripped = absl::StripAsciiWhitespace(name);
  return absl::StrContainsIgnoreCase(stripped, "authorization") ||
         absl::StrContainsIgnoreCase(stripped, "auth_token") ||
         absl::StrContainsIgnoreCase(stripped, "x-amz-security-token") ||
         absl::StrContainsIgnoreCase(stripped, "cookie") ||
         absl::StrContainsIgnoreCase(stripped, "encryption-key") ||
         absl::EqualsIgnoreCase(stripped, "proxy-authenticate") ||
         absl::EqualsIgnoreCase(stripped, "www-authenticate") ||
         absl::EqualsIgnoreCase(stripped, "x-api-key");
}

std::string_view RedactHeaderValueForLog(std::string_view name,
                                         std::string_view value) {
  return IsSensitiveHttpHeader(name) ? std::string_view("[REDACTED]") : value;
}

std::string RedactH1HeaderLineForLog(std::string_view line) {
  size_t colon = line.find(':');
  if (colon == std::string_view::npos) return std::string(line);
  std::string_view name = line.substr(0, colon);
  if (!IsValidHttpToken(name) || !IsSensitiveHttpHeader(name)) {
    return std::string(line);
  }
  return absl::StrCat(name, ": [REDACTED]");
}

void HeaderMap::ClearHeader(std::string_view field_name) {
  ABSL_DCHECK(IsLowercase(field_name));
  headers_.erase(field_name);
}

void HeaderMap::SetHeader(std::string_view field_name,
                          std::string_view field_value) {
  ABSL_DCHECK(IsLowercase(field_name));
  headers_.insert_or_assign(field_name, field_value);
}

void HeaderMap::SetHeader(std::string_view field_name,
                          std::string field_value) {
  ABSL_DCHECK(IsLowercase(field_name));
  headers_.insert_or_assign(field_name, std::move(field_value));
}

void HeaderMap::CombineHeader(std::string_view field_name,
                              std::string_view field_value) {
  ABSL_DCHECK(IsLowercase(field_name));
  if (auto it = headers_.find(field_name); it != headers_.end()) {
    if (!field_value.empty() && field_value != it->second) {
      it->second =
          absl::StrCat(it->second, it->second.empty() ? "" : ",", field_value);
    }
  } else {
    headers_.emplace(field_name, std::string(field_value));
  }
}

Result<std::pair<std::string_view, std::string_view>> ValidateHttpHeader(
    std::string_view field_name, std::string_view field_value) {
  // Check the header field name.
  if (field_name.empty()) {
    return absl::InvalidArgumentError("Empty HTTP header field name");
  }
  for (char c : field_name) {
    if (!kHttpTokenChars.Test(c)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid HTTP char %c in header field name: %v", c,
                          QuoteString(field_name)));
    }
  }
  // Check the header field value.
  if (!IsValidHttpHeaderValue(field_value)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid HTTP header field value: %v", QuoteString(field_value)));
  }
  return std::make_pair(field_name, StripHttpOws(field_value));
}

Result<std::pair<std::string_view, std::string_view>> ValidateHttpHeader(
    std::string_view header) {
  size_t idx = header.find(':');
  if (idx == std::string_view::npos) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid HTTP header: %v", QuoteString(header)));
  }
  return ValidateHttpHeader(header.substr(0, idx), header.substr(idx + 1));
}

size_t ParseAndSetHeaders(std::string_view data,
                          absl::FunctionRef<void(std::string_view field_name,
                                                 std::string_view field_value)>
                              set_header) {
  // Header fields must be separated in CRLF; thus data must end in LF,
  // and the individual fields are split on LF.
  if (data.empty() || *data.rbegin() != '\n') return data.size();
  for (std::string_view field : absl::StrSplit(data, '\n', absl::SkipEmpty())) {
    // Ensure the header field ends in CRLF by testing the CR part.
    if (field.empty() || *field.rbegin() != '\r') break;
    field.remove_suffix(1);
    if (StripHttpOws(field).empty()) continue;

    // Parse field-name.
    auto it = field.begin();
    for (; it != field.end() && kHttpTokenChars.Test(*it); ++it) {
      /**/
    }
    if (it == field.begin() || it == field.end() || *it != ':') {
      // Invalid header: empty token, not split by :, or no :
      continue;
    }
    std::string_view field_name = field.substr(0, it - field.begin());

    // Transform the value by dropping OWS in the field value prefix/suffix.
    field.remove_prefix(field_name.size() + 1);
    std::string_view value = StripHttpOws(field);
    if (!IsValidHttpHeaderValue(value)) continue;
    set_header(absl::AsciiStrToLower(field_name), value);
  }
  return data.size();
}

std::optional<std::tuple<size_t, size_t, size_t>> TryParseContentRangeHeader(
    const HeaderMap& headers) {
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

std::optional<size_t> TryGetContentLength(const HeaderMap& headers) {
  std::optional<size_t> content_length;
  // Extract the size of the returned content.
  if (headers.find("transfer-encoding") == headers.end() &&
      headers.find("content-encoding") == headers.end()) {
    content_length = headers.TryParseIntHeader<size_t>("content-length");
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
