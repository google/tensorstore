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

#include "tensorstore/internal/http/http_response.h"

#include <stddef.h>

#include <algorithm>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "re2/re2.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
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
inline bool IsTchar(char ch) {
  switch (ch) {
    case '!':
    case '#':
    case '$':
    case '%':
    case '&':
    case '\'':
    case '*':
    case '+':
    case '-':
    case '.':
      return true;
    default:
      return absl::ascii_isdigit(ch) || absl::ascii_isalpha(ch);
  }
}

inline bool IsOWS(char ch) { return ch == ' ' || ch == '\t'; }

absl::StatusCode HttpResponseCodeToStatusCode(const HttpResponse& response) {
  switch (response.status_code) {
    // The group of response codes indicating that the request achieved
    // the expected goal.
    case 200:  // OK
    case 201:  // Created
    case 202:  // Accepted
    case 204:  // No Content
    case 206:  // Partial Content
      return absl::StatusCode::kOk;

    // INVALID_ARGUMENT indicates a problem with how the request is constructed.
    case 400:  // Bad Request
    case 411:  // Length Required
      return absl::StatusCode::kInvalidArgument;

    // PERMISSION_DENIED indicates an authentication or an authorization issue.
    case 401:  // Unauthorized
    case 403:  // Forbidden
      return absl::StatusCode::kPermissionDenied;

    // NOT_FOUND indicates that the requested resource does not exist.
    case 404:  // Not found
    case 410:  // Gone
      return absl::StatusCode::kNotFound;

    // FAILED_PRECONDITION indicates that the request failed because some
    // of the underlying assumptions were not satisfied. The request
    // shouldn't be retried unless the external context has changed.
    case 302:  // Found
    case 303:  // See Other
    case 304:  // Not Modified
    case 307:  // Temporary Redirect
    case 412:  // Precondition Failed
    case 413:  // Payload Too Large
      return absl::StatusCode::kFailedPrecondition;

    case 416:  // Requested Range Not Satisfiable
      // The requested range had no overlap with the available range.
      // This doesn't indicate an error, but we should produce an empty response
      // body. (Not all servers do; GCS returns a short error message body.)
      return absl::StatusCode::kOutOfRange;

    // UNAVAILABLE indicates a problem that can go away if the request
    // is just retried without any modification. 308 return codes are intended
    // for write requests that can be retried. See the documentation and the
    // official library:
    // https://cloud.google.com/kvstore/docs/json_api/v1/how-tos/resumable-upload
    // https://github.com/google/apitools/blob/master/apitools/base/py/transfer.py
    // https://cloud.google.com/storage/docs/request-rate
    case 308:  // Resume Incomplete
    case 408:  // Request Timeout
    case 409:  // Conflict
    case 429:  // Too Many Requests
    case 500:  // Internal Server Error
    case 502:  // Bad Gateway
    case 503:  // Service Unavailable
    case 504:  // Gateway timeout
      return absl::StatusCode::kUnavailable;
  }

  if (response.status_code < 300) {
    return absl::StatusCode::kOk;
  }
  // All other HTTP response codes are translated to "Unknown" errors.
  return absl::StatusCode::kUnknown;
}

}  // namespace

size_t AppendHeaderData(std::multimap<std::string, std::string>& headers,
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

absl::Status HttpResponseCodeToStatus(const HttpResponse& response,
                                      SourceLocation loc) {
  auto code = HttpResponseCodeToStatusCode(response);
  if (code == absl::StatusCode::kOk) {
    return absl::OkStatus();
  }

  const auto pos = (std::max)(response.payload.size(),
                              static_cast<std::string::size_type>(256));
  auto message = tensorstore::StrCat(
      "HTTP response code: ", response.status_code,
      ((pos < response.payload.size()) ? " with body (clipped): "
                                       : " with body: "),
      response.payload.Subcord(0, pos).Flatten());

  absl::Status status(code, message);
  MaybeAddSourceLocation(status, loc);
  return status;
}

Result<std::tuple<size_t, size_t, size_t>> ParseContentRangeHeader(
    const HttpResponse& response) {
  auto it = response.headers.find("content-range");
  if (it == response.headers.end()) {
    if (response.status_code != 206) {
      // A content range header is not expected.
      return absl::UnknownError(
          tensorstore::StrCat("No Content-Range header expected with HTTP ",
                              response.status_code, " response"));
    }
    return absl::UnknownError(
        "Expected Content-Range header with HTTP 206 response");
  }
  // Expected header format:
  // "bytes <inclusive_start>-<inclusive_end>/<total_size>"
  static const RE2 kContentRangeRegex(R"(^bytes (\d+)-(\d+)(?:/(\d+))?)");
  std::tuple<size_t, size_t, size_t> result(0, 0, 0);
  if (!RE2::FullMatch(it->second, kContentRangeRegex, &std::get<0>(result),
                      &std::get<1>(result), &std::get<2>(result))) {
    return absl::UnknownError(tensorstore::StrCat(
        "Unexpected Content-Range header received: ", QuoteString(it->second)));
  }
  return result;
}

std::optional<bool> TryParseBoolHeader(
    const std::multimap<std::string, std::string>& headers,
    const std::string& header) {
  auto it = headers.find(header);
  bool result;
  if (it != headers.end() && absl::SimpleAtob(it->second, &result)) {
    return result;
  }
  return std::nullopt;
}

}  // namespace internal_http
}  // namespace tensorstore
