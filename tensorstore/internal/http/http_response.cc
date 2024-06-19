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
#include <stdint.h>

#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "re2/re2.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_http {

const char* HttpResponseCodeToMessage(const HttpResponse& response) {
  switch (response.status_code) {
    case 400:
      return "Bad Request";
    case 401:
      return "Unauthorized";
    case 402:
      return "Payment Required";
    case 403:
      return "Forbidden";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 406:
      return "Not Acceptable";
    case 407:
      return "Proxy Authentication Required";
    case 408:
      return "Request Timeout";
    case 409:
      return "Conflict";
    case 410:
      return "Gone";
    case 411:
      return "Length Required";
    case 412:
      return "Precondition Failed";
    case 413:
      return "Payload Too Large";
    case 414:
      return "URI Too Long";
    case 415:
      return "Unsupported Media Type";
    case 416:
      return "Range Not Satisfiable";
    case 417:
      return "Expectation Failed";
    case 418:
      return "I'm a teapot";
    case 421:
      return "Misdirected Request";
    case 422:
      return "Unprocessable Content";
    case 423:
      return "Locked";
    case 424:
      return "Failed Dependency";
    case 425:
      return "Too Early";
    case 426:
      return "Upgrade Required";
    case 428:
      return "Precondition Required";
    case 429:
      return "Too Many Requests";
    case 431:
      return "Request Header Fields Too Large";
    case 451:
      return "Unavailable For Legal Reasons";
    case 500:
      return "Internal Server Error";
    case 501:
      return "Not Implemented";
    case 502:
      return "Bad Gateway";
    case 503:
      return "Service Unavailable";
    case 504:
      return "Gateway Timeout";
    case 505:
      return "HTTP Version Not Supported";
    case 506:
      return "Variant Also Negotiates";
    case 507:
      return "Insufficient Storage";
    case 508:
      return "Loop Detected";
    case 510:
      return "Not Extended";
    case 511:
      return "Network Authentication Required";
    default:
      return nullptr;
  }
}

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

    // INVALID_ARGUMENT indicates a problem with how the request is
    // constructed.
    case 400:  // Bad Request
    case 411:  // Length Required
      return absl::StatusCode::kInvalidArgument;

    // PERMISSION_DENIED indicates an authentication or an authorization
    // issue.
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
      // This doesn't indicate an error, but we should produce an empty
      // response body. (Not all servers do; GCS returns a short error message
      // body.)
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

absl::Status HttpResponseCodeToStatus(const HttpResponse& response,
                                      SourceLocation loc) {
  auto code = HttpResponseCodeToStatusCode(response);
  if (code == absl::StatusCode::kOk) {
    return absl::OkStatus();
  }

  auto status_message = HttpResponseCodeToMessage(response);
  if (!status_message) status_message = "Unknown";

  absl::Status status(code, status_message);
  if (!response.payload.empty()) {
    status.SetPayload(
        "http_response_body",
        response.payload.Subcord(
            0, response.payload.size() < 256 ? response.payload.size() : 256));
  }

  MaybeAddSourceLocation(status, loc);
  status.SetPayload("http_response_code",
                    absl::Cord(tensorstore::StrCat(response.status_code)));
  return status;
}

Result<ParsedContentRange> ParseContentRangeHeader(
    const HttpResponse& response) {
  auto it = response.headers.find("content-range");
  if (it == response.headers.end()) {
    if (response.status_code != 206) {
      // A content range header is not expected.
      return absl::FailedPreconditionError(
          tensorstore::StrCat("No Content-Range header expected with HTTP ",
                              response.status_code, " response"));
    }
    return absl::FailedPreconditionError(
        "Expected Content-Range header with HTTP 206 response");
  }
  // Expected header format:
  // "bytes <inclusive_start>-<inclusive_end>/<total_size>"
  static const RE2 kContentRangeRegex(R"(^bytes (\d+)-(\d+)/(?:(\d+)|\*))");
  int64_t a, b;
  std::optional<int64_t> total_size;
  if (!RE2::FullMatch(it->second, kContentRangeRegex, &a, &b, &total_size) ||
      a > b || (total_size && b >= *total_size) ||
      b == std::numeric_limits<int64_t>::max()) {
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Unexpected Content-Range header received: ", QuoteString(it->second)));
  }
  return ParsedContentRange{a, b + 1, total_size.value_or(-1)};
}

}  // namespace internal_http
}  // namespace tensorstore
