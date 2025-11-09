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

#ifndef TENSORSTORE_INTERNAL_HTTP_HTTP_RESPONSE_H_
#define TENSORSTORE_INTERNAL_HTTP_HTTP_RESPONSE_H_

#include <stddef.h>
#include <stdint.h>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "tensorstore/internal/http/http_header.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_http {

/// HttpResponse contains the results of an HTTP request.
struct HttpResponse {
  int32_t status_code;
  absl::Cord payload;
  HeaderMap headers;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const HttpResponse& response) {
    absl::Format(&sink, "HttpResponse{code=%d, headers=%v, payload.size=%d",
                 response.status_code, response.headers,
                 response.payload.size());
    if (response.payload.size() <= 64) {
      absl::Format(&sink, ", payload=%v}", response.payload);
    } else {
      absl::Format(&sink, "}");
    }
  }
};

/// Returns the error message corresponding to the HttpResponse.status_code.
const char* HttpResponseCodeToMessage(const HttpResponse& response);

/// Returns the `absl::StatusCode` corresponding to the
/// HttpResponse.status_code.
absl::StatusCode HttpResponseCodeToStatusCode(const HttpResponse& response);

/// Returns an `absl::Status` object for a corresponding
/// HttpResponse.status_code.
absl::Status HttpResponseCodeToStatus(
    const HttpResponse& response,
    SourceLocation loc = ::tensorstore::SourceLocation::current());

struct ParsedContentRange {
  // Inclusive min byte, always >= `0`.
  int64_t inclusive_min;
  // Exclusive max byte, always > `inclusive_min`.
  int64_t exclusive_max;
  int64_t total_size;  // (equal to -1 if unknown)
};

/// Parses the "content-range" header, which can be used to determine the
/// portion of an object returned by an HTTP request (with status code 206).
Result<ParsedContentRange> ParseContentRangeHeader(
    const HttpResponse& response);

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_HTTP_RESPONSE_H_
