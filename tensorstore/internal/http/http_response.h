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

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_http {

/// AppendHeaderData parses `data` as a header and append to the set of
/// `headers`.
std::size_t AppendHeaderData(std::multimap<std::string, std::string>& headers,
                             std::string_view data);

/// HttpResponse contains the results of an HTTP request.
struct HttpResponse {
  int32_t status_code;
  absl::Cord payload;
  std::multimap<std::string, std::string> headers;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const HttpResponse& response) {
    absl::Format(&sink, "HttpResponse{code=%d, headers=<",
                 response.status_code);
    const char* sep = "";
    for (const auto& kv : response.headers) {
      sink.Append(sep);
      sink.Append(kv.first);
      sink.Append("=");
      sink.Append(kv.second);
      sep = ", ";
    }
    absl::Format(&sink, ">, body=%v}", response.payload);
  }
};

/// Returns an `absl::Status` object for a corresponding
/// HttpResponse.status_code.
absl::Status HttpResponseCodeToStatus(const HttpResponse& response);

/// Parses the "content-range" header, which can be used to determine the
/// portion of an object returned by an HTTP request (with status code 206).
/// Returned tuple fields are {start, end, total_length}
Result<std::tuple<size_t, size_t, size_t>> ParseContentRangeHeader(
    const HttpResponse& response);

/// Attempts to parse a header using SimpleAtoi.
template <typename T>
std::optional<T> TryParseIntHeader(
    const std::multimap<std::string, std::string>& headers,
    const std::string& header) {
  auto it = headers.find(header);
  T result;
  if (it != headers.end() && absl::SimpleAtoi(it->second, &result)) {
    return result;
  }
  return std::nullopt;
}

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_HTTP_RESPONSE_H_
