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
#include <string>
#include <string_view>

#include "absl/strings/cord.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/util/status.h"

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
};

/// Returns a Status object for a corresponding HttpResponse.status_code.
Status HttpResponseCodeToStatus(const HttpResponse& response);

/// Determines the portion of the response, if any, that corresponds to the
/// requested byte range, based on the presence of an HTTP 206 Partial Content
/// status code and a `Content-Range` header.
Result<ByteRange> GetHttpResponseByteRange(
    const HttpResponse& response, OptionalByteRangeRequest byte_range_request);

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_HTTP_RESPONSE_H_
