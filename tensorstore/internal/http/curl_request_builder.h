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

#ifndef TENSORSTORE_INTERNAL_HTTP_CURL_REQUEST_BUILDER_H_
#define TENSORSTORE_INTERNAL_HTTP_CURL_REQUEST_BUILDER_H_

#include <cstddef>
#include <memory>
#include <string>

#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/curl_request.h"
#include "tensorstore/kvstore/byte_range.h"

namespace tensorstore {
namespace internal_http {

/// Implements the builder pattern for CurlRequest.
class CurlRequestBuilder {
 public:
  explicit CurlRequestBuilder(std::string base_url,
                              std::shared_ptr<CurlHandleFactory> factory);

  /// Creates an http request with the given payload.
  ///
  /// This function invalidates the builder. The application should not use this
  /// builder once this function is called.
  CurlRequest BuildRequest();

  /// Adds a prefix to the user-agent string.
  CurlRequestBuilder& AddUserAgentPrefix(absl::string_view prefix);

  /// Adds request headers.
  CurlRequestBuilder& AddHeader(const std::string& header);

  /// Adds a parameter for a request.
  CurlRequestBuilder& AddQueryParameter(absl::string_view key,
                                        absl::string_view value);

  /// Changes the http method used for this request.
  CurlRequestBuilder& SetMethod(absl::string_view method);

  /// Enables sending Accept-Encoding header and transparently decoding the
  /// response.
  CurlRequestBuilder& EnableAcceptEncoding();

 private:
  CurlRequest request_;
  char const* query_parameter_separator_;
};

/// Returns an HTTP Range header for requesting the specified byte range.
std::string GetRangeHeader(OptionalByteRangeRequest byte_range);

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_CURL_REQUEST_BUILDER_H_
