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

#ifndef TENSORSTORE_INTERNAL_HTTP_HTTP_REQUEST_H_
#define TENSORSTORE_INTERNAL_HTTP_HTTP_REQUEST_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_http {

/// HttpRequest encapsulates a single HTTP request.
class HttpRequest {
 public:
  const std::string& url() const { return url_; }
  const std::string& user_agent() const { return user_agent_; }

  /// method() returns the value set via CURLOPT_CUSTOMREQUEST,
  /// NOTE that GET / POST is determined by whether the request has a body,
  /// and CURLOPT_CUSTOMREQUEST should not be used to set HEAD requests.
  const std::string& method() const { return method_; }
  const std::vector<std::string>& headers() const { return headers_; }
  const bool accept_encoding() const { return accept_encoding_; }

 private:
  friend class HttpRequestBuilder;

  std::string url_;
  std::string method_;
  std::string user_agent_;
  bool accept_encoding_ = false;
  std::vector<std::string> headers_;
};

/// Implements the builder pattern for HttpRequest.
class HttpRequestBuilder {
 public:
  explicit HttpRequestBuilder(std::string base_url);

  /// Creates an http request with the given payload.
  ///
  /// This function invalidates the builder. The application should not use this
  /// builder once this function is called.
  HttpRequest BuildRequest();

  /// Adds a prefix to the user-agent string.
  HttpRequestBuilder& AddUserAgentPrefix(absl::string_view prefix);

  /// Adds request headers.
  HttpRequestBuilder& AddHeader(std::string header);

  /// Adds a parameter for a request.
  HttpRequestBuilder& AddQueryParameter(absl::string_view key,
                                        absl::string_view value);

  /// Changes the http method used for this request.
  HttpRequestBuilder& SetMethod(absl::string_view method);

  /// Enables sending Accept-Encoding header and transparently decoding the
  /// response.
  HttpRequestBuilder& EnableAcceptEncoding();

 private:
  HttpRequest request_;
  char const* query_parameter_separator_;
};

/// Returns an HTTP Range header for requesting the specified byte range.
std::string GetRangeHeader(OptionalByteRangeRequest byte_range);

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_HTTP_REQUEST_H_
