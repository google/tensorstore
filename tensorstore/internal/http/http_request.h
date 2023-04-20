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
#include <string_view>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

namespace internal_storage_s3 {
  class S3RequestBuilder;
}

namespace internal_http {

/// HttpRequest encapsulates a single HTTP request.
class HttpRequest {
 public:
  const std::string& url() const { return url_; }
  const std::string& user_agent() const { return user_agent_; }

  // HTTP method, i.e. GET, POST, PUT, HEAD, etc.
  const std::string& method() const { return method_; }
  const std::vector<std::string>& headers() const { return headers_; }
  const bool accept_encoding() const { return accept_encoding_; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const HttpRequest& request) {
    absl::Format(&sink, "HttpRequest{%s %s user_agent=%s", request.method_,
                 request.url_, request.user_agent_);
    for (const auto& v : request.headers_) {
      sink.Append(", ");
      sink.Append(v);
    }
    sink.Append("}");
  }

 private:
  friend class HttpRequestBuilder;
  friend class ::tensorstore::internal_storage_s3::S3RequestBuilder;

  std::string url_;
  std::string method_;
  std::string user_agent_;
  bool accept_encoding_ = false;
  std::vector<std::string> headers_;
};

/// Implements the builder pattern for HttpRequest.
class HttpRequestBuilder {
 public:
  /// Creates a request builder, using the specified method and url.
  ///
  /// The method should be an HTTP method, like "GET", "POST", "PUT", "HEAD",
  /// etc.
  explicit HttpRequestBuilder(std::string_view method, std::string base_url);

  /// Creates an http request with the given payload.
  ///
  /// This function invalidates the builder. The application should not use this
  /// builder once this function is called.
  HttpRequest BuildRequest();

  /// Adds a prefix to the user-agent string.
  HttpRequestBuilder& AddUserAgentPrefix(std::string_view prefix);

  /// Adds request headers.
  HttpRequestBuilder& AddHeader(std::string header);

  /// Adds a parameter for a request.
  HttpRequestBuilder& AddQueryParameter(std::string_view key,
                                        std::string_view value);

  /// Enables sending Accept-Encoding header and transparently decoding the
  /// response.
  HttpRequestBuilder& EnableAcceptEncoding();


 private:
  HttpRequest request_;
  char const* query_parameter_separator_;
};

/// Adds a `range` header to the http request if the byte_range
/// is specified.
bool AddRangeHeader(HttpRequestBuilder& request_builder,
                    OptionalByteRangeRequest byte_range);

/// Adds a `cache-control` header specifying `max-age` or `no-cache`.
bool AddCacheControlMaxAgeHeader(HttpRequestBuilder& request_builder,
                                 absl::Duration max_age);

/// `strptime`-compatible format string for the HTTP date header.
///
/// https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Date
///
/// Note that the time zone is always UTC and is specified as "GMT".
constexpr const char kHttpTimeFormat[] = "%a, %d %b %E4Y %H:%M:%S GMT";

/// Adds a `cache-control` header consistent with `staleness_bound`.
bool AddStalenessBoundCacheControlHeader(HttpRequestBuilder& request_builder,
                                         absl::Time staleness_bound);

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_HTTP_REQUEST_H_
