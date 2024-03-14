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

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/byte_range.h"

namespace tensorstore {
namespace internal_http {

/// HttpRequest encapsulates a single HTTP request.
struct HttpRequest {
  std::string method;
  std::string url;
  std::string user_agent = {};
  std::vector<std::string> headers = {};
  bool accept_encoding = false;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const HttpRequest& request) {
    absl::Format(&sink, "HttpRequest{%s %s user_agent=%s, headers=<",
                 request.method, request.url, request.user_agent);
    const char* sep = "";
    for (const auto& v : request.headers) {
      sink.Append(sep);
#ifndef NDEBUG
      // Redact authorization token in request logging.
      if (absl::StartsWithIgnoreCase(v, "authorization:")) {
        sink.Append(std::string_view(v).substr(0, 25));
        sink.Append("#####");
      } else
#endif
      {
        sink.Append(v);
      }
      sep = "  ";
    }
    sink.Append(">}");
  }
};

/// Formats a `range` header to the http request if the byte_range
/// is specified.
std::optional<std::string> FormatRangeHeader(
    OptionalByteRangeRequest byte_range);

/// Formats a `cache-control` header specifying `max-age` or `no-cache`.
std::optional<std::string> FormatCacheControlMaxAgeHeader(
    absl::Duration max_age);

/// Formats a `cache-control` header consistent with `staleness_bound`.
std::optional<std::string> FormatStalenessBoundCacheControlHeader(
    absl::Time staleness_bound);

/// Implements the builder pattern for HttpRequest.
class HttpRequestBuilder {
 public:
  using UriEncodeFunctor = absl::FunctionRef<std::string(std::string_view)>;

  /// Creates a request builder, using the specified method and url.
  ///
  /// The method should be an HTTP method, like "GET", "POST", "PUT", "HEAD".
  /// The uri_encoder is used to encode query parameters.
  HttpRequestBuilder(std::string_view method, std::string base_url)
      : HttpRequestBuilder(method, base_url,
                           internal::PercentEncodeUriComponent) {}

  HttpRequestBuilder(std::string_view method, std::string base_url,
                     UriEncodeFunctor uri_encoder);

  /// Creates an `HttpRequest` request from the builder.
  ///
  /// This function invalidates the builder. The application should not use this
  /// builder once this function is called.
  HttpRequest BuildRequest();

  /// Adds a parameter for a request.
  HttpRequestBuilder& AddQueryParameter(std::string_view key,
                                        std::string_view value);

  /// Enables sending Accept-Encoding header and transparently decoding the
  /// response.
  HttpRequestBuilder& EnableAcceptEncoding();

  /// Adds request headers.
  HttpRequestBuilder& AddHeader(std::string header);
  HttpRequestBuilder& AddHeader(std::string_view header) {
    return header.empty() ? *this : AddHeader(std::string(header));
  }
  HttpRequestBuilder& AddHeader(const char* header) {
    return AddHeader(std::string_view(header));
  }
  HttpRequestBuilder& AddHeader(std::optional<std::string> header) {
    return header ? AddHeader(std::move(*header)) : *this;
  }

  /// Adds a `range` header to the http request if the byte_range
  /// is specified.
  HttpRequestBuilder& MaybeAddRangeHeader(OptionalByteRangeRequest byte_range);

  /// Adds a `cache-control` header specifying `max-age` or `no-cache`.
  HttpRequestBuilder& MaybeAddCacheControlMaxAgeHeader(absl::Duration max_age);

  /// Adds a `cache-control` header consistent with `staleness_bound`.
  HttpRequestBuilder& MaybeAddStalenessBoundCacheControlHeader(
      absl::Time staleness_bound);

  /// Adds a 'host' header for the request url.
  HttpRequestBuilder& AddHostHeader(std::string_view host);

 private:
  absl::FunctionRef<std::string(std::string_view)> uri_encoder_;
  HttpRequest request_;
  char const* query_parameter_separator_;
};

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_HTTP_REQUEST_H_
