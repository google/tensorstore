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

#ifndef TENSORSTORE_INTERNAL_HTTP_HTTP_TRANSPORT_H_
#define TENSORSTORE_INTERNAL_HTTP_HTTP_TRANSPORT_H_

#include <stdint.h>

#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_http {

struct IssueRequestOptions {
  // Maps to CURLOPT_HTTP_VERSION
  enum class HttpVersion : uint8_t {
    kDefault = 0,
    kHttp1,
    kHttp2,
    kHttp2TLS,
    kHttp2PriorKnowledge,
  };

  IssueRequestOptions() = default;
  explicit IssueRequestOptions(absl::Cord body) : payload(std::move(body)) {}

  IssueRequestOptions&& SetPayload(absl::Cord payload) && {
    this->payload = std::move(payload);
    return std::move(*this);
  }
  IssueRequestOptions&& SetHttpVersion(HttpVersion http_version) && {
    this->http_version = http_version;
    return std::move(*this);
  }
  IssueRequestOptions&& SetRequestTimeout(absl::Duration request_timeout) && {
    this->request_timeout = request_timeout;
    return std::move(*this);
  }
  IssueRequestOptions&& SetConnectTimeout(absl::Duration connect_timeout) && {
    this->connect_timeout = connect_timeout;
    return std::move(*this);
  }

  absl::Cord payload;
  absl::Duration request_timeout = absl::ZeroDuration();
  absl::Duration connect_timeout = absl::ZeroDuration();
  HttpVersion http_version = HttpVersion::kDefault;
};

/// Interface used by the HTTP transport to signal data to caller.
class HttpResponseHandler {
 public:
  virtual ~HttpResponseHandler() = default;
  // Request has failed with an error. May occur at any time, and no
  // further methods will be invoked.
  virtual void OnFailure(absl::Status) = 0;

  // Sets the status code. Will be called before any of
  // OnResponseHeader/OnResponseBody/OnComplete.
  virtual void OnStatus(int32_t status_code) = 0;
  // Raw header content is available. May be called multiple times.
  virtual void OnResponseHeader(std::string_view data) = 0;
  // Raw body content is available. May be called multiple times.
  virtual void OnResponseBody(std::string_view data) = 0;
  // Request has completed with the provided http status code.
  virtual void OnComplete() = 0;
  // TODO: GetStopToken()
};

/// HttpTransport is an interface class for making http requests.
class HttpTransport {
 public:
  virtual ~HttpTransport() = default;

  /// IssueRequest issues the request with the provided body `payload`,
  /// returning the HttpResponse.
  Future<HttpResponse> IssueRequest(const HttpRequest& request,
                                    IssueRequestOptions options);

  /// IssueRequest issues the request with the provided body `payload`.
  /// The HttpResponseHandler is used to return data to the caller.
  /// One of the methods OnComplete/OnFailure will be invoked when the
  /// request has completed.
  virtual void IssueRequestWithHandler(
      const HttpRequest& request, IssueRequestOptions options,
      HttpResponseHandler* response_handler) = 0;
};

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_HTTP_TRANSPORT_H_
