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

#ifndef TENSORSTORE_INTERNAL_HTTP_CURL_REQUEST_H_
#define TENSORSTORE_INTERNAL_HTTP_CURL_REQUEST_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/variant.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_http {

class CurlRequest;

class CurlRequestMockContext {
 public:
  virtual ~CurlRequestMockContext();

  virtual absl::variant<absl::monostate, HttpResponse, Status> Match(
      CurlRequest* request, absl::string_view payload) = 0;
};

/// CurlReceivedHeaders is a map of received http headers.
using CurlReceivedHeaders = std::multimap<std::string, std::string>;

/// CurlRequest encapsulates a single HTTP request using CURL.
class CurlRequest {
 public:
  const std::string& url() const { return url_; }
  const std::string& user_agent() const { return user_agent_; }

  /// method() returns the value set via CURLOPT_CUSTOMREQUEST,
  /// NOTE that GET / POST is determined by whether the request has a body,
  /// and CURLOPT_CUSTOMREQUEST should not be used to set HEAD requests.
  const std::string& method() const { return method_; }
  const CurlHeaders& headers() const { return headers_; }

  /// IssueRequest issues the request with the provided body `payload`,
  /// returning the HttpResponse.
  ///
  /// IssueRequest may be called multiple times on the same request.
  Result<HttpResponse> IssueRequest(
      absl::string_view payload,
      absl::Duration request_timeout = absl::ZeroDuration(),
      absl::Duration connect_timeout = absl::ZeroDuration());

  // TESTONLY
  static void SetMockContext(CurlRequestMockContext* context);

 private:
  friend class CurlRequestBuilder;

  std::string url_;
  std::string method_;

  CurlHeaders headers_;
  std::string user_agent_;
  bool accept_encoding_ = false;
  std::shared_ptr<CurlHandleFactory> factory_;
};

// Creates a string of the request/response. If the payload is desired, for the
// 4th parameter, pass response.payload.
std::string DumpRequestResponse(const CurlRequest& request,
                                absl::string_view payload,
                                const HttpResponse& response,
                                absl::string_view response_payload);

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_CURL_REQUEST_H_
