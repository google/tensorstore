// Copyright 2024 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_HTTP_MOCK_HTTP_TRANSPORT_H_
#define TENSORSTORE_INTERNAL_HTTP_MOCK_HTTP_TRANSPORT_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_http {

/// Adds default headers for the HttpResponse.
void AddDefaultHeaders(internal_http::HttpResponse& response);

/// Applies the response to the HttpResponseHandler.
void ApplyResponseToHandler(const HttpResponse& response,
                            HttpResponseHandler* handler);
void ApplyResponseToHandler(const absl::Status& response,
                            HttpResponseHandler* handler);
void ApplyResponseToHandler(const Result<HttpResponse>& response,
                            HttpResponseHandler* handler);

/// Mocks an HttpTransport by overriding the IssueRequest method to
/// respond with a predefined set of request-response pairs supplied
/// to the constructor.
/// The first matching pair will be returned for each call, then expired.
class DefaultMockHttpTransport : public internal_http::HttpTransport {
 public:
  using Responses =
      std::vector<std::pair<std::string, internal_http::HttpResponse>>;

  /// Construct a DefaultMockHttpTransport that returns 404 for all requests.
  DefaultMockHttpTransport() = default;

  explicit DefaultMockHttpTransport(Responses url_to_response) {
    Reset(std::move(url_to_response), true);
  }
  DefaultMockHttpTransport(Responses url_to_response, bool add_headers) {
    Reset(std::move(url_to_response), add_headers);
  }
  virtual ~DefaultMockHttpTransport() = default;

  /// Initializes the list of request-response pairs.
  /// The first matching pair will be returned for each call, then expired.
  void Reset(Responses url_to_response, bool add_headers = true);

  const std::vector<HttpRequest>& requests() const { return requests_; }

  void IssueRequestWithHandler(const HttpRequest& request,
                               IssueRequestOptions options,
                               HttpResponseHandler* response_handler) override;

 private:
  absl::Mutex mutex_;
  std::vector<HttpRequest> requests_;
  Responses url_to_response_;
};

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_MOCK_HTTP_TRANSPORT_H_
