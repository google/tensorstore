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

#ifndef TENSORSTORE_INTERNAL_HTTP_CURL_TRANSPORT_H_
#define TENSORSTORE_INTERNAL_HTTP_CURL_TRANSPORT_H_

#include <memory>
#include <string_view>

#include "absl/time/time.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_http {

/// Called to initialize any CURL handles used to make requests.  The
/// definition can be overridden to set options such as certificate paths.
void InitializeCurlHandle(CURL* handle);

/// Implementation of HttpTransport which uses libcurl via the curl_multi
/// interface.
class CurlTransport : public HttpTransport {
 public:
  explicit CurlTransport(std::shared_ptr<CurlHandleFactory> factory);

  ~CurlTransport() override;

  /// IssueRequest issues the request with the provided body `payload`,
  /// returning the HttpResponse.
  Future<HttpResponse> IssueRequest(const HttpRequest& request,
                                    absl::Cord payload,
                                    absl::Duration request_timeout,
                                    absl::Duration connect_timeout) override;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

/// Returns the default CurlTransport.
std::shared_ptr<HttpTransport> GetDefaultHttpTransport();

/// Sets the default CurlTransport. Exposed for test mocking.
void SetDefaultHttpTransport(std::shared_ptr<HttpTransport> t);

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_CURL_TRANSPORT_H_
