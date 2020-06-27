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

#include "absl/time/time.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_http {

/// HttpTransport is an interface class for making http requests.
class HttpTransport {
 public:
  virtual ~HttpTransport() = default;

  /// IssueRequest issues the request with the provided body `payload`,
  /// returning the HttpResponse.
  virtual Future<HttpResponse> IssueRequest(
      const HttpRequest& request, absl::Cord payload,
      absl::Duration request_timeout = absl::ZeroDuration(),
      absl::Duration connect_timeout = absl::ZeroDuration()) = 0;
};

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_HTTP_TRANSPORT_H_
