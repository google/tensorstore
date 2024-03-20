// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/http/mock_http_transport.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_http {

void AddDefaultHeaders(internal_http::HttpResponse& response) {
  auto& headers = response.headers;
  if (headers.find("content-length") == headers.end()) {
    headers.emplace("content-length", absl::StrCat(response.payload.size()));
  }
  if (headers.find("content-type") == headers.end()) {
    headers.emplace("content-type", "application/octet-stream");
  }
}

void DefaultMockHttpTransport::Reset(
    absl::flat_hash_map<std::string, internal_http::HttpResponse>
        url_to_response,
    bool add_headers) {
  if (add_headers) {
    // Add additional headers to the response.
    for (auto& kv : url_to_response) {
      AddDefaultHeaders(kv.second);
    }
  }

  absl::MutexLock l(&mutex_);
  requests_.clear();
  url_to_response_ = std::move(url_to_response);
}

Future<internal_http::HttpResponse> DefaultMockHttpTransport::IssueRequest(
    const internal_http::HttpRequest& request, absl::Cord payload,
    absl::Duration request_timeout, absl::Duration connect_timeout) {
  absl::MutexLock l(&mutex_);
  requests_.push_back(request);

  if (auto it =
          url_to_response_.find(absl::StrCat(request.method, " ", request.url));
      it != url_to_response_.end()) {
    return it->second;
  }

  return internal_http::HttpResponse{404, absl::Cord(), {}};
}

}  // namespace internal_http
}  // namespace tensorstore
