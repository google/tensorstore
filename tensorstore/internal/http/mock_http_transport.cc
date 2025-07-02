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
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_http {

void AddDefaultHeaders(HttpResponse& response) {
  auto& headers = response.headers;
  if (headers.find("content-length") == headers.end()) {
    headers.SetHeader("content-length", absl::StrCat(response.payload.size()));
  }
  if (headers.find("content-type") == headers.end()) {
    headers.SetHeader("content-type", "application/octet-stream");
  }
}

void ApplyResponseToHandler(const HttpResponse& response,
                            HttpResponseHandler* handler) {
  handler->OnStatus(response.status_code);
  std::string headers_str;
  for (const auto& kv : response.headers) {
    handler->OnResponseHeader(kv.first, kv.second);
  }
  handler->OnHeaderBlockDone();
  auto end = response.payload.chunk_end();
  for (auto it = response.payload.chunk_begin(); it != end; ++it) {
    handler->OnResponseBody(*it);
  }
  handler->OnComplete();
}

void ApplyResponseToHandler(const absl::Status& response,
                            HttpResponseHandler* handler) {
  handler->OnFailure(response);
}

void ApplyResponseToHandler(const Result<HttpResponse>& response,
                            HttpResponseHandler* handler) {
  if (!response.ok()) {
    ApplyResponseToHandler(response.status(), handler);
  } else {
    ApplyResponseToHandler(response.value(), handler);
  }
}

void DefaultMockHttpTransport::Reset(std::vector<MockResponse> url_to_response,
                                     bool add_headers) {
  if (add_headers) {
    // Add additional headers to the response.
    for (auto& kv : url_to_response) {
      if (std::holds_alternative<HttpResponse>(kv.response)) {
        AddDefaultHeaders(std::get<HttpResponse>(kv.response));
      }
    }
  }

  absl::MutexLock l(&mutex_);
  requests_.clear();
  mock_responses_ = std::move(url_to_response);
}

void DefaultMockHttpTransport::IssueRequestWithHandler(
    const HttpRequest& request, IssueRequestOptions options,
    HttpResponseHandler* response_handler) {
  std::string key = absl::StrCat(request.method, " ", request.url);
  ABSL_LOG(INFO) << key;
  absl::MutexLock l(&mutex_);
  requests_.push_back(request);

  for (auto& mock : mock_responses_) {
    if (mock.count == 0 || mock.url != key) continue;
    if (std::holds_alternative<absl::Status>(mock.response)) {
      ApplyResponseToHandler(std::get<absl::Status>(mock.response),
                             response_handler);
    } else {
      ApplyResponseToHandler(std::get<HttpResponse>(mock.response),
                             response_handler);
    }
    mock.count--;
    return;
  }

  ABSL_LOG(INFO) << "Returning 404 for: " << request;
  return ApplyResponseToHandler(
      internal_http::HttpResponse{404, absl::Cord(key), {}}, response_handler);
}

}  // namespace internal_http
}  // namespace tensorstore
