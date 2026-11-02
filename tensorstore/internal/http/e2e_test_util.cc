// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/internal/http/e2e_test_util.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_http {

HttpRequest MakeRequest(std::string_view base_url, std::string_view method,
                        std::string_view path, RequestHeaders headers) {
  HttpRequestBuilder builder(method, absl::StrCat(base_url, path));
  for (const auto& [name, value] : headers) {
    builder.AddHeader(name, value);
  }
  return builder.BuildRequest();
}

std::string DescribePayload(std::string_view payload) {
  constexpr size_t kMaxPrinted = 64;
  if (payload.size() <= kMaxPrinted) {
    return absl::StrCat("\"", absl::CHexEscape(payload), "\"");
  }
  return absl::StrCat("\"", absl::CHexEscape(payload.substr(0, kMaxPrinted)),
                      "\"... (", payload.size(), " bytes)");
}

TransportDroppingHandler::TransportDroppingHandler(
    std::shared_ptr<HttpTransport>* transport_holder,
    Promise<int32_t> done_promise)
    : transport_holder_(transport_holder),
      done_promise_(std::move(done_promise)) {}

void TransportDroppingHandler::OnStatus(int32_t status_code) {
  status_code_ = status_code;
}

void TransportDroppingHandler::OnComplete() {
  transport_holder_->reset();
  auto promise = std::move(done_promise_);
  promise.SetResult(status_code_);
}

void TransportDroppingHandler::OnFailure(absl::Status status) {
  transport_holder_->reset();
  auto promise = std::move(done_promise_);
  promise.SetResult(std::move(status));
}

}  // namespace internal_http
}  // namespace tensorstore
