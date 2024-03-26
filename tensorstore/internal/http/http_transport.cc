
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

#include "tensorstore/internal/http/http_transport.h"

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <string_view>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "riegeli/bytes/cord_writer.h"
#include "tensorstore/internal/http/http_header.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_http {
namespace {
// Adapts the IssueRequestWithHandler api to IssueRequest.
class LegacyHttpResponseHandler : public HttpResponseHandler {
 public:
  LegacyHttpResponseHandler(Promise<HttpResponse> p);

  ~LegacyHttpResponseHandler() override = default;

  void OnFailure(absl::Status status) override;

  void OnStatus(int32_t status_code) override;
  void OnResponseHeader(std::string_view data) override;
  void OnResponseBody(std::string_view data) override;
  void OnComplete() override;

 private:
  Promise<HttpResponse> promise_;
  absl::Cord data_;
  riegeli::CordWriter<absl::Cord*> writer_;
  int32_t status_code_ = 0;
  absl::btree_multimap<std::string, std::string> headers_;
};

LegacyHttpResponseHandler::LegacyHttpResponseHandler(Promise<HttpResponse> p)
    : promise_(std::move(p)), writer_(&data_) {}

void LegacyHttpResponseHandler::OnStatus(int32_t status_code) {
  status_code_ = status_code;
}

void LegacyHttpResponseHandler::OnResponseHeader(std::string_view data) {
  AppendHeaderData(headers_, data);
  auto content_length = TryGetContentLength(headers_);
  if (content_length) {
    writer_.SetWriteSizeHint(*content_length);
  }
}

void LegacyHttpResponseHandler::OnResponseBody(std::string_view data) {
  writer_.Write(data);
}

void LegacyHttpResponseHandler::OnFailure(absl::Status status) {
  promise_.SetResult(std::move(status));
  delete this;
}

void LegacyHttpResponseHandler::OnComplete() {
  writer_.Close();
  promise_.SetResult(
      HttpResponse{status_code_, std::move(data_), std::move(headers_)});
  delete this;
}

}  // namespace

Future<HttpResponse> HttpTransport::IssueRequest(const HttpRequest& request,
                                                 IssueRequestOptions options) {
  auto pair = PromiseFuturePair<HttpResponse>::Make();
  IssueRequestWithHandler(
      request, std::move(options),
      new LegacyHttpResponseHandler(std::move(pair.promise)));
  return std::move(pair.future);
}

}  // namespace internal_http
}  // namespace tensorstore
