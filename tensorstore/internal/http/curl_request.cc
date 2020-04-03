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

#include "tensorstore/internal/http/curl_request.h"

#include "tensorstore/internal/logging.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_http {

CurlRequestMockContext::~CurlRequestMockContext() = default;

static CurlRequestMockContext* g_mock_context_ = nullptr;

/* static */
void CurlRequest::SetMockContext(CurlRequestMockContext* context) {
  g_mock_context_ = context;
}

Result<HttpResponse> CurlRequest::IssueRequest(absl::string_view payload,
                                               absl::Duration request_timeout,
                                               absl::Duration connect_timeout) {
  if (g_mock_context_) {
    // When using a mock context, we assume that the mock is
    // thread safe and not uninstalled when it might introduce
    // race conditions.
    auto match_result = g_mock_context_->Match(this, payload);
    if (absl::holds_alternative<HttpResponse>(match_result)) {
      return std::move(absl::get<HttpResponse>(match_result));
    } else if (absl::holds_alternative<Status>(match_result)) {
      return std::move(absl::get<Status>(match_result));
    }
    // mocking failed, dispatch normally.
  }

  struct HandleWrapper : public CurlPtr {
    HandleWrapper(CurlHandleFactory* factory)
        : CurlPtr(factory->CreateHandle()), factory_(factory) {}

    ~HandleWrapper() { factory_->CleanupHandle(static_cast<CurlPtr&&>(*this)); }

    CurlHandleFactory* factory_;
  };
  HandleWrapper handle(factory_.get());
  std::string response_payload;
  CurlReceivedHeaders received_headers;

  // For thread safety, don't use signals to time out name resolves (when async
  // name resolution is not supported).
  //
  // https://curl.haxx.se/libcurl/c/threadsafe.html
  CurlEasySetopt(handle.get(), CURLOPT_NOSIGNAL, 1L);

  CurlEasySetopt(handle.get(), CURLOPT_URL, url_.c_str());
  CurlEasySetopt(handle.get(), CURLOPT_HTTPHEADER, headers_.get());
  CurlEasySetopt(handle.get(), CURLOPT_USERAGENT, user_agent_.c_str());
  if (accept_encoding_) {
    CurlEasySetopt(handle.get(), CURLOPT_ACCEPT_ENCODING, "");
  }
  CurlWriteCallback write_callback(handle.get(), [&](absl::string_view data) {
    response_payload.append(data.data(), data.size());
    return data.size();
  });
  CurlHeaderCallback header_callback(handle.get(), [&](absl::string_view data) {
    return AppendHeaderData(received_headers, data);
  });

  if (!method_.empty()) {
    CurlEasySetopt(handle.get(), CURLOPT_CUSTOMREQUEST, method_.c_str());
  }

  if (!payload.empty()) {
    if (!method_.empty()) {
      TENSORSTORE_LOG("Changing custom http method [", method_, "] to POST");
    }
    CurlEasySetopt(handle.get(), CURLOPT_POST, 1);
    CurlEasySetopt(handle.get(), CURLOPT_POSTFIELDSIZE_LARGE, payload.length());
    CurlEasySetopt(handle.get(), CURLOPT_POSTFIELDS, payload.data());
  } else if (method_.empty()) {
    CurlEasySetopt(handle.get(), CURLOPT_HTTPGET, 1);
  }

  if (request_timeout > absl::ZeroDuration()) {
    auto ms = absl::ToInt64Milliseconds(request_timeout);
    CurlEasySetopt(handle.get(), CURLOPT_TIMEOUT_MS, ms > 0 ? ms : 1);
  }
  if (connect_timeout > absl::ZeroDuration()) {
    auto ms = absl::ToInt64Milliseconds(connect_timeout);
    CurlEasySetopt(handle.get(), CURLOPT_CONNECTTIMEOUT_MS, ms > 0 ? ms : 1);
  }

  auto status = CurlEasyPerform(handle.get());
  if (!status.ok()) return std::move(status);
  int32_t code = CurlGetResponseCode(handle.get());
  return HttpResponse{code, std::move(response_payload),
                      std::move(received_headers)};
}

std::string DumpRequestResponse(const CurlRequest& request,
                                absl::string_view payload,
                                const HttpResponse& response,
                                absl::string_view response_payload) {
  std::string out = request.method();
  if (out.empty()) out = payload.empty() ? "GET" : "POST";
  absl::StrAppend(&out, " ", request.url());
  if (!payload.empty()) {
    absl::StrAppend(&out, "\n\n", payload, "\n");
  }
  absl::StrAppend(&out, "\n", response.status_code);
  for (const auto& kv : response.headers) {
    absl::StrAppend(&out, "\n", kv.first, " ", kv.second);
  }
  if (!response_payload.empty()) {
    absl::StrAppend(&out, "\n\n", response_payload);
  }
  return out;
}

}  // namespace internal_http
}  // namespace tensorstore
