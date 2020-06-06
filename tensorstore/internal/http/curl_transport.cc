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

#include "tensorstore/internal/http/curl_transport.h"

#include <stdlib.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include <curl/curl.h>
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_http {
namespace {

bool CurlVerboseEnabled() {
  // Cache the result.
  static bool value = std::getenv("TENSORSTORE_CURL_VERBOSE") != nullptr;
  return value;
}

struct CurlRequestState {
  CurlHandleFactory* factory_;
  CurlPtr handle_;
  CurlHeaders headers_;
  HttpResponse response_;

  CurlRequestState(CurlHandleFactory* factory)
      : factory_(factory), handle_(factory->CreateHandle()) {
    InitializeCurlHandle(handle_.get());

    CurlEasySetopt(handle_.get(), CURLOPT_WRITEDATA, this);
    CurlEasySetopt(handle_.get(), CURLOPT_WRITEFUNCTION,
                   &CurlRequestState::CurlWriteCallback);

    CurlEasySetopt(handle_.get(), CURLOPT_HEADERDATA, this);
    CurlEasySetopt(handle_.get(), CURLOPT_HEADERFUNCTION,
                   &CurlRequestState::CurlHeaderCallback);
  }

  ~CurlRequestState() {
    CurlEasySetopt(handle_.get(), CURLOPT_WRITEDATA,
                   static_cast<void*>(nullptr));
    CurlEasySetopt(handle_.get(), CURLOPT_WRITEFUNCTION,
                   static_cast<void*>(nullptr));
    CurlEasySetopt(handle_.get(), CURLOPT_HEADERDATA,
                   static_cast<void*>(nullptr));
    CurlEasySetopt(handle_.get(), CURLOPT_HEADERFUNCTION,
                   static_cast<void (*)()>(nullptr));

    factory_->CleanupHandle(std::move(handle_));
  }

  void Setup(const HttpRequest& request, std::string_view payload,
             absl::Duration request_timeout, absl::Duration connect_timeout) {
    // For thread safety, don't use signals to time out name resolves (when
    // async name resolution is not supported).
    //
    // https://curl.haxx.se/libcurl/c/threadsafe.html
    CurlEasySetopt(handle_.get(), CURLOPT_NOSIGNAL, 1L);
    if (CurlVerboseEnabled()) {
      CurlEasySetopt(handle_.get(), CURLOPT_VERBOSE, 1L);
    }
    std::string user_agent = request.user_agent() + GetCurlUserAgentSuffix();
    CurlEasySetopt(handle_.get(), CURLOPT_USERAGENT, user_agent.c_str());

    CurlEasySetopt(handle_.get(), CURLOPT_URL, request.url().c_str());

    // Convert headers to a curl slist
    curl_slist* head = nullptr;
    for (const std::string& h : request.headers()) {
      head = curl_slist_append(head, h.c_str());
    }
    headers_.reset(head);
    CurlEasySetopt(handle_.get(), CURLOPT_HTTPHEADER, headers_.get());
    if (request.accept_encoding()) {
      CurlEasySetopt(handle_.get(), CURLOPT_ACCEPT_ENCODING, "");
    }

    if (request_timeout > absl::ZeroDuration()) {
      auto ms = absl::ToInt64Milliseconds(request_timeout);
      CurlEasySetopt(handle_.get(), CURLOPT_TIMEOUT_MS, ms > 0 ? ms : 1);
    }
    if (connect_timeout > absl::ZeroDuration()) {
      auto ms = absl::ToInt64Milliseconds(connect_timeout);
      CurlEasySetopt(handle_.get(), CURLOPT_CONNECTTIMEOUT_MS, ms > 0 ? ms : 1);
    }
    if (!request.method().empty()) {
      CurlEasySetopt(handle_.get(), CURLOPT_CUSTOMREQUEST,
                     request.method().c_str());
    }

    if (!payload.empty()) {
      if (!request.method().empty()) {
        TENSORSTORE_LOG("Changing custom http method [", request.method(),
                        "] to POST");
      }
      CurlEasySetopt(handle_.get(), CURLOPT_POST, 1);
      CurlEasySetopt(handle_.get(), CURLOPT_POSTFIELDSIZE_LARGE,
                     payload.length());
      CurlEasySetopt(handle_.get(), CURLOPT_POSTFIELDS, payload.data());
    } else if (request.method().empty()) {
      CurlEasySetopt(handle_.get(), CURLOPT_HTTPGET, 1);
    }
  }

  size_t WriteCallback(absl::string_view data) {
    response_.payload.append(data.data(), data.size());
    return data.size();
  }

  size_t HeaderCallback(absl::string_view data) {
    return AppendHeaderData(response_.headers, data);
  }

  static std::size_t CurlWriteCallback(void* contents, std::size_t size,
                                       std::size_t nmemb, void* userdata) {
    return static_cast<CurlRequestState*>(userdata)->WriteCallback(
        absl::string_view(static_cast<char const*>(contents), size * nmemb));
  }

  static std::size_t CurlHeaderCallback(void* contents, std::size_t size,
                                        std::size_t nmemb, void* userdata) {
    return static_cast<CurlRequestState*>(userdata)->HeaderCallback(
        absl::string_view(static_cast<char const*>(contents), size * nmemb));
  }
};

}  // namespace

CurlTransport::CurlTransport(std::shared_ptr<CurlHandleFactory> factory)
    : factory_(factory) {}

CurlTransport::~CurlTransport() = default;

Future<HttpResponse> CurlTransport::IssueRequest(
    const HttpRequest& request, absl::string_view payload,
    absl::Duration request_timeout, absl::Duration connect_timeout) {
  CurlRequestState state(factory_.get());
  state.Setup(request, payload, request_timeout, connect_timeout);

  auto status = CurlEasyPerform(state.handle_.get());
  if (!status.ok()) return std::move(status);
  state.response_.status_code = CurlGetResponseCode(state.handle_.get());
  return std::move(state.response_);
}

namespace {
struct GlobalTransport {
  GlobalTransport()
      : transport(
            std::make_shared<CurlTransport>(GetDefaultCurlHandleFactory())) {}

  std::shared_ptr<HttpTransport> transport;
};

ABSL_CONST_INIT absl::Mutex global_mu(absl::kConstInit);

static GlobalTransport& GetGlobalTransport() {
  static auto* g = new GlobalTransport();
  return *g;
}

}  // namespace

std::shared_ptr<HttpTransport> GetDefaultHttpTransport() {
  absl::MutexLock l(&global_mu);
  return GetGlobalTransport().transport;
}

/// Sets the default CurlTransport. Exposed for test mocking.
void SetDefaultHttpTransport(std::shared_ptr<HttpTransport> t) {
  absl::MutexLock l(&global_mu);
  if (!t) {
    t = std::make_shared<CurlTransport>(GetDefaultCurlHandleFactory());
  }
  GetGlobalTransport().transport = std::move(t);
}

}  // namespace internal_http
}  // namespace tensorstore
