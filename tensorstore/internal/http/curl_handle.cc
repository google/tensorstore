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

#include "tensorstore/internal/http/curl_handle.h"

#include <stdint.h>

#include <memory>
#include <string>
#include <string_view>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include <curl/curl.h>
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_http {
namespace {

// libcurl depends on many different SSL libraries, depending on the library
// the library might not be thread safe. We defer such considerations for
// now. https://curl.haxx.se/libcurl/c/threadsafe.html

/// Automatically initialize the libcurl library.
class CurlInitializer {
 public:
  CurlInitializer() { curl_global_init(CURL_GLOBAL_ALL); }
  // Don't call `curl_global_cleanup()` since it is pointless anyway and
  // potentially leads to order of destruction race conditions.
};

void CurlInitialize() { static CurlInitializer curl_initializer; }

/// DefaultCurlHandleFactory generates a new handle on each request.
class DefaultCurlHandleFactory : public CurlHandleFactory {
 public:
  DefaultCurlHandleFactory() { CurlInitialize(); }

  CurlPtr CreateHandle() override { return CurlPtr(curl_easy_init()); };

  void CleanupHandle(CurlPtr&& h) override { h.reset(); }

  CurlMulti CreateMultiHandle() override {
    return CurlMulti(curl_multi_init());
  }

  void CleanupMultiHandle(CurlMulti&& m) override { m.reset(); }
};

/// TODO: Implement a CurlHandleFactory which caches values.

}  // namespace

void CurlPtrCleanup::operator()(CURL* c) { curl_easy_cleanup(c); }
void CurlMultiCleanup::operator()(CURLM* m) { curl_multi_cleanup(m); }
void CurlSlistCleanup::operator()(curl_slist* s) { curl_slist_free_all(s); }

std::shared_ptr<CurlHandleFactory> GetDefaultCurlHandleFactory() {
  static std::shared_ptr<CurlHandleFactory> default_curl_handle_factory =
      std::make_shared<DefaultCurlHandleFactory>();
  return default_curl_handle_factory;
}

/// Returns the default CurlUserAgent.
std::string GetCurlUserAgentSuffix() {
  static std::string agent =
      tensorstore::StrCat("tensorstore/0.1 ", curl_version());
  return agent;
}

/// Returns a absl::Status object for a corresponding CURLcode.
absl::Status CurlCodeToStatus(CURLcode code, std::string_view detail) {
  // Constant errors:

  auto error_code = absl::StatusCode::kUnknown;
  switch (code) {
    case CURLE_OK:
      return absl::OkStatus();
    case CURLE_COULDNT_RESOLVE_PROXY:
      error_code = absl::StatusCode::kUnavailable;
      if (detail.empty()) detail = "Failed to resolve proxy";
      break;

    case CURLE_OPERATION_TIMEDOUT:
      error_code = absl::StatusCode::kDeadlineExceeded;
      if (detail.empty()) detail = "Timed out";
      break;

    case CURLE_COULDNT_RESOLVE_HOST:
    case CURLE_COULDNT_CONNECT:
    case CURLE_SEND_ERROR:
    case CURLE_RECV_ERROR:
    case CURLE_HTTP2:
    case CURLE_SSL_CONNECT_ERROR:
    case CURLE_HTTP2_STREAM:
    case CURLE_PARTIAL_FILE:
      error_code = absl::StatusCode::kUnavailable;
      break;

    case CURLE_UNSUPPORTED_PROTOCOL:
    case CURLE_URL_MALFORMAT:
      error_code = absl::StatusCode::kInvalidArgument;
      break;

    case CURLE_WRITE_ERROR:
      error_code = absl::StatusCode::kCancelled;
      break;
    default:
      break;
  }
  return absl::Status(
      error_code,
      tensorstore::StrCat("CURL error[", code, "] ", curl_easy_strerror(code),
                          detail.empty() ? "" : ": ", detail));
}

/// Returns a absl::Status object for a corresponding CURLcode.
absl::Status CurlMCodeToStatus(CURLMcode code, std::string_view detail) {
  if (code == CURLM_OK) {
    return absl::OkStatus();
  }
  return absl::InternalError(
      tensorstore::StrCat("CURLM error[", code, "] ", curl_multi_strerror(code),
                          detail.empty() ? "" : ": ", detail));
}

int32_t CurlGetResponseCode(CURL* handle) {
  long code = 0;  // NOLINT
  auto e = curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &code);
  if (e != CURLE_OK) {
    ABSL_LOG(WARNING) << "Error [" << e << "]=" << curl_easy_strerror(e)
                      << " in curl operation";
  }
  return static_cast<int32_t>(code);
}

}  // namespace internal_http
}  // namespace tensorstore
