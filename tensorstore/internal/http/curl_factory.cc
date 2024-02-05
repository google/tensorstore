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

#include "tensorstore/internal/http/curl_factory.h"

#include <memory>

#include "absl/base/call_once.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include <curl/curl.h>
#include "tensorstore/internal/http/curl_wrappers.h"

namespace tensorstore {
namespace internal_http {
namespace {

static absl::once_flag g_init;

// See curl:src/lib/curl_trc.c
static constexpr const char* kCurlTypeStrings[] = {
    ": * ",  // CURLINFO_TEXT,
    ": < ",  // CURLINFO_HEADER_IN,
    ": > ",  // CURLINFO_HEADER_OUT,
    ": ",    // CURLINFO_DATA_IN,
    ": ",    // CURLINFO_DATA_OUT,
    ": ",    // CURLINFO_SSL_DATA_IN,
    ": ",    // CURLINFO_SSL_DATA_OUT,
};

int CurlLogToAbseil(CURL* handle, curl_infotype type, char* data, size_t size,
                    void* userp) {
  switch (type) {
    case CURLINFO_TEXT:
    case CURLINFO_HEADER_OUT:
    case CURLINFO_HEADER_IN:
      break;
    default: /* nada */
      return 0;
  }
  ABSL_LOG(INFO) << handle << kCurlTypeStrings[type]
                 << std::string_view(data, size);
  return 0;
}

/// DefaultCurlHandleFactory generates a new handle on each request.
class DefaultCurlHandleFactory : public CurlHandleFactory {
 public:
  DefaultCurlHandleFactory() = default;

  CurlPtr CreateHandle() override {
    CurlPtr handle(curl_easy_init());
    curl_easy_setopt(handle.get(), CURLOPT_DEBUGFUNCTION, CurlLogToAbseil);
    return handle;
  };

  void CleanupHandle(CurlPtr&& h) override { h.reset(); }

  CurlMulti CreateMultiHandle() override {
    return CurlMulti(curl_multi_init());
  }

  void CleanupMultiHandle(CurlMulti&& m) override { m.reset(); }
};

/// TODO: Implement a CurlHandleFactory which caches values.

}  // namespace

std::shared_ptr<CurlHandleFactory> GetDefaultCurlHandleFactory() {
  // libcurl depends on many different SSL libraries, depending on the library
  // the library might not be thread safe. We defer such considerations for
  // now. https://curl.haxx.se/libcurl/c/threadsafe.html
  //
  // Automatically initialize the libcurl library.
  // Don't call `curl_global_cleanup()` since it is pointless anyway and
  // potentially leads to order of destruction race conditions.
  absl::call_once(g_init, [] { curl_global_init(CURL_GLOBAL_ALL); });

  return std::make_shared<DefaultCurlHandleFactory>();
}

}  // namespace internal_http
}  // namespace tensorstore
