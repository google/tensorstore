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

#include "tensorstore/internal/http/e2e_http2_test.h"

#include <stdint.h>

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include <curl/curl.h>
#include <curl/multi.h>
#include "tensorstore/internal/curl/curl_transport.h"
#include "tensorstore/internal/curl/curl_wrappers.h"
#include "tensorstore/internal/curl/default_factory.h"
#include "tensorstore/internal/http/http_transport.h"

namespace tensorstore {
namespace internal_http {
namespace {

bool CurlSupportsStreamRetry() {
  const auto* info = curl_version_info(CURLVERSION_NOW);
  return info && info->version_num >= 0x080200;
}

bool CurlSupportsEarlyFinalResponse() {
  const auto* info = curl_version_info(CURLVERSION_NOW);
  return info && info->version_num >= 0x080700;
}

bool CurlSupportsReentrantRequests() {
  const auto* info = curl_version_info(CURLVERSION_NOW);
  return info && info->version_num >= 0x080200;
}

bool CurlSupportsMultiplexedStreamReset() {
  const auto* info = curl_version_info(CURLVERSION_NOW);
  return info && info->version_num >= 0x080200;
}

// Caps connections per host on each curl multi handle.
//
// CurlTransport owns one multi handle per http thread, so the cap is only
// global when the test runs with --tensorstore_http_threads=1 (see BUILD).
class HostLimitedCurlHandleFactory : public DefaultCurlHandleFactory {
 public:
  HostLimitedCurlHandleFactory(Config config, int64_t max_host_connections)
      : DefaultCurlHandleFactory(std::move(config)),
        max_host_connections_(max_host_connections) {}

  CurlMulti CreateMultiHandle() override {
    CurlMulti handle = DefaultCurlHandleFactory::CreateMultiHandle();
    ABSL_CHECK_EQ(CURLM_OK,
                  curl_multi_setopt(handle.get(), CURLMOPT_MAX_HOST_CONNECTIONS,
                                    static_cast<long>(max_host_connections_)));
    return handle;
  }

 private:
  int64_t max_host_connections_;
};

Http2TestParam MakeCurlTestParam() {
  Http2TestParam param;
  param.factory =
      [](const Http2TestOptions& opts) -> std::shared_ptr<HttpTransport> {
    if (opts.max_connections_per_host > 0) {
      return std::make_shared<CurlTransport>(
          std::make_shared<HostLimitedCurlHandleFactory>(
              DefaultCurlHandleFactory::DefaultConfig(),
              opts.max_connections_per_host));
    }
    return std::make_shared<CurlTransport>(GetDefaultCurlHandleFactory());
  };
  param.supports_stream_retry = CurlSupportsStreamRetry();
  param.supports_early_final_response = CurlSupportsEarlyFinalResponse();
  param.supports_reentrant_requests = CurlSupportsReentrantRequests();
  param.supports_multiplexed_stream_reset =
      CurlSupportsMultiplexedStreamReset();
  return param;
}

INSTANTIATE_TEST_SUITE_P(CurlTransport, Http2TransportTest,
                         ::testing::Values(MakeCurlTestParam()));

}  // namespace
}  // namespace internal_http
}  // namespace tensorstore
