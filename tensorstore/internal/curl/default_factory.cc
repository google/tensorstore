// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/internal/curl/default_factory.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <optional>
#include <string>

#include "absl/base/attributes.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include <curl/curl.h>  // IWYU pragma: keep
#include "tensorstore/internal/curl/curl_factory.h"
#include "tensorstore/internal/curl/curl_wrappers.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/log/verbose_flag.h"

ABSL_FLAG(std::optional<bool>, tensorstore_curl_verbose, std::nullopt,
          "Enable curl verbose logging. "
          "Overrides TENSORSTORE_CURL_VERBOSE.");

ABSL_FLAG(std::optional<uint32_t>, tensorstore_curl_low_speed_time_seconds,
          std::nullopt,
          "Timeout threshold for low speed transfer detection. "
          "Overrides TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS.");

ABSL_FLAG(std::optional<uint32_t>, tensorstore_curl_low_speed_limit_bytes,
          std::nullopt,
          "Bytes threshold for low speed transfer detection. "
          "Overrides TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES.");

ABSL_FLAG(std::optional<std::string>, tensorstore_ca_path, std::nullopt,
          "CA path used with http connections. "
          "Overrides TENSORSTORE_CA_PATH.");

ABSL_FLAG(std::optional<std::string>, tensorstore_ca_bundle, std::nullopt,
          "CA Bundle used with http connections. "
          "Overrides TENSORSTORE_CA_BUNDLE.");

ABSL_FLAG(std::optional<uint32_t>, tensorstore_http2_max_concurrent_streams,
          std::nullopt,
          "Maximum concurrent streams for http2 connections. "
          "Overrides TENSORSTORE_HTTP2_MAX_CONCURRENT_STREAMS.");

using ::tensorstore::internal::GetFlagOrEnvValue;

namespace tensorstore {
namespace internal_http {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag curl_logging("curl");

// Concurrent CURL HTTP/2 streams.
[[maybe_unused]] int32_t GetMaxHttp2ConcurrentStreams() {
  auto limit = GetFlagOrEnvValue(FLAGS_tensorstore_http2_max_concurrent_streams,
                                 "TENSORSTORE_HTTP2_MAX_CONCURRENT_STREAMS");
  if (limit && (*limit <= 0 || *limit > 1000)) {
    ABSL_LOG(WARNING)
        << "Failed to parse TENSORSTORE_HTTP2_MAX_CONCURRENT_STREAMS: "
        << *limit;
    limit = std::nullopt;
  }
  return limit.value_or(4);  // New default streams.
}

}  // namespace

/* static */
DefaultCurlHandleFactory::Config DefaultCurlHandleFactory::DefaultConfig() {
  Config config;
  config.low_speed_time_seconds =
      GetFlagOrEnvValue(FLAGS_tensorstore_curl_low_speed_time_seconds,
                        "TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS")
          .value_or(0);
  config.low_speed_limit_bytes =
      GetFlagOrEnvValue(FLAGS_tensorstore_curl_low_speed_limit_bytes,
                        "TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES")
          .value_or(0);
  config.max_http2_concurrent_streams = GetMaxHttp2ConcurrentStreams();
  config.ca_path =
      GetFlagOrEnvValue(FLAGS_tensorstore_ca_path, "TENSORSTORE_CA_PATH");
  config.ca_bundle =
      GetFlagOrEnvValue(FLAGS_tensorstore_ca_bundle, "TENSORSTORE_CA_BUNDLE");
  config.verbose = GetFlagOrEnvValue(FLAGS_tensorstore_curl_verbose,
                                     "TENSORSTORE_CURL_VERBOSE")
                       .value_or(curl_logging.Level(0));
  config.verify_host = true;
  return config;
};

CurlPtr DefaultCurlHandleFactory::CreateHandle() {
  CurlPtr handle(curl_easy_init());
  SetLogToAbseil(handle.get());

  CurlPtrHook(handle);

  if (config_.verbose) {
    ABSL_CHECK_EQ(CURLE_OK,
                  curl_easy_setopt(handle.get(), CURLOPT_VERBOSE, 1L));
  }

  // For thread safety, don't use signals to time out name resolves (when
  // async name resolution is not supported).
  //
  // https://curl.haxx.se/libcurl/c/threadsafe.html
  ABSL_CHECK_EQ(CURLE_OK, curl_easy_setopt(handle.get(), CURLOPT_NOSIGNAL, 1L));

  // Follow curl command manpage to set up default values for low speed
  // timeout:
  // https://curl.se/docs/manpage.html#-Y
  if (config_.low_speed_time_seconds > 0 || config_.low_speed_limit_bytes > 0) {
    auto seconds = config_.low_speed_time_seconds > 0
                       ? config_.low_speed_time_seconds
                       : 30;
    auto bytes =
        config_.low_speed_limit_bytes > 0 ? config_.low_speed_limit_bytes : 1;
    ABSL_CHECK_EQ(CURLE_OK, curl_easy_setopt(handle.get(),
                                             CURLOPT_LOW_SPEED_TIME, seconds));
    ABSL_CHECK_EQ(CURLE_OK, curl_easy_setopt(handle.get(),
                                             CURLOPT_LOW_SPEED_LIMIT, bytes));
  }

  // Set ca_path or ca_bundle, if provided.
  if (config_.ca_path || config_.ca_bundle) {
    // Disable custom SSL CTX function.
    auto curle =
        curl_easy_setopt(handle.get(), CURLOPT_SSL_CTX_FUNCTION, nullptr);
    ABSL_CHECK(curle == CURLE_NOT_BUILT_IN || curle == CURLE_OK);

    if (auto& x = config_.ca_path) {
      ABSL_LOG_IF(INFO, curl_logging.Level(1)) << "Setting ca_path " << *x;
      ABSL_CHECK_EQ(CURLE_OK,
                    curl_easy_setopt(handle.get(), CURLOPT_CAPATH, x->c_str()));
    }
    if (auto& x = config_.ca_bundle) {
      ABSL_LOG_IF(INFO, curl_logging.Level(1)) << "Setting ca_bundle " << *x;
      ABSL_CHECK_EQ(CURLE_OK,
                    curl_easy_setopt(handle.get(), CURLOPT_CAINFO, x->c_str()));
    }
  }
  // Disable host verification if requested.
  if (!config_.verify_host) {
    ABSL_CHECK_EQ(CURLE_OK,
                  curl_easy_setopt(handle.get(), CURLOPT_SSL_VERIFYHOST, 0L));
    ABSL_CHECK_EQ(CURLE_OK,
                  curl_easy_setopt(handle.get(), CURLOPT_SSL_VERIFYPEER, 0L));
  }
  return handle;
};

CurlMulti DefaultCurlHandleFactory::CreateMultiHandle() {
  CurlMulti handle(curl_multi_init());

  // Without any option, the CURL library multiplexes up to 100 http/2
  // streams over a single connection. In practice there's a tradeoff
  // between concurrent streams and latency/throughput of requests.
  // Empirical tests suggest that using a small number of streams per
  // connection increases throughput of large transfers, which is common in
  // tensorstore.
  ABSL_CHECK_EQ(CURLM_OK,
                curl_multi_setopt(handle.get(), CURLMOPT_MAX_CONCURRENT_STREAMS,
                                  config_.max_http2_concurrent_streams));
  return handle;
}

// Returns the default CurlHandleFactory.
std::shared_ptr<CurlHandleFactory> GetDefaultCurlHandleFactory() {
  return std::make_shared<DefaultCurlHandleFactory>(
      DefaultCurlHandleFactory::DefaultConfig());
}

}  // namespace internal_http
}  // namespace tensorstore
