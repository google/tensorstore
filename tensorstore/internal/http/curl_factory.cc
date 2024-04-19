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

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include <curl/curl.h>  // IWYU pragma: keep
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_wrappers.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/no_destructor.h"

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

namespace tensorstore {
namespace internal_http {
namespace {

using ::tensorstore::internal::GetFlagOrEnvValue;

ABSL_CONST_INIT internal_log::VerboseFlag curl_logging("curl");

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

// Cached configuration from environment variables.
struct CurlConfig {
  bool verbose = GetFlagOrEnvValue(FLAGS_tensorstore_curl_verbose,
                                   "TENSORSTORE_CURL_VERBOSE")
                     .value_or(curl_logging.Level(0));
  int64_t low_speed_time_seconds =
      GetFlagOrEnvValue(FLAGS_tensorstore_curl_low_speed_time_seconds,
                        "TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS")
          .value_or(0);
  int64_t low_speed_limit_bytes =
      GetFlagOrEnvValue(FLAGS_tensorstore_curl_low_speed_limit_bytes,
                        "TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES")
          .value_or(0);
  std::optional<std::string> ca_path =
      GetFlagOrEnvValue(FLAGS_tensorstore_ca_path, "TENSORSTORE_CA_PATH");
  std::optional<std::string> ca_bundle =
      GetFlagOrEnvValue(FLAGS_tensorstore_ca_bundle, "TENSORSTORE_CA_BUNDLE");
  int32_t max_http2_concurrent_streams = GetMaxHttp2ConcurrentStreams();
};

const CurlConfig& CurlEnvConfig() {
  static const internal::NoDestructor<CurlConfig> curl_config{};
  return *curl_config;
}

/// DefaultCurlHandleFactory generates a new handle on each request.
class DefaultCurlHandleFactory : public CurlHandleFactory {
 public:
  DefaultCurlHandleFactory() = default;

  CurlPtr CreateHandle() override {
    CurlPtr handle(curl_easy_init());
    CurlPtrHook(handle);

    ABSL_CHECK_EQ(
        CURLE_OK,
        curl_easy_setopt(handle.get(), CURLOPT_DEBUGFUNCTION, CurlLogToAbseil));

    auto& config = CurlEnvConfig();
    if (config.verbose) {
      ABSL_CHECK_EQ(CURLE_OK,
                    curl_easy_setopt(handle.get(), CURLOPT_VERBOSE, 1L));
    }

    // For thread safety, don't use signals to time out name resolves (when
    // async name resolution is not supported).
    //
    // https://curl.haxx.se/libcurl/c/threadsafe.html
    ABSL_CHECK_EQ(CURLE_OK,
                  curl_easy_setopt(handle.get(), CURLOPT_NOSIGNAL, 1L));

    // Follow curl command manpage to set up default values for low speed
    // timeout:
    // https://curl.se/docs/manpage.html#-Y
    if (config.low_speed_time_seconds > 0 || config.low_speed_limit_bytes > 0) {
      auto seconds = config.low_speed_time_seconds > 0
                         ? config.low_speed_time_seconds
                         : 30;
      auto bytes =
          config.low_speed_limit_bytes > 0 ? config.low_speed_limit_bytes : 1;
      ABSL_CHECK_EQ(
          CURLE_OK,
          curl_easy_setopt(handle.get(), CURLOPT_LOW_SPEED_TIME, seconds));
      ABSL_CHECK_EQ(CURLE_OK, curl_easy_setopt(handle.get(),
                                               CURLOPT_LOW_SPEED_LIMIT, bytes));
    }

    // Set ca_path or ca_bundle, if provided.
    if (config.ca_path || config.ca_bundle) {
      ABSL_CHECK_EQ(
          CURLE_OK,
          curl_easy_setopt(handle.get(), CURLOPT_SSL_CTX_FUNCTION, nullptr));

      if (auto& x = config.ca_path) {
        ABSL_CHECK_EQ(CURLE_OK, curl_easy_setopt(handle.get(), CURLOPT_CAPATH,
                                                 x->c_str()));
      }
      if (auto& x = config.ca_bundle) {
        ABSL_CHECK_EQ(CURLE_OK, curl_easy_setopt(handle.get(), CURLOPT_CAINFO,
                                                 x->c_str()));
      }
    }

    return handle;
  };

  void CleanupHandle(CurlPtr&& h) override { h.reset(); }

  CurlMulti CreateMultiHandle() override {
    CurlMulti handle(curl_multi_init());

    // Without any option, the CURL library multiplexes up to 100 http/2
    // streams over a single connection. In practice there's a tradeoff
    // between concurrent streams and latency/throughput of requests.
    // Empirical tests suggest that using a small number of streams per
    // connection increases throughput of large transfers, which is common in
    // tensorstore.
    auto& config = CurlEnvConfig();
    ABSL_CHECK_EQ(CURLM_OK, curl_multi_setopt(
                                handle.get(), CURLMOPT_MAX_CONCURRENT_STREAMS,
                                config.max_http2_concurrent_streams));
    return handle;
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
