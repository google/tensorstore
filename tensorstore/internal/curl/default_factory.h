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

#ifndef TENSORSTORE_INTERNAL_CURL_DEFAULT_FACTORY_H_
#define TENSORSTORE_INTERNAL_CURL_DEFAULT_FACTORY_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "tensorstore/internal/curl/curl_factory.h"
#include "tensorstore/internal/curl/curl_wrappers.h"

namespace tensorstore {
namespace internal_http {

/// DefaultCurlHandleFactory implements a default CurlHandleFactory to create
/// curl and curl_multi handles.
class DefaultCurlHandleFactory : public CurlHandleFactory {
 public:
  /// The default configuration for the default curl factory.
  struct Config {
    int64_t low_speed_time_seconds;
    int64_t low_speed_limit_bytes;
    int32_t max_http2_concurrent_streams;
    std::optional<std::string> ca_path;
    std::optional<std::string> ca_bundle;
    bool verbose;
    bool verify_host;
  };
  static Config DefaultConfig();

  explicit DefaultCurlHandleFactory(Config config)
      : config_(std::move(config)) {
    CurlInit();
  }

  CurlPtr CreateHandle() override;
  void CleanupHandle(CurlPtr&& h) override { h.reset(); }

  CurlMulti CreateMultiHandle() override;
  void CleanupMultiHandle(CurlMulti&& m) override { m.reset(); }

 private:
  Config config_;
};

/// Returns the default CurlHandleFactory.
std::shared_ptr<CurlHandleFactory> GetDefaultCurlHandleFactory();

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CURL_DEFAULT_FACTORY_H_
