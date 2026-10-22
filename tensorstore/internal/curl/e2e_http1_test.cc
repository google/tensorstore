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

#include "tensorstore/internal/http/e2e_http1_test.h"

#include <memory>

#include <gtest/gtest.h>
#include <curl/curl.h>
#include "tensorstore/internal/curl/curl_transport.h"
#include "tensorstore/internal/curl/default_factory.h"

namespace tensorstore {
namespace internal_http {
namespace {

bool CurlSupportsEarlyFinalResponse() {
  const auto* info = curl_version_info(CURLVERSION_NOW);
  return info && info->version_num >= 0x080700;
}

Http1TestParam MakeCurlTestParam() {
  Http1TestParam param;
  param.factory = []() {
    return std::make_shared<CurlTransport>(GetDefaultCurlHandleFactory());
  };
  param.supports_early_final_response = CurlSupportsEarlyFinalResponse();
  return param;
}

// Instantiates end-to-end HTTP/1.1 test suite for CurlTransport.
INSTANTIATE_TEST_SUITE_P(CurlTransport, Http1TransportTest,
                         ::testing::Values(MakeCurlTestParam()));

}  // namespace
}  // namespace internal_http
}  // namespace tensorstore
