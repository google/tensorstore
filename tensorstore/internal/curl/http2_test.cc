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

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/base/call_once.h"
#include "absl/base/no_destructor.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "tensorstore/internal/curl/curl_transport.h"
#include "tensorstore/internal/curl/default_factory.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/http/test_httpserver.h"

using ::tensorstore::internal_http::CurlTransport;
using ::tensorstore::internal_http::DefaultCurlHandleFactory;
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_http::IssueRequestOptions;
using ::tensorstore::internal_http::TestHttpServer;

namespace {

TestHttpServer& GetHttpServer() {
  static absl::NoDestructor<TestHttpServer> testserver;

  static absl::once_flag init_once;
  absl::call_once(init_once, [&]() {
    testserver->SpawnProcess();
    static std::string http_address = testserver->http_address();
    ABSL_LOG(INFO) << "Using " << http_address;
  });

  return *testserver;
}

std::shared_ptr<HttpTransport> GetTransport() {
  auto config = DefaultCurlHandleFactory::Config();
  config.ca_bundle = GetHttpServer().GetCertPath();
  config.verify_host = false;
  return std::make_shared<CurlTransport>(
      std::make_shared<DefaultCurlHandleFactory>(std::move(config)));
}

TEST(HttpserverTest, Basic) {
  auto base_url = absl::StrFormat("https://%s", GetHttpServer().http_address());
  auto transport = GetTransport();
  GetHttpServer().MaybeLogStdoutPipe();

  // Issue request 1.
  // Using SetHttpVersion adds the HTTP1.1 to HTTP2 upgrade headers.
  {
    auto response = transport->IssueRequest(
        HttpRequestBuilder("POST", absl::StrCat(base_url, "/file"))
            .AddHeader("x-foo", "bar")
            .AddQueryParameter("name", "dragon")
            .AddQueryParameter("age", "1234")
            .EnableAcceptEncoding()
            .BuildRequest(),
        IssueRequestOptions(absl::Cord("Hello"))
            .SetConnectTimeout(absl::Seconds(10))
            .SetRequestTimeout(absl::Seconds(10)));

    // Waits for the response.
    ABSL_LOG(INFO) << response.status();

    EXPECT_EQ(200, response.value().status_code);
    EXPECT_EQ("Received 5 bytes\n", response.value().payload);
  }
  GetHttpServer().MaybeLogStdoutPipe();

  {
    auto response = transport->IssueRequest(
        HttpRequestBuilder("GET", absl::StrCat(base_url, "/missing"))
            .BuildRequest(),
        IssueRequestOptions()
            .SetConnectTimeout(absl::Seconds(10))
            .SetRequestTimeout(absl::Seconds(10)));

    // Waits for the response.
    ABSL_LOG(INFO) << response.status();

    EXPECT_EQ(404, response.value().status_code);
    EXPECT_EQ("Not Found: /missing\n", response.value().payload);
  }
  GetHttpServer().MaybeLogStdoutPipe();
}

}  // namespace
