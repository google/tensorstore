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

#include "tensorstore/internal/http/e2e_h2_interop_test.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/call_once.h"
#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/e2e_test_util.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/http/test_httpserver.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status_testutil.h"

namespace tensorstore {
namespace internal_http {
namespace {

using ::tensorstore::StatusIs;
using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Eq;
using ::testing::HasSubstr;

TestHttpServer& GetTlsHttpServer() {
  static absl::NoDestructor<TestHttpServer> testserver;
  static absl::once_flag init_once;
  absl::call_once(init_once, [&]() { testserver->SpawnProcess(); });
  return *testserver;
}

// Points the transport at the test server's self-signed certificate. Peer
// and host verification stay disabled: the certificate is not issued for the
// loopback address the tests dial.
H2InteropTestOptions TestServerTlsOptions(TestHttpServer& server) {
  H2InteropTestOptions options;
  options.ca_bundle_path = server.GetCertPath();
  options.verify_peer = false;
  options.verify_host = false;
  return options;
}

IssueRequestOptions DefaultHttp2Options(absl::Duration timeout) {
  return IssueRequestOptions()
      .SetHttpVersion(IssueRequestOptions::HttpVersion::kHttp2TLS)
      .SetRequestTimeout(timeout)
      .SetConnectTimeout(timeout);
}

IssueRequestOptions DefaultHttp2Options(absl::Cord payload,
                                        absl::Duration timeout) {
  return IssueRequestOptions(std::move(payload))
      .SetHttpVersion(IssueRequestOptions::HttpVersion::kHttp2TLS)
      .SetRequestTimeout(timeout)
      .SetConnectTimeout(timeout);
}

// Verifies that requesting a nonexistent resource returns HTTP 404.
TEST_P(H2InteropTransportTest, GetMissingResourceReturns404) {
  auto& server = GetTlsHttpServer();
  auto transport = CreateTransport(TestServerTlsOptions(server));
  const std::string url =
      absl::StrCat("https://", server.http_address(), "/missing_resource");

  HttpRequest req = HttpRequestBuilder("GET", url).BuildRequest();
  auto future =
      transport->IssueRequest(req, DefaultHttp2Options(request_timeout()));

  EXPECT_THAT(
      future.result(),
      HasStatusAndPayload(404, absl::Cord("Not Found: /missing_resource\n")));
}

// Verifies HTTP/2 PUT body upload and subsequent GET round-trip payload parity.
TEST_P(H2InteropTransportTest, PutAndGetRoundTripPreservesPayload) {
  auto& server = GetTlsHttpServer();
  auto transport = CreateTransport(TestServerTlsOptions(server));
  const std::string url =
      absl::StrCat("https://", server.http_address(), "/roundtrip_item");

  std::string test_payload = "HTTP/2 zero-copy round-trip test payload 12345";
  HttpRequest put_req = HttpRequestBuilder("PUT", url).BuildRequest();
  auto put_future = transport->IssueRequest(
      put_req,
      DefaultHttp2Options(absl::Cord(test_payload), request_timeout()));

  EXPECT_THAT(put_future.result(),
              HasStatusAndPayload(
                  200, absl::Cord(absl::StrCat("Received ", test_payload.size(),
                                               " bytes\n"))));

  HttpRequest get_req = HttpRequestBuilder("GET", url).BuildRequest();
  auto get_future =
      transport->IssueRequest(get_req, DefaultHttp2Options(request_timeout()));

  EXPECT_THAT(get_future.result(),
              HasStatusAndPayload(200, absl::Cord(test_payload)));
}

// Verifies HTTP/2 DELETE removes resource and subsequent GET returns 404.
TEST_P(H2InteropTransportTest, DeleteRemovesResource) {
  auto& server = GetTlsHttpServer();
  auto transport = CreateTransport(TestServerTlsOptions(server));
  const std::string url =
      absl::StrCat("https://", server.http_address(), "/item_to_delete");

  HttpRequest put_req = HttpRequestBuilder("PUT", url).BuildRequest();
  auto put_future = transport->IssueRequest(
      put_req,
      DefaultHttp2Options(absl::Cord("to be deleted"), request_timeout()));
  ASSERT_THAT(put_future.result(), HasStatusCode(200));

  HttpRequest del_req = HttpRequestBuilder("DELETE", url).BuildRequest();
  auto del_future =
      transport->IssueRequest(del_req, DefaultHttp2Options(request_timeout()));
  EXPECT_THAT(del_future.result(), HasStatusCode(200));

  HttpRequest get_req = HttpRequestBuilder("GET", url).BuildRequest();
  auto get_future =
      transport->IssueRequest(get_req, DefaultHttp2Options(request_timeout()));
  EXPECT_THAT(get_future.result(), HasStatusCode(404));
}

// Verifies HTTP/2 HEAD delivers Content-Length and headers with empty payload.
TEST_P(H2InteropTransportTest, HeadRequestReturnsHeadersWithoutBody) {
  auto& server = GetTlsHttpServer();
  auto transport = CreateTransport(TestServerTlsOptions(server));
  const std::string url =
      absl::StrCat("https://", server.http_address(), "/head_target");

  std::string data = "data-for-head-check";
  HttpRequest put_req = HttpRequestBuilder("PUT", url).BuildRequest();
  auto put_future = transport->IssueRequest(
      put_req, DefaultHttp2Options(absl::Cord(data), request_timeout()));
  ASSERT_THAT(put_future.result(), HasStatusCode(200));

  HttpRequest head_req = HttpRequestBuilder("HEAD", url).BuildRequest();
  auto head_future =
      transport->IssueRequest(head_req, DefaultHttp2Options(request_timeout()));

  EXPECT_THAT(
      head_future.result(),
      AllOf(HasStatusAndPayload(200, absl::Cord("")),
            HasResponseHeader("content-length", absl::StrCat(data.size()))));
}

// Verifies custom headers and query parameters are transmitted and echoed.
TEST_P(H2InteropTransportTest, EchoHeadersAndQueryParams) {
  auto& server = GetTlsHttpServer();
  auto transport = CreateTransport(TestServerTlsOptions(server));
  const std::string url =
      absl::StrCat("https://", server.http_address(), "/echo_test");

  // PATCH is used as an unhandled method that triggers _respond_echo in
  // h2_server.py while still having upload body support in libcurl.
  HttpRequest req = HttpRequestBuilder("PATCH", url)
                        .AddHeader("x-test-header", "header_value_123")
                        .AddQueryParameter("k", "v")
                        .BuildRequest();
  req.user_agent = "h2_test_agent";

  auto future = transport->IssueRequest(
      req, DefaultHttp2Options(absl::Cord("echo-payload"), request_timeout()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resp, future.result());
  EXPECT_THAT(resp.status_code, Eq(200));
  EXPECT_THAT(std::string(resp.payload),
              AllOf(HasSubstr("/echo_test?k=v"), HasSubstr("header_value_123"),
                    HasSubstr("h2_test_agent"), HasSubstr("echo-payload")));
}

// Verifies zero-copy streaming of multi-chunk payloads across flow windows.
TEST_P(H2InteropTransportTest, LargePayloadZeroCopyStreaming) {
  auto& server = GetTlsHttpServer();
  auto transport = CreateTransport(TestServerTlsOptions(server));
  const std::string url =
      absl::StrCat("https://", server.http_address(), "/large_file");

  std::string chunk1(32768, 'X');
  std::string chunk2(32768, 'Y');
  absl::Cord payload;
  payload.Append(chunk1);
  payload.Append(chunk2);

  HttpRequest req = HttpRequestBuilder("POST", url).BuildRequest();
  auto future = transport->IssueRequest(
      req, DefaultHttp2Options(payload, request_timeout()));

  EXPECT_THAT(
      future.result(),
      HasStatusAndPayload(200, absl::Cord(absl::StrCat(
                                   "Received ", payload.size(), " bytes\n"))));
}

// Verifies concurrent requests are multiplexed across a single H2 connection.
TEST_P(H2InteropTransportTest, ConcurrentMultiplexedStreamsOnSingleConnection) {
  auto& server = GetTlsHttpServer();
  auto transport = CreateTransport(TestServerTlsOptions(server));
  const std::string base_url = absl::StrCat("https://", server.http_address());

  constexpr int kNumConcurrent = 16;
  std::vector<Future<HttpResponse>> futures;
  futures.reserve(kNumConcurrent);

  for (int i = 0; i < kNumConcurrent; ++i) {
    HttpRequest req =
        HttpRequestBuilder("PATCH", absl::StrCat(base_url, "/stream/", i))
            .BuildRequest();
    std::string body = absl::StrCat("concurrent_payload_", i);
    futures.push_back(transport->IssueRequest(
        req, DefaultHttp2Options(absl::Cord(body), request_timeout())));
  }

  for (int i = 0; i < kNumConcurrent; ++i) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resp, futures[i].result());
    EXPECT_THAT(resp.status_code, Eq(200));
    EXPECT_THAT(std::string(resp.payload),
                AllOf(HasSubstr(absl::StrCat("/stream/", i)),
                      HasSubstr(absl::StrCat("concurrent_payload_", i))));
  }
}

// Verifies response headers missing mandatory :status are rejected as error.
TEST_P(H2InteropTransportTest, MalformedNoStatusResponseFails) {
  auto& server = GetTlsHttpServer();
  auto transport = CreateTransport(TestServerTlsOptions(server));
  const std::string url =
      absl::StrCat("https://", server.http_address(), "/malformed/no_status");

  HttpRequest req = HttpRequestBuilder("GET", url).BuildRequest();
  auto future =
      transport->IssueRequest(req, DefaultHttp2Options(request_timeout()));

  EXPECT_THAT(
      future.result(),
      StatusIs(AnyOf(absl::StatusCode::kUnknown, absl::StatusCode::kUnavailable,
                     absl::StatusCode::kInternal,
                     absl::StatusCode::kInvalidArgument)));
}

// Verifies gzip decompression across 2-byte DATA frames interleaved with
// 0-length DATA frames and server-initiated PING frames.
TEST_P(H2InteropTransportTest, FragmentedGzipWithZeroLengthDataAndPings) {
  auto& server = GetTlsHttpServer();
  auto transport = CreateTransport(TestServerTlsOptions(server));
  const std::string url =
      absl::StrCat("https://", server.http_address(), "/fragmented_gzip");

  HttpRequest req =
      HttpRequestBuilder("GET", url).EnableAcceptEncoding().BuildRequest();
  auto future =
      transport->IssueRequest(req, DefaultHttp2Options(request_timeout()));

  std::string expected;
  for (int i = 0; i < 8; ++i) {
    expected.append("Fragmented gzip stream payload. ");
  }
  EXPECT_THAT(future.result(), HasStatusAndPayload(200, absl::Cord(expected)));
}

// Verifies large header blocks split across HEADERS + CONTINUATION frames.
TEST_P(H2InteropTransportTest, LargeHeadersAcrossContinuationFrames) {
  auto& server = GetTlsHttpServer();
  auto transport = CreateTransport(TestServerTlsOptions(server));
  const std::string url =
      absl::StrCat("https://", server.http_address(), "/large_headers");

  HttpRequest req = HttpRequestBuilder("GET", url).BuildRequest();
  auto future =
      transport->IssueRequest(req, DefaultHttp2Options(request_timeout()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resp, future.result());
  EXPECT_THAT(resp.status_code, Eq(200));
  EXPECT_THAT(resp.payload, Eq(absl::Cord("continuation_headers_ok")));
  for (int i = 0; i < 8; ++i) {
    auto it = resp.headers.find(absl::StrCat("x-large-", i));
    ASSERT_NE(it, resp.headers.end());
    EXPECT_THAT(it->second.size(), Eq(3000));
  }
}

// Verifies transparent decompression of gzip Content-Encoding response body.
TEST_P(H2InteropTransportTest, TransparentGzipDecompression) {
  auto& server = GetTlsHttpServer();
  auto transport = CreateTransport(TestServerTlsOptions(server));
  const std::string url =
      absl::StrCat("https://", server.http_address(), "/compressed");

  HttpRequest req =
      HttpRequestBuilder("GET", url).EnableAcceptEncoding().BuildRequest();
  auto future =
      transport->IssueRequest(req, DefaultHttp2Options(request_timeout()));

  std::string expected_pattern =
      "TensorStore HTTP/2 transparent compression test payload. "
      "Repeatable string pattern for gzip compression efficiency. ";
  std::string expected_payload;
  expected_payload.reserve(expected_pattern.size() * 10);
  for (int i = 0; i < 10; ++i) {
    expected_payload.append(expected_pattern);
  }

  EXPECT_THAT(future.result(),
              HasStatusAndPayload(200, absl::Cord(expected_payload)));
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    internal::SetEnv(name_, value);
  }
  ~ScopedEnvVar() { internal::UnsetEnv(name_); }

 private:
  const char* name_;
};

// Verifies HTTP/2 requests tunnel through an HTTP CONNECT proxy when
// https_proxy / http_proxy environment variables are configured.
TEST_P(H2InteropTransportTest, WithProxy) {
  auto& server = GetTlsHttpServer();
  const std::string proxy_url = absl::StrCat("http://", server.proxy_address());
  ScopedEnvVar https_proxy("https_proxy", proxy_url.c_str());
  ScopedEnvVar http_proxy("http_proxy", proxy_url.c_str());

  auto transport = CreateTransport(TestServerTlsOptions(server));
  const std::string url =
      absl::StrCat("https://", server.http_address(), "/proxied_item");

  std::string payload = "proxied-http2-payload";
  HttpRequest put_req = HttpRequestBuilder("PUT", url).BuildRequest();
  auto put_future = transport->IssueRequest(
      put_req, DefaultHttp2Options(absl::Cord(payload), request_timeout()));
  EXPECT_THAT(put_future.result(),
              AllOf(HasStatusAndPayload(
                        200, absl::Cord(absl::StrCat(
                                 "Received ", payload.size(), " bytes\n"))),
                    HasResponseHeader("x-via-proxy", "1")));

  HttpRequest get_req = HttpRequestBuilder("GET", url).BuildRequest();
  auto get_future =
      transport->IssueRequest(get_req, DefaultHttp2Options(request_timeout()));
  EXPECT_THAT(get_future.result(),
              AllOf(HasStatusAndPayload(200, absl::Cord(payload)),
                    HasResponseHeader("x-via-proxy", "1")));
}

// Verifies proxy environment variables fail when pointing to a non-proxy
// HTTP/2 server.
TEST_P(H2InteropTransportTest, WithInvalidProxy) {
  auto& server = GetTlsHttpServer();
  const std::string base_url = absl::StrCat("https://", server.http_address());
  ScopedEnvVar https_proxy("https_proxy", base_url.c_str());
  ScopedEnvVar http_proxy("http_proxy", base_url.c_str());

  auto transport = CreateTransport(TestServerTlsOptions(server));

  HttpRequest req = HttpRequestBuilder("GET", absl::StrCat(base_url, "/proxy"))
                        .BuildRequest();
  auto future =
      transport->IssueRequest(req, DefaultHttp2Options(request_timeout()));

  // TestHttpServer's H2 port is not a proxy; should return an unavailable
  // error.
  EXPECT_THAT(future.result(), StatusIs(absl::StatusCode::kUnavailable));
}

}  // namespace
}  // namespace internal_http
}  // namespace tensorstore
