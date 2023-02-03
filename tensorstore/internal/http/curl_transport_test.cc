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

#ifdef _WIN32
#undef UNICODE
#define WIN32_LEAN_AND_MEAN
#pragma comment(lib, "ws2_32.lib")
#endif

#include "tensorstore/internal/http/curl_transport.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <optional>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/transport_test_utils.h"
#include "tensorstore/internal/thread.h"

using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::transport_test_utils::AcceptNonBlocking;
using ::tensorstore::transport_test_utils::AssertSend;
using ::tensorstore::transport_test_utils::CloseSocket;
using ::tensorstore::transport_test_utils::CreateBoundSocket;
using ::tensorstore::transport_test_utils::FormatSocketAddress;
using ::tensorstore::transport_test_utils::ReceiveAvailable;
using ::tensorstore::transport_test_utils::socket_t;
using ::testing::HasSubstr;

namespace {

class CurlTransportTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
#ifdef _WIN32
    WSADATA wsaData;
    ABSL_CHECK(WSAStartup(MAKEWORD(2, 2), &wsaData) == 0);
#endif
  }
  static void TearDownTestCase() {
#ifdef _WIN32
    WSACleanup();
#endif
  }
};

TEST_F(CurlTransportTest, Http1) {
  auto transport = ::tensorstore::internal_http::GetDefaultHttpTransport();

  // This test sets up a simple single-request tcp/ip service which allows
  // us to mock a simple http server.
  // NOTE: It would be nice to expand this to provide, e.g. HTTP2 functionality.

  auto socket = CreateBoundSocket();
  ABSL_CHECK(socket.has_value());

  auto hostport = FormatSocketAddress(*socket);
  ABSL_CHECK(!hostport.empty());

  static constexpr char kResponse[] =  //
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/html\r\n"
      "\r\n"
      "<html>\n<body>\n<h1>Hello, World!</h1>\n</body>\n</html>\n";

  // Start a thread to handle a single request.
  std::string initial_request;
  tensorstore::internal::Thread serve_thread({"serve_thread"}, [&] {
    auto client_fd = AcceptNonBlocking(*socket);
    ABSL_CHECK(client_fd.has_value());
    initial_request = ReceiveAvailable(*client_fd);
    AssertSend(*client_fd, kResponse);
    CloseSocket(*client_fd);
  });

  // Issue a request.
  auto response = transport->IssueRequest(
      HttpRequestBuilder("POST", absl::StrCat("http://", hostport, "/"))
          .AddUserAgentPrefix("test")
          .AddHeader("X-foo: bar")
          .AddQueryParameter("name", "dragon")
          .AddQueryParameter("age", "1234")
          .EnableAcceptEncoding()
          .BuildRequest(),
      absl::Cord("Hello"));

  ABSL_LOG(INFO) << response.status();

  ABSL_LOG(INFO) << "Wait on server";
  serve_thread.Join();
  CloseSocket(*socket);

  EXPECT_THAT(initial_request, HasSubstr("POST /?name=dragon&age=1234"));
  EXPECT_THAT(initial_request,
              HasSubstr(absl::StrCat("Host: ", hostport, "\r\n")));

  // User-Agent versions change based on zlib, nghttp2, and curl versions.
  EXPECT_THAT(initial_request, HasSubstr("User-Agent: testtensorstore/0.1 "));

  EXPECT_THAT(initial_request, HasSubstr("Accept: */*\r\n"));
  EXPECT_THAT(initial_request, HasSubstr("X-foo: bar\r\n"));
  EXPECT_THAT(initial_request, HasSubstr("Content-Length: 5"));
  EXPECT_THAT(initial_request,
              HasSubstr("Content-Type: application/x-www-form-urlencoded\r\n"));
  EXPECT_THAT(initial_request, HasSubstr("Hello"));

  EXPECT_EQ(200, response.value().status_code);
  EXPECT_EQ("<html>\n<body>\n<h1>Hello, World!</h1>\n</body>\n</html>\n",
            response.value().payload);
}

// Tests that resending (using CURL_SEEKFUNCTION) works correctly.
TEST_F(CurlTransportTest, Http1Resend) {
  auto transport = ::tensorstore::internal_http::GetDefaultHttpTransport();

  auto socket = CreateBoundSocket();
  ABSL_CHECK(socket.has_value());

  auto hostport = FormatSocketAddress(*socket);
  ABSL_CHECK(!hostport.empty());

  // Include content-length to allow connection reuse.
  static constexpr char kResponse[] =  //
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/html\r\n"
      "Connection: Keep-Alive\r\n"
      "Content-Length: 53\r\n"
      "\r\n"
      "<html>\n<body>\n<h1>Hello, World!</h1>\n</body>\n</html>\n";

  // The client sends 2 requests; the server sees 3 requests because
  // it closes the connection after the second request before sending
  // the response.

  std::string seen_requests[3];
  tensorstore::internal::Thread serve_thread({"serve_thread"}, [&] {
    std::optional<socket_t> client_fd = std::nullopt;

    for (int i = 0; i < 3; i++) {
      if (!client_fd.has_value()) {
        ABSL_LOG(INFO) << "S: Waiting on listen";
        client_fd = AcceptNonBlocking(*socket);
        ABSL_CHECK(client_fd.has_value());
      }

      while (seen_requests[i].empty()) {
        seen_requests[i] = ReceiveAvailable(*client_fd);
      }
      ABSL_LOG(INFO) << "S: request " << i
                     << " size=" << seen_requests[i].size();

      if (i == 1) {
        // Terminate the connection after receiving the second request
        // (simulates the race condition under which the server closes the
        // connection due to a timeout just the client is reusing the connection
        // to send another request).
        CloseSocket(*client_fd);
        client_fd = std::nullopt;
        continue;
      }

      AssertSend(*client_fd, kResponse);
    }

    // cleanup.
    CloseSocket(*client_fd);
  });

  // Issue 2 requests.
  for (int i = 0; i < 2; ++i) {
    ABSL_LOG(INFO) << "C: send " << i;

    auto future = transport->IssueRequest(
        HttpRequestBuilder("POST", absl::StrCat("http://", hostport, "/"))
            .AddUserAgentPrefix("test")
            .AddHeader("X-foo: bar")
            .AddQueryParameter("name", "dragon")
            .AddQueryParameter("age", "1234")
            .EnableAcceptEncoding()
            .BuildRequest(),
        absl::Cord("Hello"));

    ABSL_LOG(INFO) << "C: " << i << " " << future.status();

    EXPECT_EQ(200, future.value().status_code);
    EXPECT_EQ("<html>\n<body>\n<h1>Hello, World!</h1>\n</body>\n</html>\n",
              future.value().payload);
  }

  ABSL_LOG(INFO) << "Wait on server";
  serve_thread.Join();
  CloseSocket(*socket);

  for (auto& request : seen_requests) {
    using ::testing::HasSubstr;
    EXPECT_THAT(request, HasSubstr("POST /?name=dragon&age=1234"));
    EXPECT_THAT(request, HasSubstr(absl::StrCat("Host: ", hostport, "\r\n")));

    // User-Agent versions change based on zlib, nghttp2, and curl versions.
    EXPECT_THAT(request, HasSubstr("User-Agent: testtensorstore/0.1 "));

    EXPECT_THAT(request, HasSubstr("Accept: */*\r\n"));
    EXPECT_THAT(request, HasSubstr("X-foo: bar\r\n"));
    EXPECT_THAT(request, HasSubstr("Content-Length: 5"));
    EXPECT_THAT(
        request,
        HasSubstr("Content-Type: application/x-www-form-urlencoded\r\n"));
    EXPECT_THAT(request, HasSubstr("Hello"));
  }
}

}  // namespace
