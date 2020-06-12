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

#include "tensorstore/internal/http/curl_transport.h"

#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/util/executor.h"

#ifdef _WIN32

#undef UNICODE
#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")

using socket_t = SOCKET;

#else  // _WIN32

#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

using socket_t = int;
constexpr socket_t INVALID_SOCKET = -1;
int closesocket(socket_t fd) { return close(fd); }

#endif

using tensorstore::internal_http::HttpRequestBuilder;

namespace {

socket_t CreateBoundSocket() {
  auto try_open_socket = [](struct addrinfo *rp) -> socket_t {
    // Create a socket
    //
    // NOTE: On WIN32 we could use WSASocket to prevent the socket from being
    // inherited and to enable overlapped io, but for now we skip that and stick
    // with the standard C interface.
    socket_t sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (sock < 0) return INVALID_SOCKET;

    // Make 'reuse address' option available
    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char *>(&yes),
               sizeof(yes));
#ifdef SO_REUSEPORT
    setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, reinterpret_cast<char *>(&yes),
               sizeof(yes));
#endif

    // Ensure port is set to 0.
    if (rp->ai_family == AF_INET) {
      ((struct sockaddr_in *)rp->ai_addr)->sin_port = 0;
    } else if (rp->ai_family == AF_INET6) {
      ((struct sockaddr_in6 *)rp->ai_addr)->sin6_port = 0;
    }

    // bind and listen
    if (::bind(sock, rp->ai_addr, static_cast<socklen_t>(rp->ai_addrlen))) {
      closesocket(sock);
      return INVALID_SOCKET;
    }
    if (::listen(sock, 5)) {  // Listen through 5 channels
      closesocket(sock);
      return INVALID_SOCKET;
    }
    return sock;
  };

  // Get address info
  struct addrinfo hints;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  hints.ai_protocol = 0;

  struct addrinfo *result;
  if (getaddrinfo("localhost", nullptr, &hints, &result)) {
    return INVALID_SOCKET;
  }

  // Loop over the address families twice. On the first pass try and open
  // ipv4 sockets, and on the second pass try and open ipv6 sockets.
  for (auto *rp = result; rp; rp = rp->ai_next) {
    if (rp->ai_family == AF_INET) {
      auto sock = try_open_socket(rp);
      if (sock != INVALID_SOCKET) {
        freeaddrinfo(result);
        return sock;
      }
    }
  }
  for (auto *rp = result; rp; rp = rp->ai_next) {
    if (rp->ai_family == AF_INET6) {
      auto sock = try_open_socket(rp);
      if (sock != INVALID_SOCKET) {
        freeaddrinfo(result);
        return sock;
      }
    }
  }

  freeaddrinfo(result);
  return INVALID_SOCKET;
}

std::string FormatSocketAddress(socket_t sock) {
  struct sockaddr_storage peer_addr;
  socklen_t peer_len = sizeof(peer_addr);

  // Get the bound address.
  if (getsockname(sock, (struct sockaddr *)&peer_addr, &peer_len)) {
    return {};
  }

  const bool is_ipv6 = ((struct sockaddr *)&peer_addr)->sa_family == AF_INET6;
  char hbuf[1025], sbuf[32];
  if (0 == getnameinfo((struct sockaddr *)&peer_addr, peer_len, hbuf,
                       sizeof(hbuf), sbuf, sizeof(sbuf),
                       NI_NUMERICHOST | NI_NUMERICSERV)) {
    return absl::StrCat(is_ipv6 ? "[" : "", hbuf, is_ipv6 ? "]:" : ":", sbuf);
  }
  return {};
}

bool SetSocketNonBlocking(socket_t sock) {
#ifdef _WIN32
  unsigned long mode = 1;
  return (ioctlsocket(sock, FIONBIO, &mode) == 0) ? true : false;
#else
  int flags = fcntl(sock, F_GETFL, 0);
  if (flags == -1) return false;
  return (fcntl(sock, F_SETFL, flags | O_NONBLOCK) == 0) ? true : false;
#endif
}

// Waits up to 100ms for a read.
bool WaitForRead(socket_t sock) {
  fd_set read;
  struct timeval tv;
  tv.tv_sec = 0;
  tv.tv_usec = 100000;

  for (;;) {
    FD_ZERO(&read);
    FD_SET(sock, &read);

    int sel = select(sock + 1, &read, nullptr, nullptr, &tv);
    if (sel < 0) continue;
    if (FD_ISSET(sock, &read)) return true;
    return false;
  }
}

std::string AcceptAndRespond(socket_t server_fd, std::string my_response) {
  struct sockaddr_storage peer_addr;
  socklen_t peer_len = sizeof(peer_addr);

  socket_t client_fd =
      accept(server_fd, (struct sockaddr *)&peer_addr, &peer_len);
  if (client_fd < 0) return "ERROR";

  SetSocketNonBlocking(client_fd);

  constexpr size_t kBufferSize = 4096;
  char buf[kBufferSize];
  std::string request;

  for (; WaitForRead(client_fd);) {
    int r = recv(client_fd, buf, kBufferSize, 0);
    if (r <= 0) break;
    request.append(buf, r);
  }

  int err = send(client_fd, my_response.data(), my_response.size(), 0);
  if (err < 0) {
    printf("Error transmitting\n");
  }
  closesocket(client_fd);
  return request;
}

class CurlTransportTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
#ifdef _WIN32
    WSADATA wsaData;
    TENSORSTORE_CHECK(WSAStartup(MAKEWORD(2, 2), &wsaData) == 0);
#endif
  }
  static void TearDownTestCase() {
#ifdef _WIN32
    WSACleanup();
#endif
  }
};

TEST_F(CurlTransportTest, Basic) {
  auto transport = ::tensorstore::internal_http::GetDefaultHttpTransport();

  // This test sets up a simple single-request tcp/ip service which allows
  // us to mock a simple http server.
  // NOTE: It would be nice to expand this to provide, e.g. HTTP2 functionality.

  socket_t socket = CreateBoundSocket();
  TENSORSTORE_CHECK(socket != INVALID_SOCKET);

  auto hostport = FormatSocketAddress(socket);
  TENSORSTORE_CHECK(!hostport.empty());

  static constexpr char kResponse[] =  //
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/html\r\n"
      "\r\n"
      "<html>\n<body>\n<h1>Hello, World!</h1>\n</body>\n</html>\n";

  // Start a thread to handle a single request.
  std::string actual_request;
  std::thread serve_thread = std::thread([socket = socket, &actual_request] {
    actual_request = AcceptAndRespond(socket, kResponse);
    closesocket(socket);
  });

  // Issue a request.
  auto response = transport->IssueRequest(
      HttpRequestBuilder(absl::StrCat("http://", hostport, "/"))
          .AddUserAgentPrefix("test")
          .AddHeader("X-foo: bar")
          .AddQueryParameter("name", "dragon")
          .AddQueryParameter("age", "1234")
          .EnableAcceptEncoding()
          .BuildRequest(),
      "Hello");

  response.value();
  serve_thread.join();

  using ::testing::HasSubstr;
  EXPECT_THAT(actual_request, HasSubstr("/?name=dragon&age=1234"));
  EXPECT_THAT(actual_request,
              HasSubstr(absl::StrCat("Host: ", hostport, "\r\n")));

  // User-Agent versions change based on zlib, nghttp2, and curl versions.
  EXPECT_THAT(actual_request, HasSubstr("User-Agent: testtensorstore/0.1 "));

  EXPECT_THAT(actual_request, HasSubstr("Accept: */*\r\n"));
  EXPECT_THAT(actual_request, HasSubstr("Accept-Encoding: deflate, gzip\r\n"));
  EXPECT_THAT(actual_request, HasSubstr("X-foo: bar\r\n"));
  EXPECT_THAT(actual_request, HasSubstr("Content-Length: 5"));
  EXPECT_THAT(actual_request,
              HasSubstr("Content-Type: application/x-www-form-urlencoded\r\n"));
  EXPECT_THAT(actual_request, HasSubstr("Hello"));

  EXPECT_EQ(200, response.value().status_code);
  EXPECT_EQ("<html>\n<body>\n<h1>Hello, World!</h1>\n</body>\n</html>\n",
            response.value().payload);
}

}  // namespace
