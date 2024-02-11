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

#include "tensorstore/internal/http/transport_test_utils.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock.h>
#include <ws2tcpip.h>

#pragma comment(lib, "ws2_32.lib")
#else  // !_WIN32

#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>  // for TCP_NODELAY
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#endif  // _WIN32

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace transport_test_utils {
namespace {

bool IsPortAvailable(uint16_t port) {
  struct sockaddr_storage peer_addr;
  memset(&peer_addr, 0, sizeof(peer_addr));
  socklen_t peer_len = 0;

  // Succeeds if either the IPv4 or IPv6 port can be bound.
  socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (sock < 0) {
    sock = socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP);
    if (sock < 0) {
      return false;
    }
    peer_len = sizeof(sockaddr_in6);
    auto* addr = reinterpret_cast<struct sockaddr_in6*>(&peer_addr);
    *addr = {};
    addr->sin6_family = AF_INET6;
    addr->sin6_port = ntohs(port);
  } else {
    peer_len = sizeof(sockaddr_in);
    auto* addr = reinterpret_cast<struct sockaddr_in*>(&peer_addr);
    *addr = {};
    addr->sin_family = AF_INET;
    addr->sin_port = ntohs(port);
  }

#ifndef _WIN32
  // On non-WIN32, set socket as close on exec.  Ignore errors.
  fcntl(sock, F_SETFD, FD_CLOEXEC);
#endif

  // Make 'reuse address' option available
  int yes = 1;
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char*>(&yes),
             sizeof(yes));
#ifdef SO_REUSEPORT
  setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, reinterpret_cast<char*>(&yes),
             sizeof(yes));
#endif

  // Try binding to port.
  int err =
      ::bind(sock, reinterpret_cast<struct sockaddr*>(&peer_addr), peer_len);
  CloseSocket(sock);
  return err == 0;
}

}  // namespace

// Platform specific defines.
#ifdef _WIN32
static constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#else   // _WIN32
static constexpr socket_t kInvalidSocket = -1;
#endif  // _WIN32

std::optional<uint16_t> TryGetPort(socket_t sock) {
  struct sockaddr_storage peer_addr;
  socklen_t peer_len = sizeof(peer_addr);
  if (getsockname(sock, (struct sockaddr*)&peer_addr, &peer_len)) {
    return std::nullopt;
  }
  if (reinterpret_cast<struct sockaddr*>(&peer_addr)->sa_family == AF_INET) {
    return ntohs(reinterpret_cast<struct sockaddr_in*>(&peer_addr)->sin_port);
  } else if (reinterpret_cast<struct sockaddr*>(&peer_addr)->sa_family ==
             AF_INET6) {
    return ntohs(reinterpret_cast<struct sockaddr_in6*>(&peer_addr)->sin6_port);
  }
  return std::nullopt;
}

std::optional<socket_t> CreateBoundSocket(uint16_t port) {
  auto try_open_socket = [](struct addrinfo* rp, uint16_t port) -> socket_t {
    // Create a socket
    //
    // NOTE: On WIN32 we could use WSASocket to prevent the socket from being
    // inherited and to enable overlapped io, but for now we skip that and stick
    // with the standard C interface.
    socket_t sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (sock < 0) return kInvalidSocket;

#ifndef _WIN32
    // On non-WIN32, set socket as close on exec.  Ignore errors.
    fcntl(sock, F_SETFD, FD_CLOEXEC);
#endif

    // Make 'reuse address' option available
    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char*>(&yes),
               sizeof(yes));
#ifdef SO_REUSEPORT
    setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, reinterpret_cast<char*>(&yes),
               sizeof(yes));
#endif

    // Ensure port is set to 0.
    if (rp->ai_family == AF_INET) {
      ((struct sockaddr_in*)rp->ai_addr)->sin_port = htons(port);
    } else if (rp->ai_family == AF_INET6) {
      ((struct sockaddr_in6*)rp->ai_addr)->sin6_port = htons(port);
    }

    // bind and listen
    if (::bind(sock, rp->ai_addr, static_cast<socklen_t>(rp->ai_addrlen))) {
      CloseSocket(sock);
      return kInvalidSocket;
    }
    if (::listen(sock, 5)) {  // Listen through 5 channels
      CloseSocket(sock);
      return kInvalidSocket;
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

  struct addrinfo* result = nullptr;
  if (getaddrinfo("localhost", nullptr, &hints, &result)) {
    return std::nullopt;
  }

  // Loop over the address families twice. On the first pass try and open
  // ipv4 sockets, and on the second pass try and open ipv6 sockets.
  for (auto* rp = result; rp; rp = rp->ai_next) {
    if (rp->ai_family == AF_INET) {
      auto sock = try_open_socket(rp, port);
      if (sock != kInvalidSocket) {
        freeaddrinfo(result);
        return sock;
      }
    }
  }
  for (auto* rp = result; rp; rp = rp->ai_next) {
    if (rp->ai_family == AF_INET6) {
      auto sock = try_open_socket(rp, port);
      if (sock != kInvalidSocket) {
        freeaddrinfo(result);
        return sock;
      }
    }
  }

  freeaddrinfo(result);
  return std::nullopt;
}

std::string FormatSocketAddress(socket_t sock) {
  struct sockaddr_storage peer_addr;
  socklen_t peer_len = sizeof(peer_addr);

  // Get the bound address.
  if (getsockname(sock, (struct sockaddr*)&peer_addr, &peer_len)) {
    return {};
  }

  const bool is_ipv6 = ((struct sockaddr*)&peer_addr)->sa_family == AF_INET6;
  char hbuf[1025], sbuf[32];
  if (0 == getnameinfo((struct sockaddr*)&peer_addr, peer_len, hbuf,
                       sizeof(hbuf), sbuf, sizeof(sbuf),
                       NI_NUMERICHOST | NI_NUMERICSERV)) {
    return tensorstore::StrCat(is_ipv6 ? "[" : "", hbuf, is_ipv6 ? "]:" : ":",
                               sbuf);
  }
  return {};
}

bool SetSocketNonBlocking(socket_t sock) {
  // Disable NAGLE by using TCP_NODELAY.
  int yes = 1;
  setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char*>(&yes),
             sizeof(yes));

  // Also set the fd/ioctl as non-blocking.
#ifdef _WIN32
  unsigned long mode = 1;
  return (ioctlsocket(sock, FIONBIO, &mode) == 0) ? true : false;
#else
  int flags = fcntl(sock, F_GETFL, 0);
  if (flags == -1) return false;
  return (fcntl(sock, F_SETFL, flags | O_NONBLOCK) == 0) ? true : false;
#endif
}

// Waits up to 200ms for a read.
bool WaitForRead(socket_t sock) {
  fd_set read;
  struct timeval tv;
  tv.tv_sec = 0;
  tv.tv_usec = 200000;

  for (;;) {
    FD_ZERO(&read);
    FD_SET(sock, &read);

    int sel = select(sock + 1, &read, nullptr, nullptr, &tv);
    if (sel < 0) continue;
    if (FD_ISSET(sock, &read)) return true;
    return false;
  }
}

std::optional<socket_t> AcceptNonBlocking(socket_t server_fd) {
  struct sockaddr_storage peer_addr;
  socklen_t peer_len = sizeof(peer_addr);

  socket_t client_fd =
      accept(server_fd, (struct sockaddr*)&peer_addr, &peer_len);
  if (client_fd == kInvalidSocket) return std::nullopt;

  SetSocketNonBlocking(client_fd);
  return client_fd;
}

std::string ReceiveAvailable(socket_t client_fd) {
  constexpr size_t kBufferSize = 4096;
  char buf[kBufferSize];
  std::string data;

  // Read as long as there is data available.
  for (; WaitForRead(client_fd);) {
    int r = recv(client_fd, buf, kBufferSize, 0);
    if (r < 0) {
      // WaitForRead calls select() on the socket, so we
      // should not see EAGAIN nor EWOULDBLOCK.
      ABSL_LOG(INFO) << "recv error: " << get_socket_errno();
      break;
    }
    // No data here would be unexpected since WaitForRead calls select().
    ABSL_CHECK_GT(r, 0);
    data.append(buf, r);
  }
  ABSL_LOG(INFO) << "recv " << data.size() << ": " << data;
  return data;
}

int AssertSend(socket_t client_fd, std::string_view data) {
  ABSL_LOG(INFO) << "send " << data.size() << ":" << data;
  int err = send(client_fd, data.data(), data.size(), 0);
  if (err < 0) {
    ABSL_LOG(INFO) << "send error:" << get_socket_errno();
  }
  assert(err == data.size());
  return err;
}

// Close the socket
void CloseSocket(socket_t fd) {
  if (fd == kInvalidSocket) return;
#ifdef _WIN32
  closesocket(fd);
#else
  close(fd);
#endif
}

std::optional<uint16_t> TryPickUnusedPort() {
  // Used ports tracks the ports handed out by this process to avoid
  // returning duplicates as they may not be immediately opened on retrieval.
  static internal::NoDestructor<absl::flat_hash_set<uint16_t>> used_ports;

  static constexpr uint16_t kMin = 32768;
  static constexpr uint16_t kMax = 60999;

  absl::BitGen bitgen;
  for (int i = 0; i < 100; i++) {
    uint16_t port = absl::Uniform(bitgen, kMin, kMax);
    if (used_ports->count(port) != 0) continue;

    if (IsPortAvailable(port)) {
      used_ports->insert(port);
      return port;
    }
  }

  ABSL_LOG(INFO) << "No unused port found. " << used_ports->size();
  return std::nullopt;
}

}  // namespace transport_test_utils
}  // namespace tensorstore

#ifdef _WIN32
// Called via atexit()
static void TensorstoreCleanupWinsock() { WSACleanup(); }

TENSORSTORE_GLOBAL_INITIALIZER {
  WSADATA winsock;
  WSAStartup(MAKEWORD(2, 2), &winsock);
  atexit(TensorstoreCleanupWinsock);
}
#endif
