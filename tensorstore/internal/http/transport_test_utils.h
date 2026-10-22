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

#ifndef TENSORSTORE_INTERNAL_HTTP_TRANSPORT_TEST_UTILS_H_
#define TENSORSTORE_INTERNAL_HTTP_TRANSPORT_TEST_UTILS_H_

// This file includes test-only methods used in the transport tests.

#ifdef _WIN32
#undef UNICODE
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#else
#include <errno.h>
#endif

#include <stdint.h>

#include <atomic>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/thread/thread.h"

namespace tensorstore {
namespace transport_test_utils {

// Platform specific defines.
#ifdef _WIN32

using socket_t = SOCKET;
inline int get_socket_errno() { return WSAGetLastError(); }

#else  // _WIN32

using socket_t = int;
inline int get_socket_errno() { return errno; }

#endif  // _WIN32

// Creates a socket bound to localhost on an open port.
std::optional<socket_t> CreateBoundSocket(uint16_t port = 0);

// Returns the port of a bound socket.
std::optional<uint16_t> TryGetPort(socket_t sock);

// Formats a socket address as a host:port / [host]:port.
std::string FormatSocketAddress(socket_t sock);

// Sets a socket to non-blocking. Disables NAGLE.
bool SetSocketNonBlocking(socket_t sock);

// Disables Nagle, so that a write followed by a pause reaches the peer as a
// separate read, and suppresses SIGPIPE where the platform uses a socket
// option for that.
void SetSocketNoDelay(socket_t sock);

// Waits up to 200ms for a read.
bool WaitForRead(socket_t sock);

// Waits until `sock` is readable or hung up. Returns false if `stop` is set,
// `deadline` passes, or polling fails.
bool WaitForRead(socket_t sock, const std::atomic<bool>& stop,
                 absl::Time deadline = absl::InfiniteFuture());

// Accepts a client socket & sets the socke to non-blocking.
std::optional<socket_t> AcceptNonBlocking(socket_t server_fd);

// Receives available data on the socket, stopping at EOF.
std::string ReceiveAvailable(socket_t client_fd);

// Sends data on the socket, asserting that all of it was sent.
int AssertSend(socket_t client_fd, std::string_view data);

// Sends all of `data` on a blocking socket. Returns false if the peer has
// gone away; never raises SIGPIPE.
bool SendAll(socket_t sock, std::string_view data);

// Shuts down both directions, unblocking any thread waiting on `sock`.
void ShutdownSocket(socket_t sock);

// Shuts down the send direction, signalling EOF to the peer.
void ShutdownSocketSend(socket_t sock);

// Close the socket
void CloseSocket(socket_t fd);

// Attempts to pick an unused port.
std::optional<uint16_t> TryPickUnusedPort();

// Loopback TCP server that accepts on a dedicated thread and serves each
// connection on its own thread with blocking sockets.
//
// The handler owns the protocol; it should return when `stop` is set or the
// peer disconnects. The destructor stops accepting, shuts down every accepted
// socket, and joins all threads, so a handler may reference state of an
// enclosing object as long as the LoopbackServer member is declared after
// that state.
class LoopbackServer {
 public:
  using ConnectionHandler = std::function<void(
      socket_t fd, int connection_index, const std::atomic<bool>& stop)>;

  explicit LoopbackServer(ConnectionHandler handler);
  ~LoopbackServer();

  LoopbackServer(const LoopbackServer&) = delete;
  LoopbackServer& operator=(const LoopbackServer&) = delete;

  // host:port / [host]:port of the listening socket.
  const std::string& hostport() const { return hostport_; }

  int accepted_connections() const { return accepted_connections_.load(); }

 private:
  void AcceptLoop();

  ConnectionHandler handler_;
  socket_t listen_fd_;
  std::string hostport_;
  std::atomic<bool> stop_{false};
  std::atomic<int> accepted_connections_{0};
  std::vector<internal::Thread> worker_threads_;  // Accept thread only.
  absl::Mutex mu_;
  std::vector<socket_t> client_fds_ ABSL_GUARDED_BY(mu_);
  internal::Thread accept_thread_;
};

}  // namespace transport_test_utils
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_TRANSPORT_TEST_UTILS_H_
