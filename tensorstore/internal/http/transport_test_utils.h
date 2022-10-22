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

#include <cstdio>
#include <optional>
#include <string>

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
std::optional<socket_t> CreateBoundSocket();

// Formats a socket address as a host:port / [host]:port.
std::string FormatSocketAddress(socket_t sock);

// Sets a socket to non-blocking. Disables NAGLE.
bool SetSocketNonBlocking(socket_t sock);

// Waits up to 200ms for a read using select.
bool WaitForRead(socket_t sock);

// Accepts a client socket & sets the socke to non-blocking.
std::optional<socket_t> AcceptNonBlocking(socket_t server_fd);

// Receives available data on the socket.
std::string ReceiveAvailable(socket_t client_fd);

// Sends data on the socket.
int AssertSend(socket_t client_fd, std::string_view data);

// Close the socket
void CloseSocket(socket_t fd);

}  // namespace transport_test_utils
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_TRANSPORT_TEST_UTILS_H_
