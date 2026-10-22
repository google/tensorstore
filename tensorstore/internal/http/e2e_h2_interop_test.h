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

#ifndef TENSORSTORE_INTERNAL_HTTP_E2E_H2_INTEROP_TEST_H_
#define TENSORSTORE_INTERNAL_HTTP_E2E_H2_INTEROP_TEST_H_

#include <functional>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_transport.h"

namespace tensorstore {
namespace internal_http {

struct H2InteropTestOptions {
  std::string ca_bundle_path;
  bool verify_peer = false;
  bool verify_host = false;
};

using H2InteropTransportFactory =
    std::function<std::shared_ptr<HttpTransport>(const H2InteropTestOptions&)>;

// Parameter of `H2InteropTransportTest`: how to build the transport under
// test, and the deadline its requests use.
struct H2InteropTestParam {
  H2InteropTransportFactory factory;
  // Connect and request deadline. Generous because the Python server and
  // the TLS handshake are slow on loaded test machines.
  absl::Duration request_timeout = absl::Seconds(15);
};

// End-to-end tests that drive a transport over TLS (ALPN h2) against the
// Python `h2_server.py`, an HTTP/2 implementation independent of nghttp2.
//
// This suite covers interoperability, TLS, proxy tunneling, and responses
// that nghttp2 refuses to emit (e.g. a missing :status). Tests that script
// stream resets, GOAWAY, flow control, interim responses, or other
// per-stream server behavior belong in `Http2TransportTest`.
class H2InteropTransportTest
    : public ::testing::TestWithParam<H2InteropTestParam> {
 protected:
  std::shared_ptr<HttpTransport> CreateTransport(
      const H2InteropTestOptions& opts) {
    return GetParam().factory(opts);
  }
  absl::Duration request_timeout() const { return GetParam().request_timeout; }
};

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_E2E_H2_INTEROP_TEST_H_
