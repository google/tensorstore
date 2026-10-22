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

#ifndef TENSORSTORE_INTERNAL_HTTP_E2E_HTTP2_TEST_H_
#define TENSORSTORE_INTERNAL_HTTP_E2E_HTTP2_TEST_H_

#include <functional>
#include <memory>

#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_transport.h"

namespace tensorstore {
namespace internal_http {

// Transport configuration requested by an individual scripted test.
struct Http2TestOptions {
  // Maximum number of concurrent connections to a single host, or 0 for the
  // transport default. Tests that exercise stream-slot accounting and
  // GOAWAY handling set this to 1 so that the transport cannot mask a bug by
  // opening an additional connection.
  int max_connections_per_host = 0;
  // Maximum decompressed response size in bytes, or 0 for the transport
  // default.
  size_t max_decompressed_response_bytes = 0;
};

using Http2TransportFactory =
    std::function<std::shared_ptr<HttpTransport>(const Http2TestOptions&)>;

// Parameter of `Http2TransportTest`: how to build the transport under test,
// and the deadlines its requests use.
struct Http2TestParam {
  Http2TransportFactory factory;
  // Deadline for requests that are expected to complete.
  absl::Duration request_timeout = absl::Seconds(10);
  // Deadline for requests that a buggy transport would hang on. Short
  // enough that a hang fails the test quickly, long enough to avoid flakes.
  absl::Duration hang_detect_timeout = absl::Seconds(3);
  // Whether the transport transparently retries requests on REFUSED_STREAM
  // and GOAWAY.
  bool supports_stream_retry = true;
  // Whether the transport tolerates an early response arriving while the
  // request upload is still blocked on flow control.
  bool supports_early_final_response = true;
  // Whether the transport supports issuing reentrant requests from within
  // a response callback on a single multiplexed connection.
  bool supports_reentrant_requests = true;
  // Whether the transport isolates stream resets and stream decoding errors
  // without breaking or draining the shared HTTP/2 connection.
  bool supports_multiplexed_stream_reset = true;
  // Whether the transport supports configuring max_decompressed_response_bytes.
  bool supports_decompressed_size_limit = false;
  // Whether the transport enforces per-stream response header count and 1xx
  // interim response count limits.
  bool enforces_response_header_and_1xx_limits = false;
  // Whether the transport preserves deterministic FIFO stream ordering when
  // retrying unprocessed streams after GOAWAY.
  bool preserves_fifo_stream_retry_order = false;
};

// End-to-end tests that drive a transport against a scripted in-process
// HTTP/2 (h2c, prior knowledge) server, so that each test controls the exact
// frame sequence the client observes.
//
// The scripted server is deliberately cleartext and must not grow TLS
// support. TLS adds certificates, ALPN negotiation and handshake timing that
// are irrelevant to the framing behavior these tests pin down, and it would
// duplicate `H2InteropTransportTest`, which owns TLS and interop coverage.
class Http2TransportTest : public ::testing::TestWithParam<Http2TestParam> {
 protected:
  std::shared_ptr<HttpTransport> CreateTransport(
      const Http2TestOptions& opts = {}) {
    return GetParam().factory(opts);
  }
  absl::Duration request_timeout() const { return GetParam().request_timeout; }
  absl::Duration hang_detect_timeout() const {
    return GetParam().hang_detect_timeout;
  }
  bool supports_stream_retry() const {
    return GetParam().supports_stream_retry;
  }
  bool supports_early_final_response() const {
    return GetParam().supports_early_final_response;
  }
  bool supports_reentrant_requests() const {
    return GetParam().supports_reentrant_requests;
  }
  bool supports_multiplexed_stream_reset() const {
    return GetParam().supports_multiplexed_stream_reset;
  }
  bool supports_decompressed_size_limit() const {
    return GetParam().supports_decompressed_size_limit;
  }
  bool enforces_response_header_and_1xx_limits() const {
    return GetParam().enforces_response_header_and_1xx_limits;
  }
  bool preserves_fifo_stream_retry_order() const {
    return GetParam().preserves_fifo_stream_retry_order;
  }
};

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_E2E_HTTP2_TEST_H_
