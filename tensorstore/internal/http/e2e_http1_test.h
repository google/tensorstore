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

#ifndef TENSORSTORE_INTERNAL_HTTP_E2E_HTTP1_TEST_H_
#define TENSORSTORE_INTERNAL_HTTP_E2E_HTTP1_TEST_H_

#include <functional>
#include <memory>

#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_transport.h"

namespace tensorstore {
namespace internal_http {

using Http1TransportFactory = std::function<std::shared_ptr<HttpTransport>()>;

// Parameter of `Http1TransportTest`: how to build the transport under test,
// and the deadlines its requests use.
struct Http1TestParam {
  Http1TransportFactory factory;
  // Deadline for requests that are expected to complete.
  absl::Duration request_timeout = absl::Seconds(10);
  // Deadline for requests that a buggy transport would hang on. Short
  // enough that a hang fails the test quickly, long enough to avoid flakes.
  absl::Duration hang_detect_timeout = absl::Seconds(3);
  // Whether the transport tolerates an early response arriving before or
  // during request upload without desynchronizing the connection.
  bool supports_early_final_response = true;
};

class Http1TransportTest : public ::testing::TestWithParam<Http1TestParam> {
 protected:
  std::shared_ptr<HttpTransport> CreateTransport() {
    return GetParam().factory();
  }
  absl::Duration request_timeout() const { return GetParam().request_timeout; }
  absl::Duration hang_detect_timeout() const {
    return GetParam().hang_detect_timeout;
  }
  bool supports_early_final_response() const {
    return GetParam().supports_early_final_response;
  }
};

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_E2E_HTTP1_TEST_H_
