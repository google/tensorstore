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

#ifndef TENSORSTORE_INTERNAL_HTTP_E2E_TEST_UTIL_H_
#define TENSORSTORE_INTERNAL_HTTP_E2E_TEST_UTIL_H_

// Matchers and helpers shared by the end-to-end HTTP transport test suites.

#include <stdint.h>

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_http {

using RequestHeaders =
    absl::Span<const std::pair<std::string_view, std::string_view>>;

// Builds a `method` request for `base_url` + `path` with extra `headers`.
HttpRequest MakeRequest(std::string_view base_url, std::string_view method,
                        std::string_view path, RequestHeaders headers = {});

// Formats `payload` for matcher output, abbreviating long payloads.
std::string DescribePayload(std::string_view payload);

// Matches an OK `Result<HttpResponse>` with the given status code and
// payload. `expected_payload` may be anything convertible to `std::string`,
// including `absl::Cord`.
MATCHER_P2(HasStatusAndPayload, expected_code, expected_payload,
           absl::StrCat(negation ? "isn't" : "is",
                        " an OK HttpResponse with status_code ",
                        ::testing::PrintToString(expected_code),
                        " and payload ",
                        DescribePayload(std::string(expected_payload)))) {
  if (!arg.ok()) {
    *result_listener << "which failed with: " << arg.status();
    return false;
  }
  *result_listener << "whose status_code is " << arg->status_code
                   << " and payload is "
                   << DescribePayload(std::string(arg->payload));
  return arg->status_code == expected_code &&
         arg->payload == std::string(expected_payload);
}

// Matches an OK `Result<HttpResponse>` with the given status code.
MATCHER_P(HasStatusCode, expected_code,
          absl::StrCat(negation ? "isn't" : "is",
                       " an OK HttpResponse with status_code ",
                       ::testing::PrintToString(expected_code))) {
  if (!arg.ok()) {
    *result_listener << "which failed with: " << arg.status();
    return false;
  }
  *result_listener << "whose status_code is " << arg->status_code;
  return arg->status_code == expected_code;
}

// Matches an OK `Result<HttpResponse>` whose `header_name` header has
// `expected_value`. `header_name` must be lowercase.
MATCHER_P2(HasResponseHeader, header_name, expected_value,
           absl::StrCat(negation ? "isn't" : "is",
                        " an OK HttpResponse with header ",
                        ::testing::PrintToString(header_name), ": ",
                        ::testing::PrintToString(expected_value))) {
  if (!arg.ok()) {
    *result_listener << "which failed with: " << arg.status();
    return false;
  }
  auto it = arg->headers.find(header_name);
  if (it == arg->headers.end()) {
    *result_listener << "which lacks header " << header_name;
    return false;
  }
  *result_listener << "whose header " << header_name << " is "
                   << ::testing::PrintToString(it->second);
  return it->second == expected_value;
}

// Matches an OK `Result<HttpResponse>` without a `header_name` header.
// `header_name` must be lowercase.
MATCHER_P(LacksResponseHeader, header_name,
          absl::StrCat(negation ? "isn't" : "is",
                       " an OK HttpResponse without header ",
                       ::testing::PrintToString(header_name))) {
  if (!arg.ok()) {
    *result_listener << "which failed with: " << arg.status();
    return false;
  }
  auto it = arg->headers.find(header_name);
  if (it != arg->headers.end()) {
    *result_listener << "whose header " << header_name << " is "
                     << ::testing::PrintToString(it->second);
    return false;
  }
  return true;
}

// Response handler that destroys the transport from inside its completion
// callbacks, then resolves `done_promise` with the status code or error.
// Used to verify that a transport tolerates destruction from a callback.
class TransportDroppingHandler : public HttpResponseHandler {
 public:
  TransportDroppingHandler(std::shared_ptr<HttpTransport>* transport_holder,
                           Promise<int32_t> done_promise);

  void OnStatus(int32_t status_code) override;
  void OnResponseHeader(std::string_view, std::string_view) override {}
  void OnHeaderBlockDone() override {}
  void OnResponseBody(std::string_view) override {}
  void OnComplete() override;
  void OnFailure(absl::Status status) override;

 private:
  std::shared_ptr<HttpTransport>* transport_holder_;
  Promise<int32_t> done_promise_;
  int32_t status_code_ = 0;
};

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_E2E_TEST_UTIL_H_
