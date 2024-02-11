// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/http/curl_wrappers.h"

#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal_http::CurlCodeToStatus;
using ::tensorstore::internal_http::CurlMCodeToStatus;

TEST(CurlFactoryTest, CurlCodeToStatus) {
  struct {
    CURLcode curl;
    absl::StatusCode expected;
  } expected_codes[]{
      {CURLE_OK, absl::StatusCode::kOk},
      {CURLE_RECV_ERROR, absl::StatusCode::kUnavailable},
      {CURLE_SEND_ERROR, absl::StatusCode::kUnavailable},
      {CURLE_PARTIAL_FILE, absl::StatusCode::kUnavailable},
      {CURLE_SSL_CONNECT_ERROR, absl::StatusCode::kUnavailable},
      {CURLE_COULDNT_RESOLVE_HOST, absl::StatusCode::kUnavailable},
      {CURLE_COULDNT_RESOLVE_PROXY, absl::StatusCode::kUnavailable},
      {CURLE_COULDNT_CONNECT, absl::StatusCode::kUnavailable},
      {CURLE_REMOTE_ACCESS_DENIED, absl::StatusCode::kPermissionDenied},
      {CURLE_OPERATION_TIMEDOUT, absl::StatusCode::kDeadlineExceeded},
      {CURLE_ABORTED_BY_CALLBACK, absl::StatusCode::kAborted},
      {CURLE_FAILED_INIT, absl::StatusCode::kUnknown},
      {CURLE_GOT_NOTHING, absl::StatusCode::kUnavailable},
      {CURLE_AGAIN, absl::StatusCode::kUnknown},
      {CURLE_HTTP2, absl::StatusCode::kUnavailable},
      {CURLE_BAD_DOWNLOAD_RESUME, absl::StatusCode::kInternal},
      {CURLE_RANGE_ERROR, absl::StatusCode::kInternal},
      {CURLE_UNSUPPORTED_PROTOCOL, absl::StatusCode::kUnavailable},
  };

  for (auto const& t : expected_codes) {
    auto actual = CurlCodeToStatus(t.curl, {});
    EXPECT_EQ(t.expected, actual.code()) << "CURL code=" << t.curl;
  }
}

TEST(CurlFactoryTest, CurlMCodeToStatus) {
  struct {
    CURLMcode curl;
    absl::StatusCode expected;
  } expected_codes[]{
      {CURLM_OK, absl::StatusCode::kOk},
      {CURLM_BAD_HANDLE, absl::StatusCode::kInternal},
      {CURLM_BAD_EASY_HANDLE, absl::StatusCode::kInternal},
      {CURLM_OUT_OF_MEMORY, absl::StatusCode::kInternal},
      {CURLM_INTERNAL_ERROR, absl::StatusCode::kInternal},
  };
  for (auto const& t : expected_codes) {
    auto actual = CurlMCodeToStatus(t.curl, {});
    EXPECT_EQ(t.expected, actual.code()) << "CURLM code=" << t.curl;
  }
}

}  // namespace
