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

#include "tensorstore/internal/http/http_response.h"

#include <set>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::IsOkAndHolds;
using ::tensorstore::internal_http::HttpResponse;


TEST(HttpResponseCodeToStatusTest, AllCodes) {
  using ::tensorstore::internal_http::HttpResponseCodeToStatus;

  // OK responses
  absl::flat_hash_set<int> seen;
  for (auto code : {200, 201, 204, 206}) {
    seen.insert(code);
    EXPECT_TRUE(HttpResponseCodeToStatus({code, {}, {}}).ok()) << code;
  }
  for (auto code : {400, 411}) {
    seen.insert(code);
    EXPECT_EQ(absl::StatusCode::kInvalidArgument,
              HttpResponseCodeToStatus({code, {}, {}}).code())
        << code;
  }
  for (auto code : {401, 403}) {
    seen.insert(code);
    EXPECT_EQ(absl::StatusCode::kPermissionDenied,
              HttpResponseCodeToStatus({code, {}, {}}).code())
        << code;
  }
  for (auto code : {404, 410}) {
    seen.insert(code);
    EXPECT_EQ(absl::StatusCode::kNotFound,
              HttpResponseCodeToStatus({code, {}, {}}).code())
        << code;
  }
  for (auto code : {302, 303, 304, 307, 412, 413}) {
    seen.insert(code);
    EXPECT_EQ(absl::StatusCode::kFailedPrecondition,
              HttpResponseCodeToStatus({code, {}, {}}).code())
        << code;
  }
  for (auto code : {416}) {
    seen.insert(code);
    EXPECT_EQ(absl::StatusCode::kOutOfRange,
              HttpResponseCodeToStatus({code, {}, {}}).code())
        << code;
  }
  for (auto code : {308, 408, 409, 429, 500, 502, 503, 504}) {
    seen.insert(code);
    EXPECT_EQ(absl::StatusCode::kUnavailable,
              HttpResponseCodeToStatus({code, {}, {}}).code())
        << code;
  }

  for (int i = 300; i < 600; i++) {
    if (seen.count(i) > 0) continue;
    // All other errors are translated to kUnknown.
    EXPECT_EQ(absl::StatusCode::kUnknown,
              HttpResponseCodeToStatus({i, {}, {}}).code())
        << i;
  }
}


}  // namespace
