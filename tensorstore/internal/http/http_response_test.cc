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

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "tensorstore/util/status.h"

namespace {

using ::tensorstore::internal_http::AppendHeaderData;

TEST(AppendHeaderData, BadHeaders) {
  std::multimap<std::string, std::string> headers;

  EXPECT_EQ(0, AppendHeaderData(headers, ""));           // empty
  EXPECT_EQ(2, AppendHeaderData(headers, "\r\n"));       // empty
  EXPECT_EQ(8, AppendHeaderData(headers, "foo: bar"));   // no CRLF
  EXPECT_EQ(5, AppendHeaderData(headers, "foo\r\n"));    // no :
  EXPECT_EQ(7, AppendHeaderData(headers, "fo@: \r\n"));  // invalid token

  EXPECT_TRUE(headers.empty());
}

TEST(AppendHeaderData, GoodHeaders) {
  // Default
  {
    std::multimap<std::string, std::string> headers;
    EXPECT_EQ(10, AppendHeaderData(headers, "bar: baz\r\n"));
    EXPECT_FALSE(headers.empty());
    ASSERT_EQ(1, headers.count("bar"));

    auto range = headers.equal_range("bar");
    EXPECT_EQ("baz", range.first->second);
  }

  // No value is fine, too.
  {
    std::multimap<std::string, std::string> headers;
    EXPECT_EQ(6, AppendHeaderData(headers, "foo:\r\n"));
    ASSERT_EQ(1, headers.count("foo"));

    auto range = headers.equal_range("foo");
    EXPECT_EQ("", range.first->second);
  }

  // Remove OWS in field-value.
  {
    std::multimap<std::string, std::string> headers;
    EXPECT_EQ(16, AppendHeaderData(headers, "bAr: \t  baz  \t\r\n"));
    ASSERT_EQ(1, headers.count("bar"));

    auto range = headers.equal_range("bar");
    EXPECT_EQ("baz", range.first->second);
  }

  // Order is preserved.
  {
    std::multimap<std::string, std::string> headers;
    EXPECT_EQ(16, AppendHeaderData(headers, "bAr: \t  one  \t\r\n"));
    EXPECT_EQ(10, AppendHeaderData(headers, "bar: two\r\n"));

    ASSERT_EQ(2, headers.count("bar"));

    auto range = headers.equal_range("bar");
    EXPECT_EQ("one", range.first->second);
    ++range.first;
    EXPECT_EQ("two", range.first->second);
  }
}

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
  for (auto code : {302, 303, 304, 307, 412, 413, 416}) {
    seen.insert(code);
    EXPECT_EQ(absl::StatusCode::kFailedPrecondition,
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
