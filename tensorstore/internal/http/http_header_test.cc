// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/http/http_header.h"

#include <stddef.h>

#include <optional>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_http::AppendHeaderData;
using ::tensorstore::internal_http::HeaderMap;
using ::tensorstore::internal_http::TryParseContentRangeHeader;
using ::tensorstore::internal_http::ValidateHttpHeader;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Optional;
using ::testing::Pair;

TEST(ValidateHttpHeaderTest, Valid) {
  EXPECT_THAT(ValidateHttpHeader("a!#$%&'*+-.^_`|~3X: b\xfe"),
              Optional(Pair("a!#$%&'*+-.^_`|~3X", "b\xfe")));
  EXPECT_THAT(ValidateHttpHeader("host: foo-bar.example.com"),
              Optional(Pair("host", "foo-bar.example.com")));
  EXPECT_THAT(ValidateHttpHeader("empty:"), Optional(Pair("empty", "")));
}

TEST(ValidateHttpHeaderTest, Invalid) {
  EXPECT_THAT(ValidateHttpHeader("a"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateHttpHeader("a: \n"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateHttpHeader(": b\n"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(SetHeaderTest, Basic) {
  HeaderMap headers;
  headers.SetHeader("x", "a");
  EXPECT_THAT(headers, ElementsAre(Pair("x", "a")));

  headers.SetHeader("x", "b");
  headers.SetHeader("x", "c");
  EXPECT_THAT(headers, ElementsAre(Pair("x", "c")));
}

TEST(CombineHeaderTest, Basic) {
  HeaderMap headers;
  headers.CombineHeader("x", "a");
  EXPECT_THAT(headers, ElementsAre(Pair("x", "a")));

  headers.CombineHeader("x", "b");
  headers.CombineHeader("x", "c");
  EXPECT_THAT(headers, ElementsAre(Pair("x", "a,b,c")));
}

TEST(AppendHeaderData, BadHeaders) {
  HeaderMap headers;

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
    HeaderMap headers;
    EXPECT_EQ(10, AppendHeaderData(headers, "bar: baz\r\n"));

    EXPECT_THAT(headers, ElementsAre(Pair("bar", "baz")));
  }

  // No value is fine, too.
  {
    HeaderMap headers;
    EXPECT_EQ(6, AppendHeaderData(headers, "foo:\r\n"));

    EXPECT_THAT(headers, ElementsAre(Pair("foo", "")));
  }

  // Remove OWS in field-value.
  {
    HeaderMap headers;
    EXPECT_EQ(16, AppendHeaderData(headers, "bAr: \t  baz  \t\r\n"));

    EXPECT_THAT(headers, ElementsAre(Pair("bar", "baz")));
  }

  // Order is preserved.
  {
    HeaderMap headers;
    EXPECT_EQ(16, AppendHeaderData(headers, "bAr: \t  one  \t\r\n"));
    EXPECT_EQ(10, AppendHeaderData(headers, "bar: two\r\n"));

    EXPECT_THAT(headers, ElementsAre(Pair("bar", "one,two")));
  }
}

TEST(TryParse, ContentRangeHeader) {
  EXPECT_THAT(TryParseContentRangeHeader(
                  HeaderMap{{"content-range", "bytes 10-20/100"}}),
              Optional(Eq(std::tuple<size_t, size_t, size_t>(10, 20, 100))));

  EXPECT_THAT(
      TryParseContentRangeHeader(HeaderMap{{"content-range", "bytes 10-20/*"}}),
      Optional(Eq(std::tuple<size_t, size_t, size_t>(10, 20, 0))));

  EXPECT_THAT(
      TryParseContentRangeHeader(HeaderMap{{"content-range", "bytes 10-20"}}),
      Optional(Eq(std::tuple<size_t, size_t, size_t>(10, 20, 0))));

  EXPECT_THAT(TryParseContentRangeHeader(
                  HeaderMap{{"content-range", "bytes 1-abc/100"}}),
              Eq(std::nullopt));
}

TEST(TryParse, BoolHeader) {
  HeaderMap headers{{"false-header", "0"}, {"true-header", "true"}};

  EXPECT_THAT(headers.TryParseBoolHeader("true-header"), Optional(Eq(true)));
  EXPECT_THAT(headers.TryParseBoolHeader("false-header"), Optional(Eq(false)));
  EXPECT_THAT(headers.TryParseBoolHeader("missing-header"), Eq(std::nullopt));
}

}  // namespace
