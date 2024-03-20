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
#include <string>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_http::AppendHeaderData;
using ::tensorstore::internal_http::TryParseBoolHeader;
using ::tensorstore::internal_http::TryParseContentRangeHeader;
using ::tensorstore::internal_http::TryParseIntHeader;
using ::tensorstore::internal_http::ValidateHttpHeader;

TEST(ValidateHttpHeaderTest, Valid) {
  TENSORSTORE_EXPECT_OK(ValidateHttpHeader("a!#$%&'*+-.^_`|~3X: b\xfe"));
}

TEST(ValidateHttpHeaderTest, Invalid) {
  EXPECT_THAT(ValidateHttpHeader("a"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateHttpHeader("a: \n"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(AppendHeaderData, BadHeaders) {
  absl::btree_multimap<std::string, std::string> headers;

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
    absl::btree_multimap<std::string, std::string> headers;
    EXPECT_EQ(10, AppendHeaderData(headers, "bar: baz\r\n"));
    EXPECT_FALSE(headers.empty());
    ASSERT_EQ(1, headers.count("bar"));

    auto range = headers.equal_range("bar");
    EXPECT_EQ("baz", range.first->second);
  }

  // No value is fine, too.
  {
    absl::btree_multimap<std::string, std::string> headers;
    EXPECT_EQ(6, AppendHeaderData(headers, "foo:\r\n"));
    ASSERT_EQ(1, headers.count("foo"));

    auto range = headers.equal_range("foo");
    EXPECT_EQ("", range.first->second);
  }

  // Remove OWS in field-value.
  {
    absl::btree_multimap<std::string, std::string> headers;
    EXPECT_EQ(16, AppendHeaderData(headers, "bAr: \t  baz  \t\r\n"));
    ASSERT_EQ(1, headers.count("bar"));

    auto range = headers.equal_range("bar");
    EXPECT_EQ("baz", range.first->second);
  }

  // Order is preserved.
  {
    absl::btree_multimap<std::string, std::string> headers;
    EXPECT_EQ(16, AppendHeaderData(headers, "bAr: \t  one  \t\r\n"));
    EXPECT_EQ(10, AppendHeaderData(headers, "bar: two\r\n"));

    ASSERT_EQ(2, headers.count("bar"));

    auto range = headers.equal_range("bar");
    EXPECT_EQ("one", range.first->second);
    ++range.first;
    EXPECT_EQ("two", range.first->second);
  }
}

TEST(TryParse, ContentRangeHeader) {
  EXPECT_THAT(
      TryParseContentRangeHeader({{"content-range", "bytes 10-20/100"}}),
      ::testing::Optional(
          testing::Eq(std::tuple<size_t, size_t, size_t>(10, 20, 100))));

  EXPECT_THAT(TryParseContentRangeHeader({{"content-range", "bytes 10-20/*"}}),
              ::testing::Optional(
                  testing::Eq(std::tuple<size_t, size_t, size_t>(10, 20, 0))));

  EXPECT_THAT(TryParseContentRangeHeader({{"content-range", "bytes 10-20"}}),
              ::testing::Optional(
                  testing::Eq(std::tuple<size_t, size_t, size_t>(10, 20, 0))));

  EXPECT_THAT(
      TryParseContentRangeHeader({{"content-range", "bytes 1-abc/100"}}),
      ::testing::Eq(std::nullopt));
}

TEST(TryParse, BoolHeader) {
  EXPECT_THAT(TryParseBoolHeader({{"bool-header", "true"}}, "bool-header"),
              ::testing::Optional(testing::Eq(true)));
}

TEST(TryParse, IntHeader) {
  EXPECT_THAT(TryParseIntHeader<size_t>({{"int-header", "100"}}, "int-header"),
              ::testing::Optional(testing::Eq(100)));
}

}  // namespace
