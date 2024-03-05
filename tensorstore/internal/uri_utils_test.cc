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

#include "tensorstore/internal/uri_utils.h"

#include <string_view>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::tensorstore::internal::AsciiSet;
using ::tensorstore::internal::ParseGenericUri;
using ::tensorstore::internal::ParseHostname;
using ::tensorstore::internal::PercentDecode;
using ::tensorstore::internal::PercentEncodeReserved;
using ::tensorstore::internal::PercentEncodeUriComponent;
using ::tensorstore::internal::PercentEncodeUriPath;

namespace {

TEST(PercentDecodeTest, NoOp) {
  std::string_view s = "abcd %zz %%";
  EXPECT_THAT(PercentDecode(s), ::testing::Eq(s));
}

TEST(PercentDecodeTest, EscapeSequenceInMiddle) {
  EXPECT_THAT(PercentDecode("abc%20efg"), ::testing::Eq("abc efg"));
}

TEST(PercentDecodeTest, EscapeSequenceAtEnd) {
  EXPECT_THAT(PercentDecode("abc%20"), ::testing::Eq("abc "));
}

TEST(PercentDecodeTest, EscapeSequenceLetter) {
  EXPECT_THAT(PercentDecode("abc%fF"), ::testing::Eq("abc\xff"));
}

TEST(PercentEncodeReservedTest, Basic) {
  constexpr AsciiSet kMyUnreservedChars{
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "0123456789/."};

  std::string_view s =
      "abcdefghijklmnopqrstuvwxyz"
      "/ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "/01234.56789";

  EXPECT_THAT(PercentEncodeReserved(s, kMyUnreservedChars), ::testing::Eq(s));

  std::string_view t = "-_!~*'()";

  EXPECT_THAT(PercentEncodeReserved(t, kMyUnreservedChars),
              ::testing::Eq("%2D%5F%21%7E%2A%27%28%29"));
}

TEST(PercentEncodeUriPathTest, NoOp) {
  std::string_view s =
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "0123456789"
      "-_.!~*'():@&=+$,;/";
  EXPECT_THAT(PercentEncodeUriPath(s), ::testing::Eq(s));
}

TEST(PercentEncodeUriPathTest, Percent) {
  EXPECT_THAT(PercentEncodeUriPath("%"), ::testing::Eq("%25"));
}

TEST(PercentEncodeUriPathTest, NonAscii) {
  EXPECT_THAT(PercentEncodeUriPath("\xff"), ::testing::Eq("%FF"));
}

TEST(PercentEncodeUriComponentTest, NoOp) {
  std::string_view s =
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "0123456789"
      "-_.!~*'()";
  EXPECT_THAT(PercentEncodeUriComponent(s), ::testing::Eq(s));
}

TEST(PercentEncodeUriComponentTest, Percent) {
  EXPECT_THAT(PercentEncodeUriComponent("%"), ::testing::Eq("%25"));
}

TEST(PercentEncodeUriComponentTest, NonAscii) {
  EXPECT_THAT(PercentEncodeUriComponent("\xff"), ::testing::Eq("%FF"));
}

TEST(ParseGenericUriTest, PathOnly) {
  auto parsed = ParseGenericUri("/abc/def");
  EXPECT_EQ("", parsed.scheme);
  EXPECT_EQ("/abc/def", parsed.authority_and_path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("", parsed.fragment);
}

TEST(ParseGenericUriTest, GsScheme) {
  auto parsed = ParseGenericUri("gs://bucket/path");
  EXPECT_EQ("gs", parsed.scheme);
  EXPECT_EQ("bucket/path", parsed.authority_and_path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("", parsed.fragment);
}

TEST(ParseGenericUriTest, SchemeAuthorityPathQuery) {
  auto parsed = ParseGenericUri("http://host:port/path?query");
  EXPECT_EQ("http", parsed.scheme);
  EXPECT_EQ("host:port/path", parsed.authority_and_path);
  EXPECT_EQ("query", parsed.query);
  EXPECT_EQ("", parsed.fragment);
}

TEST(ParseGenericUriTest, SchemeAuthorityPathFragment) {
  auto parsed = ParseGenericUri("http://host:port/path#fragment");
  EXPECT_EQ("http", parsed.scheme);
  EXPECT_EQ("host:port/path", parsed.authority_and_path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("fragment", parsed.fragment);
}

TEST(ParseGenericUriTest, SchemeAuthorityPathQueryFragment) {
  auto parsed = ParseGenericUri("http://host:port/path?query#fragment");
  EXPECT_EQ("http", parsed.scheme);
  EXPECT_EQ("host:port/path", parsed.authority_and_path);
  EXPECT_EQ("query", parsed.query);
  EXPECT_EQ("fragment", parsed.fragment);
}

// Tests that any "?" after the first "#" is treated as part of the fragment.
TEST(ParseGenericUriTest, SchemeAuthorityPathFragmentQuery) {
  auto parsed = ParseGenericUri("http://host:port/path#fragment?query");
  EXPECT_EQ("http", parsed.scheme);
  EXPECT_EQ("host:port/path", parsed.authority_and_path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("fragment?query", parsed.fragment);
}

TEST(ParseGenericUriTest, S3Scheme) {
  auto parsed = ParseGenericUri("s3://bucket/path");
  EXPECT_EQ("s3", parsed.scheme);
  EXPECT_EQ("bucket/path", parsed.authority_and_path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("", parsed.fragment);
}

TEST(ParseHostname, Basic) {
  static constexpr std::pair<std::string_view, std::string_view> kCases[] = {
      {"host.without.port", "host.without.port"},
      {"host.with.port:1234", "host.with.port"},
      {"localhost:1234/foo/bar", "localhost"},
      {"localhost/foo/bar", "localhost"},
      {"[::1]:0/foo/bar", "::1"},
  };
  for (const auto& [authority_and_path, hostname] : kCases) {
    EXPECT_THAT(ParseHostname(authority_and_path), ::testing::Eq(hostname))
        << authority_and_path;
  }
}

}  // namespace
