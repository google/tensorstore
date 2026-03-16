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

#include "tensorstore/internal/uri/parse.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::tensorstore::internal_uri::HostPort;
using ::tensorstore::internal_uri::ParseGenericUri;
using ::tensorstore::internal_uri::SplitHostPort;

namespace {

TEST(ParseGenericUriTest, InvalidPathOnly) {
  auto parsed = ParseGenericUri("/abc/def");
  EXPECT_EQ("", parsed.scheme);
  EXPECT_EQ("", parsed.authority);
  EXPECT_EQ("/abc/def", parsed.authority_and_path);
  EXPECT_EQ("/abc/def", parsed.path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("", parsed.fragment);
  EXPECT_FALSE(parsed.has_authority_delimiter);
}

TEST(ParseGenericUriTest, FileScheme1) {
  auto parsed = ParseGenericUri("file:///abc/def");
  EXPECT_EQ("file", parsed.scheme);
  EXPECT_EQ("", parsed.authority);
  EXPECT_EQ("/abc/def", parsed.authority_and_path);
  EXPECT_EQ("/abc/def", parsed.path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("", parsed.fragment);
  EXPECT_TRUE(parsed.has_authority_delimiter);
}

TEST(ParseGenericUriTest, FileSchemeWindows) {
  auto parsed = ParseGenericUri("file:///C:/Users/me/temp");
  EXPECT_EQ("file", parsed.scheme);
  EXPECT_EQ("", parsed.authority);
  EXPECT_EQ("/C:/Users/me/temp", parsed.authority_and_path);
  EXPECT_EQ("/C:/Users/me/temp", parsed.path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("", parsed.fragment);
  EXPECT_TRUE(parsed.has_authority_delimiter);
}

TEST(ParseGenericUriTest, FooScheme) {
  auto parsed = ParseGenericUri("foo:/abc/def");
  EXPECT_EQ("foo", parsed.scheme);
  EXPECT_EQ("", parsed.authority);
  EXPECT_EQ("/abc/def", parsed.authority_and_path);
  EXPECT_EQ("/abc/def", parsed.path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("", parsed.fragment);
  EXPECT_FALSE(parsed.has_authority_delimiter);
}

TEST(ParseGenericUriTest, GsScheme) {
  auto parsed = ParseGenericUri("gs://bucket/path");
  EXPECT_EQ("gs", parsed.scheme);
  EXPECT_EQ("bucket/path", parsed.authority_and_path);
  EXPECT_EQ("bucket", parsed.authority);
  EXPECT_EQ("/path", parsed.path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("", parsed.fragment);
  EXPECT_TRUE(parsed.has_authority_delimiter);
}

TEST(ParseGenericUriTest, SchemeAuthorityNoPath) {
  auto parsed = ParseGenericUri("http://host:port");
  EXPECT_EQ("http", parsed.scheme);
  EXPECT_EQ("host:port", parsed.authority_and_path);
  EXPECT_EQ("host:port", parsed.authority);
  EXPECT_EQ("", parsed.path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("", parsed.fragment);
  EXPECT_TRUE(parsed.has_authority_delimiter);
}

TEST(ParseGenericUriTest, SchemeAuthorityRootPath) {
  auto parsed = ParseGenericUri("http://host:port/");
  EXPECT_EQ("http", parsed.scheme);
  EXPECT_EQ("host:port/", parsed.authority_and_path);
  EXPECT_EQ("host:port", parsed.authority);
  EXPECT_EQ("/", parsed.path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("", parsed.fragment);
  EXPECT_TRUE(parsed.has_authority_delimiter);
}

TEST(ParseGenericUriTest, SchemeAuthorityPathQuery) {
  auto parsed = ParseGenericUri("http://host:port/path?query");
  EXPECT_EQ("http", parsed.scheme);
  EXPECT_EQ("host:port/path", parsed.authority_and_path);
  EXPECT_EQ("host:port", parsed.authority);
  EXPECT_EQ("/path", parsed.path);
  EXPECT_EQ("query", parsed.query);
  EXPECT_EQ("", parsed.fragment);
  EXPECT_TRUE(parsed.has_authority_delimiter);
}

TEST(ParseGenericUriTest, SchemeAuthorityPathFragment) {
  auto parsed = ParseGenericUri("http://host:port/path#fragment");
  EXPECT_EQ("http", parsed.scheme);
  EXPECT_EQ("host:port/path", parsed.authority_and_path);
  EXPECT_EQ("host:port", parsed.authority);
  EXPECT_EQ("/path", parsed.path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("fragment", parsed.fragment);
  EXPECT_TRUE(parsed.has_authority_delimiter);
}

TEST(ParseGenericUriTest, SchemeAuthorityPathQueryFragment) {
  auto parsed = ParseGenericUri("http://host:port/path?query#fragment");
  EXPECT_EQ("http", parsed.scheme);
  EXPECT_EQ("host:port/path", parsed.authority_and_path);
  EXPECT_EQ("host:port", parsed.authority);
  EXPECT_EQ("/path", parsed.path);
  EXPECT_EQ("query", parsed.query);
  EXPECT_EQ("fragment", parsed.fragment);
  EXPECT_TRUE(parsed.has_authority_delimiter);
}

// Tests that any "?" after the first "#" is treated as part of the fragment.
TEST(ParseGenericUriTest, SchemeAuthorityPathFragmentQuery) {
  auto parsed = ParseGenericUri("http://host:port/path#fragment?query");
  EXPECT_EQ("http", parsed.scheme);
  EXPECT_EQ("host:port/path", parsed.authority_and_path);
  EXPECT_EQ("host:port", parsed.authority);
  EXPECT_EQ("/path", parsed.path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("fragment?query", parsed.fragment);
  EXPECT_TRUE(parsed.has_authority_delimiter);
}

TEST(ParseGenericUriTest, S3Scheme) {
  auto parsed = ParseGenericUri("s3://bucket/path");
  EXPECT_EQ("s3", parsed.scheme);
  EXPECT_EQ("bucket/path", parsed.authority_and_path);
  EXPECT_EQ("bucket", parsed.authority);
  EXPECT_EQ("/path", parsed.path);
  EXPECT_EQ("", parsed.query);
  EXPECT_EQ("", parsed.fragment);
  EXPECT_TRUE(parsed.has_authority_delimiter);
}

TEST(ParseGenericUriTest, Basic) {
  static constexpr std::pair<std::string_view, std::string_view> kCases[] = {
      {"http://host.without.port", "host.without.port"},
      {"http://host.with.port:1234", "host.with.port:1234"},
      {"http://localhost:1234/foo/bar", "localhost:1234"},
      {"http://localhost/foo/bar", "localhost"},
      {"http://[::1]", "[::1]"},
      {"http://[::1]:0", "[::1]:0"},
      {"http://[::1]:0/foo/bar", "[::1]:0"},
  };
  for (const auto& [uri, authority] : kCases) {
    EXPECT_THAT(ParseGenericUri(uri).authority, ::testing::Eq(authority));
  }
}

TEST(ParseHostPortTest, Basic) {
  // Non-IPv6 literals
  EXPECT_THAT(SplitHostPort("127.0.0.1"),
              ::testing::Optional(HostPort{"127.0.0.1"}));
  EXPECT_THAT(SplitHostPort("127.0.0.1:1"),
              ::testing::Optional(HostPort{"127.0.0.1", "1"}));

  EXPECT_THAT(SplitHostPort("host.without.port"),
              ::testing::Optional(HostPort{"host.without.port"}));
  EXPECT_THAT(SplitHostPort("host.without.port::1"),
              ::testing::Optional(HostPort{"host.without.port::1"}));

  EXPECT_THAT(SplitHostPort("host.with.port:1234"),
              ::testing::Optional(HostPort{"host.with.port", "1234"}));
  EXPECT_THAT(SplitHostPort("localhost:1234"),
              ::testing::Optional(HostPort{"localhost", "1234"}));
  EXPECT_THAT(SplitHostPort("localhost"),
              ::testing::Optional(HostPort{"localhost"}));
  // IPv6 literals
  EXPECT_THAT(SplitHostPort("::1"), ::testing::Optional(HostPort{"::1"}));
  EXPECT_THAT(SplitHostPort("[::1]:1"),
              ::testing::Optional(HostPort{"[::1]", "1"}));
  EXPECT_THAT(SplitHostPort("[::1"), ::testing::Eq(std::nullopt));
  EXPECT_THAT(SplitHostPort("[::1]::1"), ::testing::Eq(std::nullopt));
}

}  // namespace
