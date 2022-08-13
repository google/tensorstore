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

#include "tensorstore/internal/path.h"

#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal::EnsureDirectoryPath;
using ::tensorstore::internal::EnsureNonDirectoryPath;
using ::tensorstore::internal::JoinPath;
using ::tensorstore::internal::ParseGenericUri;
using ::tensorstore::internal::PathDirnameBasename;
using ::tensorstore::internal::PercentDecode;
using ::tensorstore::internal::PercentEncodeUriComponent;
using ::tensorstore::internal::PercentEncodeUriPath;

TEST(PathTest, JoinPath) {
  EXPECT_EQ("/foo/bar", JoinPath("/foo", "bar"));
  EXPECT_EQ("/foo/bar", JoinPath("/foo/", "bar"));
  EXPECT_EQ("/foo/bar", JoinPath("/foo", "/bar"));
  EXPECT_EQ("/foo//bar", JoinPath("/foo/", "/bar"));

  EXPECT_EQ("foo/bar", JoinPath("foo", "bar"));
  EXPECT_EQ("foo/bar", JoinPath("foo", "/bar"));

  EXPECT_EQ("/bar", JoinPath("", "/bar"));
  EXPECT_EQ("bar", JoinPath("", "bar"));
  EXPECT_EQ("/foo", JoinPath("/foo", ""));

  EXPECT_EQ("/foo/bar/baz//blah/blink/biz",
            JoinPath("/foo/bar/baz/", "/blah/blink/biz"));
  EXPECT_EQ("/foo/bar/baz/blah", JoinPath("/foo", "bar", "baz", "blah"));

  EXPECT_EQ("http://foo/bar/baz", JoinPath("http://foo/", "bar", "baz"));
}

TEST(PathTest, JoinPath_MixedArgs) {
  constexpr const char kFoo[] = "/foo";
  std::string_view foo_view("/foo");
  std::string foo("/foo");

  EXPECT_EQ("/foo/bar", JoinPath(foo_view, "bar"));
  EXPECT_EQ("/foo/bar", JoinPath(foo, "bar"));
  EXPECT_EQ("/foo/bar", JoinPath(kFoo, "/bar"));
}

TEST(PathTest, PathDirnameBasename) {
  EXPECT_EQ("/a/b", PathDirnameBasename("/a/b/bar").first);
  EXPECT_EQ("bar", PathDirnameBasename("/a/b/bar").second);

  EXPECT_EQ("a/b", PathDirnameBasename("a/b/bar").first);
  EXPECT_EQ("bar", PathDirnameBasename("a/b/bar").second);

  EXPECT_EQ("", PathDirnameBasename("bar").first);
  EXPECT_EQ("bar", PathDirnameBasename("bar").second);

  EXPECT_EQ("/", PathDirnameBasename("/bar").first);
  EXPECT_EQ("bar", PathDirnameBasename("/bar").second);

  EXPECT_EQ("//a/b", PathDirnameBasename("//a/b///bar").first);
  EXPECT_EQ("bar", PathDirnameBasename("//a/b///bar").second);

  EXPECT_EQ("/", PathDirnameBasename("///bar").first);
  EXPECT_EQ("bar", PathDirnameBasename("///bar").second);
}

TEST(EnsureDirectoryPathTest, EmptyString) {
  std::string path = "";
  EnsureDirectoryPath(path);
  EXPECT_EQ("", path);
}

TEST(EnsureDirectoryPathTest, SingleSlash) {
  std::string path = "/";
  EnsureDirectoryPath(path);
  EXPECT_EQ("", path);
}

TEST(EnsureDirectoryPathTest, NonEmptyWithoutSlash) {
  std::string path = "abc";
  EnsureDirectoryPath(path);
  EXPECT_EQ("abc/", path);
}

TEST(EnsureDirectoryPathTest, NonEmptyWithSlash) {
  std::string path = "abc/";
  EnsureDirectoryPath(path);
  EXPECT_EQ("abc/", path);
}

TEST(EnsureNonDirectoryPathTest, EmptyString) {
  std::string path = "";
  EnsureNonDirectoryPath(path);
  EXPECT_EQ("", path);
}

TEST(EnsureNonDirectoryPathTest, SingleSlash) {
  std::string path = "/";
  EnsureNonDirectoryPath(path);
  EXPECT_EQ("", path);
}

TEST(EnsureNonDirectoryPathTest, NonEmptyWithoutSlash) {
  std::string path = "abc";
  EnsureNonDirectoryPath(path);
  EXPECT_EQ("abc", path);
}

TEST(EnsureNonDirectoryPathTest, NonEmptyWithSlash) {
  std::string path = "abc/";
  EnsureNonDirectoryPath(path);
  EXPECT_EQ("abc", path);
}

TEST(EnsureNonDirectoryPathTest, NonEmptyWithSlashes) {
  std::string path = "abc////";
  EnsureNonDirectoryPath(path);
  EXPECT_EQ("abc", path);
}

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

}  // namespace
