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

#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/internal/ascii_set.h"
#include "tensorstore/internal/testing/on_windows.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::IsOkAndHolds;
using ::tensorstore::StatusIs;
using ::tensorstore::internal::AsciiSet;
using ::tensorstore::internal::FileUriToOsPath;
using ::tensorstore::internal::HostPort;
using ::tensorstore::internal::OsPathToFileUri;
using ::tensorstore::internal::ParseGenericUri;
using ::tensorstore::internal::PercentDecode;
using ::tensorstore::internal::PercentEncodeKvStoreUriPath;
using ::tensorstore::internal::PercentEncodeReserved;
using ::tensorstore::internal::PercentEncodeUriComponent;
using ::tensorstore::internal::PercentEncodeUriPath;
using ::tensorstore::internal::SplitHostPort;
using ::tensorstore::internal_testing::OnWindows;
using ::testing::StrEq;

namespace tensorstore::internal {

// Avoid a public implementation operator==.
inline bool operator==(const HostPort& a, const HostPort& b) {
  return std::tie(a.host, a.port) == std::tie(b.host, b.port);
}

}  // namespace tensorstore::internal

namespace {

std::string Get7BitAscii() {
  std::string tmp;
  tmp.reserve(128);
  for (int i = 0; i < 128; i++) {
    tmp.push_back(static_cast<char>(i));
  }
  return tmp;
}

TEST(PercentDecodeTest, NoOp) {
  std::string_view s = "abcd %zz %%";
  EXPECT_THAT(PercentDecode(s), StrEq(s));
}

TEST(PercentDecodeTest, EscapeSequenceInMiddle) {
  EXPECT_THAT(PercentDecode("abc%20efg"), StrEq("abc efg"));
}

TEST(PercentDecodeTest, EscapeSequenceAtEnd) {
  EXPECT_THAT(PercentDecode("abc%20"), StrEq("abc "));
}

TEST(PercentDecodeTest, EscapeSequenceLetter) {
  EXPECT_THAT(PercentDecode("abc%fF"), StrEq("abc\xff"));
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

  EXPECT_THAT(PercentEncodeReserved(s, kMyUnreservedChars), StrEq(s));

  std::string_view t = "-_!~*'()";

  EXPECT_THAT(PercentEncodeReserved(t, kMyUnreservedChars),
              StrEq("%2D%5F%21%7E%2A%27%28%29"));
}

TEST(PercentEncodeUriPathTest, Ascii) {
  EXPECT_THAT(
      PercentEncodeUriPath(Get7BitAscii()),
      StrEq("%00%01%02%03%04%05%06%07%08%09%0A%0B%0C%0D%0E%0F%10%11%12%13%14%"
            "15%16%17%18%19%1A%1B%1C%1D%1E%1F%20!%22%23$%25&'()*+,-./"
            "0123456789:;%3C=%3E%3F@ABCDEFGHIJKLMNOPQRSTUVWXYZ%5B%5C%5D%5E_%"
            "60abcdefghijklmnopqrstuvwxyz%7B%7C%7D~%7F"));
}

TEST(PercentEncodeUriPathTest, NoOp) {
  std::string_view s =
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "0123456789"
      "-_.!~*'():@&=+$,;/";
  EXPECT_THAT(PercentEncodeUriPath(s), StrEq(s));
}

TEST(PercentEncodeUriPathTest, NonAscii) {
  EXPECT_THAT(PercentEncodeUriPath("\xff"), StrEq("%FF"));
}

TEST(PercentEncodeKvStoreUriPathTest, Ascii) {
  EXPECT_THAT(
      PercentEncodeKvStoreUriPath(Get7BitAscii()),
      StrEq("%00%01%02%03%04%05%06%07%08%09%0A%0B%0C%0D%0E%0F%10%11%12%13%14%"
            "15%16%17%18%19%1A%1B%1C%1D%1E%1F%20!%22%23$%25&'()*+,-./"
            "0123456789:;%3C=%3E%3F%40ABCDEFGHIJKLMNOPQRSTUVWXYZ%5B%5C%5D%5E_%"
            "60abcdefghijklmnopqrstuvwxyz%7B%7C%7D~%7F"));
}

TEST(PercentEncodeKvStoreUriPathTest, NoOp) {
  std::string_view s =
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "0123456789"
      "-_.!~*'():&=+$,;/";
  EXPECT_THAT(PercentEncodeKvStoreUriPath(s), StrEq(s));
}

TEST(PercentEncodeKvStoreUriPathTest, NonAscii) {
  EXPECT_THAT(PercentEncodeKvStoreUriPath("\xff"), StrEq("%FF"));
}

TEST(PercentEncodeUriComponentTest, Ascii) {
  EXPECT_THAT(
      PercentEncodeUriComponent(Get7BitAscii()),
      StrEq("%00%01%02%03%04%05%06%07%08%09%0A%0B%0C%0D%0E%0F%10%11%12%13%14%"
            "15%16%17%18%19%1A%1B%1C%1D%1E%1F%20!%22%23%24%25%26'()*%2B%2C-.%"
            "2F0123456789%3A%3B%3C%3D%3E%3F%40ABCDEFGHIJKLMNOPQRSTUVWXYZ%5B%5C%"
            "5D%5E_%60abcdefghijklmnopqrstuvwxyz%7B%7C%7D~%7F"));
}

TEST(PercentEncodeUriComponentTest, NoOp) {
  std::string_view s =
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "0123456789"
      "-_.!~*'()";
  EXPECT_THAT(PercentEncodeUriComponent(s), StrEq(s));
}

TEST(PercentEncodeUriComponentTest, NonAscii) {
  EXPECT_THAT(PercentEncodeUriComponent("\xff"), StrEq("%FF"));
}

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

TEST(OsPathToUriPathTest, Basic) {
  EXPECT_THAT(OsPathToFileUri(""),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OsPathToFileUri("foo"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OsPathToFileUri("foo/"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OsPathToFileUri("/"), IsOkAndHolds("file:///"));
  EXPECT_THAT(OsPathToFileUri("/foo"), IsOkAndHolds("file:///foo"));
  EXPECT_THAT(OsPathToFileUri("/foo/"), IsOkAndHolds("file:///foo/"));

  EXPECT_THAT(OsPathToFileUri("c:/tmp"),
              OnWindows(IsOkAndHolds("file:///c:/tmp"),
                        StatusIs(absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(OsPathToFileUri("c:/tmp/"),
              OnWindows(IsOkAndHolds("file:///c:/tmp/"),
                        StatusIs(absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(OsPathToFileUri("c:\\tmp\\foo"),
              OnWindows(IsOkAndHolds("file:///c:/tmp/foo"),
                        StatusIs(absl::StatusCode::kInvalidArgument)));

  EXPECT_THAT(OsPathToFileUri("//server/share/tmp"),
              IsOkAndHolds(OnWindows("file://server/share/tmp",
                                     "file:///server/share/tmp")));

  EXPECT_THAT(OsPathToFileUri("\\\\server\\share\\tmp"),
              OnWindows(IsOkAndHolds("file://server/share/tmp"),
                        StatusIs(absl::StatusCode::kInvalidArgument)));
}

TEST(UriPathToOsPathTest, Basic) {
  auto ToPath = [](std::string_view uri) {
    return FileUriToOsPath(ParseGenericUri(uri));
  };

  EXPECT_THAT(ToPath(""), StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ToPath("foo"), StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ToPath("foo/"), StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(ToPath("file:///"), IsOkAndHolds("/"));
  EXPECT_THAT(ToPath("file:///foo"), IsOkAndHolds("/foo"));
  EXPECT_THAT(ToPath("file:///foo/"), IsOkAndHolds("/foo/"));
  EXPECT_THAT(ToPath("file:///c:/tmp"),
              IsOkAndHolds(OnWindows("c:/tmp", "/c:/tmp")));
  EXPECT_THAT(ToPath("file:///c:/tmp/"),
              IsOkAndHolds(OnWindows("c:/tmp/", "/c:/tmp/")));

  EXPECT_THAT(ToPath("file://server/share/tmp"),
              OnWindows(IsOkAndHolds("//server/share/tmp"),
                        StatusIs(absl::StatusCode::kInvalidArgument)));
}

}  // namespace
