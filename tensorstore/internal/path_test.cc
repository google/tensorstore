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
#include "tensorstore/internal/testing/on_windows.h"

namespace {

using ::tensorstore::internal::EnsureDirectoryPath;
using ::tensorstore::internal::EnsureNonDirectoryPath;
using ::tensorstore::internal::IsAbsolutePath;
using ::tensorstore::internal::JoinPath;
using ::tensorstore::internal::LexicalNormalizePath;
using ::tensorstore::internal::PathDirnameBasename;
using ::tensorstore::internal::PathRootName;
using ::tensorstore::internal_testing::OnWindows;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::StrEq;

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
  EXPECT_THAT(PathDirnameBasename("/a/b/bar"), Pair("/a/b", "bar"));
  EXPECT_THAT(PathDirnameBasename("/a/b/bar/"), Pair("/a/b/bar", ""));
  EXPECT_THAT(PathDirnameBasename(""), Pair("", ""));
  EXPECT_THAT(PathDirnameBasename("/"), Pair("/", ""));
  EXPECT_THAT(PathDirnameBasename("a/b/bar"), Pair("a/b", "bar"));
  EXPECT_THAT(PathDirnameBasename("bar"), Pair("", "bar"));
  EXPECT_THAT(PathDirnameBasename("/bar"), Pair("/", "bar"));
  EXPECT_THAT(PathDirnameBasename("//a/b///bar"), Pair("//a/b", "bar"));
  EXPECT_THAT(PathDirnameBasename("///bar"), Pair("/", "bar"));

  // NOTE: Should return the drive-letter as the base name in these on
  // windows.
  EXPECT_THAT(PathDirnameBasename("C:"),
              OnWindows(Pair("C:", ""), Pair("", "C:")));
  EXPECT_THAT(PathDirnameBasename("C:bar"),
              OnWindows(Pair("C:", "bar"), Pair("", "C:bar")));
  EXPECT_THAT(PathDirnameBasename("C:/bar"),
              OnWindows(Pair("C:/", "bar"), Pair("C:", "bar")));

  // NOTE: Should return the server/share in these on windows.
  EXPECT_THAT(PathDirnameBasename("//server/share"),
              OnWindows(Pair("//server/", "share"), Pair("//server", "share")));
  EXPECT_THAT(PathDirnameBasename("//server/"),
              OnWindows(Pair("//server/", ""), Pair("//server", "")));

  // Not actually a valid network share.
  EXPECT_THAT(PathDirnameBasename("//server"), Pair("/", "server"));
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

TEST(PathTest, IsAbsolutePath) {
  EXPECT_FALSE(IsAbsolutePath(""));
  EXPECT_FALSE(IsAbsolutePath("a/b"));
  EXPECT_FALSE(IsAbsolutePath("C:"));

  EXPECT_TRUE(IsAbsolutePath("/"));
  EXPECT_TRUE(IsAbsolutePath("/tmp/bar"));

  EXPECT_TRUE(IsAbsolutePath("//server/share"));
  EXPECT_THAT(IsAbsolutePath("C:\\"), OnWindows(true, false));
}

TEST(PathTest, LexicalNormalizePath) {
  EXPECT_THAT(LexicalNormalizePath("/"), StrEq("/"));
  EXPECT_THAT(LexicalNormalizePath("a/b/c"), StrEq("a/b/c"));
  EXPECT_THAT(LexicalNormalizePath("/a/b/c"), StrEq("/a/b/c"));
  EXPECT_THAT(LexicalNormalizePath("a/b/c/"), StrEq("a/b/c/"));
  EXPECT_THAT(LexicalNormalizePath("/a/b/c/"), StrEq("/a/b/c/"));
  EXPECT_THAT(LexicalNormalizePath("a\\b\\c/"), StrEq("a/b/c/"));
  EXPECT_THAT(LexicalNormalizePath("C:a/b\\c\\"), StrEq("C:a/b/c/"));

  // self .
  EXPECT_THAT(LexicalNormalizePath("a/b/./c"), StrEq("a/b/c"));
  EXPECT_THAT(LexicalNormalizePath("./a/b/c/"), StrEq("a/b/c/"));
  EXPECT_THAT(LexicalNormalizePath("a/b/c/./"), StrEq("a/b/c/"));
  EXPECT_THAT(LexicalNormalizePath("a/b/c/."), StrEq("a/b/c/"));

  // parent ..
  EXPECT_THAT(LexicalNormalizePath("a/b/bb/../c/"), StrEq("a/b/c/"));
  EXPECT_THAT(LexicalNormalizePath("a/b/c/bb/.."), StrEq("a/b/c/"));
  EXPECT_THAT(LexicalNormalizePath("../a/b/c"), StrEq("../a/b/c"));
  EXPECT_THAT(LexicalNormalizePath("/../a/b/c"), StrEq("/a/b/c"));

  // Windows drive-letter paths.
  EXPECT_THAT(LexicalNormalizePath("C:a/b/."), StrEq("C:a/b/"));
  EXPECT_THAT(LexicalNormalizePath("C:/a/b/."), StrEq("C:/a/b/"));
  EXPECT_THAT(LexicalNormalizePath("C:a/b/bb/../c/"), StrEq("C:a/b/c/"));
  EXPECT_THAT(LexicalNormalizePath("C:/a/b/bb/../c/"), StrEq("C:/a/b/c/"));

  // Not valid networks shares.
  EXPECT_THAT(LexicalNormalizePath("//path"), StrEq("/path"));
  EXPECT_THAT(LexicalNormalizePath("///a////b//c////"), StrEq("/a/b/c/"));
}

TEST(PathTest, LexicalNormalizePath_Windows) {
  // Identical on Windows and non-windows, even with path root.
  EXPECT_THAT(LexicalNormalizePath("C:a/b\\c\\"), StrEq("C:a/b/c/"));

  // Relative paths at the beginning behave differently under windows
  // since there is a root-name available.
  EXPECT_THAT(LexicalNormalizePath("C:\\a/b\\c\\"), StrEq("C:/a/b/c/"));
  EXPECT_THAT(LexicalNormalizePath("C:./a"), OnWindows("C:a", "C:./a"));
  EXPECT_THAT(LexicalNormalizePath("C:/../a/b/c"),
              StrEq(OnWindows("C:/a/b/c", "a/b/c")));

  // Handle windows networks shares.
  EXPECT_THAT(LexicalNormalizePath("\\\\share\\path\\sub"),
              StrEq(OnWindows("\\\\share/path/sub", "/share/path/sub")));
  EXPECT_THAT(LexicalNormalizePath("//share/path/sub"),
              StrEq(OnWindows("//share/path/sub", "/share/path/sub")));

  // Slashes may be normalized differently under windows since a network
  // share may be detected.
  EXPECT_THAT(LexicalNormalizePath("//a////b//c////"),
              StrEq(OnWindows("//a/b/c/", "/a/b/c/")));
}

TEST(PathTest, PathRootName) {
  EXPECT_THAT(PathRootName(""), IsEmpty());
  EXPECT_THAT(PathRootName("/"), IsEmpty());
  EXPECT_THAT(PathRootName("a/b/c"), IsEmpty());
  EXPECT_THAT(PathRootName("/a/b/c"), IsEmpty());

  // Never a valid network share.
  EXPECT_THAT(PathRootName("///a/b/c"), IsEmpty());

  // Absolute and relative paths return the drive-letter.
  EXPECT_THAT(PathRootName("C:\\a/b\\c\\"), StrEq(OnWindows("C:", "")));
  EXPECT_THAT(PathRootName("C:a/b\\c\\"), StrEq(OnWindows("C:", "")));

  // Windows network shares return the share name.
  EXPECT_THAT(PathRootName("\\\\share\\path\\sub"),
              StrEq(OnWindows("\\\\share", "")));
  EXPECT_THAT(PathRootName("//share//path//sub"),
              StrEq(OnWindows("//share", "")));
}

}  // namespace
