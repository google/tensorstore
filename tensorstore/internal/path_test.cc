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
using ::tensorstore::internal::PathDirnameBasename;

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

}  // namespace
