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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

using tensorstore::internal::CreateURI;
using tensorstore::internal::JoinPath;
using tensorstore::internal::ParseURI;

namespace {

TEST(PathTest, JoinPath) {
  EXPECT_EQ("/foo/bar", JoinPath("/foo", "bar"));
  EXPECT_EQ("/foo/bar", JoinPath("/foo/", "bar"));
  EXPECT_EQ("/foo/bar", JoinPath("/foo", "/bar"));
  EXPECT_EQ("/foo/bar", JoinPath("/foo/", "/bar"));

  EXPECT_EQ("foo/bar", JoinPath("foo", "bar"));
  EXPECT_EQ("foo/bar", JoinPath("foo", "/bar"));

  EXPECT_EQ("/bar", JoinPath("", "/bar"));
  EXPECT_EQ("bar", JoinPath("", "bar"));
  EXPECT_EQ("/foo", JoinPath("/foo", ""));

  EXPECT_EQ("/foo/bar/baz/blah/blink/biz",
            JoinPath("/foo/bar/baz/", "/blah/blink/biz"));
  EXPECT_EQ("/foo/bar/baz/blah", JoinPath("/foo", "bar", "baz", "blah"));

  EXPECT_EQ("http://foo/bar/baz", JoinPath("http://foo/", "bar", "baz"));
}

TEST(PathTest, JoinPath_MixedArgs) {
  constexpr const char kFoo[] = "/foo";
  absl::string_view foo_view("/foo");
  std::string foo("/foo");

  EXPECT_EQ("/foo/bar", JoinPath(foo_view, "bar"));
  EXPECT_EQ("/foo/bar", JoinPath(foo, "bar"));
  EXPECT_EQ("/foo/bar", JoinPath(kFoo, "/bar"));
}

#define EXPECT_PARSE_URI(uri, scheme, host, path)                  \
  do {                                                             \
    EXPECT_EQ(uri, CreateURI(scheme, host, path));                 \
    absl::string_view s, h, p;                                     \
    absl::string_view u(uri);                                      \
    ParseURI(u, &s, &h, &p);                                       \
    EXPECT_EQ(scheme, s) << "s=" << s << " h=" << h << " p=" << p; \
    EXPECT_EQ(host, h) << "s=" << s << " h=" << h << " p=" << p;   \
    EXPECT_EQ(path, p) << "s=" << s << " h=" << h << " p=" << p;   \
    EXPECT_LE(u.begin(), s.begin());                               \
    EXPECT_GE(u.end(), s.begin());                                 \
    EXPECT_LE(u.begin(), s.end());                                 \
    EXPECT_GE(u.end(), s.end());                                   \
    EXPECT_LE(u.begin(), h.begin());                               \
    EXPECT_GE(u.end(), h.begin());                                 \
    EXPECT_LE(u.begin(), h.end());                                 \
    EXPECT_GE(u.end(), h.end());                                   \
    EXPECT_LE(u.begin(), p.begin());                               \
    EXPECT_GE(u.end(), p.begin());                                 \
    EXPECT_LE(u.begin(), p.end());                                 \
    EXPECT_GE(u.end(), p.end());                                   \
  } while (0)

TEST(PathTest, ParseURI) {
  EXPECT_PARSE_URI("http://foo", "http", "foo", "");
  EXPECT_PARSE_URI("/encrypted/://foo", "", "", "/encrypted/://foo");
  EXPECT_PARSE_URI("/usr/local/foo", "", "", "/usr/local/foo");
  EXPECT_PARSE_URI("file:///usr/local/foo", "file", "", "/usr/local/foo");
  EXPECT_PARSE_URI("local.file:///usr/local/foo", "local.file", "",
                   "/usr/local/foo");

  EXPECT_PARSE_URI("a-b:///foo", "a-b", "", "/foo");
  EXPECT_PARSE_URI("a=b:///foo", "", "", "a=b:///foo");

  EXPECT_PARSE_URI(":///foo", "", "", ":///foo");
  EXPECT_PARSE_URI("9dfd:///foo", "", "", "9dfd:///foo");
  EXPECT_PARSE_URI("file:", "", "", "file:");
  EXPECT_PARSE_URI("file:/", "", "", "file:/");
  EXPECT_PARSE_URI("hdfs://localhost:8020/path/to/file", "hdfs",
                   "localhost:8020", "/path/to/file");
  EXPECT_PARSE_URI("hdfs://localhost:8020", "hdfs", "localhost:8020", "");
  EXPECT_PARSE_URI("hdfs://localhost:8020/", "hdfs", "localhost:8020", "/");
}

TEST(PathTest, ParseURIMissingParams) {
  absl::string_view s, h, p;
  ParseURI("http://foo/bar", &s, nullptr, nullptr);
  EXPECT_EQ("http", s);

  ParseURI("http://foo/bar", nullptr, &h, nullptr);
  EXPECT_EQ("foo", h);

  ParseURI("http://foo/bar", nullptr, nullptr, &p);
  EXPECT_EQ("/bar", p);
}

}  // namespace
