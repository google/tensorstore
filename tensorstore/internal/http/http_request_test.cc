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

#include "tensorstore/internal/http/http_request.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"

namespace {

using ::tensorstore::internal_http::HttpRequestBuilder;

TEST(HttpRequestBuilder, BuildRequest) {
  auto request = HttpRequestBuilder("GET", "http://127.0.0.1:0/")
                     .AddUserAgentPrefix("test")
                     .AddHeader("X-foo: bar")
                     .AddQueryParameter("name", "dragon")
                     .AddQueryParameter("age", "1234")
                     .EnableAcceptEncoding()
                     .BuildRequest()
                     .value();

  EXPECT_EQ("http://127.0.0.1:0/?name=dragon&age=1234", request.url());
  EXPECT_TRUE(request.accept_encoding());
  EXPECT_EQ("test", request.user_agent());
  EXPECT_EQ("GET", request.method());
  EXPECT_THAT(request.headers(), testing::ElementsAre("X-foo: bar"));
}

TEST(HttpRequestBuilder, AddCacheControlMaxAgeHeader) {
  {
    HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
    builder.MaybeAddCacheControlMaxAgeHeader(absl::InfiniteDuration());
    EXPECT_THAT(builder.BuildRequest().value().headers(), ::testing::IsEmpty());
  }
  {
    HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
    builder.MaybeAddCacheControlMaxAgeHeader(absl::ZeroDuration());
    EXPECT_THAT(builder.BuildRequest().value().headers(),
                ::testing::ElementsAre("cache-control: no-cache"));
  }
  {
    HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
    builder.MaybeAddCacheControlMaxAgeHeader(absl::Seconds(10));
    EXPECT_THAT(builder.BuildRequest().value().headers(),
                ::testing::ElementsAre("cache-control: max-age=10"));
  }
  {
    HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
    builder.MaybeAddCacheControlMaxAgeHeader(-absl::Seconds(10));
    EXPECT_THAT(builder.BuildRequest().value().headers(),
                ::testing::ElementsAre("cache-control: no-cache"));
  }
}

TEST(HttpRequestBuilder, AddStalenessBoundCacheControlHeader) {
  {
    HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
    builder.MaybeAddStalenessBoundCacheControlHeader(absl::InfinitePast());
    EXPECT_THAT(builder.BuildRequest().value().headers(), ::testing::IsEmpty());
  }
  {
    // staleness is in the future => no-cache.
    HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
    builder.MaybeAddStalenessBoundCacheControlHeader(absl::InfiniteFuture());
    EXPECT_THAT(builder.BuildRequest().value().headers(),
                ::testing::ElementsAre("cache-control: no-cache"));
  }
  {
    const absl::Time kFutureTime = absl::Now() + absl::Minutes(525600);
    HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
    builder.MaybeAddStalenessBoundCacheControlHeader(kFutureTime);
    EXPECT_THAT(builder.BuildRequest().value().headers(),
                ::testing::ElementsAre("cache-control: no-cache"));
  }
  {
    // staleness is in the past => max-age
    HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
    builder.MaybeAddStalenessBoundCacheControlHeader(
        absl::Now() - absl::Milliseconds(5900));
    EXPECT_THAT(builder.BuildRequest().value().headers(),
                ::testing::ElementsAre(::testing::AnyOf("cache-control: max-age=4",
                                       ::testing::AnyOf("cache-control: max-age=5"))));
  }
}

TEST(HttpRequestBuilder, AddRangeHeader) {
  {
    HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
    builder.AddRangeHeader({});
    EXPECT_THAT(builder.BuildRequest().value().headers(), ::testing::IsEmpty());
  }
  {
    HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
    builder.AddRangeHeader({1});
    EXPECT_THAT(builder.BuildRequest().value().headers(),
                ::testing::ElementsAre("Range: bytes=1-"));
  }
  {
    HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
    builder.AddRangeHeader({1, 2});
    EXPECT_THAT(builder.BuildRequest().value().headers(),
                ::testing::ElementsAre("Range: bytes=1-1"));
  }
}

}  // namespace
