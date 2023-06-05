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
                     .BuildRequest();

  EXPECT_EQ("http://127.0.0.1:0/?name=dragon&age=1234", request.url());
  EXPECT_TRUE(request.accept_encoding());
  EXPECT_EQ("test", request.user_agent());
  EXPECT_EQ("GET", request.method());
  EXPECT_THAT(request.headers(), testing::ElementsAre("X-foo: bar"));
}

TEST(HttpRequestBuilder, AddCacheControlMaxAgeHeader) {
  HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
  bool result;
  builder.AddCacheControlMaxAgeHeader(absl::InfiniteDuration(), result);
  EXPECT_FALSE(result);
  builder.AddCacheControlMaxAgeHeader(absl::ZeroDuration(), result);
  EXPECT_TRUE(result);
  builder.AddCacheControlMaxAgeHeader(absl::Seconds(10), result);
  EXPECT_TRUE(result);
  builder.AddCacheControlMaxAgeHeader(-absl::Seconds(10), result);
  EXPECT_TRUE(result);

  auto request = builder.BuildRequest();

  EXPECT_THAT(request.headers(),
              testing::ElementsAre("cache-control: no-cache",
                                   "cache-control: max-age=10",
                                   "cache-control: no-cache"));
}

TEST(HttpRequestBuilder, AddStalenessBoundCacheControlHeader) {
  const absl::Time kFutureTime = absl::Now() + absl::Minutes(525600);
  HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
  bool result;
  builder.AddStalenessBoundCacheControlHeader(absl::InfinitePast(), result);
  EXPECT_FALSE(result);
  // staleness is in the future => no-cache.
  builder.AddStalenessBoundCacheControlHeader(absl::InfiniteFuture(), result);
  EXPECT_TRUE(result);
  builder.AddStalenessBoundCacheControlHeader(kFutureTime, result);
  EXPECT_TRUE(result);
  // staleness is in the past => max-age
  builder.AddStalenessBoundCacheControlHeader(
      absl::Now() - absl::Milliseconds(5900), result);
  EXPECT_TRUE(result);

  auto request = builder.BuildRequest();

  EXPECT_THAT(
      request.headers(),
      testing::ElementsAre("cache-control: no-cache", "cache-control: no-cache",
                           ::testing::AnyOf("cache-control: max-age=5",
                                            "cache-control: max-age=4")));
}

TEST(HttpRequestBuilder, AddRangeHeader) {
  HttpRequestBuilder builder("GET", "http://127.0.0.1:0/");
  bool result;
  builder.AddRangeHeader({}, result);
  EXPECT_FALSE(result);
  builder.AddRangeHeader({1}, result);
  EXPECT_TRUE(result);
  builder.AddRangeHeader({1, 2}, result);
  EXPECT_TRUE(result);

  auto request = builder.BuildRequest();
  EXPECT_THAT(request.headers(),
              testing::ElementsAre("Range: bytes=1-", "Range: bytes=1-1"));
}

}  // namespace
