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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/queue_testutil.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/status_testutil.h"

namespace {
namespace kvstore = ::tensorstore::kvstore;
using ::tensorstore::Future;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Promise;
using ::tensorstore::PromiseFuturePair;
using ::tensorstore::StorageGeneration;
using ::tensorstore::internal::MatchesKvsReadResult;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_http::SetDefaultHttpTransport;

class MyMockTransport : public HttpTransport {
 public:
  Future<HttpResponse> IssueRequest(const HttpRequest& request,
                                    absl::Cord payload,
                                    absl::Duration request_timeout,
                                    absl::Duration connect_timeout) override {
    auto [promise, future] = PromiseFuturePair<HttpResponse>::Make();
    requests_.push({request, promise});
    return future;
  }

  struct Request {
    HttpRequest request;
    Promise<HttpResponse> promise;
  };

  tensorstore::internal::ConcurrentQueue<Request> requests_;
};

struct DefaultHttpTransportSetter {
  DefaultHttpTransportSetter(std::shared_ptr<HttpTransport> transport) {
    SetDefaultHttpTransport(transport);
  }
  ~DefaultHttpTransportSetter() { SetDefaultHttpTransport(nullptr); }
};

class HttpKeyValueStoreTest : public ::testing::Test {
 public:
  std::shared_ptr<MyMockTransport> mock_transport =
      std::make_shared<MyMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};
};

TEST(DescribeKeyTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());

  EXPECT_EQ("https://example.com/my/path/xyz",
            store.driver->DescribeKey("/my/path/xyz"));
}

TEST_F(HttpKeyValueStoreTest, UnconditionalReadUncachedWithEtag) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());
  auto read_future = kvstore::Read(store, "abc");
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
  EXPECT_THAT(request.request.headers(),
              ::testing::ElementsAre("cache-control: no-cache"));
  request.promise.SetResult(
      HttpResponse{200, absl::Cord("value"), {{"etag", "\"xyz\""}}});
  EXPECT_THAT(read_future.result(),
              MatchesKvsReadResult(absl::Cord("value"),
                                   StorageGeneration::FromString("xyz")));
}

TEST_F(HttpKeyValueStoreTest, ReadNotFound) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());
  auto read_future = kvstore::Read(store, "abc");
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
  EXPECT_THAT(request.request.headers(),
              ::testing::ElementsAre("cache-control: no-cache"));
  request.promise.SetResult(HttpResponse{404, absl::Cord()});
  EXPECT_THAT(read_future.result(), MatchesKvsReadResultNotFound());
}

TEST_F(HttpKeyValueStoreTest, UnconditionalReadWeakEtag) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());
  auto read_future = kvstore::Read(store, "abc");
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
  EXPECT_THAT(request.request.headers(),
              ::testing::ElementsAre("cache-control: no-cache"));
  request.promise.SetResult(
      HttpResponse{200, absl::Cord("value"), {{"etag", "W/\"xyz\""}}});
  EXPECT_THAT(
      read_future.result(),
      MatchesKvsReadResult(absl::Cord("value"), StorageGeneration::Invalid()));
}

TEST_F(HttpKeyValueStoreTest, ReadByteRange) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());
  kvstore::ReadOptions options;
  options.byte_range.inclusive_min = 10;
  options.byte_range.exclusive_max = 20;
  auto read_future = kvstore::Read(store, "abc", options);
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
  EXPECT_THAT(
      request.request.headers(),
      ::testing::ElementsAre("cache-control: no-cache", "Range: bytes=10-19"));
  request.promise.SetResult(HttpResponse{
      206, absl::Cord("valueabcde"), {{"content-range", "bytes 10-19/50"}}});
  EXPECT_THAT(read_future.result(),
              MatchesKvsReadResult(absl::Cord("valueabcde"),
                                   StorageGeneration::Invalid()));
}

TEST_F(HttpKeyValueStoreTest, ReadWithStalenessBound) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());
  kvstore::ReadOptions options;
  options.staleness_bound = absl::Now() - absl::Milliseconds(4500);
  auto read_future = kvstore::Read(store, "abc", options);
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
  EXPECT_THAT(request.request.headers(),
              ::testing::ElementsAre("cache-control: max-age=4"));
  request.promise.SetResult(HttpResponse{200, absl::Cord("value")});
  EXPECT_THAT(
      read_future.result(),
      MatchesKvsReadResult(absl::Cord("value"), StorageGeneration::Invalid()));
}

TEST_F(HttpKeyValueStoreTest, IfEqualSatisfied) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());
  kvstore::ReadOptions options;
  options.if_equal = StorageGeneration::FromString("xyz");
  auto read_future = kvstore::Read(store, "abc", options);
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
  EXPECT_THAT(
      request.request.headers(),
      ::testing::ElementsAre("cache-control: no-cache", "if-match: \"xyz\""));
  request.promise.SetResult(HttpResponse{200, absl::Cord("value")});
  EXPECT_THAT(read_future.result(), MatchesKvsReadResult(absl::Cord("value")));
}

TEST_F(HttpKeyValueStoreTest, IfEqualNotSatisfied) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());
  kvstore::ReadOptions options;
  options.if_equal = StorageGeneration::FromString("xyz");
  auto read_future = kvstore::Read(store, "abc", options);
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
  EXPECT_THAT(
      request.request.headers(),
      ::testing::ElementsAre("cache-control: no-cache", "if-match: \"xyz\""));
  request.promise.SetResult(HttpResponse{412});
  EXPECT_THAT(read_future.result(),
              MatchesKvsReadResult(kvstore::ReadResult::kUnspecified,
                                   StorageGeneration::Unknown()));
}

TEST_F(HttpKeyValueStoreTest, IfNotEqualSatisfied) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());
  kvstore::ReadOptions options;
  options.if_not_equal = StorageGeneration::FromString("xyz");
  auto read_future = kvstore::Read(store, "abc", options);
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
  EXPECT_THAT(request.request.headers(),
              ::testing::ElementsAre("cache-control: no-cache",
                                     "if-none-match: \"xyz\""));
  request.promise.SetResult(HttpResponse{200, absl::Cord("value")});
  EXPECT_THAT(
      read_future.result(),
      MatchesKvsReadResult(absl::Cord("value"), StorageGeneration::Invalid()));
}

TEST_F(HttpKeyValueStoreTest, IfNotEqualNotSatisfied) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());
  kvstore::ReadOptions options;
  options.if_not_equal = StorageGeneration::FromString("xyz");
  auto read_future = kvstore::Read(store, "abc", options);
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
  EXPECT_THAT(request.request.headers(),
              ::testing::ElementsAre("cache-control: no-cache",
                                     "if-none-match: \"xyz\""));
  request.promise.SetResult(HttpResponse{304});
  EXPECT_THAT(read_future.result(),
              MatchesKvsReadResult(kvstore::ReadResult::kUnspecified,
                                   StorageGeneration::FromString("xyz")));
}

TEST_F(HttpKeyValueStoreTest, Retry) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());

  auto read_future = kvstore::Read(store, "abc");
  {
    auto request = mock_transport->requests_.pop();
    EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
    EXPECT_THAT(request.request.headers(),
                ::testing::ElementsAre("cache-control: no-cache"));
    request.promise.SetResult(HttpResponse{503, absl::Cord()});
  }
  {
    auto request = mock_transport->requests_.pop();
    EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
    EXPECT_THAT(request.request.headers(),
                ::testing::ElementsAre("cache-control: no-cache"));
    request.promise.SetResult(
        HttpResponse{200, absl::Cord("value"), {{"etag", "\"xyz\""}}});
  }
  EXPECT_THAT(read_future.result(),
              MatchesKvsReadResult(absl::Cord("value"),
                                   StorageGeneration::FromString("xyz")));
}

TEST_F(HttpKeyValueStoreTest, RetryMax) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open(
          {{"driver", "http"},
           {"base_url", "https://example.com/my/path/"},
           {"context", {{"http_request_retries", {{"max_retries", 1}}}}}})
          .result());

  auto read_future = kvstore::Read(store, "abc");
  {
    auto request = mock_transport->requests_.pop();
    EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
    EXPECT_THAT(request.request.headers(),
                ::testing::ElementsAre("cache-control: no-cache"));
    request.promise.SetResult(HttpResponse{503, absl::Cord()});
  }
  EXPECT_THAT(read_future.result(), MatchesStatus(absl::StatusCode::kAborted));
}

TEST_F(HttpKeyValueStoreTest, Date) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());

  kvstore::ReadOptions options;
  options.staleness_bound = absl::InfinitePast();
  auto read_future = kvstore::Read(store, "abc", options);
  auto response_date = absl::UnixEpoch() + absl::Seconds(100);
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
  EXPECT_THAT(request.request.headers(), ::testing::ElementsAre());
  request.promise.SetResult(HttpResponse{
      200,
      absl::Cord("value"),
      {{"date", absl::FormatTime(tensorstore::internal_http::kHttpTimeFormat,
                                 response_date, absl::UTCTimeZone())}}});
  EXPECT_THAT(
      read_future.result(),
      MatchesKvsReadResult(absl::Cord("value"), StorageGeneration::Invalid(),
                           response_date));
}

TEST_F(HttpKeyValueStoreTest, DateSkew) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());

  kvstore::ReadOptions options;
  options.staleness_bound = absl::Now() - absl::Milliseconds(5500);
  auto read_future = kvstore::Read(store, "abc", options);
  auto response_date = absl::UnixEpoch() + absl::Seconds(100);
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
  EXPECT_THAT(request.request.headers(),
              ::testing::ElementsAre("cache-control: max-age=5"));
  request.promise.SetResult(HttpResponse{
      200,
      absl::Cord("value"),
      {{"date", absl::FormatTime(tensorstore::internal_http::kHttpTimeFormat,
                                 response_date, absl::UTCTimeZone())}}});
  EXPECT_THAT(
      read_future.result(),
      MatchesKvsReadResult(absl::Cord("value"), StorageGeneration::Invalid(),
                           options.staleness_bound));
}

TEST_F(HttpKeyValueStoreTest, Query) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open("https://example.com/my/path/?query=value").result());
  auto read_future = kvstore::Read(store, "abc");
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc?query=value",
            request.request.url());
  EXPECT_THAT(request.request.headers(),
              ::testing::ElementsAre("cache-control: no-cache"));
  request.promise.SetResult(HttpResponse{200, absl::Cord("value")});
  EXPECT_THAT(read_future.result(), MatchesKvsReadResult(absl::Cord("value")));
}

TEST_F(HttpKeyValueStoreTest, InvalidDate) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open("https://example.com/my/path/").result());

  auto read_future = kvstore::Read(store, "abc");
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc", request.request.url());
  EXPECT_THAT(request.request.headers(),
              ::testing::ElementsAre("cache-control: no-cache"));
  request.promise.SetResult(
      HttpResponse{200, absl::Cord("value"), {{"date", "xyz"}}});
  EXPECT_THAT(read_future.result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid \"date\" response header: \"xyz\""));
}

TEST_F(HttpKeyValueStoreTest, ExtraHeaders) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "http"},
                     {"base_url", "https://example.com/my/path/?query=value"},
                     {"headers", {"a!#$%&'*+-.^_`|~3X: b\xfe"}}})
          .result());
  auto read_future = kvstore::Read(store, "abc");
  auto request = mock_transport->requests_.pop();
  EXPECT_EQ("https://example.com/my/path/abc?query=value",
            request.request.url());
  EXPECT_THAT(request.request.headers(),
              ::testing::ElementsAre("a!#$%&'*+-.^_`|~3X: b\xfe",
                                     "cache-control: no-cache"));
  request.promise.SetResult(HttpResponse{200, absl::Cord("value")});
  EXPECT_THAT(read_future.result(), MatchesKvsReadResult(absl::Cord("value")));
}

TEST(UrlTest, UrlRoundtrip) {
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "http"},
       {"base_url", "https://example.com:8080"},
       {"path", "/abc"}},
      "https://example.com:8080/abc");
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "http"},
       {"base_url", "https://example.com:8080"},
       {"path", "/abc def"}},
      "https://example.com:8080/abc%20def");
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "http"},
       {"base_url", "http://example.com:8080"},
       {"path", "/abc def"}},
      "http://example.com:8080/abc%20def");
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "http"},
       {"base_url", "https://example.com:8080?query=value"},
       {"path", "/abc def"}},
      "https://example.com:8080/abc%20def?query=value");
}

TEST(UrlTest, InvalidUri) {
  EXPECT_THAT(kvstore::Spec::FromUrl("http://example.com#fragment"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: Fragment identifier not supported"));
}

TEST(SpecTest, InvalidScheme) {
  EXPECT_THAT(
      kvstore::Open({{"driver", "http"}, {"base_url", "file:///abc"}}).result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
}
TEST(SpecTest, MissingScheme) {
  EXPECT_THAT(kvstore::Open({{"driver", "http"}, {"base_url", "abc"}}).result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}
TEST(SpecTest, InvalidFragment) {
  EXPECT_THAT(kvstore::Open({{"driver", "http"},
                             {"base_url", "https://example.com#fragment"}})
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(SpecTest, InvalidHeader) {
  EXPECT_THAT(kvstore::Open({{"driver", "http"},
                             {"base_url", "https://example.com"},
                             {"headers", {"a"}}})
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(SpecTest, SpecRoundtrip) {
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.check_write_read = false;
  options.check_data_persists = false;
  options.full_spec = {{"driver", "http"},
                       {"base_url", "https://example.com?query"},
                       {"headers", {"a: b"}},
                       {"path", "/abc"}};
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(SpecTest, NormalizeSpecRelativePath) {
  tensorstore::internal::TestKeyValueStoreSpecRoundtripNormalize(
      {{"driver", "http"},
       {"base_url", "https://example.com/my/path?query=value"},
       {"path", "abc"}},
      {{"driver", "http"},
       {"base_url", "https://example.com?query=value"},
       {"path", "/my/path/abc"}});
}

TEST(SpecTest, NormalizeSpecAbsolutePath) {
  tensorstore::internal::TestKeyValueStoreSpecRoundtripNormalize(
      {{"driver", "http"},
       {"base_url", "https://example.com/?query=value"},
       {"path", "/abc"}},
      {{"driver", "http"},
       {"base_url", "https://example.com?query=value"},
       {"path", "/abc"}});
}

TEST(SpecTest, NormalizeSpecInvalidAbsolutePath) {
  EXPECT_THAT(
      kvstore::Open({{"driver", "http"},
                     {"base_url", "https://example.com/my/path?query=value"},
                     {"path", "/abc"}})
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot specify absolute path \"/abc\" in conjunction with "
                    "base URL \".*\" that includes a path component"));
}

}  // namespace
