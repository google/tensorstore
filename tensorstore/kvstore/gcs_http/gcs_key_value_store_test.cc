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

#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/http/mock_http_transport.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/oauth2/google_auth_provider.h"
#include "tensorstore/internal/oauth2/google_auth_test_utils.h"
#include "tensorstore/internal/thread/schedule_at.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/gcs_http/gcs_mock.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;

using ::tensorstore::CompletionNotifyingReceiver;
using ::tensorstore::Context;
using ::tensorstore::Future;
using ::tensorstore::GCSMockStorageBucket;
using ::tensorstore::KeyRange;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::StorageGeneration;
using ::tensorstore::internal::MatchesListEntry;
using ::tensorstore::internal::ScheduleAt;
using ::tensorstore::internal_http::ApplyResponseToHandler;
using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpResponseHandler;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_http::IssueRequestOptions;
using ::tensorstore::internal_http::SetDefaultHttpTransport;
using ::tensorstore::internal_oauth2::GoogleAuthTestScope;

static constexpr char kUriScheme[] = "gs";
static constexpr char kDriver[] = "gcs";

// Responds to a "metadata.google.internal" request.
class MetadataMockHelper {
 public:
  tensorstore::Result<HttpResponse> GetResponse(const HttpRequest& request) {
    auto parsed = tensorstore::internal::ParseGenericUri(request.url);

    if (!absl::StartsWith(parsed.authority_and_path,
                          "metadata.google.internal/")) {
      return absl::UnimplementedError("Mock cannot satisfy the request.");
    }

    // Respond with the GCE OAuth2 token
    constexpr char kOAuthPath[] =
        "metadata.google.internal/computeMetadata/v1/"
        "instance/service-accounts/user@nowhere.com/token";
    if (absl::StartsWith(parsed.authority_and_path, kOAuthPath)) {
      return HttpResponse{
          200,
          absl::Cord(
              R"({ "token_type" : "refresh", "access_token": "abc", "expires_in": 3600 })")};
    }

    // Respond with the GCE context metadata.
    constexpr char kServiceAccountPath[] =
        "metadata.google.internal/computeMetadata/v1/"
        "instance/service-accounts/default/";
    if (absl::StartsWith(parsed.authority_and_path, kServiceAccountPath)) {
      return HttpResponse{
          200, absl::Cord(
                   R"({ "email": "user@nowhere.com", "scopes": [ "test" ] })")};
    }

    // Pretend to run on GCE.
    return HttpResponse{200, absl::Cord()};
  }

  GoogleAuthTestScope google_auth_test_scope;
};

class MyMockTransport : public HttpTransport {
 public:
  void IssueRequestWithHandler(const HttpRequest& request,
                               IssueRequestOptions options,
                               HttpResponseHandler* response_handler) override {
    ApplyResponseToHandler(
        [&]() -> Result<HttpResponse> {
          auto result = metadata_mock_.GetResponse(request);
          if (result.ok()) return result;

          // Next, try each bucket until there is a success.
          for (auto* bucket : buckets_) {
            result = bucket->IssueRequest(request, options.payload);
            if (result.ok()) break;
          }
          return result;
        }(),
        response_handler);
  }

  MetadataMockHelper metadata_mock_;
  std::vector<GCSMockStorageBucket*> buckets_;
};

struct DefaultHttpTransportSetter {
  DefaultHttpTransportSetter(std::shared_ptr<HttpTransport> transport) {
    SetDefaultHttpTransport(transport);
    tensorstore::internal_oauth2::ResetSharedGoogleAuthProvider();
  }
  ~DefaultHttpTransportSetter() {
    tensorstore::internal_oauth2::ResetSharedGoogleAuthProvider();
    SetDefaultHttpTransport(nullptr);
  }
};

Context DefaultTestContext() {
  // Opens the gcs driver with small exponential backoff values.
  return Context{Context::Spec::FromJson({{"gcs_request_retries",
                                           {{"max_retries", 4},
                                            {"initial_delay", "1ms"},
                                            {"max_delay", "5ms"}}}})
                     .value()};
}

TEST(GcsKeyValueStoreTest, BadBucketNames) {
  auto context = DefaultTestContext();
  for (auto bucket :
       {"a", "_abc", "abc_", "ABC", "a..b", "a.-.b",
        "a."
        "0123456789123456789012345678912345678901234567891234567890"
        "1234567891234567890123456789123456789012345678912345678901"
        "23456789123456789.b"}) {
    EXPECT_FALSE(
        kvstore::Open({{"driver", kDriver}, {"bucket", bucket}}, context)
            .result())
        << "bucket: " << bucket;
  }
  for (auto bucket : {"abc", "abc.1-2_3.abc"}) {
    EXPECT_TRUE(
        kvstore::Open({{"driver", kDriver}, {"bucket", bucket}}, context)
            .result())
        << "bucket: " << bucket;
  }
}

TEST(GcsKeyValueStoreTest, BadObjectNames) {
  auto mock_transport = std::make_shared<MyMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  GCSMockStorageBucket bucket("my-bucket");
  mock_transport->buckets_.push_back(&bucket);

  auto context = DefaultTestContext();

  // https://www.googleapis.com/kvstore/v1/b/my-project/o/test
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", kDriver}, {"bucket", "my-bucket"}}, context)
          .result());
  EXPECT_THAT(kvstore::Read(store, ".").result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(kvstore::Read(store, "..").result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(kvstore::Read(store, ".well-known/acme-challenge").result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(kvstore::Read(store, "foo\nbar").result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(kvstore::Read(store, "foo\rbar").result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_not_equal =
        StorageGeneration::FromString("abc123");
    EXPECT_THAT(kvstore::Read(store, "abc", options).result(),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }
}

TEST(GcsKeyValueStoreTest, Basic) {
  // Setup mocks for:
  // https://www.googleapis.com/kvstore/v1/b/my-bucket/o/test
  auto mock_transport = std::make_shared<MyMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  GCSMockStorageBucket bucket("my-bucket");
  mock_transport->buckets_.push_back(&bucket);

  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", kDriver}, {"bucket", "my-bucket"}}, context)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, store.spec());
  EXPECT_THAT(spec.ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(
                  MatchesJson({{"driver", kDriver}, {"bucket", "my-bucket"}})));
  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST(GcsKeyValueStoreTest, Retry) {
  for (int max_retries : {2, 3, 4}) {
    for (bool fail : {false, true}) {
      ABSL_LOG(INFO) << max_retries << (fail ? " fail" : " success");

      // Setup mocks for:
      // https://www.googleapis.com/kvstore/v1/b/my-bucket/o/test
      auto mock_transport = std::make_shared<MyMockTransport>();
      DefaultHttpTransportSetter mock_transport_setter{mock_transport};

      GCSMockStorageBucket bucket("my-bucket");
      mock_transport->buckets_.push_back(&bucket);

      auto context = Context::Default();
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto store, kvstore::Open({{"driver", kDriver},
                                     {"bucket", "my-bucket"},
                                     {"context",
                                      {
                                          {"gcs_request_retries",
                                           {{"max_retries", max_retries},
                                            {"initial_delay", "1ms"},
                                            {"max_delay", "10ms"}}},
                                      }}},
                                    context)
                          .result());
      if (fail) {
        bucket.TriggerErrors(max_retries + 1);
        EXPECT_THAT(kvstore::Read(store, "x").result(),
                    MatchesStatus(absl::StatusCode::kAborted));
      } else {
        bucket.TriggerErrors(max_retries - 2);
        TENSORSTORE_EXPECT_OK(kvstore::Read(store, "x").result());
      }
    }
  }
}

TEST(GcsKeyValueStoreTest, List) {
  // Setup mocks for:
  // https://www.googleapis.com/kvstore/v1/b/my-bucket/o/test
  auto mock_transport = std::make_shared<MyMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  GCSMockStorageBucket bucket("my-bucket");
  mock_transport->buckets_.push_back(&bucket);

  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", kDriver}, {"bucket", "my-bucket"}}, context)
          .result());

  // Listing an empty bucket via `List` works.
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        CompletionNotifyingReceiver{&notification,
                                    tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();
    EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_done",
                                            "set_stopping"));
  }

  // Listing an empty bucket via `ListFuture` works.
  EXPECT_THAT(ListFuture(store, {}).result(),
              ::testing::Optional(::testing::ElementsAre()));

  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/b", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/d", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/x", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/y", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/e", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/f", absl::Cord("xyz")));

  // Listing the entire stream via `List` works.
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        CompletionNotifyingReceiver{&notification,
                                    tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();
    EXPECT_THAT(
        log, ::testing::UnorderedElementsAre(
                 "set_starting", "set_value: a/d", "set_value: a/c/z/f",
                 "set_value: a/c/y", "set_value: a/c/z/e", "set_value: a/c/x",
                 "set_value: a/b", "set_done", "set_stopping"));
  }

  // Listing the entire stream via `ListFuture` works.
  EXPECT_THAT(ListFuture(store, {}).result(),
              ::testing::Optional(::testing::UnorderedElementsAre(
                  MatchesListEntry("a/d"), MatchesListEntry("a/c/z/f"),
                  MatchesListEntry("a/c/y"), MatchesListEntry("a/c/z/e"),
                  MatchesListEntry("a/c/x"), MatchesListEntry("a/b"))));

  // Listing a subset of the stream via `List` works.
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {KeyRange::Prefix("a/c/")}),
        CompletionNotifyingReceiver{&notification,
                                    tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();
    EXPECT_THAT(log, ::testing::UnorderedElementsAre(
                         "set_starting", "set_value: a/c/z/f",
                         "set_value: a/c/y", "set_value: a/c/z/e",
                         "set_value: a/c/x", "set_done", "set_stopping"));
  }

  // Cancellation immediately after starting yields nothing..
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        CompletionNotifyingReceiver{
            &notification, tensorstore::CancelOnStartingReceiver{{&log}}});
    notification.WaitForNotification();
    EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_done",
                                            "set_stopping"));
  }

  // Cancellation in the middle of the stream stops the stream.
  // However it may not do it immediately.
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        CompletionNotifyingReceiver{
            &notification, tensorstore::CancelAfterNReceiver<2>{{&log}}});
    notification.WaitForNotification();
    EXPECT_THAT(log, ::testing::Contains("set_starting"));
    EXPECT_THAT(log, ::testing::Contains("set_done"));
    EXPECT_THAT(log, ::testing::Contains("set_stopping"));
    // We cannot guarantee that a single value is returned, so just verify
    // that any of the following exists and the size.
    EXPECT_LE(4, log.size());
    EXPECT_THAT(
        log, ::testing::Contains(::testing::AnyOf(
                 "set_value: a/d", "set_value: a/c/z/f", "set_value: a/c/y",
                 "set_value: a/c/z/e", "set_value: a/c/x", "set_value: a/b")));
  }

  // Listing a subset of the stream via `ListFuture` works.
  EXPECT_THAT(ListFuture(store, {KeyRange::Prefix("a/c/")}).result(),
              ::testing::Optional(::testing::UnorderedElementsAre(
                  MatchesListEntry("a/c/z/f"), MatchesListEntry("a/c/y"),
                  MatchesListEntry("a/c/z/e"), MatchesListEntry("a/c/x"))));
}

TEST(GcsKeyValueStoreTest, SpecRoundtrip) {
  auto mock_transport = std::make_shared<MyMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  GCSMockStorageBucket bucket("my-bucket");
  mock_transport->buckets_.push_back(&bucket);

  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.full_spec = {{"driver", kDriver}, {"bucket", "my-bucket"}};
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(GcsKeyValueStoreTest, InvalidSpec) {
  auto mock_transport = std::make_shared<MyMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  auto context = DefaultTestContext();

  // Test with extra key.
  EXPECT_THAT(
      kvstore::Open(
          {{"driver", kDriver}, {"bucket", "my-bucket"}, {"extra", "key"}},
          context)
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test with missing `"bucket"` key.
  EXPECT_THAT(kvstore::Open({{"driver", kDriver}}, context).result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test with invalid `"bucket"` key.
  EXPECT_THAT(
      kvstore::Open({{"driver", kDriver}, {"bucket", 5}}, context).result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(GcsKeyValueStoreTest, RequestorPays) {
  auto mock_transport = std::make_shared<MyMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  GCSMockStorageBucket bucket1("my-bucket1");
  GCSMockStorageBucket bucket2("my-bucket2", "myproject");
  mock_transport->buckets_.push_back(&bucket1);
  mock_transport->buckets_.push_back(&bucket2);

  const auto TestWrite = [&](Context context, auto bucket2_status_matcher) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store1, kvstore::Open({{"driver", kDriver},
                                    {"bucket", "my-bucket1"},
                                    {"context",
                                     {
                                         {"gcs_request_retries",
                                          {{"max_retries", 3},
                                           {"initial_delay", "1ms"},
                                           {"max_delay", "10ms"}}},
                                     }}},
                                   context)
                         .result());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store2, kvstore::Open({{"driver", kDriver},
                                    {"bucket", "my-bucket2"},
                                    {"context",
                                     {
                                         {"gcs_request_retries",
                                          {{"max_retries", 3},
                                           {"initial_delay", "1ms"},
                                           {"max_delay", "10ms"}}},
                                     }}},
                                   context)
                         .result());
    TENSORSTORE_EXPECT_OK(kvstore::Write(store1, "abc", absl::Cord("xyz")));
    EXPECT_THAT(kvstore::Write(store2, "abc", absl::Cord("xyz")).status(),
                bucket2_status_matcher);
  };

  TestWrite(Context::Default(),
            MatchesStatus(absl::StatusCode::kInvalidArgument));
  TestWrite(Context(Context::Spec::FromJson(
                        {{"gcs_user_project", {{"project_id", "badproject"}}}})
                        .value()),
            MatchesStatus(absl::StatusCode::kInvalidArgument));
  TestWrite(Context(Context::Spec::FromJson(
                        {{"gcs_user_project", {{"project_id", "myproject"}}}})
                        .value()),
            absl::OkStatus());
}

TEST(GcsKeyValueStoreTest, DeletePrefix) {
  auto mock_transport = std::make_shared<MyMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  GCSMockStorageBucket bucket("my-bucket");
  mock_transport->buckets_.push_back(&bucket);

  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", kDriver}, {"bucket", "my-bucket"}}, context)
          .result());
  tensorstore::internal::TestKeyValueStoreDeletePrefix(store);
}

TEST(GcsKeyValueStoreTest, DeleteRange) {
  auto mock_transport = std::make_shared<MyMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  GCSMockStorageBucket bucket("my-bucket");
  bucket.SetErrorRate(0.02);
  mock_transport->buckets_.push_back(&bucket);

  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", kDriver}, {"bucket", "my-bucket"}}, context)
          .result());
  tensorstore::internal::TestKeyValueStoreDeleteRange(store);
}

TEST(GcsKeyValueStoreTest, DeleteRangeToEnd) {
  auto mock_transport = std::make_shared<MyMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  GCSMockStorageBucket bucket("my-bucket");
  mock_transport->buckets_.push_back(&bucket);

  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", kDriver}, {"bucket", "my-bucket"}}, context)
          .result());
  tensorstore::internal::TestKeyValueStoreDeleteRangeToEnd(store);
}

TEST(GcsKeyValueStoreTest, DeleteRangeFromBeginning) {
  auto mock_transport = std::make_shared<MyMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  GCSMockStorageBucket bucket("my-bucket");
  mock_transport->buckets_.push_back(&bucket);

  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", kDriver}, {"bucket", "my-bucket"}}, context)
          .result());
  tensorstore::internal::TestKeyValueStoreDeleteRangeFromBeginning(store);
}

class MyDeleteRangeCancellationMockTransport : public MyMockTransport {
 public:
  void IssueRequestWithHandler(const HttpRequest& request,
                               IssueRequestOptions options,
                               HttpResponseHandler* response_handler) final {
    if (request.method == "DELETE") {
      cancellation_notification_.WaitForNotification();
      ++total_delete_requests_;
    }
    MyMockTransport::IssueRequestWithHandler(request, std::move(options),
                                             response_handler);
  }

  std::atomic<size_t> total_delete_requests_{0};
  absl::Notification cancellation_notification_;
};

TEST(GcsKeyValueStoreTest, DeleteRangeCancellation) {
  auto mock_transport =
      std::make_shared<MyDeleteRangeCancellationMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  GCSMockStorageBucket bucket("my-bucket");
  mock_transport->buckets_.push_back(&bucket);

  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open(
          {
              {"driver", kDriver},
              {"bucket", "my-bucket"},
              {"context", {{"gcs_request_concurrency", {{"limit", 1}}}}},
          },
          context)
          .result());
  for (std::string key : {"a/b", "a/c/a", "a/c/b", "a/c/d", "a/d"}) {
    TENSORSTORE_ASSERT_OK(kvstore::Write(store, key, absl::Cord()));
  }

  // Issue DeleteRange request and then immediately cancel it by dropping the
  // future.
  {
    [[maybe_unused]] auto future =
        kvstore::DeleteRange(store, tensorstore::KeyRange{"a/ba", "a/ca"});
  }
  mock_transport->cancellation_notification_.Notify();
  // FIXME(jbms): Unfortunately our cancellation mechanism does not permit
  // waiting for the cancellation to complete, so we must use a delay as a
  // hack. Note that if the delay is not long enough, the test may pass
  // spuriously, but it won't fail spuriously.
  absl::SleepFor(absl::Milliseconds(100));
  EXPECT_GE(1, mock_transport->total_delete_requests_.load());
  EXPECT_THAT(ListFuture(store).result(),
              ::testing::Optional(::testing::SizeIs(::testing::Ge(4))));
}

class MyConcurrentMockTransport : public MyMockTransport {
 public:
  size_t reset() {
    absl::MutexLock lock(&concurrent_request_mutex_);
    cur_concurrent_requests_ = 0;
    return std::exchange(max_concurrent_requests_, 0);
  }

  void IssueRequestWithHandler(const HttpRequest& request,
                               IssueRequestOptions options,
                               HttpResponseHandler* response_handler) final {
    auto parsed = tensorstore::internal::ParseGenericUri(request.url);

    // Don't do concurrency test on auth requests, as those don't happen
    // concurrently.
    if (absl::StartsWith(parsed.authority_and_path,
                         "metadata.google.internal/")) {
      MyMockTransport::IssueRequestWithHandler(request, std::move(options),
                                               response_handler);
      return;
    }

    {
      absl::MutexLock lock(&concurrent_request_mutex_);
      ++cur_concurrent_requests_;
      max_concurrent_requests_ =
          std::max(max_concurrent_requests_, cur_concurrent_requests_);
    }

    /// Schedule the completion 5ms in the future.
    auto op = tensorstore::PromiseFuturePair<HttpResponse>::Make();
    ScheduleAt(absl::Now() + absl::Milliseconds(5),
               [this, r = request, o = std::move(options), response_handler] {
                 absl::MutexLock lock(&concurrent_request_mutex_);
                 --cur_concurrent_requests_;
                 MyMockTransport::IssueRequestWithHandler(r, std::move(o),
                                                          response_handler);
               });
  }

  size_t cur_concurrent_requests_ = 0;
  size_t max_concurrent_requests_ = 0;
  absl::Mutex concurrent_request_mutex_;
};

TEST(GcsKeyValueStoreTest, Concurrency) {
  auto mock_transport = std::make_shared<MyConcurrentMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  GCSMockStorageBucket bucket("my-bucket");
  mock_transport->buckets_.push_back(&bucket);

  const auto TestConcurrency = [&](size_t limit) {
    auto context = DefaultTestContext();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        kvstore::Open(
            {
                {"driver", kDriver},
                {"bucket", "my-bucket"},
                {"context",
                 {{"gcs_request_concurrency", {{"limit", limit}}}}} /**/
            },
            context)
            .result());

    std::vector<tensorstore::Future<kvstore::ReadResult>> futures;
    for (size_t i = 0; i < 10 * limit; ++i) {
      futures.push_back(kvstore::Read(store, "abc"));
    }
    for (const auto& future : futures) {
      future.Wait();
    }
  };

  TestConcurrency(1);
  EXPECT_EQ(1, mock_transport->reset());

  TestConcurrency(2);
  EXPECT_EQ(2, mock_transport->reset());

  TestConcurrency(3);
  EXPECT_EQ(3, mock_transport->reset());
}

class MyRateLimitedMockTransport : public MyMockTransport {
 public:
  std::tuple<absl::Time, absl::Time, size_t> reset() {
    absl::MutexLock l(&request_timing_mutex_);
    return {min_time_, max_time_, std::exchange(count_, 0)};
  }

  void IssueRequestWithHandler(const HttpRequest& request,
                               IssueRequestOptions options,
                               HttpResponseHandler* response_handler) final {
    auto parsed = tensorstore::internal::ParseGenericUri(request.url);
    if (absl::StartsWith(parsed.authority_and_path,
                         "metadata.google.internal/")) {
      MyMockTransport::IssueRequestWithHandler(request, std::move(options),
                                               response_handler);
      return;
    }

    // Measure the inter-request interval on non-auth requests.
    {
      absl::MutexLock l(&request_timing_mutex_);
      max_time_ = absl::Now();
      if (count_++ == 0) {
        min_time_ = max_time_;
      }
    }
    MyMockTransport::IssueRequestWithHandler(request, std::move(options),
                                             response_handler);
  }

  absl::Time min_time_;
  absl::Time max_time_;
  size_t count_;
  absl::Mutex request_timing_mutex_;
};

TEST(GcsKeyValueStoreTest, RateLimited) {
  auto mock_transport = std::make_shared<MyRateLimitedMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  GCSMockStorageBucket bucket("my-bucket");
  mock_transport->buckets_.push_back(&bucket);

  const auto TestRateLimiting = [&](size_t limit) {
    tensorstore::Context context{
        tensorstore::Context::Spec::FromJson(
            {
                {"gcs_request_concurrency", {{"limit", 128}}}, /**/
                {"data_copy_concurrency", {{"limit", 128}}},   /**/
                {"experimental_gcs_rate_limiter",
                 {{"read_rate", limit},
                  {"write_rate", limit},
                  {"doubling_time", "20m"}}},
            })
            .value()};

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, /**/
                                    kvstore::Open(
                                        {
                                            {"driver", kDriver},
                                            {"bucket", "my-bucket"},
                                        },
                                        context)
                                        .result());

    kvstore::Read(store, "xyz").Wait();
    mock_transport->reset();
    std::vector<tensorstore::Future<kvstore::ReadResult>> futures;
    for (size_t i = 0; i < 100; ++i) {
      futures.push_back(kvstore::Read(store, "abc"));
    }
    for (const auto& future : futures) {
      future.Wait();
    }
    auto t = mock_transport->reset();
    return std::get<1>(t) - std::get<0>(t);
  };

  // Target is 100 requests per test.
  [[maybe_unused]] auto a = TestRateLimiting(10);  // ~100ms to start.
  [[maybe_unused]] auto b = TestRateLimiting(1000);

#if 0
  // Unfortunately this is flaky
  EXPECT_THAT(b, testing::Lt(a));
#endif
}

TEST(GcsKeyValueStoreTest, UrlRoundtrip) {
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", kDriver}, {"bucket", "my-bucket"}, {"path", "abc"}},
      tensorstore::StrCat(kUriScheme, "://my-bucket/abc"));
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", kDriver}, {"bucket", "my-bucket"}, {"path", "abc def"}},
      tensorstore::StrCat(kUriScheme, "://my-bucket/abc%20def"));
}

TEST(GcsKeyValueStoreTest, InvalidUri) {
  EXPECT_THAT(
      kvstore::Spec::FromUrl(tensorstore::StrCat(kUriScheme, "://bucket:xyz")),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*: Invalid GCS bucket name: \"bucket:xyz\""));
  EXPECT_THAT(kvstore::Spec::FromUrl(
                  tensorstore::StrCat(kUriScheme, "://bucket?query")),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: Query string not supported"));
  EXPECT_THAT(kvstore::Spec::FromUrl(
                  tensorstore::StrCat(kUriScheme, "://bucket#fragment")),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: Fragment identifier not supported"));
}

TEST(GcsKeyValueStoreTest, BatchRead) {
  auto mock_transport = std::make_shared<MyMockTransport>();
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  GCSMockStorageBucket bucket("my-bucket");
  mock_transport->buckets_.push_back(&bucket);

  auto context = DefaultTestContext();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", kDriver}, {"bucket", "my-bucket"}}, context)
          .result());

  tensorstore::internal::BatchReadGenericCoalescingTestOptions options;
  options.coalescing_options = tensorstore::internal_kvstore_batch::
      kDefaultRemoteStorageCoalescingOptions;
  options.metric_prefix = "/tensorstore/kvstore/gcs/";
  tensorstore::internal::TestBatchReadGenericCoalescing(store, options);
}

}  // namespace
