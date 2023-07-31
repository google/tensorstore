// Copyright 2023 The TensorStore Authors
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
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/schedule_at.h"
#include "tensorstore/kvstore/s3/s3_request_builder.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/internal/subprocess.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace kvstore = ::tensorstore::kvstore;
using ::tensorstore::CompletionNotifyingReceiver;
using ::tensorstore::Context;
using ::tensorstore::Future;
using ::tensorstore::KeyRange;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::StorageGeneration;
using ::tensorstore::internal::SpawnSubprocess;
using ::tensorstore::internal::Subprocess;
using ::tensorstore::internal::SubprocessOptions;
using ::tensorstore::TimestampedStorageGeneration;
using ::tensorstore::StrCat;
using ::tensorstore::internal::ScheduleAt;
using ::tensorstore::internal_http::GetDefaultHttpTransport;
using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http::SetDefaultHttpTransport;
using ::tensorstore::internal_kvstore_s3::S3Credentials;
using ::tensorstore::internal_kvstore_s3::S3RequestBuilder;
using ::tensorstore::internal::GetEnv;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::UnsetEnv;


ABSL_FLAG(std::string, localstack_binary, "",
          "Path to the localstack");

namespace {

static constexpr char kAwsAccessKeyId[] = "LSIAQAAAAAAVNCBMPNSG";
static constexpr char kAwsSecretKeyId[] = "localstackdontcare";
static constexpr char kBucket[] = "testbucket";
static constexpr char kAwsRegion[] = "af-south-1";
static constexpr char kUriScheme[] = "s3";
static constexpr char kDriver[] = "s3";
static constexpr char kLocalStackEndpoint[] = "http://localhost:4566";
/// sha256 hash of an empty string
static constexpr char kEmptySha256[] =
  "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";


class LocalStackProcess {
public:
  LocalStackProcess() = default;

  void SpawnProcess() {
    if (running) return;
    ABSL_LOG(INFO) << "Spawning localstack: " << endpoint_url();
    {
      SubprocessOptions options{absl::GetFlag(FLAGS_localstack_binary), {"start", "-d"}};
      TENSORSTORE_CHECK_OK_AND_ASSIGN(auto spawn_proc, SpawnSubprocess(options));
      TENSORSTORE_CHECK_OK_AND_ASSIGN(auto join_value, spawn_proc.Join());
      assert(join_value == 0);
    }

    running = true;
  }

  void StopProcess() {
    if (!running) return;
    ABSL_LOG(INFO) << "Shutting localstack down: " << endpoint_url();
    {
      SubprocessOptions options{absl::GetFlag(FLAGS_localstack_binary), {"stop"}};
      TENSORSTORE_CHECK_OK_AND_ASSIGN(auto stop_proc, SpawnSubprocess(options));
      TENSORSTORE_CHECK_OK_AND_ASSIGN(auto join_value, stop_proc.Join());
      assert(join_value == 0);
    }
    running = false;
  }

  void CreateBucket() {
    auto value = absl::Cord{
      absl::StrFormat(R"(<?xml version="1.0" encoding="UTF-8"?>
                       <CreateBucketConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
                       <LocationConstraint>%s</LocationConstraint>
                       </CreateBucketConfiguration>)",
                       kAwsRegion)};

    auto request = S3RequestBuilder("PUT", endpoint_url()).BuildRequest(
      absl::StrFormat("%s.s3.amazonaws.com", kBucket),
      S3Credentials{}, kAwsRegion, kEmptySha256, absl::Now()
    );

    for (auto deadline = absl::Now() + absl::Seconds(10);;) {
      absl::SleepFor(absl::Milliseconds(200));
      auto response = GetDefaultHttpTransport()->IssueRequest(
                        request, value, absl::Seconds(15), absl::Seconds(15));

      if (response.status().ok() && response.value().status_code == 200) {
        ABSL_LOG(INFO) << "Created bucket " << kBucket << "  " << response.value();
        running = true;
        break;
      }
      if(absl::Now() < deadline && absl::IsUnavailable(response.status())) {
          continue;
      }
    }
  }

  std::string endpoint_url() { return kLocalStackEndpoint; }

  bool running = false;
  std::optional<Subprocess> child;
};

class LocalStackFixture : public ::testing::Test {
protected:
  static LocalStackProcess process;
  // Environment variables to save and restore during setup and teardown
  static std::map<std::string, std::optional<std::string>> saved_vars;

  static void SetUpTestSuite() {
    for(auto &pair: saved_vars) {
        pair.second = GetEnv(pair.first.c_str());
        UnsetEnv(pair.first.c_str());
    }

    SetEnv("AWS_ACCESS_KEY_ID", kAwsAccessKeyId);
    SetEnv("AWS_SECRET_KEY_ID", kAwsSecretKeyId);

    process.SpawnProcess();
    process.CreateBucket();
  }

  static void TearDownTestSuite() {
    process.StopProcess();

    for(auto &pair: saved_vars) {
        if(pair.second) {
            SetEnv(pair.first.c_str(), pair.second.value().c_str());
        }
    }

  }
};

LocalStackProcess LocalStackFixture::process;

std::map<std::string, std::optional<std::string>> LocalStackFixture::saved_vars{
    {"AWS_ACCESS_KEY_ID", std::nullopt},
    {"AWS_SECRET_ACCESS_KEY", std::nullopt},
 };


Context DefaultTestContext() {
  // Opens the s3 driver with small exponential backoff values.
  return Context{Context::Spec::FromJson({{"s3_request_retries",
                                           {{"max_retries", 3},
                                            {"initial_delay", "1ms"},
                                            {"max_delay", "10ms"}}}})
                     .value()};
}

TEST_F(LocalStackFixture, Basic) {
  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"aws_region", kAwsRegion},
                     {"driver", kDriver},
                     {"bucket", kBucket},
                     {"endpoint", process.endpoint_url()},
                     {"host", absl::StrFormat("%s.s3.%s.localstack.localhost.com", kBucket, kAwsRegion)},
                     {"path", "tensorstore/test/"}}, context)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, store.spec());
  EXPECT_THAT(spec.ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(
                  MatchesJson({{"aws_region", kAwsRegion},
                               {"driver", kDriver},
                               {"bucket", kBucket},
                               {"endpoint", process.endpoint_url()},
                               {"host", absl::StrFormat("%s.s3.%s.localstack.localhost.com", kBucket, kAwsRegion)},
                               {"path", "tensorstore/test/"},
                               {"profile", "default"},
                               {"requester_pays", false}})));

  tensorstore::internal::TestKeyValueStoreBasicFunctionality(store);
}


TEST(S3KeyValueStoreTest, BadBucketNames) {
  auto context = DefaultTestContext();
  for (auto bucket :
       {"a", "_abc", "abc_", "ABC", "a..b", "a.-.b"}) {
    EXPECT_FALSE(
        kvstore::Open({{"driver", kDriver}, {"bucket", bucket}, {"endpoint", "https://i.dont.exist"}}, context)
            .result())
        << "bucket: " << bucket;
  }
  for (auto bucket : {"abc", "abc.1-2-3.abc",
        "a."
        "0123456789123456789012345678912345678901234567891234567890"
        "1234567891234567890123456789123456789012345678912345678901"
        "23456789123456789.B"}) {
    EXPECT_TRUE(
        kvstore::Open({{"driver", kDriver}, {"bucket", bucket}, {"endpoint", "https://i.dont.exist"}}, context)
            .result())
        << "bucket: " << bucket;
  }
}


} // namespace
