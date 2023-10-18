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

#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/transport_test_utils.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/subprocess.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/s3/credentials/aws_credential_provider.h"
#include "tensorstore/kvstore/s3/s3_request_builder.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

// When provided with --localstack_binary, localstack_test will start
// localstack in host mode (via package localstack[runtime]).
//
// When provided with --localstack_endpoint, localstack_test will connect
// to a running localstack instance.
ABSL_FLAG(std::string, localstack_endpoint, "", "Localstack endpoint");
ABSL_FLAG(std::string, localstack_binary, "", "Path to the localstack");

namespace kvstore = ::tensorstore::kvstore;

using ::tensorstore::Context;
using ::tensorstore::MatchesJson;
using ::tensorstore::internal::GetEnvironmentMap;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::SpawnSubprocess;
using ::tensorstore::internal::Subprocess;
using ::tensorstore::internal::SubprocessOptions;
using ::tensorstore::internal_http::GetDefaultHttpTransport;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_kvstore_s3::AwsCredentials;
using ::tensorstore::internal_kvstore_s3::S3RequestBuilder;
using ::tensorstore::transport_test_utils::TryPickUnusedPort;

namespace {

static constexpr char kAwsAccessKeyId[] = "LSIAQAAAAAAVNCBMPNSG";
static constexpr char kAwsSecretKeyId[] = "localstackdontcare";
static constexpr char kBucket[] = "testbucket";
static constexpr char kAwsRegion[] = "af-south-1";
/// sha256 hash of an empty string
static constexpr char kEmptySha256[] =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

class LocalStackProcess {
 public:
  LocalStackProcess() = default;
  ~LocalStackProcess() { StopProcess(); }

  void SpawnProcess() {
    if (child_) return;

    // NOTE: We may need to add in a retry loop for port selection to avoid
    // flaky tests.
    http_port = TryPickUnusedPort().value_or(4566);

    ABSL_LOG(INFO) << "Spawning localstack: " << endpoint_url();
    SubprocessOptions options{absl::GetFlag(FLAGS_localstack_binary),
                              {"start", "--host"}};

    // See https://docs.localstack.cloud/references/configuration/
    // for the allowed environment variables for localstack.
    options.env.emplace(GetEnvironmentMap());
    auto &env = *options.env;
    env["GATEWAY_LISTEN"] = absl::StrFormat("localhost:%d", http_port);
    env["LOCALSTACK_HOST"] =
        absl::StrFormat("localhost.localstack.cloud:%d", http_port);
    env["SERVICES"] = "s3";

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto spawn_proc, SpawnSubprocess(options));

    // Once the process is running, start a gRPC server on the provided port.
    absl::SleepFor(absl::Milliseconds(300));
    auto status = spawn_proc.Join(/*block=*/false).status();

    // The process may fail due to an in-use port, or something else.
    ABSL_CHECK(absl::IsUnavailable(status))
        << "Failed to spawn localstack: " << status;

    child_.emplace(std::move(spawn_proc));
  }

  void StopProcess() {
    if (child_) {
      child_->Kill().IgnoreError();
      auto join_result = child_->Join();
      if (!join_result.ok()) {
        ABSL_LOG(ERROR) << "Joining storage_testbench subprocess failed: "
                        << join_result.status();
      }
    }
  }

  std::string endpoint_url() {
    return absl::StrFormat("http://localhost:%d", http_port);
  }

  int http_port = 0;
  std::optional<Subprocess> child_;
};

class LocalStackFixture : public ::testing::Test {
 protected:
  static LocalStackProcess process;

  static void SetUpTestSuite() {
    SetEnv("AWS_ACCESS_KEY_ID", kAwsAccessKeyId);
    SetEnv("AWS_SECRET_KEY_ID", kAwsSecretKeyId);

    if (absl::GetFlag(FLAGS_localstack_endpoint).empty()) {
      ABSL_CHECK(!absl::GetFlag(FLAGS_localstack_binary).empty());

      process.SpawnProcess();
    }

    MaybeCreateBucket();
  }

  static void TearDownTestSuite() { process.StopProcess(); }

  // Attempts to create the kBucket bucket on the localstack host.
  static void MaybeCreateBucket() {
    auto value = absl::Cord{absl::StrFormat(
        R"(<?xml version="1.0" encoding="UTF-8"?>)"
        R"(<CreateBucketConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">)"
        R"(<LocationConstraint>%s</LocationConstraint>)"
        R"(</CreateBucketConfiguration>)",
        kAwsRegion)};

    auto request =
        S3RequestBuilder("PUT", endpoint_url())
            .BuildRequest(absl::StrFormat("%s.s3.amazonaws.com", kBucket),
                          AwsCredentials{}, kAwsRegion, kEmptySha256,
                          absl::Now());

    ::tensorstore::Future<HttpResponse> response;
    for (auto deadline = absl::Now() + absl::Seconds(5);;) {
      absl::SleepFor(absl::Milliseconds(100));
      response = GetDefaultHttpTransport()->IssueRequest(
          request, value, absl::Seconds(15), absl::Seconds(15));

      // Failed to make the request; retry.
      if (absl::Now() < deadline && absl::IsUnavailable(response.status())) {
        continue;
      }
      break;
    }

    // Log the response, but don't fail the process on error.
    if (!response.status().ok()) {
      ABSL_LOG(INFO) << "Create bucket error: " << response.status();
    } else {
      ABSL_LOG(INFO) << "Create bucket response: " << kBucket << "  "
                     << response.value();
    }
  }

  static std::string endpoint_url() {
    if (absl::GetFlag(FLAGS_localstack_endpoint).empty()) {
      return process.endpoint_url();
    }
    return absl::GetFlag(FLAGS_localstack_endpoint);
  }

  static std::string host() {
    return absl::StrFormat("%s.s3.%s.localstack.localhost.com", kBucket,
                           kAwsRegion);
  }
};

LocalStackProcess LocalStackFixture::process;

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
      auto store, kvstore::Open({{"aws_region", kAwsRegion},
                                 {"driver", "s3"},
                                 {"bucket", kBucket},
                                 {"endpoint", endpoint_url()},
                                 {"host", host()},
                                 {"path", "tensorstore/test/"}},
                                context)
                      .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, store.spec());
  EXPECT_THAT(
      spec.ToJson(tensorstore::IncludeDefaults{false}),
      ::testing::Optional(MatchesJson({{"aws_region", kAwsRegion},
                                       {"driver", "s3"},
                                       {"bucket", kBucket},
                                       {"endpoint", endpoint_url()},
                                       {"host", host()},
                                       {"path", "tensorstore/test/"}})));

  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

}  // namespace
