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

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/aws/aws_credentials.h"
#include "tensorstore/internal/aws/credentials/common.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/http/default_transport.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/http/transport_test_utils.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/os/subprocess.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/s3/s3_endpoint.h"
#include "tensorstore/kvstore/s3/s3_request_builder.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

// When provided with --localstack_binary, localstack_test will start
// localstack in host mode (via package localstack[runtime]).
//
// When provided with --localstack_endpoint, localstack_test will connect
// to a running localstack instance.
//
// To run directly against an Amazon S3 endpoint, run with
//   --aws_region=us-west-2
//   --aws_bucket=...
//   --localstack_endpoint=https://s3.us-west-2.amazonaws.com

ABSL_FLAG(std::string, localstack_endpoint, "", "Localstack endpoint");
ABSL_FLAG(std::string, localstack_binary, "", "Path to the localstack");

// --localstack_timeout is the time the process will wait for localstack.
ABSL_FLAG(absl::Duration, localstack_timeout, absl::Seconds(15),
          "Time to wait for localstack process to start serving requests");

// --host_header can override the host: header used for signing.
// It can be, for example, s3.us-east-1.localstack.localhost.com
ABSL_FLAG(std::string, host_header, "", "Host header to use for signing");

// --binary_mode selects whether the `--localstack_binary` is localstack
// binary or whether it is a moto binary.
ABSL_FLAG(std::string, binary_mode, "",
          "Selects options for starting --localstack_binary. Valid values are "
          "[moto]. Assumes localstack otherwise.");

// AWS bucket, region, and path.
ABSL_FLAG(std::string, aws_bucket, "testbucket",
          "The S3 bucket used for the test.");

ABSL_FLAG(std::string, aws_region, "us-east-1",
          "The S3 region used for the test.");

ABSL_FLAG(std::string, aws_path, "tensorstore/test/",
          "The S3 path used for the test.");

namespace kvstore = ::tensorstore::kvstore;

using ::tensorstore::Context;
using ::tensorstore::MatchesJson;
using ::tensorstore::internal::GetEnv;
using ::tensorstore::internal::GetEnvironmentMap;
using ::tensorstore::internal::KeyValueStoreOpsTestParameters;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::SpawnSubprocess;
using ::tensorstore::internal::Subprocess;
using ::tensorstore::internal::SubprocessOptions;
using ::tensorstore::internal_aws::AwsCredentials;
using ::tensorstore::internal_aws::GetAwsCredentials;
using ::tensorstore::internal_http::GetDefaultHttpTransport;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::IssueRequestOptions;
using ::tensorstore::internal_kvstore_s3::IsAwsS3Endpoint;
using ::tensorstore::internal_kvstore_s3::S3RequestBuilder;
using ::tensorstore::transport_test_utils::TryPickUnusedPort;

namespace {

// localstack account id 42
static constexpr char kAwsAccessKeyId[] = "LSIAQAAAAAAVNCBMPNSG";
static constexpr char kAwsSecretKeyId[] = "localstackdontcare";

// sha256 hash of an empty string
static constexpr char kEmptySha256[] =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

std::string Bucket() { return absl::GetFlag(FLAGS_aws_bucket); }
std::string Region() { return absl::GetFlag(FLAGS_aws_region); }
std::string Path() { return absl::GetFlag(FLAGS_aws_path); }

SubprocessOptions SetupLocalstackOptions(int http_port) {
  // See https://docs.localstack.cloud/references/configuration/
  // for the allowed environment variables for localstack.
  SubprocessOptions options{absl::GetFlag(FLAGS_localstack_binary),
                            {"start", "--host"}};
  options.env.emplace(GetEnvironmentMap());
  auto& env = *options.env;
  env["GATEWAY_LISTEN"] = absl::StrFormat("localhost:%d", http_port);
  env["LOCALSTACK_HOST"] =
      absl::StrFormat("localhost.localstack.cloud:%d", http_port);
  env["SERVICES"] = "s3";
  env["AWS_DEFAULT_REGION"] = Region();
  env["PYTHONUNBUFFERED"] = "1";
  return options;
}

SubprocessOptions SetupMotoOptions(int http_port) {
  // See https://docs.getmoto.org/en/latest/docs/getting_started.html
  // and https://docs.getmoto.org/en/latest/docs/server_mode.html
  SubprocessOptions options{absl::GetFlag(FLAGS_localstack_binary),
                            {absl::StrFormat("-p%d", http_port)}};
  options.env.emplace(GetEnvironmentMap());
  auto& env = *options.env;
  ABSL_CHECK(!Region().empty());
  env["AWS_DEFAULT_REGION"] = Region();
  env["PYTHONUNBUFFERED"] = "1";
  return options;
}

// NOTE: Support minio as well, which needs temporary directories.
// https://min.io/docs/minio/linux/reference/minio-server/minio-server.html
// minio server --address :12123  /tmp/minio

class LocalStackProcess {
 public:
  LocalStackProcess() = default;
  ~LocalStackProcess() { StopProcess(); }

  void SpawnProcess() {
    if (child_) return;

    const auto start_child = [this] {
      http_port = TryPickUnusedPort().value_or(0);
      ABSL_CHECK(http_port > 0);

      SubprocessOptions options =  //
          (absl::GetFlag(FLAGS_binary_mode) == "moto")
              ? SetupMotoOptions(http_port)
              : SetupLocalstackOptions(http_port);

      ABSL_LOG(INFO) << "Spawning: " << endpoint_url();

      absl::SleepFor(absl::Milliseconds(10));
      TENSORSTORE_CHECK_OK_AND_ASSIGN(auto spawn_proc,
                                      SpawnSubprocess(options));
      return spawn_proc;
    };

    Subprocess spawn_proc = start_child();

    // Give the child process several seconds to start.
    auto deadline = absl::Now() + absl::Seconds(10);
    while (absl::Now() < deadline) {
      absl::SleepFor(absl::Milliseconds(250));
      auto join_result = spawn_proc.Join(/*block=*/false);

      if (join_result.ok()) {
        // Process has terminated. Restart.
        spawn_proc = start_child();
        continue;
      } else if (absl::IsUnavailable(join_result.status())) {
        // Child is running.
        child_.emplace(std::move(spawn_proc));
        return;
      }
      // TODO: Also check the http port?
      //  * Running on http://127.0.0.1:46471
    }

    // Deadline has expired & there's nothing to show for it.
    ABSL_LOG(FATAL) << "Failed to start process";
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

Context DefaultTestContext() {
  // Opens the s3 driver with small exponential backoff values.
  ::nlohmann::json json_spec{
      {"s3_request_retries",
       {{"max_retries", 3}, {"initial_delay", "1ms"}, {"max_delay", "10ms"}}}};

  if (!IsAwsS3Endpoint(absl::GetFlag(FLAGS_localstack_endpoint))) {
    ::nlohmann::json environment_spec{
        {"type", "environment"},  //
    };
    json_spec["aws_credentials"] = environment_spec;
  }
  return Context{Context::Spec::FromJson(json_spec).value()};
}

class LocalStackFixture : public ::testing::Test {
 public:
  inline static LocalStackProcess process;
  inline static bool is_set_up = false;

  static void SetUpTestSuite() {
    if (is_set_up) return;
    is_set_up = true;
    bool is_aws_endpoint =
        IsAwsS3Endpoint(absl::GetFlag(FLAGS_localstack_endpoint));
    ABSL_LOG_IF(INFO, is_aws_endpoint)
        << "localstack_test connecting to Amazon using bucket: " << Bucket();

    // Ensure that environment credentials are installed except when
    // connecting to an aws endpoint.
    if (!is_aws_endpoint && (!GetEnv("AWS_ACCESS_KEY_ID").has_value() ||
                             !GetEnv("AWS_SECRET_ACCESS_KEY").has_value())) {
      ABSL_LOG(INFO) << "Installing environment credentials AWS_ACCESS_KEY_ID="
                     << kAwsAccessKeyId
                     << " AWS_SECRET_ACCESS_KEY=" << kAwsSecretKeyId;
      SetEnv("AWS_ACCESS_KEY_ID", kAwsAccessKeyId);
      SetEnv("AWS_SECRET_ACCESS_KEY", kAwsSecretKeyId);
    }

    ABSL_CHECK(!Bucket().empty());

    if (absl::GetFlag(FLAGS_localstack_endpoint).empty()) {
      ABSL_CHECK(!absl::GetFlag(FLAGS_localstack_binary).empty());
      process.SpawnProcess();
    }

    if (!is_aws_endpoint) {
      // Avoid creating the bucket when connecting directly to aws.
      ABSL_CHECK(!Region().empty());
      MaybeCreateBucket();
    }
  }

  static AwsCredentials GetEnvironmentCredentials() {
    auto provider = tensorstore::internal_aws::MakeEnvironment();
    auto credentials_future = GetAwsCredentials(provider.get());
    ABSL_CHECK(credentials_future.status().ok());

    return std::move(credentials_future.result()).value();
  }

  static std::string endpoint_url() {
    if (absl::GetFlag(FLAGS_localstack_endpoint).empty()) {
      return process.endpoint_url();
    }
    return absl::GetFlag(FLAGS_localstack_endpoint);
  }

  // Attempts to create the kBucket bucket on the localstack host.
  static void MaybeCreateBucket() {
    // Location constraints must not be provided for us-east-1
    // https://docs.aws.amazon.com/AmazonS3/latest/API/API_CreateBucket.html
    absl::Cord value;
    if (Region() != "us-east-1") {
      value = absl::Cord{absl::StrFormat(
          R"(<?xml version="1.0" encoding="UTF-8"?>)"
          R"(<CreateBucketConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">)"
          R"(<LocationConstraint>%s</LocationConstraint>)"
          R"(</CreateBucketConfiguration>)",
          Region())};
    }

    auto request =
        S3RequestBuilder("PUT",
                         absl::StrFormat("%s/%s", endpoint_url(), Bucket()))
            .BuildRequest(absl::GetFlag(FLAGS_host_header),
                          GetEnvironmentCredentials(), Region(), kEmptySha256,
                          absl::Now());

    ABSL_LOG(INFO) << "Create bucket request: " << request;

    ::tensorstore::Future<HttpResponse> response;
    // Repeat  until available, up to `--localstack_timeout` seconds (about 15).
    for (auto deadline = absl::Now() + absl::GetFlag(FLAGS_localstack_timeout);
         absl::Now() < deadline;) {
      absl::SleepFor(absl::Milliseconds(250));
      response = GetDefaultHttpTransport()->IssueRequest(
          request, IssueRequestOptions(value)
                       .SetRequestTimeout(absl::Seconds(15))
                       .SetConnectTimeout(absl::Seconds(15)));

      // Failed to make the request; retry.
      if (response.status().ok() || !absl::IsUnavailable(response.status())) {
        break;
      }
    }

    // Log the response, but don't fail the process on error.
    if (!response.status().ok()) {
      ABSL_LOG(INFO) << "Create bucket error: " << response.status();
    } else {
      ABSL_LOG(INFO) << "Create bucket response: " << Bucket() << "  "
                     << response.value() << "\n"
                     << response.value().payload;
    }
  }

  static ::nlohmann::json GetJsonSpec(std::string path = "") {
    ::nlohmann::json json_spec{
        {"driver", "s3"},                      //
        {"bucket", Bucket()},                  //
        {"endpoint", endpoint_url()},          //
        {"path", absl::StrCat(Path(), path)},  //
    };
    if (!Region().empty()) {
      json_spec["aws_region"] = Region();
    }
    if (!absl::GetFlag(FLAGS_host_header).empty()) {
      json_spec["host_header"] = absl::GetFlag(FLAGS_host_header);
    }
    return json_spec;
  }

  static tensorstore::Result<tensorstore::KvStore> OpenStore(
      tensorstore::Context context, std::string path = "") {
    return kvstore::Open(GetJsonSpec(path), context).result();
  }
};

TENSORSTORE_GLOBAL_INITIALIZER {
  KeyValueStoreOpsTestParameters params;
  params.test_name = "Basic";
  params.get_store = [](auto callback) {
    LocalStackFixture::SetUpTestSuite();
    auto context = DefaultTestContext();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                     LocalStackFixture::OpenStore(context));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, store.spec());
    EXPECT_THAT(
        spec.ToJson(tensorstore::IncludeDefaults{false}),
        ::testing::Optional(MatchesJson(LocalStackFixture::GetJsonSpec())));
    callback(store);
  };
  params.test_list_without_prefix = false;
  RegisterKeyValueStoreOpsTests(params);
}

TEST_F(LocalStackFixture, BatchRead) {
  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   OpenStore(context, "batch_read/"));

  tensorstore::internal::BatchReadGenericCoalescingTestOptions options;
  options.coalescing_options = tensorstore::internal_kvstore_batch::
      kDefaultRemoteStorageCoalescingOptions;
  options.metric_prefix = "/tensorstore/kvstore/s3/";
  tensorstore::internal::TestBatchReadGenericCoalescing(store, options);
}

TEST_F(LocalStackFixture, ConcurrentWrites) {
  // NOTE: Some S3-compatible object stores don't support if-match,
  // so only enable the concurrent test when the UseConditionalWrite heuristic
  // is true.
  if (!IsAwsS3Endpoint(endpoint_url())) {
    GTEST_SKIP() << "Concurrent writes test skipped for " << endpoint_url();
    return;
  }

  tensorstore::internal::TestConcurrentWritesOptions options;
  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   OpenStore(context, "concurrent_writes/"));
  options.get_store = [&] { return store; };
  options.num_iterations = 0x7f;
  tensorstore::internal::TestConcurrentWrites(options);
}

}  // namespace
