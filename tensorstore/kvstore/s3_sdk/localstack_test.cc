// Copyright 2024 The TensorStore Authors
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
#include "absl/synchronization/notification.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include <aws/core/Aws.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/utils/threading/Executor.h>
#include <aws/core/utils/stream/ResponseStream.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/CreateBucketConfiguration.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/CreateBucketResult.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>

#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/http/transport_test_utils.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/os/subprocess.h"
#include "tensorstore/internal/thread/thread_pool.h"
#include "tensorstore/kvstore/s3_sdk/cord_streambuf.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/result.h"

#include "tensorstore/kvstore/s3_sdk/s3_context.h"

// When provided with --localstack_binary, localstack_test will start
// localstack in host mode (via package localstack[runtime]).
//
// When provided with --localstack_endpoint, localstack_test will connect
// to a running localstack instance.
ABSL_FLAG(std::string, localstack_endpoint, "", "Localstack endpoint");
ABSL_FLAG(std::string, localstack_binary, "", "Path to the localstack");

// --localstack_timeout is the time the process will wait for localstack.
ABSL_FLAG(absl::Duration, localstack_timeout, absl::Seconds(15),
          "Time to wait for localstack process to start serving requests");

// --host_header can override the host: header used for signing.
// It can be, for example, s3.af-south-1.localstack.localhost.com
ABSL_FLAG(std::string, host_header, "", "Host header to use for signing");

// --binary_mode selects whether the `--localstack_binary` is localstack
// binary or whether it is a moto binary.
ABSL_FLAG(std::string, binary_mode, "",
          "Selects options for starting --localstack_binary. Valid values are "
          "[moto]. Assumes localstack otherwise.");

// AWS bucket, region, and path.
ABSL_FLAG(std::string, aws_bucket, "testbucket",
          "The S3 bucket used for the test.");

ABSL_FLAG(std::string, aws_region, "af-south-1",
          "The S3 region used for the test.");

ABSL_FLAG(std::string, aws_path, "tensorstore/test/",
          "The S3 path used for the test.");

using ::Aws::MakeUnique;
using ::Aws::MakeShared;
using ::Aws::Utils::Stream::DefaultUnderlyingStream;

using ::tensorstore::Context;
using ::tensorstore::MatchesJson;
using ::tensorstore::internal::GetEnv;
using ::tensorstore::internal::GetEnvironmentMap;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::SpawnSubprocess;
using ::tensorstore::internal::Subprocess;
using ::tensorstore::internal::SubprocessOptions;
using ::tensorstore::internal_http::GetDefaultHttpTransport;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::IssueRequestOptions;
using ::tensorstore::transport_test_utils::TryPickUnusedPort;

using ::tensorstore::internal_kvstore_s3::AwsContext;
using ::tensorstore::internal_kvstore_s3::CordStreamBuf;
using ::tensorstore::internal_kvstore_s3::CordBackedResponseStreamFactory;

namespace {

static constexpr char kAwsTag[] = "AWS";
static constexpr char kAwsAccessKeyId[] = "LSIAQAAAAAAVNCBMPNSG";
static constexpr char kAwsSecretKeyId[] = "localstackdontcare";

/// sha256 hash of an empty string
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
      absl::SleepFor(absl::Milliseconds(500));
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


class LocalStackFixture : public ::testing::Test {
 protected:
  static std::shared_ptr<AwsContext> context;
  static LocalStackProcess process;
  static std::shared_ptr<Aws::S3::S3Client> client;

  static void SetUpTestSuite() {
    if (!GetEnv("AWS_ACCESS_KEY_ID") || !GetEnv("AWS_SECRET_KEY_ID")) {
      SetEnv("AWS_ACCESS_KEY_ID", kAwsAccessKeyId);
      SetEnv("AWS_SECRET_KEY_ID", kAwsSecretKeyId);
    }

    ABSL_CHECK(!Bucket().empty());

    if (absl::GetFlag(FLAGS_localstack_endpoint).empty()) {
      ABSL_CHECK(!absl::GetFlag(FLAGS_localstack_binary).empty());
      process.SpawnProcess();
    }

    if (!absl::StrContains(absl::GetFlag(FLAGS_localstack_endpoint),
                           "amazonaws.com")) {
      // Only try to create the bucket when not connecting to aws.
      ABSL_CHECK(!Region().empty());
      MaybeCreateBucket();
    } else {
      ABSL_LOG(INFO) << "localstack_test connecting to Amazon using bucket:"
                     << Bucket();
    }

    CreateClient();
  }

  // Create client for use by test cases
  static void CreateClient() {
      // Offload AWS Client tasks onto a Tensorstore executor
    class TensorStoreExecutor : public Aws::Utils::Threading::Executor {
      public:
        TensorStoreExecutor(): executor_(::tensorstore::internal::DetachedThreadPool(4)) {}
      protected:
        bool SubmitToThread(std::function<void()> && fn) override {
          ::tensorstore::WithExecutor(executor_, std::move(fn))();
          return true;
        }
      private:
        ::tensorstore::Executor executor_;
    };

    auto config = Aws::Client::ClientConfiguration{};
    config.endpointOverride = endpoint_url();
    config.region = Region();
    config.executor = Aws::MakeShared<TensorStoreExecutor>(kAwsTag);
    client = std::make_shared<Aws::S3::S3Client>(
      config,
      Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Always,
      false);
  }

  static void TearDownTestSuite() {
    client.reset();
    context.reset();
    process.StopProcess();
  }

  static std::string endpoint_url() {
    if (absl::GetFlag(FLAGS_localstack_endpoint).empty()) {
      return process.endpoint_url();
    }
    return absl::GetFlag(FLAGS_localstack_endpoint);
  }

  // Attempts to create the kBucket bucket on the localstack host.
  static void MaybeCreateBucket() {
    // Create a separate client for creating the bucket
    // Without anonymous credentials bucket creation fails with 400 IllegalRegionConstraint
    auto cfg = Aws::Client::ClientConfiguration{};
    cfg.endpointOverride = endpoint_url();
    cfg.region = Region();
    auto create_client = std::make_shared<Aws::S3::S3Client>(Aws::Auth::AWSCredentials(), cfg);

    auto create_request = Aws::S3::Model::CreateBucketRequest{};
    create_request.SetBucket(Bucket());

    if (cfg.region != "us-east-1") {
        auto bucket_cfg = Aws::S3::Model::CreateBucketConfiguration{};
        bucket_cfg.SetLocationConstraint(
                Aws::S3::Model::BucketLocationConstraintMapper::GetBucketLocationConstraintForName(
                        cfg.region));
        create_request.SetCreateBucketConfiguration(bucket_cfg);
    }

    auto outcome = create_client->CreateBucket(create_request);
    if (!outcome.IsSuccess()) {
        auto err = outcome.GetError();
        ABSL_LOG(INFO) << "Error: CreateBucket: " <<
                          err.GetExceptionName() << ": " << err.GetMessage();
    }
    else {
        ABSL_LOG(INFO) << "Created bucket " << Bucket() <<
                          " in AWS Region " << Region();
    }
  }
};

LocalStackProcess LocalStackFixture::process;
std::shared_ptr<Aws::S3::S3Client> LocalStackFixture::client = nullptr;
std::shared_ptr<AwsContext> LocalStackFixture::context = tensorstore::internal_kvstore_s3::GetAwsContext();

TEST_F(LocalStackFixture, BasicSync) {
  std::string payload = "this is a test";

  // Put an object
  auto put_request = Aws::S3::Model::PutObjectRequest{};
  put_request.SetResponseStreamFactory(CordBackedResponseStreamFactory);
  put_request.SetBucket(Bucket());
  put_request.SetKey("portunus");
  put_request.SetBody(MakeShared<DefaultUnderlyingStream>(kAwsTag, MakeUnique<CordStreamBuf>(kAwsTag, absl::Cord{payload})));
  auto put_outcome = client->PutObject(put_request);
  EXPECT_TRUE(put_outcome.IsSuccess());

  // Put the same object with a different key
  put_request = Aws::S3::Model::PutObjectRequest{};
  put_request.SetResponseStreamFactory(CordBackedResponseStreamFactory);
  put_request.SetBucket(Bucket());
  put_request.SetKey("portunus0");
  put_request.SetBody(MakeShared<DefaultUnderlyingStream>(kAwsTag, MakeUnique<CordStreamBuf>(kAwsTag, absl::Cord{payload})));
  put_outcome = client->PutObject(put_request);
  EXPECT_TRUE(put_outcome.IsSuccess());

  // List the objects
  auto list_request = Aws::S3::Model::ListObjectsV2Request{};
  list_request.SetResponseStreamFactory(CordBackedResponseStreamFactory);
  list_request.SetBucket(Bucket());
  list_request.SetMaxKeys(1);
  auto continuation_token = Aws::String{};
  Aws::Vector<Aws::S3::Model::Object> objects;

  do {
    if (!continuation_token.empty()) {
      list_request.SetContinuationToken(continuation_token);
    }

    auto outcome = client->ListObjectsV2(list_request);
    EXPECT_TRUE(outcome.IsSuccess());

    auto page_objects = outcome.GetResult().GetContents();
    objects.insert(objects.end(), page_objects.begin(), page_objects.end());
    continuation_token = outcome.GetResult().GetNextContinuationToken();
  } while (!continuation_token.empty());


  EXPECT_EQ(objects.size(), 2);

  for (const auto &object: objects) {
    EXPECT_EQ(object.GetSize(), payload.size());
  }

  // Get the contents of the key
  auto get_request = Aws::S3::Model::GetObjectRequest{};
  get_request.SetResponseStreamFactory(CordBackedResponseStreamFactory);
  get_request.SetBucket(Bucket());
  get_request.SetKey("portunus");
  auto get_outcome = client->GetObject(get_request);
  EXPECT_TRUE(get_outcome.IsSuccess());
  std::string result;
  std::getline(get_outcome.GetResult().GetBody(), result);
  EXPECT_EQ(result, payload);
}

TEST_F(LocalStackFixture, BasicAsync) {
  struct TestCallbacks {
    // Data relevant to GET and PUT
    std::string key;
    std::string payload;

    // Results and notifications
    bool put_succeeded = false;
    std::optional<std::string> get_result;
    absl::Notification done;

    void do_put() {
      auto put_request = Aws::S3::Model::PutObjectRequest{};
      put_request.SetBucket(Bucket());
      put_request.SetKey(key);
      put_request.SetResponseStreamFactory(CordBackedResponseStreamFactory);
      put_request.SetBody(MakeShared<DefaultUnderlyingStream>(kAwsTag, MakeUnique<CordStreamBuf>(kAwsTag, absl::Cord{payload})));
      client->PutObjectAsync(put_request, [this](
        const auto *, const auto &, const auto & outcome, const auto &) {
          this->on_put(outcome);
      });
    }

    void on_put(const Aws::S3::Model::PutObjectOutcome & outcome) {
      if(outcome.IsSuccess()) {
        put_succeeded = true;
        do_get();
      } else {
        done.Notify();
      }
    }

    void do_get() {
      auto get_request = Aws::S3::Model::GetObjectRequest{};
      get_request.SetResponseStreamFactory(CordBackedResponseStreamFactory);
      get_request.SetBucket(Bucket());
      get_request.SetKey(key);
      client->GetObjectAsync(get_request, [this](
        const auto *, const auto &, auto outcome, const auto &) {
          this->on_get(std::move(outcome));
        });
    }

    void on_get(Aws::S3::Model::GetObjectOutcome outcome) {
      if(outcome.IsSuccess()) {
        std::string buffer;
        std::getline(outcome.GetResult().GetBody(), buffer);
        get_result = buffer;
      }

      done.Notify();
    }
  };

  auto callbacks = TestCallbacks{"key", "value"};
  callbacks.do_put();
  EXPECT_TRUE(callbacks.done.WaitForNotificationWithTimeout(absl::Milliseconds(10)));
  EXPECT_TRUE(callbacks.put_succeeded);
  EXPECT_TRUE(callbacks.get_result.has_value());
  EXPECT_EQ(callbacks.get_result.value(), callbacks.payload);
}



}  // namespace
