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

#include <fstream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/aws/aws_credentials.h"
#include "tensorstore/internal/aws/credentials/common.h"
#include "tensorstore/internal/aws/http_mocking.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::IsOk;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::JoinPath;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::UnsetEnv;
using ::tensorstore::internal_aws::AwsCredentialsProvider;
using ::tensorstore::internal_aws::DisableAwsHttpMocking;
using ::tensorstore::internal_aws::EnableAwsHttpMocking;
using ::tensorstore::internal_aws::GetAwsCredentials;
using ::tensorstore::internal_aws::MakeAnonymous;
using ::tensorstore::internal_aws::MakeDefault;
using ::tensorstore::internal_aws::MakeDefaultWithAnonymous;

static constexpr char kAccessKeyId[] = "ASIA1234567890";
static constexpr char kSecretKey[] = "1234567890abcdef";
static constexpr char kSessionToken[] = "abcdef123456790";

class TestData
    : public tensorstore::internal_testing::ScopedTemporaryDirectory {
 public:
  std::string WriteCredentialsFile() {
    auto p = JoinPath(path(), "aws_credentials");
    std::ofstream ofs(p);
    ofs << "[default]\n"
           "aws_access_key_id=AKIAIOSFODNN7EXAMPLE\n"
           "aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
           "aws_session_token=abcdef1234567890 \n"
           "\n"
           "[alice]\n"
           "aws_access_key_id = AKIAIOSFODNN6EXAMPLE\n"
           "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY\n"
           "\n";
    ofs.close();
    return p;
  }
  std::string WriteConfigFile() {
    auto p = JoinPath(path(), "aws_config");
    std::ofstream ofs(p);
    ofs << "[default]\n"
           "region=us-west-2\n"
           "output=json\n"
           "\n";
    ofs.close();
    return p;
  }
};

class DefaultCredentialProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    expiry = absl::Now() + absl::Seconds(200);
    EnableAwsHttpMocking({});

    for (const char* var : {
             "AWS_ACCESS_KEY_ID",
             "AWS_SECRET_ACCESS_KEY",
             "AWS_SESSION_TOKEN",
             "AWS_CONFIG_FILE",
             "AWS_SHARED_CREDENTIALS_FILE",
             "AWS_PROFILE",
             "AWS_CONTAINER_CREDENTIALS_FULL_URI",
             "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
         }) {
      UnsetEnv(var);
    }
  }
  void TearDown() override { DisableAwsHttpMocking(); }

  absl::Time expiry;
};

TEST_F(DefaultCredentialProviderTest, Anonymous) {
  AwsCredentialsProvider provider = MakeAnonymous();
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future, IsOk());
  EXPECT_TRUE(credentials_future.result()->IsAnonymous());
}

TEST_F(DefaultCredentialProviderTest, DefaultWithoutFallback) {
  AwsCredentialsProvider provider = MakeDefault({});
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future,
              MatchesStatus(absl::StatusCode::kInternal, ".*aws-c-.*"));
}

TEST_F(DefaultCredentialProviderTest, DefaultWithFallback) {
  AwsCredentialsProvider provider = MakeDefaultWithAnonymous({});
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future, IsOk());
  EXPECT_TRUE(credentials_future.result()->IsAnonymous());
}

TEST_F(DefaultCredentialProviderTest, FromEnvironment) {
  SetEnv("AWS_ACCESS_KEY_ID", kAccessKeyId);
  SetEnv("AWS_SECRET_ACCESS_KEY", kSecretKey);
  SetEnv("AWS_SESSION_TOKEN", kSessionToken);

  AwsCredentialsProvider provider = MakeDefault({});
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future, IsOk());

  auto credentials = std::move(credentials_future.result()).value();
  EXPECT_EQ(credentials.GetAccessKeyId(), kAccessKeyId);
  EXPECT_EQ(credentials.GetSecretAccessKey(), kSecretKey);
  EXPECT_EQ(credentials.GetSessionToken(), kSessionToken);
}

TEST_F(DefaultCredentialProviderTest, FromProfile) {
  TestData test_data;
  SetEnv("AWS_SHARED_CREDENTIALS_FILE",
         test_data.WriteCredentialsFile().c_str());
  SetEnv("AWS_CONFIG_FILE", test_data.WriteConfigFile().c_str());

  AwsCredentialsProvider provider = MakeDefault({});
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future, IsOk());

  auto credentials = std::move(credentials_future.result()).value();
  EXPECT_EQ(credentials.GetAccessKeyId(), "AKIAIOSFODNN7EXAMPLE");
  EXPECT_EQ(credentials.GetSecretAccessKey(),
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  EXPECT_EQ(credentials.GetSessionToken(), "abcdef1234567890");
}

#if 0
// Default credentials provider chain does not include support for mocking
// the IMDS calls.
TEST_F(DefaultCredentialProviderTest, FromImds) {
  EnableAwsHttpMocking(DefaultImdsCredentialFlow(
      kApiToken, kAccessKey, kSecretKey, kSessionToken, expiry));

  AwsCredentialsProvider provider = MakeDefault({});
  auto credentials_future = provider.GetCredentials();
  ASSERT_THAT(credentials_future, IsOk());

  auto credentials = std::move(credentials_future.result()).value();
  EXPECT_EQ(credentials.GetAccessKeyId(), "AKIAIOSFODNN7EXAMPLE");
  EXPECT_EQ(credentials.GetSecretAccessKey(),
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  EXPECT_EQ(credentials.GetSessionToken(), "abcdef1234567890");
}
#endif

}  // namespace
