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

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/aws/aws_credentials.h"
#include "tensorstore/internal/aws/credentials/common.h"
#include "tensorstore/internal/aws/credentials/test_utils.h"
#include "tensorstore/internal/aws/http_mocking.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/http_header.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::IsOk;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::UnsetEnv;
using ::tensorstore::internal_aws::AwsCredentialsProvider;
using ::tensorstore::internal_aws::DefaultImdsCredentialFlow;
using ::tensorstore::internal_aws::DisableAwsHttpMocking;
using ::tensorstore::internal_aws::EnableAwsHttpMocking;
using ::tensorstore::internal_aws::GetAwsCredentials;
using ::tensorstore::internal_aws::MakeImds;
using ::tensorstore::internal_http::HeaderMap;
using ::tensorstore::internal_http::HttpResponse;

namespace {

// The default endpoint for the IMDS is http://169.254.169.254:80, and the
// aws_c_auth library does not support changing this endpoint.

static constexpr char kApiToken[] = "1234567890";
static constexpr char kAccessKey[] = "ASIA1234567890";
static constexpr char kSecretKey[] = "1234567890abcdef";
static constexpr char kSessionToken[] = "abcdef123456790";

class ImdsCredentialsProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    expiry = absl::Now() + absl::Seconds(200);
    UnsetEnv("AWS_EC2_METADATA_SERVICE_ENDPOINT");
  }
  void TearDown() override { DisableAwsHttpMocking(); }

  absl::Time expiry;
};

TEST_F(ImdsCredentialsProviderTest, Basic) {
  EnableAwsHttpMocking(DefaultImdsCredentialFlow(
      kApiToken, kAccessKey, kSecretKey, kSessionToken, expiry));

  AwsCredentialsProvider provider = MakeImds();
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future, IsOk());

  auto credentials = std::move(credentials_future.result()).value();
  EXPECT_EQ(credentials.GetAccessKeyId(), kAccessKey);
  EXPECT_EQ(credentials.GetSecretAccessKey(), kSecretKey);
  EXPECT_EQ(credentials.GetSessionToken(), kSessionToken);
}

TEST_F(ImdsCredentialsProviderTest, WithEnvironmentVariable) {
  // NOTE: If the aws_c_auth library is updated to use the environment variable
  // to determine the endpoint, this test will fail.
  SetEnv("AWS_EC2_METADATA_SERVICE_ENDPOINT", "http://localhost:1234/");

  EnableAwsHttpMocking(DefaultImdsCredentialFlow(
      kApiToken, kAccessKey, kSecretKey, kSessionToken, expiry));

  AwsCredentialsProvider provider = MakeImds();
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future, IsOk());

  auto credentials = std::move(credentials_future.result()).value();
  EXPECT_EQ(credentials.GetAccessKeyId(), kAccessKey);
  EXPECT_EQ(credentials.GetSecretAccessKey(), kSecretKey);
  EXPECT_EQ(credentials.GetSessionToken(), kSessionToken);
}

TEST_F(ImdsCredentialsProviderTest, NoIamRolesInSecurityCredentials) {
  EnableAwsHttpMocking({
      {"PUT http://169.254.169.254:80/latest/api/token",
       HttpResponse{200, absl::Cord{kApiToken}}},
      {"GET "
       "http://169.254.169.254:80/latest/meta-data/iam/security-credentials/",
       HttpResponse{200, absl::Cord{""},
                    HeaderMap{{"x-aws-ec2-metadata-token", kApiToken}}}},
      // second call to GetAwsCredentials()
      {"PUT http://169.254.169.254:80/latest/api/token",
       HttpResponse{200, absl::Cord{kApiToken}}},
      {"GET "
       "http://169.254.169.254:80/latest/meta-data/iam/security-credentials/",
       HttpResponse{200, absl::Cord{""},
                    HeaderMap{{"x-aws-ec2-metadata-token", kApiToken}}}},
  });

  AwsCredentialsProvider provider = MakeImds();
  auto credentials_future = GetAwsCredentials(provider.get());
  EXPECT_THAT(credentials_future,
              MatchesStatus(absl::StatusCode::kInternal, ".*aws-c-auth.*"));
}

TEST_F(ImdsCredentialsProviderTest, UnsuccessfulJsonResponse) {
  // Test that "Code" != "Success" parsing succeeds
  EnableAwsHttpMocking(
      {{"PUT http://169.254.169.254:80/latest/api/token",
        HttpResponse{200, absl::Cord{kApiToken}}},
       {"GET http://169.254.169.254:80/latest/meta-data/iam/",
        HttpResponse{200, absl::Cord{"info"},
                     HeaderMap{{"x-aws-ec2-metadata-token", kApiToken}}}},
       {"GET "
        "http://169.254.169.254:80/latest/meta-data/iam/security-credentials/",
        HttpResponse{200, absl::Cord{"mock-iam-role"},
                     HeaderMap{{"x-aws-ec2-metadata-token", kApiToken}}}},
       {"GET "
        "http://169.254.169.254:80/latest/meta-data/iam/security-credentials/"
        "mock-iam-role",
        HttpResponse{200, absl::Cord(R"({"Code": "EntirelyUnsuccessful"})"),
                     HeaderMap{{"x-aws-ec2-metadata-token", kApiToken}}}}});

  AwsCredentialsProvider provider = MakeImds();
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future,
              MatchesStatus(absl::StatusCode::kInternal, ".*aws-c-auth.*"));
  // NOTE: The error message is not propagated to the status yet.
}

}  // namespace
