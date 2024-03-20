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

#include "tensorstore/kvstore/s3/credentials/ec2_credential_provider.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/mock_http_transport.h"
#include "tensorstore/kvstore/s3/credentials/test_utils.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::UnsetEnv;

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_http::DefaultMockHttpTransport;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_kvstore_s3::DefaultEC2MetadataFlow;
using ::tensorstore::internal_kvstore_s3::EC2MetadataCredentialProvider;

static constexpr char kDefaultEndpoint[] = "http://169.254.169.254";
static constexpr char kCustomEndpoint[] = "http://custom.endpoint";
static constexpr char kApiToken[] = "1234567890";
static constexpr char kAccessKey[] = "ASIA1234567890";
static constexpr char kSecretKey[] = "1234567890abcdef";
static constexpr char kSessionToken[] = "abcdef123456790";

class EC2MetadataCredentialProviderTest : public ::testing::Test {
 protected:
  void SetUp() override { UnsetEnv("AWS_EC2_METADATA_SERVICE_ENDPOINT"); }
};

TEST_F(EC2MetadataCredentialProviderTest, CredentialRetrievalFlow) {
  auto expiry = absl::Now() + absl::Seconds(200);

  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultEC2MetadataFlow(kDefaultEndpoint, kApiToken, kAccessKey,
                             kSecretKey, kSessionToken, expiry));
  auto provider =
      std::make_shared<EC2MetadataCredentialProvider>("", mock_transport);
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
  ASSERT_EQ(provider->GetEndpoint(), kDefaultEndpoint);
  ASSERT_EQ(credentials.access_key, kAccessKey);
  ASSERT_EQ(credentials.secret_key, kSecretKey);
  ASSERT_EQ(credentials.session_token, kSessionToken);
  // expiry less the 60s leeway
  ASSERT_EQ(credentials.expires_at, expiry - absl::Seconds(60));
}

TEST_F(EC2MetadataCredentialProviderTest, EnvironmentVariableMetadataServer) {
  SetEnv("AWS_EC2_METADATA_SERVICE_ENDPOINT", kCustomEndpoint);
  auto expiry = absl::Now() + absl::Seconds(200);

  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultEC2MetadataFlow(kCustomEndpoint, kApiToken, kAccessKey, kSecretKey,
                             kSessionToken, expiry));
  auto provider =
      std::make_shared<EC2MetadataCredentialProvider>("", mock_transport);
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
  ASSERT_EQ(provider->GetEndpoint(), kCustomEndpoint);
  ASSERT_EQ(credentials.access_key, kAccessKey);
  ASSERT_EQ(credentials.secret_key, kSecretKey);
  ASSERT_EQ(credentials.session_token, kSessionToken);
  // expiry less the 60s leeway
  ASSERT_EQ(credentials.expires_at, expiry - absl::Seconds(60));
}

TEST_F(EC2MetadataCredentialProviderTest, InjectedMetadataServer) {
  auto expiry = absl::Now() + absl::Seconds(200);

  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultEC2MetadataFlow(kCustomEndpoint, kApiToken, kAccessKey, kSecretKey,
                             kSessionToken, expiry));
  auto provider = std::make_shared<EC2MetadataCredentialProvider>(
      kCustomEndpoint, mock_transport);
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
  ASSERT_EQ(provider->GetEndpoint(), kCustomEndpoint);
  ASSERT_EQ(credentials.access_key, kAccessKey);
  ASSERT_EQ(credentials.secret_key, kSecretKey);
  ASSERT_EQ(credentials.session_token, kSessionToken);
  // expiry less the 60s leeway
  ASSERT_EQ(credentials.expires_at, expiry - absl::Seconds(60));
}

TEST_F(EC2MetadataCredentialProviderTest, NoIamRolesInSecurityCredentials) {
  auto url_to_response = absl::flat_hash_map<std::string, HttpResponse>{
      {"POST http://169.254.169.254/latest/api/token",
       HttpResponse{200, absl::Cord{kApiToken}}},
      {"GET http://169.254.169.254/latest/meta-data/iam/security-credentials/",
       HttpResponse{
           200, absl::Cord{""}, {{"x-aws-ec2-metadata-token", kApiToken}}}},
  };

  auto mock_transport =
      std::make_shared<DefaultMockHttpTransport>(std::move(url_to_response));
  auto provider =
      std::make_shared<EC2MetadataCredentialProvider>("", mock_transport);
  ASSERT_FALSE(provider->GetCredentials());
  ASSERT_EQ(provider->GetEndpoint(), kDefaultEndpoint);
  EXPECT_THAT(provider->GetCredentials().status().ToString(),
              ::testing::HasSubstr("Empty EC2 Role list"));
}

TEST_F(EC2MetadataCredentialProviderTest, UnsuccessfulJsonResponse) {
  // Test that "Code" != "Success" parsing succeeds
  auto url_to_response = absl::flat_hash_map<std::string, HttpResponse>{
      {"POST http://169.254.169.254/latest/api/token",
       HttpResponse{200, absl::Cord{kApiToken}}},
      {"GET http://169.254.169.254/latest/meta-data/iam/",
       HttpResponse{
           200, absl::Cord{"info"}, {{"x-aws-ec2-metadata-token", kApiToken}}}},
      {"GET http://169.254.169.254/latest/meta-data/iam/security-credentials/",
       HttpResponse{200,
                    absl::Cord{"mock-iam-role"},
                    {{"x-aws-ec2-metadata-token", kApiToken}}}},
      {"GET "
       "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
       "mock-iam-role",
       HttpResponse{200,
                    absl::Cord(R"({"Code": "EntirelyUnsuccessful"})"),
                    {{"x-aws-ec2-metadata-token", kApiToken}}}}};

  auto mock_transport =
      std::make_shared<DefaultMockHttpTransport>(std::move(url_to_response));
  auto provider =
      std::make_shared<EC2MetadataCredentialProvider>("", mock_transport);
  auto credentials = provider->GetCredentials();

  EXPECT_THAT(credentials.status(), MatchesStatus(absl::StatusCode::kNotFound));
  EXPECT_THAT(credentials.status().ToString(),
              ::testing::AllOf(
                  ::testing::HasSubstr("EC2Metadata request"),
                  ::testing::HasSubstr(
                      "failed with {\"Code\": \"EntirelyUnsuccessful\"}")));
}

}  // namespace
