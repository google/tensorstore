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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/kvstore/s3/credentials/test_utils.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::internal::UnsetEnv;
using ::tensorstore::internal::SetEnv;

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_kvstore_s3::DefaultEC2MetadataFlow;
using ::tensorstore::internal_kvstore_s3::EC2MetadataCredentialProvider;
using ::tensorstore::internal_kvstore_s3::EC2MetadataMockTransport;

static constexpr char endpoint[] = "http://169.254.169.254";
static constexpr char api_token[] = "1234567890";
static constexpr char access_key[] = "ASIA1234567890";
static constexpr char secret_key[] = "1234567890abcdef";
static constexpr char session_token[] = "abcdef123456790";

class EC2MetadataCredentialProviderTest : public ::testing::Test {
 protected:
  void SetUp() override { UnsetEnv("AWS_EC2_METADATA_SERVICE_ENDPOINT"); }
};


TEST_F(EC2MetadataCredentialProviderTest, CredentialRetrievalFlow) {
  auto expiry = absl::Now() + absl::Seconds(200);
  auto url_to_response =
      DefaultEC2MetadataFlow(endpoint, api_token, access_key, secret_key,
                             session_token, expiry);

  auto mock_transport =
      std::make_shared<EC2MetadataMockTransport>(url_to_response);
  auto provider =
      std::make_shared<EC2MetadataCredentialProvider>(endpoint, mock_transport);
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
  ASSERT_EQ(credentials.access_key, access_key);
  ASSERT_EQ(credentials.secret_key, secret_key);
  ASSERT_EQ(credentials.session_token, session_token);
  // expiry less the 60s leeway
  ASSERT_EQ(credentials.expires_at, expiry - absl::Seconds(60));
}

TEST_F(EC2MetadataCredentialProviderTest, EnvironmentVariableMetadataServer) {
  SetEnv("AWS_EC2_METADATA_SERVICE_ENDPOINT", "http://endpoint");
  auto expiry = absl::Now() + absl::Seconds(200);
  auto url_to_response =
      DefaultEC2MetadataFlow("http://endpoint", api_token, access_key, secret_key,
                             session_token, expiry);

  auto mock_transport =
      std::make_shared<EC2MetadataMockTransport>(url_to_response);
  auto provider =
      std::make_shared<EC2MetadataCredentialProvider>("", mock_transport);
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
  ASSERT_EQ(credentials.access_key, access_key);
  ASSERT_EQ(credentials.secret_key, secret_key);
  ASSERT_EQ(credentials.session_token, session_token);
  // expiry less the 60s leeway
  ASSERT_EQ(credentials.expires_at, expiry - absl::Seconds(60));
}


TEST_F(EC2MetadataCredentialProviderTest, NoIamRolesInSecurityCredentials) {
  auto url_to_response = absl::flat_hash_map<std::string, HttpResponse>{
      {"POST http://169.254.169.254/latest/api/token",
       HttpResponse{200, absl::Cord{api_token}}},
      {"GET http://169.254.169.254/latest/meta-data/iam/security-credentials/",
       HttpResponse{
           200, absl::Cord{""}, {{"x-aws-ec2-metadata-token", api_token}}}},
  };

  auto mock_transport =
      std::make_shared<EC2MetadataMockTransport>(url_to_response);
  auto provider =
      std::make_shared<EC2MetadataCredentialProvider>(endpoint, mock_transport);
  ASSERT_FALSE(provider->GetCredentials());
  EXPECT_THAT(provider->GetCredentials().status().ToString(),
              ::testing::HasSubstr("Empty EC2 Role list"));
}

TEST_F(EC2MetadataCredentialProviderTest, UnsuccessfulJsonResponse) {
  // Test that "Code" != "Success" parsing succeeds
  auto url_to_response = absl::flat_hash_map<std::string, HttpResponse>{
      {"POST http://169.254.169.254/latest/api/token",
       HttpResponse{200, absl::Cord{api_token}}},
      {"GET http://169.254.169.254/latest/meta-data/iam/",
       HttpResponse{200,
                    absl::Cord{"info"},
                    {{"x-aws-ec2-metadata-token", api_token}}}},
      {"GET http://169.254.169.254/latest/meta-data/iam/security-credentials/",
       HttpResponse{200,
                    absl::Cord{"mock-iam-role"},
                    {{"x-aws-ec2-metadata-token", api_token}}}},
      {"GET "
       "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
       "mock-iam-role",
       HttpResponse{200,
                    absl::Cord(R"({"Code": "EntirelyUnsuccessful"})"),
                    {{"x-aws-ec2-metadata-token", api_token}}}}};

  auto mock_transport =
      std::make_shared<EC2MetadataMockTransport>(url_to_response);
  auto provider =
      std::make_shared<EC2MetadataCredentialProvider>(endpoint, mock_transport);
  auto credentials = provider->GetCredentials();

  EXPECT_THAT(credentials.status(), MatchesStatus(absl::StatusCode::kNotFound));
  EXPECT_THAT(credentials.status().ToString(),
              ::testing::AllOf(
                  ::testing::HasSubstr("EC2Metadata request"),
                  ::testing::HasSubstr(
                      "failed with {\"Code\": \"EntirelyUnsuccessful\"}")));
}

}  // namespace
