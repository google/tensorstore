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

#include "tensorstore/kvstore/s3/aws_metadata_credential_provider.h"

#include "absl/container/flat_hash_map.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Future;
using ::tensorstore::internal_http::GetDefaultHttpTransport;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_kvstore_s3::EC2MetadataCredentialProvider;
using ::tensorstore::MatchesStatus;

class EC2MetadataMockTransport : public HttpTransport {
 public:
  EC2MetadataMockTransport(
      const absl::flat_hash_map<std::string, HttpResponse>& url_to_response)
      : url_to_response_(url_to_response) {}

  Future<HttpResponse> IssueRequest(const HttpRequest& request,
                                    absl::Cord payload,
                                    absl::Duration request_timeout,
                                    absl::Duration connect_timeout) override {
    ABSL_LOG(INFO) << request;
    auto it = url_to_response_.find(
        tensorstore::StrCat(request.method, " ", request.url));
    if (it != url_to_response_.end()) {
      return it->second;
    }
    return HttpResponse{404, absl::Cord(), {}};
  }

  const absl::flat_hash_map<std::string, HttpResponse>& url_to_response_;
};


TEST(EC2MetadataCredentialProviderTest, CredentialRetrievalFlow) {
    auto url_to_response = absl::flat_hash_map<std::string, HttpResponse>{
        {"POST http://http://169.254.169.254/latest/api/token",
         HttpResponse{200, absl::Cord{"1234567890"}}},
        {"GET http://http://169.254.169.254/latest/meta-data/iam/",
         HttpResponse{200, absl::Cord{"info"},
                      {{"x-aws-ec2-metadata-token", "1234567890"}}}},
        {"GET http://169.254.169.254/latest/meta-data/iam/security-credentials/",
         HttpResponse{200, absl::Cord{"mock-iam-role"},
                      {{"x-aws-ec2-metadata-token", "1234567890"}}}},
        {"GET http://169.254.169.254/latest/meta-data/iam/security-credentials/mock-iam-role",
         HttpResponse{200,
                      absl::Cord(R"({
                                    "Code": "Success",
                                    "LastUpdated": "2023-09-21T12:42:12Z",
                                    "Type": "AWS-HMAC",
                                    "AccessKeyId": "ASIA1234567890",
                                    "SecretAccessKey": "1234567890abcdef",
                                    "Token": "abcdef123456790",
                                    "Expiration": "2023-09-21T12:42:12Z"
                                 })"),
                      {{"x-aws-ec2-metadata-token", "1234567890"}}}}
    };

    auto mock_transport = std::make_shared<EC2MetadataMockTransport>(url_to_response);
    auto provider = std::make_shared<EC2MetadataCredentialProvider>(mock_transport);
    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
    ASSERT_EQ(credentials.access_key, "ASIA1234567890");
    ASSERT_EQ(credentials.secret_key, "1234567890abcdef");
    ASSERT_EQ(credentials.session_token, "abcdef123456790");
}

/// https://hackingthe.cloud/aws/exploitation/ec2-metadata-ssrf/
/// Requests to /latest/meta-data/iam/ when no IAM Role is associated
/// with the instance result in a 404. Return anonynmous credentials.
TEST(EC2MetadataCredentialProviderTest, NoIamRole) {
    auto url_to_response = absl::flat_hash_map<std::string, HttpResponse>{
        {"POST http://http://169.254.169.254/latest/api/token",
         HttpResponse{200, absl::Cord{"1234567890"}}},
        {"GET http://http://169.254.169.254/latest/meta-data/iam/",
         HttpResponse{404, {}, {{"x-aws-ec2-metadata-token", "1234567890"}}}},
    };

    auto mock_transport = std::make_shared<EC2MetadataMockTransport>(url_to_response);
    auto provider = std::make_shared<EC2MetadataCredentialProvider>(mock_transport);
    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
    ASSERT_TRUE(credentials.IsAnonymous());
}


/// https://hackingthe.cloud/aws/exploitation/ec2-metadata-ssrf/
/// Requests to /latest/meta-data/iam/ when IAM Role revoked
/// result in an empty response. Return anonymous credentials.
TEST(EC2MetadataCredentialProviderTest, RevokedIamRole) {
    auto url_to_response = absl::flat_hash_map<std::string, HttpResponse>{
        {"POST http://http://169.254.169.254/latest/api/token",
         HttpResponse{200, absl::Cord{"1234567890"}}},
        {"GET http://http://169.254.169.254/latest/meta-data/iam/",
         HttpResponse{200, absl::Cord{""}, {{"x-aws-ec2-metadata-token", "1234567890"}}}},
    };

    auto mock_transport = std::make_shared<EC2MetadataMockTransport>(url_to_response);
    auto provider = std::make_shared<EC2MetadataCredentialProvider>(mock_transport);
    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
    ASSERT_TRUE(credentials.IsAnonymous());
}


TEST(EC2MetadataCredentialProviderTest, UnsuccessfulJsonResponse) {
    // Test that "Code" != "Success" parsing succeeds
    auto url_to_response = absl::flat_hash_map<std::string, HttpResponse>{
        {"POST http://http://169.254.169.254/latest/api/token",
         HttpResponse{200, absl::Cord{"1234567890"}}},
        {"GET http://http://169.254.169.254/latest/meta-data/iam/",
         HttpResponse{200, absl::Cord{"info"},
                      {{"x-aws-ec2-metadata-token", "1234567890"}}}},
        {"GET http://169.254.169.254/latest/meta-data/iam/security-credentials/",
         HttpResponse{200, absl::Cord{"mock-iam-role"},
                      {{"x-aws-ec2-metadata-token", "1234567890"}}}},
        {"GET http://169.254.169.254/latest/meta-data/iam/security-credentials/mock-iam-role",
         HttpResponse{200, absl::Cord(R"({"Code": "EntirelyUnsuccessful"})"),
                      {{"x-aws-ec2-metadata-token", "1234567890"}}}}
    };

    auto mock_transport = std::make_shared<EC2MetadataMockTransport>(url_to_response);
    auto provider = std::make_shared<EC2MetadataCredentialProvider>(mock_transport);
    auto credentials = provider->GetCredentials();

    EXPECT_THAT(credentials.status(), MatchesStatus(absl::StatusCode::kNotFound));
    EXPECT_THAT(credentials.status().ToString(),
                ::testing::AllOf(
                  ::testing::HasSubstr("EC2Metadata request"),
                  ::testing::HasSubstr("failed with {\"Code\": \"EntirelyUnsuccessful\"}")));
  }

} // namespace