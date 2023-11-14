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

#include "tensorstore/kvstore/s3/credentials/default_credential_provider.h"

#include <gtest/gtest.h>

#include <fstream>
#include <new>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/kvstore/s3/credentials/test_utils.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Future;
using ::tensorstore::internal::JoinPath;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_kvstore_s3::DefaultAwsCredentialsProvider;
using ::tensorstore::internal_kvstore_s3::DefaultEC2MetadataFlow;
using ::tensorstore::internal_kvstore_s3::EC2MetadataMockTransport;
using Options =
    ::tensorstore::internal_kvstore_s3::DefaultAwsCredentialsProvider::Options;

class CredentialFileFactory
    : public tensorstore::internal::ScopedTemporaryDirectory {
 public:
  std::string WriteCredentialsFile() {
    auto p = JoinPath(path(), "aws_config");
    std::ofstream ofs(p);
    ofs << "[alice]\n"
           "aws_access_key_id = AKIAIOSFODNN6EXAMPLE\n"
           "aws_secret_access_key = "
           "wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY\n"
           "aws_session_token = abcdef1234567890\n"
           "\n";
    ofs.close();
    return p;
  }
};

/// Test configuration of FileCredentialProvider from
/// DefaultAwsCredentialsProvider::Options
TEST(DefaultCredentialProviderTest, ConfigureFileProviderFromOptions) {
  auto factory = CredentialFileFactory{};
  auto credentials_file = factory.WriteCredentialsFile();

  auto provider = std::make_unique<DefaultAwsCredentialsProvider>(
      Options{credentials_file, "alice"});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials,
                                   provider->GetCredentials());
  EXPECT_EQ(credentials.access_key, "AKIAIOSFODNN6EXAMPLE");
  EXPECT_EQ(credentials.secret_key, "wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY");
  EXPECT_EQ(credentials.session_token, "abcdef1234567890");
}

/// Test configuration of EC2MetaDataProvider from
/// DefaultAwsCredentialsProvider::Options
TEST(DefaultCredentialProviderTest, ConfigureEC2ProviderFromOptions) {
  auto now = absl::Now();
  auto stuck_clock = [&]() -> absl::Time { return now; };
  auto expiry = now + absl::Seconds(200);
  auto url_to_response = DefaultEC2MetadataFlow(
      "1234", "ASIA1234567890", "1234567890abcdef", "token", expiry);

  auto mock_transport =
      std::make_shared<EC2MetadataMockTransport>(url_to_response);

  auto provider = std::make_unique<DefaultAwsCredentialsProvider>(
      Options{{}, {}, mock_transport}, stuck_clock);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials,
                                   provider->GetCredentials());
  EXPECT_EQ(credentials.access_key, "ASIA1234567890");
  EXPECT_EQ(credentials.secret_key, "1234567890abcdef");
  EXPECT_EQ(credentials.session_token, "token");
  ASSERT_EQ(credentials.expires_at, expiry - absl::Seconds(60));

  /// Force failure on credential retrieval
  url_to_response = absl::flat_hash_map<std::string, HttpResponse>{
      {"POST http://169.254.169.254/latest/api/token",
       HttpResponse{404, absl::Cord{""}}},
  };

  // But we're not expired, so we get the original credentials
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(credentials, provider->GetCredentials());
  EXPECT_EQ(credentials.access_key, "ASIA1234567890");
  EXPECT_EQ(credentials.secret_key, "1234567890abcdef");
  EXPECT_EQ(credentials.session_token, "token");
  ASSERT_EQ(credentials.expires_at, expiry - absl::Seconds(60));

  // Force expiry and retrieve new credentials
  now += absl::Seconds(300);
  url_to_response = DefaultEC2MetadataFlow("1234", "ASIA1234567890",
                                           "1234567890abcdef", "TOKEN", expiry);

  // A new set of credentials is returned
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(credentials, provider->GetCredentials());
  EXPECT_EQ(credentials.access_key, "ASIA1234567890");
  EXPECT_EQ(credentials.secret_key, "1234567890abcdef");
  EXPECT_EQ(credentials.session_token, "TOKEN");
  ASSERT_EQ(credentials.expires_at, expiry - absl::Seconds(60));

  /// Force failure on credential retrieval
  url_to_response = absl::flat_hash_map<std::string, HttpResponse>{
      {"POST http://169.254.169.254/latest/api/token",
       HttpResponse{404, absl::Cord{""}}},
  };

  // Anonymous credentials
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(credentials, provider->GetCredentials());
  EXPECT_EQ(credentials.access_key, "");
  EXPECT_EQ(credentials.secret_key, "");
  EXPECT_EQ(credentials.session_token, "");
  ASSERT_EQ(credentials.expires_at, absl::InfiniteFuture());
}

}  // namespace