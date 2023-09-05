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

#include "tensorstore/kvstore/s3/aws_credential_provider.h"

#include <fstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::internal::GetEnv;
using ::tensorstore::internal::JoinPath;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::UnsetEnv;
using ::tensorstore::internal_http::GetDefaultHttpTransport;
using ::tensorstore::internal_kvstore_s3::GetAwsCredentialProvider;

class TestData : public tensorstore::internal::ScopedTemporaryDirectory {
 public:
  std::string WriteCredentialsFile() {
    auto p = JoinPath(path(), "aws_config");
    std::ofstream ofs(p);
    ofs << "discarded_value = 500\n"
           "\n"
           "[default]\n"
           "aws_access_key_id =AKIAIOSFODNN7EXAMPLE\n"
           "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
           "aws_session_token= abcdef1234567890 \n"
           "\n"
           "[alice]\n"
           "aws_access_key_id = AKIAIOSFODNN6EXAMPLE\n"
           "aws_secret_access_key = "
           "wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY\n"
           "\n";
    ofs.close();
    return p;
  }
};

class AwsCredentialProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Make sure that env vars are not set.
    for (const char* var :
         {"AWS_SHARED_CREDENTIALS_FILE", "AWS_ACCESS_KEY_ID",
          "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"}) {
      UnsetEnv(var);
    }
  }
};

TEST_F(AwsCredentialProviderTest, ProviderNoCredentials) {
  ASSERT_FALSE(GetAwsCredentialProvider("", GetDefaultHttpTransport()).ok());
  SetEnv("AWS_ACCESS_KEY_ID", "foo");
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto provider, GetAwsCredentialProvider("", GetDefaultHttpTransport()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials,
                                   provider->GetCredentials());
  ASSERT_EQ(credentials.access_key, "foo");
  ASSERT_TRUE(credentials.secret_key.empty());
  ASSERT_TRUE(credentials.session_token.empty());
}

TEST_F(AwsCredentialProviderTest, ProviderAwsCredentialsFromEnv) {
  SetEnv("AWS_ACCESS_KEY_ID", "foo");
  SetEnv("AWS_SECRET_ACCESS_KEY", "bar");
  SetEnv("AWS_SESSION_TOKEN", "qux");
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto provider, GetAwsCredentialProvider("", GetDefaultHttpTransport()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials,
                                   provider->GetCredentials());
  ASSERT_EQ(credentials.access_key, "foo");
  ASSERT_EQ(credentials.secret_key, "bar");
  ASSERT_EQ(credentials.session_token, "qux");
}

TEST_F(AwsCredentialProviderTest, ProviderAwsCredentialsFromFileDefault) {
  TestData test_data;
  std::string credentials_filename = test_data.WriteCredentialsFile();

  SetEnv("AWS_SHARED_CREDENTIALS_FILE", credentials_filename.c_str());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto provider, GetAwsCredentialProvider("", GetDefaultHttpTransport()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials,
                                   provider->GetCredentials());
  ASSERT_EQ(credentials.access_key, "AKIAIOSFODNN7EXAMPLE");
  ASSERT_EQ(credentials.secret_key, "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  ASSERT_EQ(credentials.session_token, "abcdef1234567890");
}

TEST_F(AwsCredentialProviderTest,
       ProviderAwsCredentialsFromFileProfileOverride) {
  TestData test_data;
  std::string credentials_filename = test_data.WriteCredentialsFile();

  SetEnv("AWS_SHARED_CREDENTIALS_FILE", credentials_filename.c_str());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto provider,
      GetAwsCredentialProvider("alice", GetDefaultHttpTransport()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials,
                                   provider->GetCredentials());
  ASSERT_EQ(credentials.access_key, "AKIAIOSFODNN6EXAMPLE");
  ASSERT_EQ(credentials.secret_key, "wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY");
  ASSERT_EQ(credentials.session_token, "");
}

TEST_F(AwsCredentialProviderTest, ProviderAwsCredentialsFromFileProfileEnv) {
  TestData test_data;
  std::string credentials_filename = test_data.WriteCredentialsFile();

  SetEnv("AWS_SHARED_CREDENTIALS_FILE", credentials_filename.c_str());
  SetEnv("AWS_PROFILE", "alice");
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto provider, GetAwsCredentialProvider("", GetDefaultHttpTransport()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials,
                                   provider->GetCredentials());
  ASSERT_EQ(credentials.access_key, "AKIAIOSFODNN6EXAMPLE");
  ASSERT_EQ(credentials.secret_key, "wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY");
  ASSERT_EQ(credentials.session_token, "");
}

TEST_F(AwsCredentialProviderTest,
       ProviderAwsCredentialsFromFileInvalidProfileEnv) {
  TestData test_data;
  std::string credentials_filename = test_data.WriteCredentialsFile();

  SetEnv("AWS_SHARED_CREDENTIALS_FILE", credentials_filename.c_str());
  SetEnv("AWS_PROFILE", "bob");
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto provider, GetAwsCredentialProvider("", GetDefaultHttpTransport()));
  auto result = provider->GetCredentials();
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(
      result.status().message(),
      ::testing::HasSubstr("Profile [bob] not found in credentials file"));
}

}  // namespace
