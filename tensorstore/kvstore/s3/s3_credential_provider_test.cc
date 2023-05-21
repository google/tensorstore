// Copyright 2020 The TensorStore Authors
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

#include "tensorstore/kvstore/s3/s3_credential_provider.h"

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>


#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::StrCat;
using ::tensorstore::internal::JoinPath;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::UnsetEnv;
using ::tensorstore::internal_auth_s3::CredentialProvider;
using ::tensorstore::internal_auth_s3::GetS3CredentialProvider;


class TestData : public tensorstore::internal::ScopedTemporaryDirectory {
 public:
  std::string WriteCredentialsFile() {
    auto p = JoinPath(path(), "aws_config");
    std::ofstream ofs(p);
    ofs <<
        "discarded_value = 500\n"
        "\n"
        "[default]\n"
        "aws_access_key_id =AKIAIOSFODNN7EXAMPLE\n"
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
        "aws_session_token= abcdef1234567890 \n"
        "\n"
        "[alice]\n"
        "aws_access_key_id = AKIAIOSFODNN6EXAMPLE\n"
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY\n";
    return p;
  }
};

class S3CredentialProviderTest : public ::testing::Test {
protected:
 TestData test_data;
 std::string credentials_filename;

 void SetUp() override {
    UnsetEnv("AWS_SHARED_CREDENTIALS_FILE");
    UnsetEnv("AWS_ACCESS_KEY_ID");
    UnsetEnv("AWS_SECRET_ACCESS_KEY");
    UnsetEnv("AWS_SESSION_TOKEN");
    UnsetEnv("AWS_DEFAULT_PROFILE");
    UnsetEnv("AWS_PROFILE");

    credentials_filename = test_data.WriteCredentialsFile();
 }
};

TEST_F(S3CredentialProviderTest, ProviderNoCredentials) {
    ASSERT_FALSE(GetS3CredentialProvider().ok());
    SetEnv("AWS_ACCESS_KEY_ID", "foo");
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto provider, GetS3CredentialProvider());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
    ASSERT_EQ(credentials.GetAccessKey(), "foo");
    ASSERT_TRUE(credentials.GetSecretKey().empty());
    ASSERT_TRUE(credentials.GetSessionToken().empty());
}

TEST_F(S3CredentialProviderTest, ProviderS3CredentialsFromEnv) {
    SetEnv("AWS_ACCESS_KEY_ID", "foo");
    SetEnv("AWS_SECRET_ACCESS_KEY", "bar");
    SetEnv("AWS_SESSION_TOKEN", "qux");
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto provider, GetS3CredentialProvider());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
    ASSERT_EQ(credentials.GetAccessKey(), "foo");
    ASSERT_EQ(credentials.GetSecretKey(), "bar");
    ASSERT_EQ(credentials.GetSessionToken(), "qux");
}

TEST_F(S3CredentialProviderTest, ProviderS3CredentialsFromFileDefault) {
    SetEnv("AWS_SHARED_CREDENTIALS_FILE", credentials_filename.c_str());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto provider, GetS3CredentialProvider());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
    ASSERT_EQ(credentials.GetAccessKey(), "AKIAIOSFODNN7EXAMPLE");
    ASSERT_EQ(credentials.GetSecretKey(), "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
    ASSERT_EQ(credentials.GetSessionToken(), "abcdef1234567890");
}

TEST_F(S3CredentialProviderTest, ProviderS3CredentialsFromFileProfileOverride) {
    SetEnv("AWS_SHARED_CREDENTIALS_FILE", credentials_filename.c_str());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto provider, GetS3CredentialProvider("alice"));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
    ASSERT_EQ(credentials.GetAccessKey(), "AKIAIOSFODNN6EXAMPLE");
    ASSERT_EQ(credentials.GetSecretKey(), "wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY");
    ASSERT_EQ(credentials.GetSessionToken(), "");
}

TEST_F(S3CredentialProviderTest, ProviderS3CredentialsFromFileProfileDefaultEnv) {
    SetEnv("AWS_SHARED_CREDENTIALS_FILE", credentials_filename.c_str());
    SetEnv("AWS_DEFAULT_PROFILE", "alice");
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto provider, GetS3CredentialProvider());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
    ASSERT_EQ(credentials.GetAccessKey(), "AKIAIOSFODNN6EXAMPLE");
    ASSERT_EQ(credentials.GetSecretKey(), "wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY");
    ASSERT_EQ(credentials.GetSessionToken(), "");
}

TEST_F(S3CredentialProviderTest, ProviderS3CredentialsFromFileProfileEnv) {
    SetEnv("AWS_SHARED_CREDENTIALS_FILE", credentials_filename.c_str());
    SetEnv("AWS_PROFILE", "alice");
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto provider, GetS3CredentialProvider());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
    ASSERT_EQ(credentials.GetAccessKey(), "AKIAIOSFODNN6EXAMPLE");
    ASSERT_EQ(credentials.GetSecretKey(), "wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY");
    ASSERT_EQ(credentials.GetSessionToken(), "");
}

TEST_F(S3CredentialProviderTest, ProviderS3CredentialsFromFileDefaultProfileOverridesProfileEnv) {
    SetEnv("AWS_SHARED_CREDENTIALS_FILE", credentials_filename.c_str());
    SetEnv("AWS_DEFAULT_PROFILE", "alice");
    SetEnv("AWS_PROFILE", "default");
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto provider, GetS3CredentialProvider());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto credentials, provider->GetCredentials());
    ASSERT_EQ(credentials.GetAccessKey(), "AKIAIOSFODNN6EXAMPLE");
    ASSERT_EQ(credentials.GetSecretKey(), "wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY");
    ASSERT_EQ(credentials.GetSessionToken(), "");
}

}
