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

#include <stdint.h>

#include <fstream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/aws/aws_credentials.h"
#include "tensorstore/internal/aws/credentials/common.h"
#include "tensorstore/internal/aws/http_mocking.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::IsOk;
using ::tensorstore::internal::JoinPath;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::UnsetEnv;
using ::tensorstore::internal_aws::AwsCredentialsProvider;
using ::tensorstore::internal_aws::DisableAwsHttpMocking;
using ::tensorstore::internal_aws::EnableAwsHttpMocking;
using ::tensorstore::internal_aws::GetAwsCredentials;
using ::tensorstore::internal_aws::MakeProfile;

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

class FileCredentialProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    EnableAwsHttpMocking({});
    for (const char* var : {
             "AWS_CONFIG_FILE",
             "AWS_SHARED_CREDENTIALS_FILE",
             "AWS_PROFILE",
         }) {
      UnsetEnv(var);
    }
  }
  void TearDown() override { DisableAwsHttpMocking(); }
};

TEST_F(FileCredentialProviderTest, DefaultSharedCredentialsFile) {
  // NOTE: When either of AWS_SHARED_CREDENTIALS_FILE and AWS_CONFIG_FILE
  // are missing, the default location of ~/.aws/... causes aws_c_auth to
  // look for $HOME via aws_get_config_file_path which can fail when HOME
  // is unset or when pw_dir (from getpwuid) is unset.
  TestData test_data;
  SetEnv("AWS_SHARED_CREDENTIALS_FILE",
         test_data.WriteCredentialsFile().c_str());
  SetEnv("AWS_CONFIG_FILE", test_data.WriteConfigFile().c_str());

  AwsCredentialsProvider provider = MakeProfile({}, {}, {});
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future, IsOk());

  auto credentials = std::move(credentials_future.result()).value();
  ASSERT_EQ(credentials.GetAccessKeyId(), "AKIAIOSFODNN7EXAMPLE");
  ASSERT_EQ(credentials.GetSecretAccessKey(),
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  ASSERT_EQ(credentials.GetSessionToken(), "abcdef1234567890");
}

TEST_F(FileCredentialProviderTest, DefaultSharedCredentialsFile_WithProfile) {
  TestData test_data;
  SetEnv("AWS_SHARED_CREDENTIALS_FILE",
         test_data.WriteCredentialsFile().c_str());
  SetEnv("AWS_CONFIG_FILE", test_data.WriteConfigFile().c_str());

  {
    AwsCredentialsProvider provider = MakeProfile("alice", {}, {});
    auto credentials_future = GetAwsCredentials(provider.get());
    ASSERT_THAT(credentials_future, IsOk());

    auto credentials = std::move(credentials_future.result()).value();
    ASSERT_EQ(credentials.GetAccessKeyId(), "AKIAIOSFODNN6EXAMPLE");
    ASSERT_EQ(credentials.GetSecretAccessKey(),
              "wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY");
    ASSERT_EQ(credentials.GetSessionToken(), "");
  }

  SetEnv("AWS_PROFILE", "alice");
  {
    AwsCredentialsProvider provider = MakeProfile({}, {}, {});
    auto credentials_future = GetAwsCredentials(provider.get());
    ASSERT_THAT(credentials_future, IsOk());

    auto credentials = std::move(credentials_future.result()).value();
    ASSERT_EQ(credentials.GetAccessKeyId(), "AKIAIOSFODNN6EXAMPLE");
    ASSERT_EQ(credentials.GetSecretAccessKey(),
              "wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY");
    ASSERT_EQ(credentials.GetSessionToken(), "");
  }
}

TEST_F(FileCredentialProviderTest, ProviderAwsCredentialsFromFileOverride) {
  TestData test_data;
  auto credentials_filename = test_data.WriteCredentialsFile();
  auto config_filename = test_data.WriteConfigFile();

  {
    AwsCredentialsProvider provider =
        MakeProfile({}, credentials_filename, config_filename);
    auto credentials_future = GetAwsCredentials(provider.get());
    ASSERT_THAT(credentials_future, IsOk());

    auto credentials = std::move(credentials_future.result()).value();
    ASSERT_EQ(credentials.GetAccessKeyId(), "AKIAIOSFODNN7EXAMPLE");
    ASSERT_EQ(credentials.GetSecretAccessKey(),
              "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
    ASSERT_EQ(credentials.GetSessionToken(), "abcdef1234567890");
  }

  {
    AwsCredentialsProvider provider =
        MakeProfile("alice", credentials_filename, config_filename);
    auto credentials_future = GetAwsCredentials(provider.get());
    ASSERT_THAT(credentials_future, IsOk());

    auto credentials = std::move(credentials_future.result()).value();
    ASSERT_EQ(credentials.GetAccessKeyId(), "AKIAIOSFODNN6EXAMPLE");
    ASSERT_EQ(credentials.GetSecretAccessKey(),
              "wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY");
    ASSERT_EQ(credentials.GetSessionToken(), "");
  }
}

}  // namespace
