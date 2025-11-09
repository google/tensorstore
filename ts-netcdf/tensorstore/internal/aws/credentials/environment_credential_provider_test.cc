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
#include "absl/time/time.h"
#include "tensorstore/internal/aws/aws_credentials.h"
#include "tensorstore/internal/aws/credentials/common.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::IsOk;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::UnsetEnv;
using ::tensorstore::internal_aws::AwsCredentialsProvider;
using ::tensorstore::internal_aws::GetAwsCredentials;
using ::tensorstore::internal_aws::MakeEnvironment;

class EnvironmentCredentialProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Make sure that env vars are not set.
    for (const char* var : {
             "AWS_ACCESS_KEY_ID",
             "AWS_SECRET_ACCESS_KEY",
             "AWS_SESSION_TOKEN",
         }) {
      UnsetEnv(var);
    }
  }
};

TEST_F(EnvironmentCredentialProviderTest, Basic) {
  SetEnv("AWS_ACCESS_KEY_ID", "foo");
  SetEnv("AWS_SECRET_ACCESS_KEY", "bar");
  SetEnv("AWS_SESSION_TOKEN", "qux");

  AwsCredentialsProvider provider = MakeEnvironment();
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future, IsOk());

  auto credentials = std::move(credentials_future.result()).value();
  ASSERT_EQ(credentials.GetAccessKeyId(), "foo");
  ASSERT_EQ(credentials.GetSecretAccessKey(), "bar");
  ASSERT_EQ(credentials.GetSessionToken(), "qux");
  ASSERT_EQ(credentials.GetExpiration(), absl::InfiniteFuture());
}

}  // namespace
