// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/kvstore/s3/aws_credentials.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::IsOk;
using ::tensorstore::internal_kvstore_s3::AwsCredentials;
using ::tensorstore::internal_kvstore_s3::AwsCredentialsProvider;
using ::tensorstore::internal_kvstore_s3::GetAwsCredentials;

TEST(CredentialsProviderTest, Nullptr) {
  AwsCredentialsProvider provider;

  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future, IsOk());

  auto credentials = std::move(credentials_future.result()).value();
  ASSERT_EQ(credentials.GetAccessKeyId(), "");
  ASSERT_EQ(credentials.GetSecretAccessKey(), "");
  ASSERT_EQ(credentials.GetSessionToken(), "");
  ASSERT_EQ(credentials.GetExpiration(), absl::InfiniteFuture());
}

TEST(CredentialsTest, Basic) {
  AwsCredentials credentials;

  ASSERT_EQ(credentials.GetAccessKeyId(), "");
  ASSERT_EQ(credentials.GetSecretAccessKey(), "");
  ASSERT_EQ(credentials.GetSessionToken(), "");
  ASSERT_EQ(credentials.GetExpiration(), absl::InfiniteFuture());
}

}  // namespace
