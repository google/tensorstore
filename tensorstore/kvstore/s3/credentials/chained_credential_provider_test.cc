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

#include <string>

#include "tensorstore/kvstore/s3/credentials/chained_credential_provider.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/test_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::Future;
using ::tensorstore::Result;
using ::tensorstore::internal_kvstore_s3::AwsCredentials;
using ::tensorstore::internal_kvstore_s3::AwsCredentialProvider;
using ::tensorstore::internal_kvstore_s3::ChainedCredentialProvider;

namespace {

class TestCredentialProvider : public AwsCredentialProvider {
 public:
  AwsCredentials credentials_;
  bool expired_;
  bool found_;
  TestCredentialProvider(std::string access_key="")
    : credentials_{access_key}, expired_(false), found_(true) {}

  Result<AwsCredentials> GetCredentials() override {
    if(found_) { expired_ = true; return credentials_; }
    return absl::NotFoundError("Credentials Not Found");
  }
  bool IsExpired() override { return expired_; }
};

TEST(ChainedCredentialProviderTest, EmptyProvider) {
  auto provider = ChainedCredentialProvider({});
  auto credentials = provider.GetCredentials();
  ASSERT_FALSE(credentials.ok());
  ASSERT_TRUE(provider.IsExpired());
}

// Tests that Credential Retrieval results in IsExpiry
// proxying the correct encapsulated provider
TEST(ChainedCredentialProviderTest, ChainedGetAndExpiryLogic) {
  auto one = std::make_unique<TestCredentialProvider>("key1");
  auto two = std::make_unique<TestCredentialProvider>("key2");
  auto one_ptr = one.get();
  auto two_ptr = two.get();

  auto providers = std::vector<std::unique_ptr<AwsCredentialProvider>>();
  providers.emplace_back(std::move(one));
  providers.emplace_back(std::move(two));
  auto provider = ChainedCredentialProvider(std::move(providers));
  ASSERT_EQ(provider.IsExpired(), true);

  // First call for credentials result in one's credentials
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto credentials, provider.GetCredentials());
  ASSERT_EQ(credentials.access_key, "key1");
  ASSERT_EQ(provider.IsExpired(), true);
  ASSERT_EQ(one_ptr->expired_, true);
  ASSERT_EQ(two_ptr->expired_, false);

  // Chained Provider proxies one for IsExpired calls
  one_ptr->expired_ = false;
  ASSERT_EQ(provider.IsExpired(), false);

  // Disable one's credentials and get new credentials (two)
  one_ptr->found_ = false;
  TENSORSTORE_CHECK_OK_AND_ASSIGN(credentials, provider.GetCredentials());
  ASSERT_EQ(credentials.access_key, "key2");
  ASSERT_EQ(provider.IsExpired(), true);
  ASSERT_EQ(one_ptr->expired_, false);
  ASSERT_EQ(two_ptr->expired_, true);

  // Chained Provider proxies two for IsExpired calls
  two_ptr->expired_ = false;
  ASSERT_EQ(provider.IsExpired(), false);

  // Disable two's credentials and get new credentials
  two_ptr->found_ = false;
  auto result = provider.GetCredentials();
  ASSERT_FALSE(result.ok());
  ASSERT_EQ(provider.IsExpired(), true);
}


}  // namespace
