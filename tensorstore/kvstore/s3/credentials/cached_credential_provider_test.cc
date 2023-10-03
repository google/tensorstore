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

#include "tensorstore/kvstore/s3/credentials/cached_credential_provider.h"
#include "tensorstore/kvstore/s3/credentials/expiry_credential_provider.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/test_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::Future;
using ::tensorstore::Result;
using ::tensorstore::internal_kvstore_s3::AwsCredentials;
using ::tensorstore::internal_kvstore_s3::CachedCredentialProvider;
using ::tensorstore::internal_kvstore_s3::ExpiryCredentialProvider;
using ::tensorstore::MatchesStatus;

namespace {

class TestCredentialProvider : public ExpiryCredentialProvider {
 public:
    int iteration = 0;
    absl::FunctionRef<absl::Time()> clock_;

    TestCredentialProvider(absl::FunctionRef<absl::Time()> clock) :
        ExpiryCredentialProvider(clock), clock_(clock)  {}

    Result<AwsCredentials> GetCredentials() override {
        this->SetExpiration(clock_() + absl::Seconds(2));
        return AwsCredentials{"key", std::to_string(++iteration)};
    }
};


TEST(CachedCredentialProviderTest, ExpiringProvider) {
    auto utc = absl::UTCTimeZone();
    auto frozen_time = absl::FromCivil(absl::CivilSecond(2023, 9, 6, 0, 4, 03), utc);
    auto stuck_clock = [&frozen_time]() -> absl::Time { return frozen_time; };

    auto test_provider = std::make_unique<TestCredentialProvider>(stuck_clock);
    auto test_prov_ptr = test_provider.get();
    auto cached_provider = CachedCredentialProvider{std::move(test_provider)};

    // Base case, no credentials have been retrieved yet
    ASSERT_TRUE(cached_provider.IsExpired());
    ASSERT_EQ(cached_provider.ExpiresAt(), absl::InfinitePast());
    ASSERT_EQ(test_prov_ptr->iteration, 0);

    // Retrieve some credentials
    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto credentials, cached_provider.GetCredentials());
    ASSERT_EQ(credentials.access_key, "key");
    ASSERT_EQ(credentials.secret_key, "1");
    ASSERT_EQ(test_prov_ptr->iteration, 1);
    ASSERT_EQ(cached_provider.ExpiresAt(), frozen_time + absl::Seconds(2));
    ASSERT_FALSE(test_prov_ptr->IsExpired());
    ASSERT_FALSE(cached_provider.IsExpired());

    // Idempotent when underlying credentials not expired
    TENSORSTORE_CHECK_OK_AND_ASSIGN(credentials, cached_provider.GetCredentials());
    ASSERT_EQ(credentials.access_key, "key");
    ASSERT_EQ(credentials.secret_key, "1");
    ASSERT_EQ(test_prov_ptr->iteration, 1);
    ASSERT_EQ(cached_provider.ExpiresAt(), frozen_time + absl::Seconds(2));
    ASSERT_FALSE(test_prov_ptr->IsExpired());
    ASSERT_FALSE(cached_provider.IsExpired());

    // Advance clock forward
    frozen_time += absl::Seconds(2.5);
    ASSERT_TRUE(test_prov_ptr->IsExpired());
    ASSERT_TRUE(cached_provider.IsExpired());

    // A new set of credentials is retrieved due to expiry
    TENSORSTORE_CHECK_OK_AND_ASSIGN(credentials, cached_provider.GetCredentials());
    ASSERT_EQ(credentials.access_key, "key");
    ASSERT_EQ(credentials.secret_key, "2");
    ASSERT_EQ(test_prov_ptr->iteration, 2);
    ASSERT_EQ(cached_provider.ExpiresAt(), frozen_time + absl::Seconds(2));
    ASSERT_FALSE(test_prov_ptr->IsExpired());
    ASSERT_FALSE(cached_provider.IsExpired());
}

}  // namespace
