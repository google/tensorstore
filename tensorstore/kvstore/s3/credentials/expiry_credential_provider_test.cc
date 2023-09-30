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

#include "tensorstore/kvstore/s3/credentials/expiry_credential_provider.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/test_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::Future;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::internal_kvstore_s3::AwsCredentials;
using ::tensorstore::internal_kvstore_s3::ExpiryCredentialProvider;

namespace {

class TestCredentialProvider : public ExpiryCredentialProvider {
 public:
  TestCredentialProvider(const absl::FunctionRef<absl::Time()> & clock=absl::Now)
    : ExpiryCredentialProvider(clock) {}

  Result<AwsCredentials> GetCredentials() override {
    return AwsCredentials{};
  }
};

TEST(ExpiryCredentialProviderTest, TestExpiry) {
  /// Configure provider with a stuck clock
  auto provider = TestCredentialProvider{[]() {
    return absl::FromCivil(absl::CivilSecond(2023, 9, 6, 0, 4, 03), absl::UTCTimeZone());
  }};
  ASSERT_TRUE(provider.IsExpired());
  EXPECT_THAT(provider.ExpiresAt(), absl::InfinitePast());

  auto utc = absl::UTCTimeZone();
  auto zero_sec = absl::Seconds(0);
  auto one_sec = absl::Seconds(1);

  /// One second before clock
  provider.SetExpiration(absl::FromCivil(absl::CivilSecond(2023, 9, 6, 0, 4, 02), utc), zero_sec);
  ASSERT_TRUE(provider.IsExpired());

  // Matches clock exactly
  provider.SetExpiration(absl::FromCivil(absl::CivilSecond(2023, 9, 6, 0, 4, 03), utc), zero_sec);
  ASSERT_FALSE(provider.IsExpired());

  // One second after clock
  provider.SetExpiration(absl::FromCivil(absl::CivilSecond(2023, 9, 6, 0, 4, 04), utc), zero_sec);
  ASSERT_FALSE(provider.IsExpired());

  /// Test expiry with duration window
  // One second before clock
  provider.SetExpiration(absl::FromCivil(absl::CivilSecond(2023, 9, 6, 0, 4, 03), utc), one_sec);
  ASSERT_TRUE(provider.IsExpired());

  // Matches clock exactly
  provider.SetExpiration(absl::FromCivil(absl::CivilSecond(2023, 9, 6, 0, 4, 04), utc), one_sec);
  ASSERT_FALSE(provider.IsExpired());

  // One second after clock
  provider.SetExpiration(absl::FromCivil(absl::CivilSecond(2023, 9, 6, 0, 4, 05), utc), one_sec);
  ASSERT_FALSE(provider.IsExpired());
}


}  // namespace
