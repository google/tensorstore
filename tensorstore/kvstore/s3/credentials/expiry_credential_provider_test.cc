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

#include "tensorstore/kvstore/s3/credentials/expiry_credential_provider.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

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
  TestCredentialProvider(
      const absl::FunctionRef<absl::Time()>& clock = absl::Now)
      : ExpiryCredentialProvider(clock) {}

  Result<AwsCredentials> GetCredentials() override { return AwsCredentials{}; }
};

TEST(ExpiryCredentialProviderTest, TestExpiry) {
  auto utc = absl::UTCTimeZone();

  // window durations
  auto window_zero = absl::Seconds(0);
  auto window_one = absl::Seconds(1);

  // Timestamps
  auto two_seconds =
      absl::FromCivil(absl::CivilSecond(2023, 9, 6, 0, 4, 02), utc);
  auto three_seconds =
      absl::FromCivil(absl::CivilSecond(2023, 9, 6, 0, 4, 03), utc);
  auto four_seconds =
      absl::FromCivil(absl::CivilSecond(2023, 9, 6, 0, 4, 04), utc);
  auto five_seconds =
      absl::FromCivil(absl::CivilSecond(2023, 9, 6, 0, 4, 05), utc);

  auto frozen_time = three_seconds;
  auto stuck_clock = [&frozen_time]() -> absl::Time { return frozen_time; };

  /// Configure provider with a stuck clock
  auto provider = TestCredentialProvider{stuck_clock};
  ASSERT_TRUE(provider.IsExpired());
  ASSERT_EQ(provider.ExpiresAt(), absl::InfinitePast());

  /// One second before clock
  provider.SetExpiration(two_seconds, window_zero);
  ASSERT_TRUE(provider.IsExpired());
  ASSERT_EQ(provider.ExpiresAt(), two_seconds);

  // Matches clock exactly
  provider.SetExpiration(three_seconds, window_zero);
  ASSERT_FALSE(provider.IsExpired());
  ASSERT_EQ(provider.ExpiresAt(), three_seconds);

  // One second after clock
  provider.SetExpiration(four_seconds, window_zero);
  ASSERT_FALSE(provider.IsExpired());
  ASSERT_EQ(provider.ExpiresAt(), four_seconds);

  /// Test expiry with duration window
  // One second before clock
  provider.SetExpiration(three_seconds, window_one);
  ASSERT_TRUE(provider.IsExpired());
  ASSERT_EQ(provider.ExpiresAt(), two_seconds);

  // Matches clock exactly
  provider.SetExpiration(four_seconds, window_one);
  ASSERT_FALSE(provider.IsExpired());
  ASSERT_EQ(provider.ExpiresAt(), three_seconds);

  // One second after clock
  provider.SetExpiration(five_seconds, window_one);
  ASSERT_FALSE(provider.IsExpired());
  ASSERT_EQ(provider.ExpiresAt(), four_seconds);
}

}  // namespace
