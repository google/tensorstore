// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/internal/grpc/clientauth/access_token_cache.h"

#include <atomic>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/grpc/clientauth/access_token.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Future;
using ::tensorstore::IsOkAndHolds;
using ::tensorstore::MakeReadyFuture;
using ::tensorstore::PromiseFuturePair;
using ::tensorstore::internal_grpc::AccessToken;
using ::tensorstore::internal_grpc::AccessTokenCache;

TEST(AccessTokenCacheTest, CachingAndRefresh) {
  absl::Time t0 = absl::Now();
  std::atomic<int> calls{0};
  auto cache = AccessTokenCache::Create([&]() -> Future<AccessToken> {
    int c = ++calls;
    if (c == 1) {
      return MakeReadyFuture<AccessToken>(
          AccessToken{"token1", t0 + absl::Minutes(10)});
    }
    return MakeReadyFuture<AccessToken>(
        AccessToken{"token2", t0 + absl::Minutes(20)});
  });

  // Initial fetch triggers refresh.
  EXPECT_THAT(cache->GetAccessToken(t0),
              IsOkAndHolds(AccessToken{"token1", t0 + absl::Minutes(10)}));
  EXPECT_EQ(1, calls.load());

  // Cached hit outside refresh slack (> 5 minutes before expiration).
  EXPECT_THAT(cache->GetAccessToken(t0 + absl::Minutes(4)),
              IsOkAndHolds(AccessToken{"token1", t0 + absl::Minutes(10)}));
  EXPECT_EQ(1, calls.load());

  // Within refresh slack (<= 5 minutes before expiration), returns cached
  // token while triggering background refresh.
  EXPECT_THAT(cache->GetAccessToken(t0 + absl::Minutes(6)),
              IsOkAndHolds(AccessToken{"token1", t0 + absl::Minutes(10)}));
  EXPECT_EQ(2, calls.load());

  // Subsequent request returns refreshed token2.
  EXPECT_THAT(cache->GetAccessToken(t0 + absl::Minutes(11)),
              IsOkAndHolds(AccessToken{"token2", t0 + absl::Minutes(20)}));
  EXPECT_EQ(2, calls.load());
}

TEST(AccessTokenCacheTest, DeduplicatesPendingRequests) {
  absl::Time t0 = absl::Now();
  std::atomic<int> calls{0};
  auto pair = PromiseFuturePair<AccessToken>::Make();
  auto cache = AccessTokenCache::Create([&]() -> Future<AccessToken> {
    ++calls;
    return pair.future;
  });

  auto f1 = cache->AsyncGetAccessToken(t0);
  auto f2 = cache->AsyncGetAccessToken(t0);
  EXPECT_EQ(1, calls.load());
  EXPECT_FALSE(f1.ready());
  EXPECT_FALSE(f2.ready());

  pair.promise.SetResult(AccessToken{"token_async", t0 + absl::Minutes(10)});
  EXPECT_THAT(f1.result(),
              IsOkAndHolds(AccessToken{"token_async", t0 + absl::Minutes(10)}));
  EXPECT_THAT(f2.result(),
              IsOkAndHolds(AccessToken{"token_async", t0 + absl::Minutes(10)}));
}

}  // namespace
