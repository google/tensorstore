// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_OAUTH2_REFRESHABLE_AUTH_PROVIDER_H_
#define TENSORSTORE_INTERNAL_OAUTH2_REFRESHABLE_AUTH_PROVIDER_H_

#include <functional>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/oauth2/auth_provider.h"
#include "tensorstore/internal/oauth2/bearer_token.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_oauth2 {

/// Base class for auth providers that support refreshing.
class RefreshableAuthProvider : public AuthProvider {
 public:
  explicit RefreshableAuthProvider(std::function<absl::Time()> clock = {});

  /// Returns the short-term authentication bearer token.
  ///
  /// Safe for concurrent use by multiple threads.
  Result<BearerTokenWithExpiration> GetToken()
      ABSL_LOCKS_EXCLUDED(mutex_) override;

  /// Checks if the token is valid.
  bool IsValid() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(&mutex_);
    return IsValidInternal();
  }

  /// Checks if the token is expired.
  bool IsExpired() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(&mutex_);
    return IsExpiredInternal();
  }

 protected:
  // Generate a new BearerTokenWithExpiration.
  // Guaranteed to be called under lock.
  virtual Result<BearerTokenWithExpiration> Refresh()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) = 0;

  bool IsExpiredInternal() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return clock_() > (token_.expiration - kExpirationMargin);
  }

  bool IsValidInternal() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return !token_.token.empty() && !IsExpiredInternal();
  }

  absl::Time GetCurrentTime() { return clock_(); }

 private:
  std::function<absl::Time()> clock_;  // mock time.

  absl::Mutex mutex_;
  BearerTokenWithExpiration token_ ABSL_GUARDED_BY(mutex_) = {
      {}, absl::InfinitePast()};
};

}  // namespace internal_oauth2
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OAUTH2_REFRESHABLE_AUTH_PROVIDER_H_
