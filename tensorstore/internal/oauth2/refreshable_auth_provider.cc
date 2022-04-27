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

#include "tensorstore/internal/oauth2/refreshable_auth_provider.h"

#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_oauth2 {

RefreshableAuthProvider::RefreshableAuthProvider(
    std::function<absl::Time()> clock)
    : clock_(clock ? std::move(clock) : &absl::Now) {}

Result<AuthProvider::BearerTokenWithExpiration>
RefreshableAuthProvider::GetToken() {
  absl::MutexLock lock(&mutex_);
  if (!IsValidInternal()) {
    auto status = Refresh();
    TENSORSTORE_RETURN_IF_ERROR(status);
  }
  return BearerTokenWithExpiration{access_token_, expiration_};
}

}  // namespace internal_oauth2
}  // namespace tensorstore
