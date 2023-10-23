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

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

namespace {}  // namespace

bool CachedCredentialProvider::IsExpired() {
  absl::ReaderMutexLock lock(&mutex_);
  return IsExpiredLocked(credentials_);
}

bool CachedCredentialProvider::IsExpiredLocked(
    const AwsCredentials& credentials) {
  return credentials.IsAnonymous() || provider_->IsExpired();
}

Result<absl::Time> CachedCredentialProvider::ExpiresAt() {
  absl::ReaderMutexLock lock(&mutex_);
  TENSORSTORE_ASSIGN_OR_RETURN(auto expires, provider_->ExpiresAt());
  if (credentials_.IsAnonymous()) return absl::InfinitePast();
  return expires;
}

Result<AwsCredentials> CachedCredentialProvider::GetCredentials() {
  absl::WriterMutexLock lock(&mutex_);
  if (!IsExpiredLocked(credentials_)) {
    return credentials_;
  }

  TENSORSTORE_ASSIGN_OR_RETURN(credentials_, provider_->GetCredentials());
  return credentials_;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
