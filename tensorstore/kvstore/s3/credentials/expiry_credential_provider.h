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

#ifndef TENSORSTORE_KVSTORE_S3_EXPIRY_CREDENTIAL_PROVIDER_H
#define TENSORSTORE_KVSTORE_S3_EXPIRY_CREDENTIAL_PROVIDER_H

#include "absl/functional/function_ref.h"
#include "absl/time/time.h"
#include "tensorstore/util/result.h"
#include "tensorstore/kvstore/s3/credentials/aws_credential_provider.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Abstract class providing credentials with a time-based expiry.
/// Implementers overriding this class should call SetExpiration
/// when overriding the GetCredentials function.
/// In addition to an expiry date, SetExpiration can also be supplied
/// with a duration argument introduces jitter into the expiry to prevent
/// retrieval of expired credentials.
/// Constructed by default with absl::Now as a clock function,
/// but can be supplied with a custom clock for testing purposes.
class ExpiryCredentialProvider : public AwsCredentialProvider {
  private:
    absl::Time expiration_;
    absl::FunctionRef<absl::Time()> clock_;

  protected:
    absl::Time clock() { return clock_(); }

  public:
    ExpiryCredentialProvider(absl::FunctionRef<absl::Time()> clock=absl::Now)
        : expiration_(absl::InfinitePast()), clock_(std::move(clock)) {}
    void SetExpiration(const absl::Time & expiration,
                       const absl::Duration & window=absl::Seconds(0)) {
      expiration_ = expiration;
      if(window > absl::Seconds(0)) {
        expiration_ -= window;
      }
    }
    virtual bool IsExpired() override { return expiration_ < clock_(); };
    virtual Result<absl::Time> ExpiresAt() override { return expiration_; };
};

} // namespace internal_kvstore_s3
} // namespace tenstorstore

#endif // TENSORSTORE_KVSTORE_S3_EXPIRY_CREDENTIAL_PROVIDER_H