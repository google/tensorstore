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

#ifndef TENSORSTORE_KVSTORE_S3_CREDENTIALS_FILE_CREDENTIAL_PROVIDER_H
#define TENSORSTORE_KVSTORE_S3_CREDENTIALS_FILE_CREDENTIAL_PROVIDER_H

#include <string>

#include "tensorstore/kvstore/s3/credentials/aws_credential_provider.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Obtains S3 credentials from a profile in a file, usually
/// `~/.aws/credentials` or a file specified in AWS_SHARED_CREDENTIALS_FILE. A
/// desired profile may be specified in the constructor: This value should be
/// derived from the s3 json spec.
/// However, if profile is passed as an empty string, the profile is obtained
/// from AWS_DEFAULT_PROFILE, AWS_PROFILE before finally defaulting to
/// "default".
class FileCredentialProvider : public AwsCredentialProvider {
 private:
  // desired profile
  std::string profile_;
  bool retrieved_;

 public:
  FileCredentialProvider(std::string profile)
    : profile_(std::move(profile)), retrieved_(false) {}
  Result<AwsCredentials> GetCredentials() override;
  // Shared Credentials never expire once retrieved
  bool IsExpired() override { return !retrieved_; }
  Result<absl::Time> ExpiresAt() override {
    return retrieved_ ? absl::InfiniteFuture() : absl::InfinitePast();
  }
};

} // namespace internal_kvstore_s3
} // namespace tensorstore

#endif // TENSORSTORE_KVSTORE_S3_CREDENTIALS_FILE_CREDENTIAL_PROVIDER_H