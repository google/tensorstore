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

#include "tensorstore/kvstore/s3/credentials/default_credential_provider.h"

#include <memory>
#include <string>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/kvstore/s3/credentials/ec2_credential_provider.h"
#include "tensorstore/kvstore/s3/credentials/environment_credential_provider.h"
#include "tensorstore/kvstore/s3/credentials/file_credential_provider.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

DefaultAwsCredentialsProvider::DefaultAwsCredentialsProvider(
    Options options, absl::FunctionRef<absl::Time()> clock)
    : options_(std::move(options)),
      clock_(clock),
      credentials_{{}, {}, {}, absl::InfinitePast()} {}

Result<AwsCredentials> DefaultAwsCredentialsProvider::GetCredentials() {
  {
    absl::ReaderMutexLock lock(&mutex_);
    if (credentials_.expires_at > clock_()) {
      return credentials_;
    }
  }

  absl::WriterMutexLock lock(&mutex_);

  // Refresh existing credentials
  if (provider_) {
    auto credentials_result = provider_->GetCredentials();
    if (credentials_result.ok()) {
      credentials_ = credentials_result.value();
      return credentials_;
    }
  }

  // Return credentials in this order:
  // 1. AWS Environment Variables, e.g. AWS_ACCESS_KEY_ID
  provider_ = std::make_unique<EnvironmentCredentialProvider>();
  auto credentials_result = provider_->GetCredentials();
  if (credentials_result.ok()) {
    credentials_ = credentials_result.value();
    return credentials_;
  }

  // 2. Shared Credential File, e.g. $HOME/.aws/credentials
  provider_ = std::make_unique<FileCredentialProvider>(options_.filename,
                                                       options_.profile);
  credentials_result = provider_->GetCredentials();
  if (credentials_result.ok()) {
    credentials_ = credentials_result.value();
    return credentials_;
  }

  // 3. EC2 Metadata Server
  provider_ =
      std::make_unique<EC2MetadataCredentialProvider>(options_.transport);
  credentials_result = provider_->GetCredentials();
  if (credentials_result.ok()) {
    credentials_ = credentials_result.value();
    return credentials_;
  }

  // 4. Anonymous credentials
  provider_ = nullptr;
  credentials_ = AwsCredentials::Anonymous();
  return credentials_;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
