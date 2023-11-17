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

#ifndef TENSORSTORE_KVSTORE_S3_CREDENTIALS_ENVIRONMENT_CREDENTIAL_PROVIDER_H_
#define TENSORSTORE_KVSTORE_S3_CREDENTIALS_ENVIRONMENT_CREDENTIAL_PROVIDER_H_

#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Provides credentials from the following environment variables:
/// AWS_ACCESS_KEY_ID, AWS_SECRET_KEY_ID, AWS_SESSION_TOKEN
class EnvironmentCredentialProvider : public AwsCredentialProvider {
 public:
  Result<AwsCredentials> GetCredentials() override;
};

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_CREDENTIALS_ENVIRONMENT_CREDENTIAL_PROVIDER_H_
