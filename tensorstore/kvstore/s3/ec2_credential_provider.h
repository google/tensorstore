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

#ifndef TENSORSTORE_KVSTORE_S3_EC2_CREDENTIAL_PROVIDER_H
#define TENSORSTORE_KVSTORE_S3_EC2_CREDENTIAL_PROVIDER_H

#include "tensorstore/kvstore/s3/aws_credential_provider.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Provides S3 credentials from the EC2 Metadata server
/// if running within AWS
class EC2MetadataCredentialProvider : public AwsCredentialProvider {
 private:
  std::shared_ptr<internal_http::HttpTransport> transport_;

 public:
  EC2MetadataCredentialProvider(
      std::shared_ptr<internal_http::HttpTransport> transport)
      : transport_(std::move(transport)) {}

  Result<AwsCredentials> GetCredentials() override;
};

} // internal_kvstore_s3
} // tensorstore


#endif // TENSORSTORE_KVSTORE_S3_EC2_CREDENTIAL_PROVIDER_H