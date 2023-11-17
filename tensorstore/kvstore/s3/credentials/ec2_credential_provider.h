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

#ifndef TENSORSTORE_KVSTORE_S3_CREDENTIALS_EC2_CREDENTIAL_PROVIDER_H_
#define TENSORSTORE_KVSTORE_S3_CREDENTIALS_EC2_CREDENTIAL_PROVIDER_H_

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Provide S3 credentials from the EC2 Metadata server
class EC2MetadataCredentialProvider : public AwsCredentialProvider {
 public:
  EC2MetadataCredentialProvider(
      std::string_view endpoint,
      std::shared_ptr<internal_http::HttpTransport> transport)
      : endpoint_(endpoint), transport_(std::move(transport)) {}

  Result<AwsCredentials> GetCredentials() override;
  inline const std::string& GetEndpoint() const { return endpoint_; }

 private:
  std::string endpoint_;
  std::shared_ptr<internal_http::HttpTransport> transport_;
};

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_CREDENTIALS_EC2_CREDENTIAL_PROVIDER_H_
