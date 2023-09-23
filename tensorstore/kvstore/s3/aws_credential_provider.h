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

#ifndef TENSORSTORE_KVSTORE_S3_AWS_CREDENTIAL_PROVIDER
#define TENSORSTORE_KVSTORE_S3_AWS_CREDENTIAL_PROVIDER

#include <functional>
#include <memory>
#include <string_view>
#include <utility>

#include "tensorstore/kvstore/s3/aws_credentials.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Base class for S3 Credential Providers
///
/// Implementers should override GetCredentials
class AwsCredentialProvider {
 public:
  virtual ~AwsCredentialProvider() = default;
  virtual Result<AwsCredentials> GetCredentials() = 0;
};

/// Provides anonymous credentials
class AnonymousCredentialProvider : public AwsCredentialProvider {
  public:
    Result<AwsCredentials> GetCredentials() override {
      return AwsCredentials{};
    }
};

using AwsCredentialProviderFn =
    std::function<Result<std::unique_ptr<AwsCredentialProvider>>()>;

void RegisterAwsCredentialProviderProvider(AwsCredentialProviderFn provider,
                                           int priority);

Result<std::unique_ptr<AwsCredentialProvider>> GetAwsCredentialProvider(
    std::string_view profile,
    std::shared_ptr<internal_http::HttpTransport> transport);

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_AWS_CREDENTIAL_PROVIDER
