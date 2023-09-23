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
#include <string_view>

#include "absl/time/time.h"
#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Base class for S3 Credential Providers
///
/// Implementers should override GetCredentials, IsExpired and,
/// if the credential source supports specifying an expiry date, ExpiresAt.
class AwsCredentialProvider {
 public:
  virtual ~AwsCredentialProvider() = default;
  virtual Result<AwsCredentials> GetCredentials() = 0;
  virtual Result<absl::Time> ExpiresAt() {
    return absl::UnimplementedError("AwsCredentialProvider::ExpiresAt");
  };
  virtual bool IsExpired() = 0;
};

/// Provides anonymous credentials
class StaticCredentialProvider : public AwsCredentialProvider {
  private:
    AwsCredentials credentials_;
  public:
    StaticCredentialProvider(std::string_view access_key="",
                             std::string_view secret_key="",
                             std::string_view session_token="")
      : credentials_{std::string{access_key},
                     std::string{secret_key},
                     std::string{session_token}} {}

    Result<AwsCredentials> GetCredentials() override { return credentials_; }
    bool IsExpired() override { return false; }
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
