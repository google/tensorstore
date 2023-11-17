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

#ifndef TENSORSTORE_KVSTORE_S3_CREDENTIALS_DEFAULT_CREDENTIAL_PROVIDER_H_
#define TENSORSTORE_KVSTORE_S3_CREDENTIALS_DEFAULT_CREDENTIAL_PROVIDER_H_

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "absl/base/thread_annotations.h"
#include "absl/functional/function_ref.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// A provider that implements a Default strategy for retrieving
/// and caching a set of AWS credentials from the following sources:
///
/// 1. Environment variables such as AWS_ACCESS_KEY_ID
/// 2. Shared Credential Files such as ~/.aws/credentials
/// 3. The EC2 Metadata server
///
/// The cached credentials are returned until they expire,
/// at which point the original source is queried again to
/// obtain fresher credentials
class DefaultAwsCredentialsProvider : public AwsCredentialProvider {
 public:
  /// Options to configure the provider. These include the:
  ///
  /// 1. Shared Credential Filename
  /// 2. Shared Credential File Profile
  /// 3. EC2 Metadata Server Endpoint
  /// 3. Http Transport for querying the EC2 Metadata Server
  struct Options {
    std::string filename;
    std::string profile;
    std::string endpoint;
    std::shared_ptr<internal_http::HttpTransport> transport;
  };

  DefaultAwsCredentialsProvider(
      Options options = {{}, {}, {}, internal_http::GetDefaultHttpTransport()},
      absl::FunctionRef<absl::Time()> clock = absl::Now);
  Result<AwsCredentials> GetCredentials() override;

 private:
  Options options_;
  absl::FunctionRef<absl::Time()> clock_;
  absl::Mutex mutex_;
  std::unique_ptr<AwsCredentialProvider> provider_ ABSL_GUARDED_BY(mutex_);
  AwsCredentials credentials_ ABSL_GUARDED_BY(mutex_);
};

using AwsCredentialProviderFn =
    std::function<Result<std::unique_ptr<AwsCredentialProvider>>()>;

void RegisterAwsCredentialProviderProvider(AwsCredentialProviderFn provider,
                                           int priority);

Result<std::unique_ptr<AwsCredentialProvider>> GetAwsCredentialProvider(
    std::string_view filename, std::string_view profile,
    std::string_view metadata_endpoint,
    std::shared_ptr<internal_http::HttpTransport> transport);

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_CREDENTIALS_DEFAULT_CREDENTIAL_PROVIDER_H_
