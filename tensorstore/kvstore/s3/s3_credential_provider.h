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

#ifndef TENSORSTORE_KVSTORE_S3_S3_CREDENTIAL_PROVIDER_H
#define TENSORSTORE_KVSTORE_S3_S3_CREDENTIAL_PROVIDER_H

#include <functional>
#include <string>
#include <string_view>
#include <map>
#include <memory>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Holds S3 credentials
///
/// Contains the access key, secret key and session token.
/// An empty access key implies anonymous access,
/// while the presence of a session token implies the use of
/// short-lived STS credentials
/// https://docs.aws.amazon.com/STS/latest/APIReference/welcome.html
struct S3Credentials {
  /// AWS_ACCESS_KEY_ID
  std::string access_key;
  /// AWS_SECRET_KEY_ID
  std::string secret_key;
  /// AWS_SESSION_TOKEN
  std::string session_token;

  bool IsAnonymous() const { return access_key.empty(); }
};

/// Base class for S3 Credential Providers
///
/// Implementers should override GetCredentials
class CredentialProvider {
 public:
  virtual ~CredentialProvider() = default;
  virtual Result<S3Credentials> GetCredentials() = 0;
};

/// Provides credentials from the following environment variables:
/// AWS_ACCESS_KEY_ID, AWS_SECRET_KEY_ID, AWS_SESSION_TOKEN
class EnvironmentCredentialProvider : public CredentialProvider {
 private:
  S3Credentials credentials_;
 public:
  EnvironmentCredentialProvider(const S3Credentials & credentials)
    : credentials_(credentials) {}
  virtual Result<S3Credentials> GetCredentials() override
    { return credentials_; }
};

/// Obtains S3 credentials from a profile in a file, usually `~/.aws/credentials`
/// or a file specified in AWS_SHARED_CREDENTIALS_FILE.
/// A desired profile may be specified in the constructor: This value should be
/// derived from the s3 json spec.
/// However, if profile is passed as an empty string, the profile is obtained from
/// AWS_DEFAULT_PROFILE, AWS_PROFILE before finally defaulting to "default".
class FileCredentialProvider : public CredentialProvider {
 private:
  absl::Mutex mutex_;
  std::string filename_;
  std::string profile_;
 public:
  FileCredentialProvider(std::string_view filename, std::string_view profile) :
    filename_(filename), profile_(profile) {}
  virtual Result<S3Credentials> GetCredentials() override;
};

/// Provides S3 credentials from the EC2 Metadata server
/// if running within AWS
class EC2MetadataCredentialProvider : public CredentialProvider {
 private:
  std::shared_ptr<internal_http::HttpTransport> transport_;
 public:
  EC2MetadataCredentialProvider(std::shared_ptr<internal_http::HttpTransport> transport)
    : transport_(transport) {}
  virtual Result<S3Credentials> GetCredentials() override
    { return absl::UnimplementedError("EC2 Metadata Server"); }
};

using S3CredentialProvider =
    std::function<Result<std::unique_ptr<CredentialProvider>>()>;

void RegisterS3CredentialProviderProvider(S3CredentialProvider provider, int priority);

Result<std::unique_ptr<CredentialProvider>> GetS3CredentialProvider(
    std::string_view profile="",
    std::shared_ptr<internal_http::HttpTransport> transport =
        internal_http::GetDefaultHttpTransport());

} // namespace internal_kvstore_s3
} // namespace tensorstore

#endif // TENSORSTORE_KVSTORE_S3_S3_CREDENTIAL_PROVIDER_H
