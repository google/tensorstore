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

#ifndef TENSORSTORE_KVSTORE_S3_S3_CREDENTIAL_PROVIDER_H_
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

using ::tensorstore::Result;

namespace tensorstore {
namespace internal_auth_s3 {


struct S3Credentials {
  std::string access_key;
  std::string secret_key;
  std::string session_token;

  bool IsAnonymous() const { return !access_key.empty(); }
};

class CredentialProvider {
 public:
  virtual ~CredentialProvider() = default;
  virtual Result<S3Credentials> GetCredentials() = 0;
};

class EnvironmentCredentialProvider : public CredentialProvider {
 private:
  S3Credentials credentials_;
 public:
  EnvironmentCredentialProvider(const S3Credentials & credentials)
    : credentials_(credentials) {}
  virtual Result<S3Credentials> GetCredentials() override
    { return credentials_; }
};

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

} // namespace internal_auth_s3
} // namespace tensorstore

#endif // TENSORSTORE_KVSTORE_S3_S3_CREDENTIAL_PROVIDER_H