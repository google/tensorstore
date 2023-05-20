// Copyright 2020 The TensorStore Authors
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

#include <string>
#include <string_view>
#include <map>

#include "tensorstore/util/result.h"

using ::tensorstore::Result;

namespace tensorstore {
namespace internal_storage_s3 {


class S3Credentials {
private:
 std::string access_key_;
 std::string secret_key_;
 std::string session_token_;

public:
  explicit S3Credentials() {}
  explicit S3Credentials(std::string_view access_key, std::string_view secret_key, std::string_view session_token)
    : access_key_(access_key), secret_key_(secret_key), session_token_(session_token)
    {}

  void SetAccessKey(std::string_view access_key) { access_key_ = access_key; }
  void SetSecretKey(std::string_view secret_key) { secret_key_ = secret_key; }
  void SetSessionToken(std::string_view session_token) { session_token_ = session_token; }

  const std::string & GetAccessKey() const { return access_key_; }
  const std::string & GetSecretKey() const { return secret_key_; }
  const std::string & GetSessionToken() const { return session_token_; }

  Result<S3Credentials> MakeResult() {
    if(!access_key_.empty() && !secret_key_.empty()) {
        return std::move(*this);
    } else if(access_key_.empty()) {
        return absl::InvalidArgumentError("No access key was found");
    } else if (secret_key_.empty()) {
        return absl::InvalidArgumentError("No secret key was found");
    }

    return absl::InternalError("S3Credentials in invalid state");
}
};

struct S3CredentialContext {
  std::string profile_;
};

class S3CredentialSource {
public:
 virtual Result<S3Credentials> GetCredentials(const S3CredentialContext & context) = 0;
 virtual std::string Provenance() const = 0;
};

class EnvironmentCredentialSource : public S3CredentialSource {
public:
 Result<S3Credentials> GetCredentials(const S3CredentialContext & context) override;
 std::string Provenance() const override { return "Environment Variables"; }
};

class FileCredentialSource : public S3CredentialSource {
public:
 Result<S3Credentials> GetCredentials(const S3CredentialContext & context) override;
 std::string Provenance() const override { return "Credentials File"; }
};

class EC2MetadataCredentialSource : public S3CredentialSource {
public:
 Result<S3Credentials> GetCredentials(const S3CredentialContext & context) override
    { return S3Credentials().MakeResult(); }
 std::string Provenance() const override { return "EC2 Metadata Server"; }
};

class S3CredentialProvider {
private:
  std::vector<std::unique_ptr<S3CredentialSource>> sources_;
  S3CredentialContext context_;

public:
  void SetProfile(std::string_view profile) { context_.profile_ = profile; }

  S3CredentialProvider() {}
  static S3CredentialProvider DefaultS3CredentialProvider() {
    auto provider = S3CredentialProvider();
    provider.sources_.emplace_back(std::make_unique<EnvironmentCredentialSource>());
    provider.sources_.emplace_back(std::make_unique<FileCredentialSource>());
    provider.sources_.emplace_back(std::make_unique<EC2MetadataCredentialSource>());
    return provider;
  }

  Result<S3Credentials> GetCredentials() const;
};

Result<S3Credentials> GetS3Credentials(const std::string & profile="");

}
}

#endif
