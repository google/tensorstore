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

#ifndef TENSORSTORE_KVSTORE_S3_CREDENTIALS_FILE_CREDENTIAL_PROVIDER_H_
#define TENSORSTORE_KVSTORE_S3_CREDENTIALS_FILE_CREDENTIAL_PROVIDER_H_

#include <string>
#include <string_view>

#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Obtains S3 credentials from a profile in a file, usually
/// `~/.aws/credentials` or a file specified in AWS_SHARED_CREDENTIALS_FILE.
/// A filename or desired profile may be specified in the constructor:
/// These values should be derived from the s3 json spec.
/// However, if filename is passed as an empty string, the filename is
/// obtained from AWS_SHARED_CREDENTIAL_FILE before defaulting to
/// `.aws/credentials`.
///
/// When profile is empty, the profile indicated by
/// AWS_DEFAULT_PROFILE is used, if set, or "default".
class FileCredentialProvider : public AwsCredentialProvider {
 private:
  std::string filename_;
  std::string profile_;

 public:
  FileCredentialProvider(std::string_view filename, std::string_view profile);

  Result<AwsCredentials> GetCredentials() override;
  inline const std::string& GetFileName() const { return filename_; }
  inline const std::string& GetProfile() const { return profile_; }
};

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_CREDENTIALS_FILE_CREDENTIAL_PROVIDER_H_
