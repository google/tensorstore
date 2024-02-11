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

#ifndef TENSORSTORE_KVSTORE_S3_CREDENTIALS_AWS_CREDENTIALS_H_
#define TENSORSTORE_KVSTORE_S3_CREDENTIALS_AWS_CREDENTIALS_H_

#include <string>

#include "absl/time/time.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Holds AWS credentials
///
/// Contains the access key, secret key, session token and expiry
/// An empty access key implies anonymous access,
/// while the presence of a session token implies the use of
/// short-lived STS credentials
/// https://docs.aws.amazon.com/STS/latest/APIReference/welcome.html
struct AwsCredentials {
  /// AWS_ACCESS_KEY_ID
  std::string access_key;
  /// AWS_SECRET_KEY_ID
  std::string secret_key;
  /// AWS_SESSION_TOKEN
  std::string session_token;
  /// Expiration date
  absl::Time expires_at = absl::InfinitePast();

  /// Anonymous credentials that do not expire
  static AwsCredentials Anonymous() {
    return AwsCredentials{{}, {}, {}, absl::InfiniteFuture()};
  }

  bool IsAnonymous() const { return access_key.empty(); }
};

/// Base class for S3 Credential Providers
///
/// Implementers should override GetCredentials.
class AwsCredentialProvider {
 public:
  virtual ~AwsCredentialProvider() = default;
  virtual Result<AwsCredentials> GetCredentials() = 0;
};

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_CREDENTIALS_AWS_CREDENTIALS_H_
