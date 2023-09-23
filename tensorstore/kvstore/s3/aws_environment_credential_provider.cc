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

#include "tensorstore/kvstore/s3/aws_environment_credential_provider.h"
#include "tensorstore/internal/env.h"

using ::tensorstore::internal::GetEnv;

namespace tensorstore {
namespace internal_kvstore_s3 {

namespace {

// AWS user identifier
constexpr char kEnvAwsAccessKeyId[] = "AWS_ACCESS_KEY_ID";
// AWS user password
constexpr char kEnvAwsSecretAccessKey[] = "AWS_SECRET_ACCESS_KEY";
// AWS session token
constexpr char kEnvAwsSessionToken[] = "AWS_SESSION_TOKEN";

} // namespace

Result<AwsCredentials> EnvironmentCredentialProvider::GetCredentials() {
// 1. Obtain credentials from environment variables
  if (auto access_key = GetEnv(kEnvAwsAccessKeyId); access_key) {
    ABSL_LOG_FIRST_N(INFO, 1)
        << "Using Environment Variable " << kEnvAwsAccessKeyId;
    AwsCredentials credentials;
    credentials.access_key = *access_key;
    auto secret_key = GetEnv(kEnvAwsSecretAccessKey);

    if (secret_key.has_value()) {
      credentials.secret_key = *secret_key;
    }

    auto session_token = GetEnv(kEnvAwsSessionToken);

    if (session_token.has_value()) {
      credentials.session_token = *session_token;
    }

    return credentials;
  }

  return absl::NotFoundError("AWS Credentials not found in environment variables");
}

} // namespace internal_kvstore_s3
} // namespace tensorstore
