// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_S3_CREDENTIALS_COMMON_H_
#define TENSORSTORE_KVSTORE_S3_CREDENTIALS_COMMON_H_

#include <string_view>

#include "tensorstore/internal/aws/aws_credentials.h"

namespace tensorstore {
namespace internal_aws {

/// Returns a credentials provider that uses the AWS default credentials chain.
AwsCredentialsProvider MakeDefault(std::string_view profile_name_override);

/// Returns a credentials provider that uses the AWS default credentials chain
/// with anonymous credentials as a fallback.
AwsCredentialsProvider MakeDefaultWithAnonymous(
    std::string_view profile_name_override);

/// Returns anonymous credentials.
AwsCredentialsProvider MakeAnonymous();

/// Returns credentials from the environment.
AwsCredentialsProvider MakeEnvironment();

/// Returns credentials from the profile, which may include delegated
/// credentials.
AwsCredentialsProvider MakeProfile(std::string_view profile_name_override,
                                   std::string_view credentials_file_override,
                                   std::string_view config_file_override);

/// Returns credentials from the EC2 Metadata Service (IMDS).
AwsCredentialsProvider MakeImds();

/// Returns credentials from the ECS Role Credentials Service
AwsCredentialsProvider MakeEcsRole(std::string_view endpoint,
                                   std::string_view auth_token_file_path,
                                   std::string_view auth_token);

/// NOTE: Add additional credentials providers such as STS, x509, SSO, Cognito,
/// etc.

/// Returns a credentials provider that caches the credentials.
AwsCredentialsProvider MakeCache(AwsCredentialsProvider provider);

}  // namespace internal_aws
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_CREDENTIALS_COMMON_H_
