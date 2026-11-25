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

#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorstore/internal/aws/aws_credentials.h"

namespace tensorstore {
namespace internal_aws {

/// Returns a credentials provider that uses the AWS default credentials chain.
AwsCredentialsProvider MakeDefault(std::string_view profile_name_override);

/// Returns a credentials provider that uses the AWS default credentials chain,
/// falling back to anonymous credentials if initial resolution fails.
/// Whichever source (default chain or anonymous) is selected on the first
/// resolution is latched for all subsequent requests. Note that a transient
/// startup failure in the default chain latches anonymous credentials for the
/// lifetime of this provider.
AwsCredentialsProvider MakeDefaultWithAnonymous(
    std::string_view profile_name_override);

/// Returns a credentials provider that queries non-null `providers` in order
/// (coalescing concurrent callers into a single resolution) and latches onto
/// the first provider that resolves credentials; all subsequent requests
/// delegate exclusively to that provider. Callbacks run on the completing
/// candidate's thread (or synchronously on the caller's thread). If every
/// provider fails on an attempt, the last provider's error is returned without
/// latching.
AwsCredentialsProvider MakeFirstResolved(
    absl::Span<const AwsCredentialsProvider> providers);

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

/// Returns a credentials provider that caches resolved credentials, refreshing
/// expiring credentials 5 minutes before expiration and non-expiring
/// credentials after `refresh_time` (or never when `absl::InfiniteDuration()`).
/// Returns `provider` unchanged if `provider` is null or
/// `refresh_time <= absl::ZeroDuration()`.
AwsCredentialsProvider MakeCache(
    AwsCredentialsProvider provider,
    absl::Duration refresh_time = absl::InfiniteDuration());

}  // namespace internal_aws
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_CREDENTIALS_COMMON_H_
