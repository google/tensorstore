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

#ifndef TENSORSTORE_KVSTORE_S3_CREDENTIALS_AWS_CREDENTIALS_H_
#define TENSORSTORE_KVSTORE_S3_CREDENTIALS_AWS_CREDENTIALS_H_

#include <string_view>

#include "absl/time/time.h"
#include <aws/auth/credentials.h>
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_aws {

/// IntrusivePtr traits for aws_credentials_provider.
struct AwsCredentialsProviderTraits {
  template <typename U>
  using pointer = U *;
  static void increment(aws_credentials_provider *p) noexcept {
    aws_credentials_provider_acquire(p);
  }
  static void decrement(aws_credentials_provider *p) noexcept {
    aws_credentials_provider_release(p);
  }
};

/// Wrapper for an aws_credentials_provider.
using AwsCredentialsProvider =
    internal::IntrusivePtr<aws_credentials_provider,
                           AwsCredentialsProviderTraits>;

/// IntrusivePtr traits for aws_credentials.
struct AwsCredentialsTraits {
  template <typename U>
  using pointer = U *;
  static void increment(aws_credentials *p) noexcept {
    aws_credentials_acquire(p);
  }
  static void decrement(aws_credentials *p) noexcept {
    aws_credentials_release(p);
  }
};

/// Wrapper for an aws_credentials.
///
/// Contains the access key, secret key, session token and expiry.
/// An empty access key implies anonymous access, while the presence of a
/// session token implies the use of short-lived STS credentials
/// https://docs.aws.amazon.com/STS/latest/APIReference/welcome.html
class AwsCredentials
    : public internal::IntrusivePtr<aws_credentials, AwsCredentialsTraits> {
  using Base = internal::IntrusivePtr<aws_credentials, AwsCredentialsTraits>;

 public:
  using Base::Base;

  static AwsCredentials Make(std::string_view access_key_id,
                             std::string_view secret_access_key,
                             std::string_view session_token,
                             absl::Time expiration = absl::InfiniteFuture());

  std::string_view GetAccessKeyId() const;
  std::string_view GetSecretAccessKey() const;
  std::string_view GetSessionToken() const;
  absl::Time GetExpiration() const;

  bool IsAnonymous() const;

  using Base::get;
};

/// Retrieves AWS credentials from an AWS credentials provider.
Future<AwsCredentials> GetAwsCredentials(aws_credentials_provider *provider);

}  // namespace internal_aws
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_CREDENTIALS_AWS_CREDENTIALS_H_
