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

#include "tensorstore/kvstore/s3/credentials/default_credential_provider.h"

#include <memory>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/kvstore/s3/credentials/ec2_credential_provider.h"
#include "tensorstore/kvstore/s3/credentials/environment_credential_provider.h"
#include "tensorstore/kvstore/s3/credentials/file_credential_provider.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {
namespace {

/// Return a ChainedCredentialProvider that attempts to retrieve credentials
/// from
/// 1. AWS Environment Variables, e.g. AWS_ACCESS_KEY_ID
/// 2. Shared Credential File, e.g. $HOME/.aws/credentials
/// 3. EC2 Metadata Server
Result<std::unique_ptr<AwsCredentialProvider>> GetDefaultAwsCredentialProvider(
    std::string_view filename,
    std::string_view profile,
    std::string_view metadata_endpoint,
    std::shared_ptr<internal_http::HttpTransport> transport) {
  return std::make_unique<DefaultAwsCredentialsProvider>(
      DefaultAwsCredentialsProvider::Options{
          std::string{filename}, std::string{profile}, std::string{metadata_endpoint}, transport});
}

struct AwsCredentialProviderRegistry {
  std::vector<std::pair<int, AwsCredentialProviderFn>> providers;
  absl::Mutex mutex;
};

AwsCredentialProviderRegistry& GetAwsProviderRegistry() {
  static internal::NoDestructor<AwsCredentialProviderRegistry> registry;
  return *registry;
}

}  // namespace

void RegisterAwsCredentialProviderProvider(AwsCredentialProviderFn provider,
                                           int priority) {
  auto& registry = GetAwsProviderRegistry();
  absl::WriterMutexLock lock(&registry.mutex);
  registry.providers.emplace_back(priority, std::move(provider));
  std::sort(registry.providers.begin(), registry.providers.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
}

/// Obtain a credential provider from a series of registered and default
/// providers
///
/// Providers are returned in the following order:
/// 1. Any registered providers that supply valid credentials
/// 2. Environment variable provider if valid credential can be obtained from
///    AWS_* environment variables
/// 3. File provider containing credentials from the $HOME/.aws/credentials
/// file.
///    The `profile` variable overrides the default profile in this file.
/// 4. EC2 Metadata server. The `transport` variable overrides the default
///    HttpTransport.
Result<std::unique_ptr<AwsCredentialProvider>> GetAwsCredentialProvider(
    std::string_view filename,
    std::string_view profile,
    std::string_view metadata_endpoint,
    std::shared_ptr<internal_http::HttpTransport> transport) {
  auto& registry = GetAwsProviderRegistry();
  absl::WriterMutexLock lock(&registry.mutex);
  for (const auto& provider : registry.providers) {
    auto credentials = provider.second();
    if (credentials.ok()) return credentials;
  }

  return GetDefaultAwsCredentialProvider(filename, profile, metadata_endpoint, transport);
}

DefaultAwsCredentialsProvider::DefaultAwsCredentialsProvider(
    Options options, absl::FunctionRef<absl::Time()> clock)
    : options_(std::move(options)),
      clock_(clock),
      credentials_{{}, {}, {}, absl::InfinitePast()} {}

Result<AwsCredentials> DefaultAwsCredentialsProvider::GetCredentials() {
  {
    absl::ReaderMutexLock lock(&mutex_);
    if (credentials_.expires_at > clock_()) {
      return credentials_;
    }
  }

  absl::WriterMutexLock lock(&mutex_);

  // Refresh existing credentials
  if (provider_) {
    auto credentials_result = provider_->GetCredentials();
    if (credentials_result.ok()) {
      credentials_ = credentials_result.value();
      return credentials_;
    }
  }

  // Return credentials in this order:
  // 1. AWS Environment Variables, e.g. AWS_ACCESS_KEY_ID
  provider_ = std::make_unique<EnvironmentCredentialProvider>();
  auto credentials_result = provider_->GetCredentials();
  if (credentials_result.ok()) {
    credentials_ = credentials_result.value();
    return credentials_;
  }

  // 2. Shared Credential File, e.g. $HOME/.aws/credentials
  provider_ = std::make_unique<FileCredentialProvider>(
                    options_.filename, options_.profile);
  credentials_result = provider_->GetCredentials();
  if (credentials_result.ok()) {
    credentials_ = credentials_result.value();
    return credentials_;
  }

  // 3. EC2 Metadata Server
  provider_ =
      std::make_unique<EC2MetadataCredentialProvider>(
        options_.endpoint, options_.transport);
  credentials_result = provider_->GetCredentials();
  if (credentials_result.ok()) {
    credentials_ = credentials_result.value();
    return credentials_;
  }

  // 4. Anonymous credentials
  provider_ = nullptr;
  credentials_ = AwsCredentials::Anonymous();
  return credentials_;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
