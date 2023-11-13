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

#include "tensorstore/kvstore/s3/credentials/aws_credential_provider.h"

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/kvstore/s3/credentials/default_credential_provider.h"
#include "tensorstore/util/result.h"

using ::tensorstore::Result;

namespace tensorstore {
namespace internal_kvstore_s3 {
namespace {

/// Return a ChainedCredentialProvider that attempts to retrieve credentials
/// from
/// 1. AWS Environment Variables, e.g. AWS_ACCESS_KEY_ID
/// 2. Shared Credential File, e.g. $HOME/.aws/credentials
/// 3. EC2 Metadata Server
Result<std::unique_ptr<AwsCredentialProvider>> GetDefaultAwsCredentialProvider(
    std::string_view profile,
    std::shared_ptr<internal_http::HttpTransport> transport) {
  return std::make_unique<DefaultAwsCredentialsProvider>(
      DefaultAwsCredentialsProvider::Options{
          {}, std::string{profile}, transport});
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
    std::string_view profile,
    std::shared_ptr<internal_http::HttpTransport> transport) {
  auto& registry = GetAwsProviderRegistry();
  absl::WriterMutexLock lock(&registry.mutex);
  for (const auto& provider : registry.providers) {
    auto credentials = provider.second();
    if (credentials.ok()) return credentials;
  }

  return GetDefaultAwsCredentialProvider(profile, transport);
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
