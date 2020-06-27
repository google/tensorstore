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

#include "tensorstore/internal/oauth2/google_auth_provider.h"

#include <fstream>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/internal/oauth2/fixed_token_auth_provider.h"
#include "tensorstore/internal/oauth2/gce_auth_provider.h"
#include "tensorstore/internal/oauth2/google_service_account_auth_provider.h"
#include "tensorstore/internal/oauth2/oauth2_auth_provider.h"
#include "tensorstore/internal/oauth2/oauth_utils.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/retry.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

using tensorstore::internal::GetEnv;
using tensorstore::internal::JoinPath;
using tensorstore::internal_http::HttpRequestBuilder;

namespace tensorstore {
namespace internal_oauth2 {
namespace {

// The environment variable to override token generation for testing.
constexpr char kGoogleAuthTokenForTesting[] = "GOOGLE_AUTH_TOKEN_FOR_TESTING";

// The environment variable pointing to the file with local
// Application Default Credentials.
constexpr char kGoogleApplicationCredentials[] =
    "GOOGLE_APPLICATION_CREDENTIALS";

// The environment variable which can override '~/.config/gcloud' if set.
constexpr char kCloudSdkConfig[] = "CLOUDSDK_CONFIG";

// The default path to the gcloud config folder, relative to the home folder.
constexpr char kGCloudConfigFolder[] = ".config/gcloud/";

// The name of the well-known credentials JSON file in the gcloud config folder.
constexpr char kWellKnownCredentialsFile[] =
    "application_default_credentials.json";

// The Google OAuth2 URL to retrieve the auth bearer token via OAuth with a
// refresh token.
constexpr char kOAuthV3Url[] = "https://www.googleapis.com/oauth2/v3/token";

/// Returns whether the given path points to a readable file.
bool IsFile(const std::string& filename) {
  std::ifstream fstream(filename.c_str());
  return fstream.good();
}

/// Returns whether this is running on GCE.
bool IsRunningOnGce(internal_http::HttpTransport* transport) {
  HttpRequestBuilder request_builder(
      JoinPath("http://", GceMetadataHostname()));

  request_builder.AddHeader("Metadata-Flavor: Google");
  auto request = request_builder.BuildRequest();

  const auto issue_request = [&request, transport]() -> Status {
    TENSORSTORE_ASSIGN_OR_RETURN(auto response,
                                 transport->IssueRequest(request, {}).result());
    return internal_http::HttpResponseCodeToStatus(response);
  };
  auto status = internal::RetryWithBackoff(
      issue_request, 3, absl::Milliseconds(10), absl::Seconds(1));

  return status.ok();
}

/// Returns the credentials file name from the env variable.
Result<std::string> GetEnvironmentVariableFileName() {
  auto env = GetEnv(kGoogleApplicationCredentials);
  if (!env || !IsFile(*env)) {
    return absl::NotFoundError(absl::StrCat("$", kGoogleApplicationCredentials,
                                            " is not set or corrupt."));
  }
  return *env;
}

/// Returns the well known file produced by command 'gcloud auth login'.
Result<std::string> GetWellKnownFileName() {
  std::string result;

  auto config_dir_override = GetEnv(kCloudSdkConfig);
  if (config_dir_override) {
    result = JoinPath(*config_dir_override, kWellKnownCredentialsFile);
  } else {
    // Determine the home dir path.
    auto home_dir = GetEnv("HOME");
    if (!home_dir) {
      // failed_precondition?
      return absl::NotFoundError("Could not read $HOME.");
    }
    result =
        JoinPath(*home_dir, kGCloudConfigFolder, kWellKnownCredentialsFile);
  }
  if (!IsFile(result)) {
    return absl::NotFoundError(
        absl::StrCat("Could not find the credentials file in the "
                     "standard gcloud location [",
                     result, "]"));
  }
  return result;
}

struct AuthProviderRegistry {
  std::vector<std::pair<int, GoogleAuthProvider>> providers;
  absl::Mutex mutex;
};

AuthProviderRegistry& GetGoogleAuthProviderRegistry() {
  static internal::NoDestructor<AuthProviderRegistry> registry;
  return *registry;
}

Result<std::unique_ptr<AuthProvider>> GetDefaultGoogleAuthProvider(
    std::shared_ptr<internal_http::HttpTransport> transport) {
  std::unique_ptr<AuthProvider> result;

  // 1. Check to see if the test environment variable is set.
  auto var = GetEnv(kGoogleAuthTokenForTesting);
  if (var) {
    TENSORSTORE_LOG("Using GOOGLE_AUTH_TOKEN_FOR_TESTING");
    result.reset(new FixedTokenAuthProvider(std::move(*var)));
    return std::move(result);
  }

  // 2. Attempt to read the well-known credentials file.
  Status status;
  auto credentials_filename = GetEnvironmentVariableFileName();
  if (!credentials_filename) {
    TENSORSTORE_LOG("Credentials file not found. ",
                    credentials_filename.status());

    credentials_filename = GetWellKnownFileName();
    if (!credentials_filename.ok()) {
      TENSORSTORE_LOG("Credentials file not found. ",
                      credentials_filename.status());
    }
  }

  if (credentials_filename.ok()) {
    TENSORSTORE_LOG("Using credentials at ", *credentials_filename);

    std::ifstream credentials_fstream(*credentials_filename);
    auto json = ::nlohmann::json::parse(credentials_fstream, nullptr, false);

    auto refresh_token = internal_oauth2::ParseRefreshToken(json);
    if (refresh_token.ok()) {
      TENSORSTORE_LOG("Using OAuth2 AuthProvider");
      result.reset(new OAuth2AuthProvider(*refresh_token, kOAuthV3Url,
                                          std::move(transport)));
      return std::move(result);
    }

    auto service_account =
        internal_oauth2::ParseGoogleServiceAccountCredentials(json);
    if (service_account.ok()) {
      TENSORSTORE_LOG("Using ServiceAccount AuthProvider");
      result.reset(new GoogleServiceAccountAuthProvider(*service_account,
                                                        std::move(transport)));
      return std::move(result);
    }

    status = absl::UnknownError(
        absl::StrCat("Unexpected content of the JSON credentials file: ",
                     *credentials_filename));
  }

  // 3. Running on GCE?
  if (IsRunningOnGce(transport.get())) {
    TENSORSTORE_LOG("Running on GCE, using GCE Auth Provider");
    result.reset(new GceAuthProvider(std::move(transport)));
    return std::move(result);
  }

  // Return a failure code.
  TENSORSTORE_RETURN_IF_ERROR(status);
  return absl::NotFoundError(
      "Could not locate the credentials file and not running on GCE.");
}

}  // namespace

void RegisterGoogleAuthProvider(GoogleAuthProvider provider, int priority) {
  auto& registry = GetGoogleAuthProviderRegistry();
  absl::WriterMutexLock lock(&registry.mutex);
  registry.providers.emplace_back(priority, std::move(provider));
  std::sort(registry.providers.begin(), registry.providers.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
}

Result<std::unique_ptr<AuthProvider>> GetGoogleAuthProvider(
    std::shared_ptr<internal_http::HttpTransport> transport) {
  {
    auto& registry = GetGoogleAuthProviderRegistry();
    absl::ReaderMutexLock lock(&registry.mutex);
    for (const auto& provider : registry.providers) {
      auto auth_result = provider.second();
      if (auth_result.ok()) return auth_result;
    }
  }
  return internal_oauth2::GetDefaultGoogleAuthProvider(std::move(transport));
}

}  // namespace internal_oauth2
}  // namespace tensorstore
