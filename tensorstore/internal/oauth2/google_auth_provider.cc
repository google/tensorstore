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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/internal/oauth2/fixed_token_auth_provider.h"
#include "tensorstore/internal/oauth2/gce_auth_provider.h"
#include "tensorstore/internal/oauth2/google_service_account_auth_provider.h"
#include "tensorstore/internal/oauth2/oauth2_auth_provider.h"
#include "tensorstore/internal/oauth2/oauth_utils.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_oauth2 {
namespace {

using ::tensorstore::internal::GetEnv;
using ::tensorstore::internal::JoinPath;
using ::tensorstore::internal_http::HttpRequestBuilder;

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
// generated with: gcloud auth application-default login
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

/// Returns the credentials file name from the env variable.
Result<std::string> GetEnvironmentVariableFileName() {
  auto env = GetEnv(kGoogleApplicationCredentials);
  if (!env || !IsFile(*env)) {
    return absl::NotFoundError(tensorstore::StrCat(
        "$", kGoogleApplicationCredentials, " is not set or corrupt."));
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
        tensorstore::StrCat("Could not find the credentials file in the "
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
    ABSL_LOG(INFO) << "Using GOOGLE_AUTH_TOKEN_FOR_TESTING";
    result.reset(new FixedTokenAuthProvider(std::move(*var)));
    return std::move(result);
  }

  // 2. Attempt to read the well-known credentials file.
  absl::Status status;
  auto credentials_filename = GetEnvironmentVariableFileName();
  if (!credentials_filename) {
    credentials_filename = GetWellKnownFileName();
  }

  if (credentials_filename.ok()) {
    ABSL_LOG(INFO) << "Using credentials at " << *credentials_filename;

    std::ifstream credentials_fstream(*credentials_filename);
    auto json = ::nlohmann::json::parse(credentials_fstream, nullptr, false);

    auto refresh_token = internal_oauth2::ParseRefreshToken(json);
    if (refresh_token.ok()) {
      ABSL_LOG(INFO) << "Using OAuth2 AuthProvider";
      result.reset(new OAuth2AuthProvider(*refresh_token, kOAuthV3Url,
                                          std::move(transport)));
      return std::move(result);
    }

    auto service_account =
        internal_oauth2::ParseGoogleServiceAccountCredentials(json);
    if (service_account.ok()) {
      ABSL_LOG(INFO) << "Using ServiceAccount AuthProvider";
      result.reset(new GoogleServiceAccountAuthProvider(*service_account,
                                                        std::move(transport)));
      return std::move(result);
    }

    status = absl::UnknownError(
        tensorstore::StrCat("Unexpected content of the JSON credentials file: ",
                            *credentials_filename));
  }

  // 3. Running on GCE?
  if (auto gce_service_account =
          GceAuthProvider::GetDefaultServiceAccountInfoIfRunningOnGce(
              transport.get());
      gce_service_account.ok()) {
    ABSL_LOG(INFO) << "Running on GCE, using service account "
                   << gce_service_account->email;
    result.reset(
        new GceAuthProvider(std::move(transport), *gce_service_account));
    return std::move(result);
  }
  if (!credentials_filename.ok()) {
    ABSL_LOG(ERROR)
        << credentials_filename.status().message()
        << ". You may specify a credentials file using $"
        << kGoogleApplicationCredentials
        << ", or to use Google application default credentials, run: "
           "gcloud auth application-default login";
  }

  // Return a failure code.
  TENSORSTORE_RETURN_IF_ERROR(status);
  return absl::NotFoundError(
      "Could not locate the credentials file and not running on GCE.");
}

struct SharedGoogleAuthProviderState {
  absl::Mutex mutex;
  std::optional<Result<std::shared_ptr<AuthProvider>>> auth_provider
      ABSL_GUARDED_BY(mutex);
};

SharedGoogleAuthProviderState& GetSharedGoogleAuthProviderState() {
  static internal::NoDestructor<SharedGoogleAuthProviderState> state;
  return *state;
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

Result<std::shared_ptr<AuthProvider>> GetSharedGoogleAuthProvider() {
  auto& state = GetSharedGoogleAuthProviderState();
  absl::MutexLock lock(&state.mutex);
  if (!state.auth_provider) {
    state.auth_provider.emplace(GetGoogleAuthProvider());
  }
  return *state.auth_provider;
}

void ResetSharedGoogleAuthProvider() {
  auto& state = GetSharedGoogleAuthProviderState();
  absl::MutexLock lock(&state.mutex);
  state.auth_provider = std::nullopt;
}

}  // namespace internal_oauth2
}  // namespace tensorstore
