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

#include "tensorstore/kvstore/s3/aws_credential_provider.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/result.h"

using ::tensorstore::Result;
using ::tensorstore::internal::GetEnv;
using ::tensorstore::internal::JoinPath;

namespace tensorstore {
namespace internal_kvstore_s3 {
namespace {

// For reference, see the latest AWS environment variables used by the cli:
// https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html

// AWS user identifier
constexpr char kEnvAwsAccessKeyId[] = "AWS_ACCESS_KEY_ID";
constexpr char kCfgAwsAccessKeyId[] = "aws_access_key_id";

// AWS user password
constexpr char kEnvAwsSecretAccessKey[] = "AWS_SECRET_ACCESS_KEY";
constexpr char kCfgAwsSecretAccessKeyId[] = "aws_secret_access_key";

// AWS session token
constexpr char kEnvAwsSessionToken[] = "AWS_SESSION_TOKEN";
constexpr char kCfgAwsSessionToken[] = "aws_session_token";

// AWS Profile environment variables
constexpr char kEnvAwsProfile[] = "AWS_PROFILE";
constexpr char kDefaultProfile[] = "default";

// Credentials file environment variable
constexpr char kEnvAwsCredentialsFile[] = "AWS_SHARED_CREDENTIALS_FILE";

// Default path to the AWS credentials file, relative to the home folder
constexpr char kDefaultAwsCredentialsFilePath[] = ".aws/credentials";

/// Returns whether the given path points to a readable file.
bool IsFile(const std::string& filename) {
  std::ifstream fstream(filename.c_str());
  return fstream.good();
}

Result<std::string> GetAwsCredentialsFileName() {
  std::string result;

  auto credentials_file = GetEnv(kEnvAwsCredentialsFile);
  if (!credentials_file) {
    auto home_dir = GetEnv("HOME");
    if (!home_dir) {
      return absl::NotFoundError("Could not read $HOME");
    }
    result = JoinPath(*home_dir, kDefaultAwsCredentialsFilePath);
  } else {
    result = *credentials_file;
  }
  if (!IsFile(result)) {
    return absl::NotFoundError(
        absl::StrCat("Could not find the credentials file at "
                     "location [",
                     result, "]"));
  }
  return result;
}

Result<std::unique_ptr<AwsCredentialProvider>> GetDefaultAwsCredentialProvider(
    std::string_view profile,
    std::shared_ptr<internal_http::HttpTransport> transport) {
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

    return std::make_unique<EnvironmentCredentialProvider>(
        std::move(credentials));
  }

  // 2. Obtain credentials from AWS_SHARED_CREDENTIALS_FILE or
  // ~/.aws/credentials
  if (auto credentials_file = GetAwsCredentialsFileName();
      credentials_file.ok()) {
    std::string env_profile;  // value must not outlive view
    if (profile.empty()) {
      env_profile = GetEnv(kEnvAwsProfile).value_or(kDefaultProfile);
      profile = std::string_view(env_profile);
    }
    ABSL_LOG(INFO) << "Using File AwsCredentialProvider with profile "
                   << profile;
    return std::make_unique<FileCredentialProvider>(
        std::move(credentials_file).value(), std::string(profile));
  }

  // 3. Obtain credentials from EC2 Metadata server
  if (false) {
    ABSL_LOG(INFO) << "Using EC2 Metadata Service AwsCredentialProvider";
    return std::make_unique<EC2MetadataCredentialProvider>(transport);
  }

  return absl::NotFoundError(
      "No credentials provided in environment variables, "
      "credentials file not found and not running on AWS.");
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

/// https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html#cli-configure-files-format
Result<AwsCredentials> FileCredentialProvider::GetCredentials() {
  absl::ReaderMutexLock lock(&mutex_);
  std::ifstream ifs(filename_);

  if (!ifs) {
    return absl::NotFoundError(
        absl::StrCat("Could not open the credentials file [", filename_, "]"));
  }

  AwsCredentials credentials;
  std::string section_name;
  std::string line;
  bool profile_found = false;

  while (std::getline(ifs, line)) {
    auto sline = absl::StripAsciiWhitespace(line);
    // Ignore empty and commented out lines
    if (sline.empty() || sline[0] == '#') continue;

    // A configuration section name has been encountered
    if (sline[0] == '[' && sline[sline.size() - 1] == ']') {
      section_name =
          absl::StripAsciiWhitespace(sline.substr(1, sline.size() - 2));
      continue;
    }

    // Look for key=value pairs if we're in the appropriate profile
    if (section_name == profile_) {
      profile_found = true;
      if (auto pos = sline.find('='); pos != std::string::npos) {
        auto key = absl::StripAsciiWhitespace(sline.substr(0, pos));
        auto value = absl::StripAsciiWhitespace(sline.substr(pos + 1));

        if (key == kCfgAwsAccessKeyId) {
          credentials.access_key = value;
        } else if (key == kCfgAwsSecretAccessKeyId) {
          credentials.secret_key = value;
        } else if (key == kCfgAwsSessionToken) {
          credentials.session_token = value;
        }
      }
    }
  }

  if (!profile_found) {
    return absl::NotFoundError(absl::StrCat("Profile [", profile_,
                                            "] not found "
                                            "in credentials file [",
                                            filename_, "]"));
  }

  return credentials;
}

void RegisterAwsCredentialProviderProvider(AwsCredentialProviderFn provider,
                                           int priority) {
  auto& registry = GetAwsProviderRegistry();
  absl::WriterMutexLock lock(&registry.mutex);
  registry.providers.emplace_back(priority, std::move(provider));
  std::sort(registry.providers.begin(), registry.providers.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
}

/// @brief Obtain a credential provider from a series of registered and default
/// providers
///
/// Providers are returned in the following order:
/// 1. Any registered providers that supply valid credentials
/// 2. Environment variable provider if valid credential can be obtained from
///    AWS_* environment variables
/// 3. File provider containing credentials from an ~/.aws/credentials file
/// 4. EC2 Metadata server
///
/// @param profile The profile to use when retrieving credentials from a
/// credentials file.
/// @param transport Optionally specify the http transport used to retreive S3
/// credentials
///                  from the EC2 metadata server.
/// @return Provider that supplies S3 Credentials
Result<std::unique_ptr<AwsCredentialProvider>> GetAwsCredentialProvider(
    std::string_view profile,
    std::shared_ptr<internal_http::HttpTransport> transport) {
  auto& registry = GetAwsProviderRegistry();
  absl::WriterMutexLock lock(&registry.mutex);
  for (const auto& provider : registry.providers) {
    auto credentials = provider.second();
    if (credentials.ok()) return credentials;
  }

  return internal_kvstore_s3::GetDefaultAwsCredentialProvider(profile,
                                                              transport);
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
