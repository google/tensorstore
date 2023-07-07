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

#include "tensorstore/kvstore/s3/s3_credential_provider.h"

#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/strip.h"
#include "absl/strings/str_split.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

using ::tensorstore::Result;

namespace tensorstore {
namespace internal_auth_s3 {

using ::tensorstore::internal::GetEnv;
using ::tensorstore::internal::JoinPath;
using ::tensorstore::internal_http::HttpRequestBuilder;

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
constexpr char kEnvAwsDefaultProfile[] = "AWS_DEFAULT_PROFILE";
constexpr char kDefaultProfile[] = "default";

// Credentials file environment variable
constexpr char kEnvAwsCredentialsFile[] = "AWS_SHARED_CREDENTIALS_FILE";

// Default path to the AWS credentials file, relative to the home folder
constexpr char kDefaultAwsDirectory[] = ".aws";
constexpr char kDefaultAwsCredentialsFile[] = "credentials";

/// Returns whether the given path points to a readable file.
bool IsFile(const std::string& filename) {
  std::ifstream fstream(filename.c_str());
  return fstream.good();
}

Result<std::string> GetS3CredentialsFileName() {
  std::string result;

  auto credentials_file = GetEnv(kEnvAwsCredentialsFile);
  if(!credentials_file) {
    auto home_dir = GetEnv("HOME");
    if(!home_dir) {
      return absl::NotFoundError("Could not read $HOME");
    }
    result = JoinPath(*home_dir, kDefaultAwsDirectory, kDefaultAwsCredentialsFile);
  } else {
    result = *credentials_file;
  }
  if(!IsFile(result)) {
    return absl::NotFoundError(
      tensorstore::StrCat("Could not find the credentials file at "
                          "location [", result, "]"));
  }
  return result;
}

/// https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html#cli-configure-files-format
Result<S3Credentials> FileCredentialProvider::GetCredentials() {
  absl::ReaderMutexLock lock(&mutex_);
  std::ifstream ifs(filename_);

  if (!ifs) {
    return absl::NotFoundError(
        tensorstore::StrCat("Could not open the credentials file "
                            "at location [", filename_, "]"));
  }

  S3Credentials credentials;
  std::string section_name;
  std::string line;

  while (std::getline(ifs, line)) {
    auto sline = absl::StripAsciiWhitespace(line);
    if(sline.empty() || sline[0] == '#') continue;

    if(sline[0] == '[' && sline[sline.size() - 1] == ']') {
      section_name = absl::StripAsciiWhitespace(sline.substr(1, sline.size() - 2));
      continue;
    }

    if(section_name == profile_) {
      std::vector<std::string_view> key_value = absl::StrSplit(sline, '=');
      if(key_value.size() != 2) continue; // Malformed, ignore
      auto key = absl::StripAsciiWhitespace(key_value[0]);
      auto value = absl::StripAsciiWhitespace(key_value[1]);

      if(key == kCfgAwsAccessKeyId) {
          credentials.SetAccessKey(value);
      } else if(key == kCfgAwsSecretAccessKeyId) {
          credentials.SetSecretKey(value);
      } else if(key == kCfgAwsSessionToken) {
          credentials.SetSessionToken(value);
      }
    }
  }

  return credentials;
}

Result<std::unique_ptr<CredentialProvider>> GetDefaultS3CredentialProvider(
    std::string_view profile,
    std::shared_ptr<internal_http::HttpTransport> transport) {

  // 1. Obtain credentials from environment variables
  if(auto access_key = GetEnv(kEnvAwsAccessKeyId); access_key.has_value()) {
    ABSL_LOG(INFO) << "Using Environment Variable S3CredentialProvider";
    S3Credentials credentials;
    credentials.SetAccessKey(*access_key);
    auto secret_key = GetEnv(kEnvAwsSecretAccessKey);

    if(secret_key.has_value()) {
      credentials.SetSecretKey(*secret_key);
    }

    auto session_token = GetEnv(kEnvAwsSessionToken);

    if(session_token.has_value()) {
      credentials.SetSessionToken(*session_token);
    }

    return std::make_unique<EnvironmentCredentialProvider>(std::move(credentials));
  }

  // 2. Obtain credentials from AWS_SHARED_CREDENTIALS_FILE or ~/.aws/credentials
  if(auto credentials_file = GetS3CredentialsFileName(); credentials_file.ok()) {
    std::optional<std::string> env_profile;  // value must not outlive view
    if(profile.empty()) {
        env_profile = GetEnv(kEnvAwsDefaultProfile);
        if(!env_profile) env_profile = GetEnv(kEnvAwsProfile);
        profile = !env_profile ? kDefaultProfile : *env_profile;
    }
    ABSL_LOG(INFO) << "Using File S3CredentialProvider with profile " << profile;
    return std::make_unique<FileCredentialProvider>(credentials_file.value(), profile);
  }

  // 3. Obtain credentials from EC2 Metadata server
  if(false) {
    ABSL_LOG(INFO) << "Using EC2 Metadata Service S3CredentialProvider";
    return std::make_unique<EC2MetadataCredentialProvider>(transport);
  }

  return absl::NotFoundError(
    "No credentials provided in environment variables, "
    "credentials file not found and not running on AWS.");
}

struct CredentialProviderRegistry {
 std::vector<std::pair<int, S3CredentialProvider>> providers;
 absl::Mutex mutex;
};

CredentialProviderRegistry& GetS3ProviderRegistry() {
 static internal::NoDestructor<CredentialProviderRegistry> registry;
 return *registry;
}


void RegisterS3CredentialProviderProvider(S3CredentialProvider provider, int priority) {
  auto& registry = GetS3ProviderRegistry();
  absl::WriterMutexLock lock(&registry.mutex);
  registry.providers.emplace_back(priority, std::move(provider));
  std::sort(registry.providers.begin(), registry.providers.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
}

Result<std::unique_ptr<CredentialProvider>> GetS3CredentialProvider(
    std::string_view profile,
    std::shared_ptr<internal_http::HttpTransport> transport) {
  auto& registry = GetS3ProviderRegistry();
  absl::WriterMutexLock lock(&registry.mutex);
  for(const auto& provider : registry.providers) {
    auto credentials = provider.second();
    if(credentials.ok()) return credentials;
  }

  return internal_auth_s3::GetDefaultS3CredentialProvider(profile, transport);
}

} // namespace internal_auth_s3
} // namespace tensorstore
