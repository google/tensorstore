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
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

using ::tensorstore::Result;

namespace tensorstore {
namespace internal_storage_s3 {

using ::tensorstore::internal::GetEnv;
using ::tensorstore::internal::JoinPath;
using ::tensorstore::internal_http::HttpRequestBuilder;

///
// https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html
// AWS user identifier environment variable
constexpr char kEnvAwsAccessKeyId[] = "AWS_ACCESS_KEY_ID";
constexpr char kCfgAwsAccessKeyId[] = "aws_access_key_id";

// AWS user password environment variable
constexpr char kEnvAwsSecretAccessKey[] = "AWS_SECRET_ACCESS_KEY";
constexpr char kCfgAwsSecretAccessKeyId[] = "aws_secret_access_key";

// AWS session token environment variable
constexpr char kEnvAwsSessionToken[] = "AWS_SESSION_TOKEN";
constexpr char kCfgAwsSessionTokenn[] = "aws_session_token";

// AWS Profile environment variables
constexpr char kEnvAwsProfile[] = "AWS_PROFILE";
constexpr char kEnvAwsDefaultProfile[] = "AWS_DEFAULT_PROFILE";

// Default AWS profile
constexpr char kDefaultAwsProfile[] = "default";

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

/// https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html#cli-configure-files-format
Result<S3Credentials> ParseCredentialsFile(
    const std::string & filename,
    std::string_view profile) {
  std::ifstream file_stream(filename);

  if (!file_stream) {
    return absl::NotFoundError(
        tensorstore::StrCat("Could not open the credentials file "
                            "in location [", filename, "]"));
  }

  S3Credentials credentials;
  std::string section_name;
  std::string line;

  while (std::getline(file_stream, line)) {
    auto sline = absl::StripAsciiWhitespace(line);
    if(sline.empty() || sline[0] == '#') continue;

    if(sline[0] == '[' && sline[sline.size() - 1] == ']') {
        section_name = absl::StripAsciiWhitespace(sline.substr(1, sline.size() - 2));
        continue;
    }

    if(section_name == profile) {
        std::vector<std::string_view> key_value = absl::StrSplit(sline, '=');
        if(key_value.size() != 2) continue; // Malformed, ignore
        auto key = absl::StripAsciiWhitespace(key_value[0]);
        auto value = absl::StripAsciiWhitespace(key_value[1]);

        if(key == kCfgAwsAccessKeyId) {
            credentials.SetAccessKey(value);
        } else if(key == kCfgAwsSecretAccessKeyId) {
            credentials.SetSecretKey(value);
        } else if(key == kCfgAwsSessionTokenn) {
            credentials.SetSessionToken(value);
        }
    }
  }

  return credentials;
}


Result<S3Credentials> EnvironmentCredentialSource::GetCredentials(const S3CredentialContext & context) {
    auto access_key = GetEnv(kEnvAwsAccessKeyId);
    S3Credentials credentials;

    if(access_key.has_value()) {
        credentials.SetAccessKey(*access_key);
        auto secret_key = GetEnv(kEnvAwsSecretAccessKey);

        if(secret_key.has_value()) {
            credentials.SetSecretKey(*secret_key);
        }

        auto session_token = GetEnv(kEnvAwsSessionToken);

        if(session_token.has_value()) {
            credentials.SetSessionToken(*session_token);
        }
    }

    return credentials.MakeResult();
}

Result<S3Credentials> FileCredentialSource::GetCredentials(const S3CredentialContext & context) {
    auto credentials_file = GetEnv(kEnvAwsCredentialsFile);

    if(!credentials_file) {
        auto home_dir = GetEnv("HOME");
        if(!home_dir) return absl::NotFoundError("Could not read $HOME");
        credentials_file = JoinPath(*home_dir, kDefaultAwsDirectory, kDefaultAwsCredentialsFile);
    }

    if(!IsFile(*credentials_file)) {
        return absl::NotFoundError(
            tensorstore::StrCat("Could not find the credentials file "
                                "at location [", *credentials_file, "]"));
    }

    std::string_view profile = context.profile_;

    if(profile.empty()) {
        auto env_profile = GetEnv(kEnvAwsDefaultProfile);
        if(!env_profile) env_profile = GetEnv(kEnvAwsProfile);
        profile = !env_profile ? kDefaultAwsProfile : *env_profile;
    }

    TENSORSTORE_ASSIGN_OR_RETURN(auto credentials,
                                 ParseCredentialsFile(*credentials_file, profile));

    return credentials.MakeResult();
}


Result<S3Credentials> GetS3CredentialsFromEnvironment() {
    auto access_key = GetEnv(kEnvAwsAccessKeyId);
    S3Credentials credentials;

    if(access_key.has_value()) {
        credentials.SetAccessKey(*access_key);
        auto secret_key = GetEnv(kEnvAwsSecretAccessKey);

        if(secret_key.has_value()) {
            credentials.SetSecretKey(*secret_key);
        }

        auto session_token = GetEnv(kEnvAwsSessionToken);

        if(session_token.has_value()) {
            credentials.SetSessionToken(*session_token);
        }
    }

    return credentials.MakeResult();
}

Result<S3Credentials> GetS3CredentialsFromConfigFileProfile(std::string profile) {
    auto credentials_file = GetEnv(kEnvAwsCredentialsFile);

    if(!credentials_file) {
        auto home_dir = GetEnv("HOME");

        if(!home_dir) {
            return absl::NotFoundError("Could not read $HOME");
        }

        credentials_file = JoinPath(*home_dir, kDefaultAwsDirectory, kDefaultAwsCredentialsFile);
    }

    if(!IsFile(*credentials_file)) {
        return absl::NotFoundError(
            tensorstore::StrCat("Could not find the credentials file in the "
                                "standard location [", *credentials_file, "]"));
    }

    if(profile.empty()) {
        auto env_profile = GetEnv(kEnvAwsDefaultProfile);
        if(!env_profile) env_profile = GetEnv(kEnvAwsProfile);
        profile = !env_profile ? kDefaultAwsProfile : *env_profile;
    }

    TENSORSTORE_ASSIGN_OR_RETURN(auto credentials,
                                 ParseCredentialsFile(*credentials_file, profile));
    return credentials.MakeResult();
}

Result<S3Credentials> GetS3CredentialsFromEC2Metadata() {
    return absl::UnimplementedError("EC2 Metadata credential extraction not implemented");
}

Result<S3Credentials> GetS3Credentials(const std::string & profile) {
    std::vector<Result<S3Credentials>> credentials = {
        GetS3CredentialsFromEnvironment(),
        GetS3CredentialsFromConfigFileProfile(profile),
        GetS3CredentialsFromEC2Metadata()};

    for(auto & credential: credentials) {
        if(credential.ok()) return credential;
    }

    std::vector<std::string> errors = {"No valid S3 credentials were found."};

    for(auto & credential: credentials) {
        errors.push_back(credential.status().ToString());
    }

    return absl::NotFoundError(absl::StrJoin(errors, "\n"));
}

Result<S3Credentials> S3CredentialProvider::GetCredentials() const {
 if(sources_.size() == 0) {
  return absl::NotFoundError("No S3 credential sources registered");
 }

 std::vector<std::string> errors = {"Unable to obtain S3 credentials"};

 for(auto & source: sources_) {
  auto credentials = source->GetCredentials(context_);
  if(credentials.ok()) return credentials;
   errors.push_back(credentials.status().ToString());
  }

  return absl::NotFoundError(absl::StrJoin(errors, "\n"));
}


}
}
