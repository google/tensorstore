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

#include "tensorstore/kvstore/s3/credentials/file_credential_provider.h"

#include <fstream>
#include <string>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/util/result.h"

using ::tensorstore::internal::GetEnv;
using ::tensorstore::internal::JoinPath;

namespace tensorstore {
namespace internal_kvstore_s3 {

namespace {

// Credentials file environment variable
static constexpr char kEnvAwsCredentialsFile[] = "AWS_SHARED_CREDENTIALS_FILE";
// Default path to the AWS credentials file, relative to the home folder
static constexpr char kDefaultAwsCredentialsFilePath[] = ".aws/credentials";
// AWS user identifier
static constexpr char kCfgAwsAccessKeyId[] = "aws_access_key_id";
// AWS user password
static constexpr char kCfgAwsSecretAccessKeyId[] = "aws_secret_access_key";
// AWS session token
static constexpr char kCfgAwsSessionToken[] = "aws_session_token";
// Discover AWS profile in environment variables
static constexpr char kEnvAwsProfile[] = "AWS_PROFILE";
// Default profile
static constexpr char kDefaultProfile[] = "default";

Result<std::string> GetAwsCredentialsFileName() {
  auto credentials_file = GetEnv(kEnvAwsCredentialsFile);
  if (!credentials_file) {
    auto home_dir = GetEnv("HOME");
    if (!home_dir) {
      return absl::NotFoundError("Could not read $HOME");
    }
    return JoinPath(*home_dir, kDefaultAwsCredentialsFilePath);
  }
  return *credentials_file;
}

}  // namespace

/// https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html#cli-configure-files-format
Result<AwsCredentials> FileCredentialProvider::GetCredentials() {
  if (filename_.empty()) {
    TENSORSTORE_ASSIGN_OR_RETURN(filename_, GetAwsCredentialsFileName());
  }

  if (profile_.empty()) {
    profile_ = GetEnv(kEnvAwsProfile).value_or(kDefaultProfile);
  }

  std::ifstream ifs(filename_.c_str());
  if (!ifs) {
    return absl::NotFoundError(
        absl::StrCat("Could not open credentials file [", filename_, "]"));
  }

  AwsCredentials credentials{};
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

  ABSL_LOG_FIRST_N(INFO, 1)
      << "Using profile [" << profile_ << "] in file [" << filename_ << "]";

  credentials.expires_at = absl::InfiniteFuture();
  return credentials;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
