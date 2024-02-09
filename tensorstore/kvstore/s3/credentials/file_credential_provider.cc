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

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/lines/line_reading.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/util/result.h"

using ::tensorstore::internal::GetEnv;
using ::tensorstore::internal::JoinPath;

namespace tensorstore {
namespace internal_kvstore_s3 {

namespace {

ABSL_CONST_INIT internal_log::VerboseFlag s3_logging("s3");

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

std::optional<std::string> GetAwsCredentialsFileName() {
  if (auto credentials_file = GetEnv(kEnvAwsCredentialsFile);
      credentials_file) {
    return credentials_file;
  }
  if (auto home_dir = GetEnv("HOME"); home_dir) {
    return JoinPath(*home_dir, kDefaultAwsCredentialsFilePath);
  }
  return std::nullopt;
}

}  // namespace

FileCredentialProvider::FileCredentialProvider(std::string_view filename,
                                               std::string_view profile)
    : filename_(filename), profile_(profile) {
  if (filename_.empty()) {
    if (auto credentials_file = GetAwsCredentialsFileName(); credentials_file) {
      filename_ = std::move(*credentials_file);
    }
  }

  if (profile_.empty()) {
    profile_ = GetEnv(kEnvAwsProfile).value_or(kDefaultProfile);
  }
}

/// https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html#cli-configure-files-format
Result<AwsCredentials> FileCredentialProvider::GetCredentials() {
  if (filename_.empty()) {
    return absl::NotFoundError("No credentials file specified");
  }

  riegeli::FdReader reader(filename_);
  if (!reader.ok()) {
    return absl::NotFoundError(
        absl::StrFormat("Could not open credentials file [%s]", filename_));
  }

  AwsCredentials credentials{};
  std::string_view line;
  bool profile_found = false;
  while (riegeli::ReadLine(reader, line)) {
    auto sline = absl::StripAsciiWhitespace(line);
    // Ignore empty and commented out lines
    if (sline.empty() || sline[0] == '#') continue;

    // A configuration section name has been encountered
    if (sline[0] == '[' && sline[sline.size() - 1] == ']') {
      if (profile_found) break;
      auto section_name =
          absl::StripAsciiWhitespace(sline.substr(1, sline.size() - 2));
      ABSL_LOG_IF(INFO, s3_logging) << "Found section name [" << section_name
                                    << "] in file [" << filename_ << "]";
      profile_found = (section_name == profile_);
      continue;
    }

    // Look for key=value pairs if we're in the appropriate profile
    if (profile_found) {
      std::pair<std::string_view, std::string_view> kv =
          absl::StrSplit(sline, absl::MaxSplits('=', 1));
      kv.first = absl::StripAsciiWhitespace(kv.first);
      kv.second = absl::StripAsciiWhitespace(kv.second);
      if (kv.first == kCfgAwsAccessKeyId) {
        credentials.access_key = kv.second;
      } else if (kv.first == kCfgAwsSecretAccessKeyId) {
        credentials.secret_key = kv.second;
      } else if (kv.first == kCfgAwsSessionToken) {
        credentials.session_token = kv.second;
      }
    }
  }

  if (!profile_found) {
    return absl::NotFoundError(
        absl::StrFormat("Profile [%s] not found in credentials file [%s]",
                        profile_, filename_));
  }

  ABSL_LOG_FIRST_N(INFO, 1)
      << "Using profile [" << profile_ << "] in file [" << filename_ << "]";

  credentials.expires_at = absl::InfiniteFuture();
  return credentials;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
