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

#include "tensorstore/kvstore/s3/credentials/ec2_credential_provider.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

// specializations
#include "tensorstore/internal/json_binding/absl_time.h"
#include "tensorstore/internal/json_binding/std_optional.h"

ABSL_FLAG(std::optional<std::string>,
          tensorstore_aws_ec2_metadata_service_endpoint, std::nullopt,
          "Endpoint to used for http access AWS metadata service. "
          "Overrides AWS_EC2_METADATA_SERVICE_ENDPOINT.");

using ::tensorstore::Result;
using ::tensorstore::internal::GetFlagOrEnvValue;
using ::tensorstore::internal::ParseJson;
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http::HttpResponseCodeToStatus;

namespace jb = tensorstore::internal_json_binding;

namespace tensorstore {
namespace internal_kvstore_s3 {

namespace {

// Metadata API token header
static constexpr char kMetadataTokenHeader[] = "x-aws-ec2-metadata-token:";

// Path to IAM credentials
static constexpr char kIamCredentialsPath[] =
    "/latest/meta-data/iam/security-credentials/";

// Requests to the above server block outside AWS
// Configure a timeout small enough not to degrade performance outside AWS
// but large enough to give the EC2Metadata enough time to respond
static constexpr absl::Duration kConnectTimeout = absl::Milliseconds(200);
static constexpr absl::Duration kDefaultTimeout = absl::Minutes(5);

// Successful EC2Metadata Security Credential Response Code
static constexpr char kSuccess[] = "Success";

// Returns the AWS_EC2_METADATA_SERVICE_ENDPOINT environment variable or the
// default metadata service endpoint of http://169.254.169.254.
//
// https://docs.aws.amazon.com/sdk-for-go/api/aws/ec2metadata/
std::string GetEC2MetadataServiceEndpoint() {
  return GetFlagOrEnvValue(FLAGS_tensorstore_aws_ec2_metadata_service_endpoint,
                           "AWS_EC2_METADATA_SERVICE_ENDPOINT")
      .value_or("http://169.254.169.254");
}

/// Represents JSON returned from
/// http://169.254.169.254/latest/meta-data/iam/security-credentials/<iam-role>/
/// where <iam-role> is usually returned as a response from a request to
/// http://169.254.169.254/latest/meta-data/iam/security-credentials/
struct EC2CredentialsResponse {
  std::string code;
  std::optional<absl::Time> last_updated;
  std::optional<std::string> type;
  std::optional<std::string> access_key_id;
  std::optional<std::string> secret_access_key;
  std::optional<std::string> token;
  std::optional<absl::Time> expiration;
};

inline constexpr auto EC2CredentialsResponseBinder = jb::Object(
    jb::Member("Code", jb::Projection(&EC2CredentialsResponse::code)),
    jb::OptionalMember("LastUpdated",
                       jb::Projection(&EC2CredentialsResponse::last_updated)),
    jb::OptionalMember("Type", jb::Projection(&EC2CredentialsResponse::type)),
    jb::OptionalMember("AccessKeyId",
                       jb::Projection(&EC2CredentialsResponse::access_key_id)),
    jb::OptionalMember(
        "SecretAccessKey",
        jb::Projection(&EC2CredentialsResponse::secret_access_key)),
    jb::OptionalMember("Token", jb::Projection(&EC2CredentialsResponse::token)),
    jb::OptionalMember("Expiration",
                       jb::Projection(&EC2CredentialsResponse::expiration)));

// Obtain a metadata token for communicating with the api server.
Result<absl::Cord> GetEC2ApiToken(std::string_view endpoint,
                                  internal_http::HttpTransport& transport) {
  // Obtain Metadata server API tokens with a TTL of 21600 seconds
  auto token_request =
      HttpRequestBuilder("POST",
                         tensorstore::StrCat(endpoint, "/latest/api/token"))
          .AddHeader("x-aws-ec2-metadata-token-ttl-seconds: 21600")
          .BuildRequest();

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto token_response,
      transport
          .IssueRequest(token_request,
                        internal_http::IssueRequestOptions()
                            .SetRequestTimeout(absl::InfiniteDuration())
                            .SetConnectTimeout(kConnectTimeout))
          .result());

  TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(token_response));
  return std::move(token_response.payload);
}

}  // namespace

/// Obtains AWS Credentials from the EC2Metadata.
///
/// https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html#instancedata-meta-data-retrieval-examples
///
/// Credential retrieval follows this flow:
///
/// 1. Post to Metadata server path "/latest/api/token" to obtain an API token
/// 2. Obtain the IAM Role name from path
///    "/latest/meta-data/iam/security-credentials".
///    The first role from a newline separated string is used.
/// 3. Obtain the associated credentials from path
///    "/latest/meta-data/iam/security-credentials/<iam-role>".
Result<AwsCredentials> EC2MetadataCredentialProvider::GetCredentials() {
  if (endpoint_.empty()) {
    endpoint_ = GetEC2MetadataServiceEndpoint();
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto api_token,
                               GetEC2ApiToken(endpoint_, *transport_));

  auto token_header = tensorstore::StrCat(kMetadataTokenHeader, api_token);

  auto iam_role_request =
      HttpRequestBuilder("GET",
                         tensorstore::StrCat(endpoint_, kIamCredentialsPath))
          .AddHeader(token_header)
          .BuildRequest();

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto iam_role_response,
      transport_->IssueRequest(iam_role_request, {}).result());

  TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(iam_role_response));

  std::vector<std::string_view> iam_roles = absl::StrSplit(
      iam_role_response.payload.Flatten(), '\n', absl::SkipWhitespace());

  if (iam_roles.empty()) {
    return absl::NotFoundError("Empty EC2 Role list");
  }

  auto iam_credentials_request_url =
      tensorstore::StrCat(endpoint_, kIamCredentialsPath, iam_roles[0]);

  auto iam_credentials_request =
      HttpRequestBuilder("GET", iam_credentials_request_url)
          .AddHeader(token_header)
          .BuildRequest();

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto iam_credentials_response,
      transport_->IssueRequest(iam_credentials_request, {}).result());

  TENSORSTORE_RETURN_IF_ERROR(
      HttpResponseCodeToStatus(iam_credentials_response));

  auto json_sv = iam_credentials_response.payload.Flatten();
  auto json_credentials = ParseJson(json_sv);

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto iam_credentials,
      jb::FromJson<EC2CredentialsResponse>(json_credentials,
                                           EC2CredentialsResponseBinder));

  if (iam_credentials.code != kSuccess) {
    return absl::NotFoundError(absl::StrCat("EC2Metadata request to [",
                                            iam_credentials_request_url,
                                            "] failed with ", json_sv));
  }

  // Introduce a leeway of 60 seconds to avoid credential expiry conditions
  auto default_timeout = absl::Now() + kDefaultTimeout;
  auto expires_at =
      iam_credentials.expiration.value_or(default_timeout) - absl::Seconds(60);

  return AwsCredentials{iam_credentials.access_key_id.value_or(""),
                        iam_credentials.secret_access_key.value_or(""),
                        iam_credentials.token.value_or(""), expires_at};
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
