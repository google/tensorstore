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


#include <optional>
#include <string>
#include <string_view>

#include "absl/time/time.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/result.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/kvstore/s3/aws_metadata_credential_provider.h"
#include "tensorstore/internal/json_binding/absl_time.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/str_cat.h"

using ::tensorstore::Result;
using ::tensorstore::internal::ParseJson;
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http::HttpResponseCodeToStatus;
using ::tensorstore::internal_kvstore_s3::AwsCredentials;

namespace jb = tensorstore::internal_json_binding;

namespace tensorstore {
namespace internal_kvstore_s3 {

namespace {

// Token ttl header
static constexpr char kTokenTtlHeader[] = "x-aws-ec2-metadata-token-ttl-seconds";
// Token header
static constexpr char kMetadataTokenHeader[] = "x-aws-ec2-metadata-token";
// Obtain Metadata server API tokens from this url
static constexpr char kTokenUrl[] = "http://169.254.169.254/latest/api/token";
// Obtain current IAM role from this url
static constexpr char kIamCredentialsUrl[] = "http://169.254.169.254/latest/meta-data/iam/security-credentials/";

// Requests to the above server block outside AWS
// Configure a timeout small enough not to degrade performance outside AWS
// but large enough to give the EC2Metadata enough time to respond
static constexpr absl::Duration kConnectTimeout = absl::Milliseconds(200);

// Successful EC2Metadata Security Credential Response Code
static constexpr char kSuccess[] = "Success";

/// Represents JSON returned from
/// http://169.254.169.254/latest/meta-data/iam/security-credentials/<iam-role>/
/// where <iam-role> is usually returned as a response from a request to
/// http://169.254.169.254/latest/meta-data/iam/security-credentials/
struct EC2CredentialsResponse {
  std::string Code;
  std::optional<absl::Time> LastUpdated;
  std::optional<std::string> Type;
  std::optional<std::string> AccessKeyId;
  std::optional<std::string> SecretAccessKey;
  std::optional<std::string> Token;
  std::optional<absl::Time> Expiration;
};

inline constexpr auto EC2CredentialsResponseBinder = jb::Object(
    jb::Member("Code", jb::Projection(&EC2CredentialsResponse::Code)),
    jb::OptionalMember("LastUpdated", jb::Projection(&EC2CredentialsResponse::LastUpdated)),
    jb::OptionalMember("Type", jb::Projection(&EC2CredentialsResponse::Type)),
    jb::OptionalMember("AccessKeyId", jb::Projection(&EC2CredentialsResponse::AccessKeyId)),
    jb::OptionalMember("SecretAccessKey", jb::Projection(&EC2CredentialsResponse::SecretAccessKey)),
    jb::OptionalMember("Token", jb::Projection(&EC2CredentialsResponse::Token)),
    jb::OptionalMember("Expiration", jb::Projection(&EC2CredentialsResponse::Expiration))
);


} // namespace

/// Obtains AWS Credentials from the EC2Metadata.
///
/// https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html#instancedata-meta-data-retrieval-examples
///
/// Credential retrieval follows this flow:
///
/// 1. Post to Metadata server path "/latest/api/token" to obtain an API token
/// 2. Obtain the IAM Role name from path "/latest/meta-data/iam/security-credentials".
///    The first role from a newline separated string is used.
/// 3. Obtain the associated credentials from path "/latest/meta-data/iam/security-credentials/<iam-role>".
Result<AwsCredentials> EC2MetadataCredentialProvider::GetCredentials() {
    // Obtain an API token for communicating with the EC2 Metadata server
    auto token_request = HttpRequestBuilder("POST", kTokenUrl)
                            .AddHeader(absl::StrCat(kTokenTtlHeader, ": 21600"))
                            .BuildRequest();

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto token_response,
        transport_->IssueRequest(token_request, {},
                                 absl::InfiniteDuration(),
                                 kConnectTimeout).result());

    TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(token_response));

    auto token_header = tensorstore::StrCat(kMetadataTokenHeader, ": ", token_response.payload);

    auto iam_role_request = HttpRequestBuilder("GET", kIamCredentialsUrl)
                            .AddHeader(token_header)
                            .BuildRequest();

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto iam_role_response,
        transport_->IssueRequest(iam_role_request, {}).result());

    TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(iam_role_response));

    std::vector<std::string_view> iam_roles = absl::StrSplit(
                        iam_role_response.payload.Flatten(), '\n',
                        absl::SkipWhitespace());

    if(iam_roles.size() == 0) {
        return absl::NotFoundError("Empty EC2 Role list");
    }

    auto iam_credentials_request_url = tensorstore::StrCat(kIamCredentialsUrl,
                                                           iam_roles[0]);

    auto iam_credentials_request = HttpRequestBuilder("GET", iam_credentials_request_url)
                                    .AddHeader(token_header)
                                    .BuildRequest();

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto iam_credentials_response,
        transport_->IssueRequest(iam_credentials_request, {}).result());

    TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(iam_credentials_response));

    auto json_sv = iam_credentials_response.payload.Flatten();
    auto json_credentials = ParseJson(json_sv);

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto iam_credentials,
        jb::FromJson<EC2CredentialsResponse>(
            json_credentials,
            EC2CredentialsResponseBinder));

    if(iam_credentials.Code != kSuccess) {
        return absl::NotFoundError(
            absl::StrCat("EC2Metadata request to [",
                         iam_credentials_request_url,
                         "] failed with ", json_sv));

    }

    return AwsCredentials{
        iam_credentials.AccessKeyId.value_or(""),
        iam_credentials.SecretAccessKey.value_or(""),
        iam_credentials.Token.value_or("")};
}

} // namespace tensorstore
} // namespace internal_kvstore_s3
