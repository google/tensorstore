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


#include <string>

#include "absl/time/time.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/kvstore/s3/aws_metadata_credential_provider.h"
#include "tensorstore/internal/json_binding/absl_time.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"

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
static constexpr char kTokenUrl[] = "http://http://169.254.169.254/latest/api/token";
// Obtain IAM status from this url
static constexpr char kIamUrl[] = "http://http://169.254.169.254/latest/meta-data/iam/";
// Obtain current IAM role from this url
static constexpr char kIamCredentialsUrl[] = "http://169.254.169.254/latest/meta-data/iam/security-credentials/";

// Requests to the above server block outside AWS
// Configure a timeout small enough not to degrade performance outside AWS
// but large enough to give the EC2Metadata enough time to respond
static constexpr absl::Duration kConnectTimeout = absl::Milliseconds(200);

/// Represents JSON returned from
/// http://http://169.254.169.254/latest/meta-data/iam/security-credentials/<iam-role>/
/// where <iam-role> is usually returned as a response from a request to
/// http://http://169.254.169.254/latest/meta-data/iam/security-credentials/
struct EC2CredentialsResponse {
  std::string Code;
  absl::Time LastUpdated;
  std::string Type;
  std::string AccessKeyId;
  std::string SecretAccessKey;
  std::string Token;
  absl::Time Expiration;

  using ToJsonOptions = IncludeDefaults;
  using FromJsonOptions = internal_json_binding::NoOptions;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(
      EC2CredentialsResponse,
      internal_kvstore_s3::EC2CredentialsResponse::FromJsonOptions,
      internal_kvstore_s3::EC2CredentialsResponse::ToJsonOptions)
};

inline constexpr auto EC2CredentialsResponseBinder = jb::Object(
    jb::Member("Code", jb::Projection(&EC2CredentialsResponse::Code)),
    jb::Member("LastUpdated", jb::Projection(&EC2CredentialsResponse::LastUpdated)),
    jb::Member("Type", jb::Projection(&EC2CredentialsResponse::Type)),
    jb::Member("AccessKeyId", jb::Projection(&EC2CredentialsResponse::AccessKeyId)),
    jb::Member("SecretAccessKey", jb::Projection(&EC2CredentialsResponse::SecretAccessKey)),
    jb::Member("Token", jb::Projection(&EC2CredentialsResponse::Token)),
    jb::Member("Expiration", jb::Projection(&EC2CredentialsResponse::Expiration))
);

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(EC2CredentialsResponse,
                                       [](auto is_loading, const auto& options,
                                          auto* obj, ::nlohmann::json* j) {
                                         return EC2CredentialsResponseBinder(
                                             is_loading, options, obj, j);
                                       })

} // namespace



Result<AwsCredentials> EC2MetadataCredentialProvider::GetCredentials() {
    /// https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html#instancedata-meta-data-retrieval-examples
    /// https://hackingthe.cloud/aws/exploitation/ec2-metadata-ssrf/

    /// Get a token for communicating with the EC2 Metadata server
    auto token_request = HttpRequestBuilder{"POST", kTokenUrl}
                            .AddHeader(absl::StrCat(kTokenTtlHeader, ": 21600"))
                            .BuildRequest();

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto token_response,
        transport_->IssueRequest(token_request, {}, absl::InfiniteDuration(), kConnectTimeout).result());

    TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(token_response));

    auto token_header = tensorstore::StrCat(kMetadataTokenHeader, ": ", token_response.payload);

    auto iam_request = HttpRequestBuilder{"GET", kIamUrl}
                            .AddHeader(token_header)
                            .BuildRequest();

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto iam_response,
        transport_->IssueRequest(iam_request, {}).result()
    )

    // No associated IAM role, implies anonymous access?
    if(iam_response.status_code == 404) {
        return AwsCredentials{};
    }

    TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(iam_response));

    // IAM Role has been revoked, assume anonymous access?
    if(iam_response.payload.empty()) {
        return AwsCredentials{};
    }

    auto iam_role_request = HttpRequestBuilder{"GET", kIamCredentialsUrl}
                            .AddHeader(token_header)
                            .BuildRequest();

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto iam_role_response,
        transport_->IssueRequest(iam_role_request, {}).result());

    TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(iam_role_response));

    auto iam_credentials_request_url = tensorstore::StrCat(kIamCredentialsUrl,
                                                           iam_role_response.payload);

    auto iam_credentials_request = HttpRequestBuilder{"GET",
                                iam_credentials_request_url}
                            .AddHeader(token_header)
        .BuildRequest();

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto iam_credentials_response,
        transport_->IssueRequest(iam_credentials_request, {}).result());

    TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(iam_credentials_response));

    auto json_sv = iam_credentials_response.payload.Flatten();

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto iam_credentials,
        EC2CredentialsResponse::FromJson(ParseJson(json_sv)));

    if(iam_credentials.Code != "Success") {
        return absl::UnauthenticatedError(
            absl::StrCat("EC2Metadata request to [",
                         iam_credentials_request_url,
                         "] failed with ", json_sv));

    }

    return AwsCredentials{
        iam_credentials.AccessKeyId,
        iam_credentials.SecretAccessKey,
        iam_credentials.Token};
}

} // namespace tensorstore
} // namespace internal_kvstore_s3
