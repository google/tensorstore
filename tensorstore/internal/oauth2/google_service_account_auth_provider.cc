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

#include "tensorstore/internal/oauth2/google_service_account_auth_provider.h"

#include <utility>

#include "absl/status/status.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

// Using Google service account credentials.
//  1. Create the service account via console.developers.google.com
//  2. Under IAM, add the role with appropriate permissions.
//  3. Download the credentials .json file, with the secret key.
//  4. Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
//
namespace tensorstore {
namespace internal_oauth2 {

using ::tensorstore::Result;
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http::HttpResponse;

// The URL to retrieve the auth bearer token via OAuth with a private key.
constexpr char kOAuthV4Url[] = "https://www.googleapis.com/oauth2/v4/token";

// The authentication token scope to request.
constexpr char kOAuthScope[] = "https://www.googleapis.com/auth/cloud-platform";

GoogleServiceAccountAuthProvider::GoogleServiceAccountAuthProvider(
    const AccountCredentials& creds,
    std::shared_ptr<internal_http::HttpTransport> transport,
    std::function<absl::Time()> clock)
    : RefreshableAuthProvider(std::move(clock)),
      creds_(creds),
      uri_(kOAuthV4Url),
      scope_(kOAuthScope),
      transport_(std::move(transport)) {}

Result<HttpResponse> GoogleServiceAccountAuthProvider::IssueRequest(
    std::string_view method, std::string_view uri, absl::Cord payload) {
  return transport_
      ->IssueRequest(
          HttpRequestBuilder(method, std::string{uri})
              .AddHeader("Content-Type: application/x-www-form-urlencoded")
              .BuildRequest(),
          std::move(payload))
      .result();
}

absl::Status GoogleServiceAccountAuthProvider::Refresh() {
  const auto now = clock_();

  // Try service account credentials.
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto body,
      internal_oauth2::BuildSignedJWTRequest(
          creds_.private_key,
          internal_oauth2::BuildJWTHeader(creds_.private_key_id),
          internal_oauth2::BuildJWTClaimBody(creds_.client_email, scope_, uri_,
                                             now, 3600 /*1 hour*/)));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto response, IssueRequest("POST", uri_, absl::Cord(std::move(body))));
  TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(response));

  TENSORSTORE_ASSIGN_OR_RETURN(auto result, internal_oauth2::ParseOAuthResponse(
                                                response.payload.Flatten()));
  expiration_ = now + absl::Seconds(result.expires_in);
  access_token_ = std::move(result.access_token);
  return absl::OkStatus();
}

}  // namespace internal_oauth2
}  // namespace tensorstore
