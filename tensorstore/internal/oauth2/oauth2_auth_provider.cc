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

#include "tensorstore/internal/oauth2/oauth2_auth_provider.h"

#include <stdint.h>

#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

using tensorstore::Result;
using tensorstore::internal_http::HttpRequestBuilder;
using tensorstore::internal_http::HttpResponse;

namespace tensorstore {
namespace internal_oauth2 {
namespace {

// Construct the refresh token payload once when the OAuth2AuthProvider
// is created & cache the value.
std::string MakePayload(const internal_oauth2::RefreshToken& creds) {
  auto client_id = internal_http::CurlEscapeString(creds.client_id);
  auto client_secret = internal_http::CurlEscapeString(creds.client_secret);
  auto refresh_token = internal_http::CurlEscapeString(creds.refresh_token);
  return absl::StrCat("grant_type=refresh_token", "&client_id=", client_id,
                      "&client_secret=", client_secret,
                      "&refresh_token=", refresh_token);
}

}  // namespace

OAuth2AuthProvider::OAuth2AuthProvider(
    const RefreshToken& creds, std::string uri,
    std::shared_ptr<internal_http::HttpTransport> transport)
    : OAuth2AuthProvider(creds, std::move(uri), std::move(transport),
                         &absl::Now) {}

OAuth2AuthProvider::OAuth2AuthProvider(
    const RefreshToken& creds, std::string uri,
    std::shared_ptr<internal_http::HttpTransport> transport,
    std::function<absl::Time()> clock)
    : refresh_payload_(MakePayload(creds)),
      uri_(std::move(uri)),
      expiration_(absl::InfinitePast()),
      transport_(std::move(transport)),
      clock_(std::move(clock)) {}

Result<AuthProvider::BearerTokenWithExpiration> OAuth2AuthProvider::GetToken() {
  if (!IsValid()) {
    auto status = Refresh();
    TENSORSTORE_RETURN_IF_ERROR(status);
  }
  return BearerTokenWithExpiration{access_token_, expiration_};
}

Result<HttpResponse> OAuth2AuthProvider::IssueRequest(absl::string_view uri,
                                                      absl::Cord payload) {
  HttpRequestBuilder request_builder(std::string{uri});
  return transport_
      ->IssueRequest(request_builder.BuildRequest(), std::move(payload))
      .result();
}

Status OAuth2AuthProvider::Refresh() {
  const auto now = clock_();
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto response, IssueRequest(uri_, absl::Cord(refresh_payload_)));
  TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(response));

  TENSORSTORE_ASSIGN_OR_RETURN(auto result, internal_oauth2::ParseOAuthResponse(
                                                response.payload.Flatten()));
  expiration_ = now + absl::Seconds(result.expires_in);
  access_token_ = std::move(result.access_token);
  return absl::OkStatus();
}

}  // namespace internal_oauth2
}  // namespace tensorstore
