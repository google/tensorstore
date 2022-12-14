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

#include "tensorstore/internal/oauth2/gce_auth_provider.h"

#include <stdint.h>

#include <optional>
#include <set>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/oauth2/oauth_utils.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_oauth2 {
namespace {

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http::HttpResponse;

// Using GCE-based credentials
// 1. Run the process on GCE.
// 2. Avoid setting GOOGLE_APPLICATION_CREDENTIALS,
//    and avoid credentials in the well-known location of
//    .config/gcloud/application_default_credentials.json
// 3. The GCE metadata service will return credentials for <self>.

constexpr static auto ServiceAccountInfoBinder = jb::Object(
    jb::Member("email",
               jb::Projection(&GceAuthProvider::ServiceAccountInfo::email,
                              jb::NonEmptyStringBinder)),

    // Note that the "scopes" attribute will always be present and contain a
    // JSON array. At minimum, for the request to succeed, the instance must
    // have been granted the scope that allows it to retrieve info from the
    // metadata server.
    jb::Member("scopes",
               jb::Projection(&GceAuthProvider::ServiceAccountInfo::scopes)),
    jb::DiscardExtraMembers);

}  // namespace

std::string GceMetadataHostname() {
  auto maybe_hostname = internal::GetEnv("GCE_METADATA_ROOT");
  if (maybe_hostname.has_value()) {
    return std::move(*maybe_hostname);
  }
  return "metadata.google.internal";
}

GceAuthProvider::GceAuthProvider(
    std::shared_ptr<internal_http::HttpTransport> transport,
    const ServiceAccountInfo& service_account_info,
    std::function<absl::Time()> clock)
    : RefreshableAuthProvider(std::move(clock)),
      service_account_email_(service_account_info.email),
      scopes_(service_account_info.scopes.begin(),
              service_account_info.scopes.end()),
      transport_(std::move(transport)) {}

Result<HttpResponse> GceAuthProvider::IssueRequest(std::string path,
                                                   bool recursive) {
  HttpRequestBuilder request_builder(
      "GET", internal::JoinPath("http://", GceMetadataHostname(), path));
  request_builder.AddHeader("Metadata-Flavor: Google");
  if (recursive) {
    request_builder.AddQueryParameter("recursive", "true");
  }
  return transport_->IssueRequest(request_builder.BuildRequest(), {}).result();
}

Result<GceAuthProvider::ServiceAccountInfo>
GceAuthProvider::GetDefaultServiceAccountInfoIfRunningOnGce(
    internal_http::HttpTransport* transport) {
  HttpRequestBuilder request_builder(
      "GET", internal::JoinPath(
                 "http://", GceMetadataHostname(),
                 "/computeMetadata/v1/instance/service-accounts/default/"));
  request_builder.AddHeader("Metadata-Flavor: Google");
  request_builder.AddQueryParameter("recursive", "true");
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto response,
      transport->IssueRequest(request_builder.BuildRequest(), {}).result());
  TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(response));
  auto info_response = internal::ParseJson(response.payload.Flatten());
  if (info_response.is_discarded()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Failed to parse service account response: ",
                            response.payload.Flatten()));
  }
  return jb::FromJson<ServiceAccountInfo>(info_response,
                                          ServiceAccountInfoBinder);
}

absl::Status GceAuthProvider::Refresh() {
  const auto now = clock_();
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto response,
      IssueRequest(
          tensorstore::StrCat("/computeMetadata/v1/instance/service-accounts/",
                              service_account_email_, "/token"),
          false));
  TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(response));
  TENSORSTORE_ASSIGN_OR_RETURN(auto result, internal_oauth2::ParseOAuthResponse(
                                                response.payload.Flatten()));
  expiration_ = now + absl::Seconds(result.expires_in);
  access_token_ = std::move(result.access_token);
  return absl::OkStatus();
}

}  // namespace internal_oauth2
}  // namespace tensorstore
