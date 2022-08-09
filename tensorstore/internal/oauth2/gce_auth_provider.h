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

#ifndef TENSORSTORE_INTERNAL_OAUTH2_GCE_AUTH_PROVIDER_H_
#define TENSORSTORE_INTERNAL_OAUTH2_GCE_AUTH_PROVIDER_H_

#include <functional>
#include <set>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/oauth2/auth_provider.h"
#include "tensorstore/internal/oauth2/oauth_utils.h"
#include "tensorstore/internal/oauth2/refreshable_auth_provider.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_oauth2 {

/// Returns the hostname of the GCE metadata server.
std::string GceMetadataHostname();

class GceAuthProvider : public RefreshableAuthProvider {
 public:
  struct ServiceAccountInfo {
    std::string email;
    std::vector<std::string> scopes;
  };

  ~GceAuthProvider() override = default;

  GceAuthProvider(std::shared_ptr<internal_http::HttpTransport> transport,
                  const ServiceAccountInfo& service_account_info,
                  std::function<absl::Time()> clock = {});

  using AuthProvider::BearerTokenWithExpiration;

  /// Returns the default GCE service account info if available.  If not running
  /// on GCE, or there is no default service account, returns `NOT_FOUND`.
  ///
  /// The ServiceAccountInfo is returned from a GCE metadata call to:
  /// "metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/",
  static Result<ServiceAccountInfo> GetDefaultServiceAccountInfoIfRunningOnGce(
      internal_http::HttpTransport* transport);

 protected:
  // Issue an http request on the provided path.
  virtual Result<internal_http::HttpResponse> IssueRequest(std::string path,
                                                           bool recursive);

 private:
  // Refresh the token from the GCE Metadata service.
  absl::Status Refresh() override ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  absl::Status RetrieveServiceAccountInfo()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  std::string service_account_email_ ABSL_GUARDED_BY(mutex_);
  std::set<std::string> scopes_ ABSL_GUARDED_BY(mutex_);

  std::shared_ptr<internal_http::HttpTransport> transport_;
};

}  // namespace internal_oauth2
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OAUTH2_GCE_AUTH_PROVIDER_H_
