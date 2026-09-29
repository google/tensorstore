// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/kvstore/gcs_grpc/default_strategy.h"

#include <memory>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "absl/log/absl_log.h"
#include "absl/strings/match.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/grpc/clientauth/channel_authentication.h"

namespace tensorstore {
namespace internal_gcs_grpc {

std::shared_ptr<internal_grpc::GrpcAuthenticationStrategy>
CreateDefaultGrpcAuthenticationStrategy(std::string_view endpoint) {
  std::string_view host = endpoint;
  if (auto pos = host.find("://"); pos != std::string_view::npos) {
    host = absl::StartsWith(host.substr(pos), ":///") ? host.substr(pos + 4)
                                                      : std::string_view{};
  }
  host = host.substr(0, host.find_first_of(":/"));

  if (absl::EndsWith(host, ".googleapis.com")) {
    // Only send `GoogleDefautCredentials` to a Google backend.
    // These are the credentials acquired from the environment variable
    // "GOOGLE_APPLICATION_CREDENTIALS"  or by using the gcloud tool:
    // `gcloud application-default login`.
    static const absl::NoDestructor kStrategy(
        internal_grpc::CreateGoogleDefaultAuthenticationStrategy());
    return *kStrategy;
  }

  // Otherwise default to insecure credentials.
  static const absl::NoDestructor kStrategy(
      internal_grpc::CreateInsecureAuthenticationStrategy());
  return *kStrategy;
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
