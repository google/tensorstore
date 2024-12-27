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

#include "absl/log/absl_log.h"
#include "absl/strings/match.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/grpc/clientauth/channel_authentication.h"

namespace tensorstore {
namespace internal_gcs_grpc {

std::shared_ptr<internal_grpc::GrpcAuthenticationStrategy>
CreateDefaultGrpcAuthenticationStrategy(std::string_view endpoint) {

  if (absl::EndsWith(endpoint, ".googleapis.com")) {
    // Only send `GoogleDefautCredentials` to a Google backend.
    // These are the credentials acquired from the environment variable
    // "GOOGLE_APPLICATION_CREDENTIALS"  or by using the gcloud tool:
    // `gcloud application-default login`.
    return internal_grpc::CreateGoogleDefaultAuthenticationStrategy();
  }

  // Otherwise default to insecure credentials.
  return internal_grpc::CreateInsecureAuthenticationStrategy();
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
