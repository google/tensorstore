// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/internal/grpc/client_credentials.h"

#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/security/credentials.h"  // third_party
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/grpc/clientauth/channel_authentication.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace {

ABSL_CONST_INIT static absl::Mutex credentials_mu(absl::kConstInit);

const internal::ContextResourceRegistration<GrpcClientCredentials>
    grpc_client_credentials_registration;

}  // namespace

// TODO: We should extend this class to permit, at least, some selection
// of grpc credentials. See grpcpp/security/credentials.h for options, such as:
//   ::grpc::experimental::LocalCredentials(LOCAL_TCP)
//   ::grpc::GoogleDefaultCredentials();

/* static */
bool GrpcClientCredentials::Use(
    tensorstore::Context context,
    std::shared_ptr<internal_grpc::GrpcAuthenticationStrategy> auth_strategy) {
  auto resource = context.GetResource<GrpcClientCredentials>().value();
  absl::MutexLock l(&credentials_mu);
  bool result = (resource->auth_strategy_ == nullptr);
  resource->auth_strategy_ = std::move(auth_strategy);
  return result;
}

/* static */
bool GrpcClientCredentials::Use(
    tensorstore::Context context,
    std::shared_ptr<::grpc::ChannelCredentials> credentials) {
  return Use(
      context,
      std::make_shared<internal_grpc::GrpcChannelCredentialsAuthentication>(
          credentials));
}

std::shared_ptr<internal_grpc::GrpcAuthenticationStrategy>
GrpcClientCredentials::Resource::GetAuthenticationStrategy() {
  absl::MutexLock l(&credentials_mu);
  if (auth_strategy_) return auth_strategy_;
  return internal_grpc::CreateInsecureAuthenticationStrategy();
}

}  // namespace tensorstore
