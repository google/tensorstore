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

#include "tensorstore/internal/grpc/clientauth/call_authentication.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_grpc {

std::shared_ptr<grpc::ChannelCredentials>
GrpcCallCredentialsAuthentication::GetChannelCredentials(
    std::string_view endpoint, grpc::ChannelArguments& arguments) const {
  return channel_creds_;
}

bool GrpcCallCredentialsAuthentication::RequiresConfigureContext() const {
  return true;
}

Future<std::shared_ptr<grpc::ClientContext>>
GrpcCallCredentialsAuthentication::ConfigureContext(
    std::shared_ptr<grpc::ClientContext> context) const {
  context->set_credentials(call_creds_);
  return std::move(context);
}

std::shared_ptr<GrpcAuthenticationStrategy>
CreateAccessTokenAuthenticationStrategy(const std::string& token,
                                        const CaInfo& ca_info) {
  grpc::SslCredentialsOptions ssl_options;
  auto cainfo = LoadCAInfo(ca_info);
  if (cainfo) ssl_options.pem_root_certs = std::move(*cainfo);
  return std::make_shared<GrpcCallCredentialsAuthentication>(
      grpc::SslCredentials(ssl_options), grpc::AccessTokenCredentials(token));
}

std::shared_ptr<GrpcAuthenticationStrategy>
CreateServiceAccountAuthenticationStrategy(const std::string& json_object,
                                           const CaInfo& ca_info) {
  grpc::SslCredentialsOptions ssl_options;
  auto cainfo = LoadCAInfo(ca_info);
  if (cainfo) ssl_options.pem_root_certs = std::move(*cainfo);
  return std::make_shared<GrpcCallCredentialsAuthentication>(
      grpc::SslCredentials(ssl_options),
      grpc::ServiceAccountJWTAccessCredentials(json_object));
}

}  // namespace internal_grpc
}  // namespace tensorstore
