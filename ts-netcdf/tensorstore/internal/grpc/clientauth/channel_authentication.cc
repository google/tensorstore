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

#include "tensorstore/internal/grpc/clientauth/channel_authentication.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_grpc {

std::shared_ptr<grpc::ChannelCredentials>
GrpcChannelCredentialsAuthentication::GetChannelCredentials(
    std::string_view endpoint, grpc::ChannelArguments& arguments) const {
  return channel_creds_;
}

bool GrpcChannelCredentialsAuthentication::RequiresConfigureContext() const {
  return false;
}

Future<std::shared_ptr<grpc::ClientContext>>
GrpcChannelCredentialsAuthentication::ConfigureContext(
    std::shared_ptr<grpc::ClientContext> context) const {
  return std::move(context);
}

/// Creates an "insecure" authentication strategy with the given token.
std::shared_ptr<GrpcAuthenticationStrategy>
CreateInsecureAuthenticationStrategy() {
  return std::make_shared<GrpcChannelCredentialsAuthentication>(
      grpc::InsecureChannelCredentials());
}

/// Creates a "google_default" authentication strategy.
std::shared_ptr<GrpcAuthenticationStrategy>
CreateGoogleDefaultAuthenticationStrategy() {
  return std::make_shared<GrpcChannelCredentialsAuthentication>(
      grpc::GoogleDefaultCredentials());
}

std::shared_ptr<GrpcAuthenticationStrategy>
CreateExternalAccountAuthenticationStrategy(
    const std::string& json_object, const std::vector<std::string>& scopes,
    const CaInfo& ca_info) {
  grpc::SslCredentialsOptions ssl_options;
  auto cainfo = LoadCAInfo(ca_info);
  if (cainfo) ssl_options.pem_root_certs = std::move(*cainfo);
  return std::make_shared<GrpcChannelCredentialsAuthentication>(
      grpc::CompositeChannelCredentials(
          grpc::SslCredentials(ssl_options),
          grpc::ExternalAccountCredentials(json_object, scopes)));
}

}  // namespace internal_grpc
}  // namespace tensorstore
