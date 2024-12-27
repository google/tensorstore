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

#ifndef TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_CHANNEL_AUTHENTICATION_H_
#define TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_CHANNEL_AUTHENTICATION_H_

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

// gRPC authentication strategy that uses the provided ChannelCredentials
// and does not require any per-call configuration.
// This is used to wrap most default credentials options.
class GrpcChannelCredentialsAuthentication : public GrpcAuthenticationStrategy {
 public:
  explicit GrpcChannelCredentialsAuthentication(
      std::shared_ptr<grpc::ChannelCredentials> channel_creds)
      : channel_creds_(std::move(channel_creds)) {}

  ~GrpcChannelCredentialsAuthentication() override = default;

  std::shared_ptr<grpc::ChannelCredentials> GetChannelCredentials(
      std::string_view endpoint,
      grpc::ChannelArguments& arguments) const override;

  bool RequiresConfigureContext() const override;

  Future<std::shared_ptr<grpc::ClientContext>> ConfigureContext(
      std::shared_ptr<grpc::ClientContext> context) const override;

 private:
  std::shared_ptr<grpc::ChannelCredentials> channel_creds_;
};

/// Creates an "insecure" authentication strategy.
std::shared_ptr<GrpcAuthenticationStrategy>
CreateInsecureAuthenticationStrategy();

/// Creates a "google_default" authentication strategy.
std::shared_ptr<GrpcAuthenticationStrategy>
CreateGoogleDefaultAuthenticationStrategy();

/// Creates an "external_account" authentication strategy with the given json
/// config and scopes.
std::shared_ptr<GrpcAuthenticationStrategy>
CreateExternalAccountAuthenticationStrategy(
    const std::string& json_object, const std::vector<std::string>& scopes,
    const CaInfo& ca_info);

}  // namespace internal_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_CHANNEL_AUTHENTICATION_H_
