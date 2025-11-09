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

#ifndef TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_CALL_AUTHENTICATION_H_
#define TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_CALL_AUTHENTICATION_H_

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

// gRPC authentication strategy that uses the provided CallCredentials as
// well as the provided ChannelCredentials.
class GrpcCallCredentialsAuthentication : public GrpcAuthenticationStrategy {
 public:
  explicit GrpcCallCredentialsAuthentication(
      std::shared_ptr<grpc::ChannelCredentials> channel_creds,
      std::shared_ptr<grpc::CallCredentials> call_creds)
      : channel_creds_(std::move(channel_creds)),
        call_creds_(std::move(call_creds)) {}

  ~GrpcCallCredentialsAuthentication() override = default;

  std::shared_ptr<grpc::ChannelCredentials> GetChannelCredentials(
      std::string_view endpoint,
      grpc::ChannelArguments& arguments) const override;

  bool RequiresConfigureContext() const override;

  Future<std::shared_ptr<grpc::ClientContext>> ConfigureContext(
      std::shared_ptr<grpc::ClientContext> context) const override;

 private:
  std::shared_ptr<grpc::ChannelCredentials> channel_creds_;
  std::shared_ptr<grpc::CallCredentials> call_creds_;
};

/// Creates an "access token" authentication strategy with the given token.
std::shared_ptr<GrpcAuthenticationStrategy>
CreateAccessTokenAuthenticationStrategy(const std::string& token,
                                        const CaInfo& ca_info);

/// Creates a "service_account" authentication strategy with the given json
/// config.
std::shared_ptr<GrpcAuthenticationStrategy>
CreateServiceAccountAuthenticationStrategy(const std::string& json_object,
                                           const CaInfo& ca_info);

}  // namespace internal_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_CALL_AUTHENTICATION_H_
