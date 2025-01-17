// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_GRPC_SERVERAUTH_STRATEGY_H_
#define TENSORSTORE_INTERNAL_GRPC_SERVERAUTH_STRATEGY_H_

#include <memory>

#include "grpcpp/security/server_credentials.h"  // third_party
#include "grpcpp/server_builder.h"  // third_party

namespace tensorstore {
namespace internal_grpc {

/// Installs gRPC Server authentication strategies.
///
/// Usage:
///     auto strategy = ...;
///     grpc::ServerBuilder builder;
///     builder.RegisterService(...);
///     strategy->AddBuilderParameters(builder);
///     builder.AddListeningPort(bind_addresses,
///                              strategy->GetServerCredentials(),
///                              &bound_port);
///     auto server = builder.BuildAndStart();
class ServerAuthenticationStrategy {
 public:
  virtual ~ServerAuthenticationStrategy() = default;

  virtual std::shared_ptr<grpc::ServerCredentials> GetServerCredentials()
      const = 0;

  virtual void AddBuilderParameters(grpc::ServerBuilder& builder) const = 0;
};

}  // namespace internal_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_SERVERAUTH_STRATEGY_H_
