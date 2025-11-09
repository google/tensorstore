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

#ifndef TENSORSTORE_INTERNAL_GRPC_SERVERAUTH_DEFAULT_STRATEGY_H_
#define TENSORSTORE_INTERNAL_GRPC_SERVERAUTH_DEFAULT_STRATEGY_H_

#include <memory>
#include <utility>
#include <vector>

#include "grpcpp/security/server_credentials.h"  // third_party
#include "grpcpp/server_builder.h"  // third_party
#include "tensorstore/internal/grpc/serverauth/strategy.h"

namespace tensorstore {
namespace internal_grpc {

class DefaultServerAuthenticationStrategy
    : public ServerAuthenticationStrategy {
 public:
  DefaultServerAuthenticationStrategy(
      std::shared_ptr<grpc::ServerCredentials> credentials)
      : credentials_(std::move(credentials)) {}

  ~DefaultServerAuthenticationStrategy() override = default;

  std::shared_ptr<grpc::ServerCredentials> GetServerCredentials()
      const override {
    return credentials_;
  }

  void AddBuilderParameters(grpc::ServerBuilder& builder) const override {}

  std::shared_ptr<grpc::ServerCredentials> credentials_;
};

/// Creates an "insecure" server authentication strategy.
std::shared_ptr<ServerAuthenticationStrategy>
CreateInsecureServerAuthenticationStrategy();

}  // namespace internal_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_SERVERAUTH_DEFAULT_STRATEGY_H_
