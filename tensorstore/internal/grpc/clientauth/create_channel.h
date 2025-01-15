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

#ifndef TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_CREATE_CHANNEL_H_
#define TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_CREATE_CHANNEL_H_

#include <memory>
#include <string>

#include "grpcpp/channel.h"  // third_party
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"

namespace tensorstore {
namespace internal_grpc {

/// Creates a gRPC channel with the given authentication strategy.
std::shared_ptr<grpc::Channel> CreateChannel(
    GrpcAuthenticationStrategy& auth_strategy, const std::string& endpoint,
    grpc::ChannelArguments& args);

}  // namespace internal_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_CREATE_CHANNEL_H_
