// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_GRPC_PEER_ADDRESS_H_
#define TENSORSTORE_INTERNAL_GRPC_PEER_ADDRESS_H_

#include <utility>

#include "grpcpp/server_context.h"  // third_party
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

// Splits the address of the peer into a portion excluding the port, of the form
// "ipv4://..."  or "ipv6://...", and the port number.
Result<std::pair<std::string, int>> GetGrpcPeerAddressAndPort(
    grpc::CallbackServerContext* context);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_PEER_ADDRESS_H_
