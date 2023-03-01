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

#include "tensorstore/internal/grpc/peer_address.h"

#include <utility>

#include "absl/status/status.h"
#include "grpcpp/server_context.h"  // third_party
#include "re2/re2.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

Result<std::pair<std::string, int>> GetGrpcPeerAddressAndPort(
    grpc::CallbackServerContext* context) {
  static LazyRE2 kPeerPattern = {"((?:ipv4|ipv6):.*):([0-9]+)"};
  std::string address;
  int port;
  if (!RE2::FullMatch(context->peer(), *kPeerPattern, &address, &port)) {
    return absl::InternalError(tensorstore::StrCat(
        "Failed to determine peer address and port: ", context->peer()));
  }
  return {std::in_place, std::move(address), port};
}

}  // namespace internal
}  // namespace tensorstore
