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

#ifndef TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_AUTHENTICATION_STRATEGY_H_
#define TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_AUTHENTICATION_STRATEGY_H_

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_grpc {

// Authentication strategy for gRPC calls which provides per-channel and
// per-call credentials.  Every method is thread-safe.
class GrpcAuthenticationStrategy {
 public:
  virtual ~GrpcAuthenticationStrategy() = default;

  virtual std::shared_ptr<grpc::ChannelCredentials> GetChannelCredentials(
      std::string_view endpoint, grpc::ChannelArguments& arguments) const = 0;

  virtual bool RequiresConfigureContext() const = 0;

  virtual Future<std::shared_ptr<grpc::ClientContext>> ConfigureContext(
      std::shared_ptr<grpc::ClientContext> context) const = 0;
};

// A struct that holds the path to a CA certificate.
struct CaInfo {
  std::string root_path;
};

std::optional<std::string> LoadCAInfo(const std::string& ca_root_path);

inline std::optional<std::string> LoadCAInfo(const CaInfo& ca_info) {
  return LoadCAInfo(ca_info.root_path);
}

}  // namespace internal_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_AUTHENTICATION_STRATEGY_H_
