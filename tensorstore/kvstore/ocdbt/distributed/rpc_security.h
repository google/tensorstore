// Copyright 2023 The TensorStore Authors
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

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/security/server_credentials.h"  // third_party
#include "grpcpp/server_context.h"  // third_party
#include "tensorstore/internal/cache_key/fwd.h"
#include "tensorstore/internal/intrusive_ptr.h"

#ifndef TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_RPC_SECURITY_H_
#define TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_RPC_SECURITY_H_

namespace tensorstore {
namespace internal_ocdbt {

class RpcSecurityMethod
    : public internal::AtomicReferenceCount<RpcSecurityMethod> {
 public:
  using Ptr = internal::IntrusivePtr<const RpcSecurityMethod>;

  virtual std::shared_ptr<grpc::ServerCredentials> GetServerCredentials()
      const = 0;
  virtual std::shared_ptr<grpc::ChannelCredentials> GetClientCredentials()
      const = 0;

  // Returns `absl::OkStatus()` by default.
  virtual absl::Status ValidateServerRequest(
      grpc::ServerContextBase* context) const;

  // Implementation should start with a call to `internal::EncodeCacheKey(out,
  // id);`, where `id` is the identifier used for JSON.
  virtual void EncodeCacheKey(std::string* out) const = 0;

  virtual ~RpcSecurityMethod();
};

RpcSecurityMethod::Ptr GetInsecureRpcSecurityMethod();

}  // namespace internal_ocdbt
namespace internal {
template <>
struct CacheKeyEncoder<internal_ocdbt::RpcSecurityMethod::Ptr> {
  static void Encode(std::string* out,
                     const internal_ocdbt::RpcSecurityMethod::Ptr& value);
};
}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_RPC_SECURITY_H_
