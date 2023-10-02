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

#include "tensorstore/kvstore/ocdbt/distributed/rpc_security.h"

#include <memory>
#include <string>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/security/server_credentials.h"  // third_party
#include "grpcpp/server_context.h"  // third_party
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security_registry.h"

namespace tensorstore {
namespace internal_ocdbt {

RpcSecurityMethodRegistry& GetRpcSecurityMethodRegistry() {
  static internal::NoDestructor<RpcSecurityMethodRegistry> registry;
  return *registry;
}

absl::Status RpcSecurityMethod::ValidateServerRequest(
    grpc::ServerContextBase* context) const {
  return absl::OkStatus();
}

namespace {

// No-op RPC security method.
class InsecureRpcSecurityMethod : public RpcSecurityMethod {
 public:
  InsecureRpcSecurityMethod() { intrusive_ptr_increment(this); }
  std::shared_ptr<grpc::ServerCredentials> GetServerCredentials()
      const override {
    return grpc::InsecureServerCredentials();
  }
  std::shared_ptr<grpc::ChannelCredentials> GetClientCredentials()
      const override {
    return grpc::InsecureChannelCredentials();
  }
  void EncodeCacheKey(std::string* out) const override {
    // This should never be called.  In the `OcdbtCoordinatorResource`, this
    // security method is represented by a null pointer.
    ABSL_UNREACHABLE();
  }
};

const RpcSecurityMethod& GetInsecureRpcSecurityMethodSingleton() {
  static internal::NoDestructor<InsecureRpcSecurityMethod> method;
  return *method;
}

}  // namespace

RpcSecurityMethod::~RpcSecurityMethod() = default;

RpcSecurityMethod::Ptr GetInsecureRpcSecurityMethod() {
  return RpcSecurityMethod::Ptr(&GetInsecureRpcSecurityMethodSingleton());
}

TENSORSTORE_DEFINE_JSON_BINDER(
    RpcSecurityMethodJsonBinder,
    [](auto is_loading, const auto& options, auto* obj, ::nlohmann::json* j) {
      if constexpr (is_loading) {
        if (j->is_discarded()) {
          *obj = nullptr;
          return absl::OkStatus();
        }
      } else {
        if (obj->get() == nullptr) {
          *j = ::nlohmann::json::value_t::discarded;
          return absl::OkStatus();
        }
      }
      namespace jb = tensorstore::internal_json_binding;
      return jb::Object(GetRpcSecurityMethodRegistry().MemberBinder("method"))(
          is_loading, options, obj, j);
    })

}  // namespace internal_ocdbt

namespace internal {
void CacheKeyEncoder<internal_ocdbt::RpcSecurityMethod::Ptr>::Encode(
    std::string* out, const internal_ocdbt::RpcSecurityMethod::Ptr& value) {
  if (!value) {
    *out += '\1';
    return;
  }
  *out += '\2';
  value->EncodeCacheKey(out);
}
}  // namespace internal

}  // namespace tensorstore
