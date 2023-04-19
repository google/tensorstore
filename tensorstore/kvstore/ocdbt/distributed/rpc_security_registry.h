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

#ifndef TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_RPC_SECURITY_REGISTRY_H_
#define TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_RPC_SECURITY_REGISTRY_H_

#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_registry.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security.h"

namespace tensorstore {
namespace internal_ocdbt {

using RpcSecurityMethodRegistry =
    internal::JsonRegistry<RpcSecurityMethod, JsonSerializationOptions,
                           JsonSerializationOptions, RpcSecurityMethod::Ptr>;

RpcSecurityMethodRegistry& GetRpcSecurityMethodRegistry();

template <typename T, typename Binder>
void RegisterRpcSecurityMethod(std::string_view id, Binder binder) {
  GetRpcSecurityMethodRegistry().Register<T>(id, binder);
}

TENSORSTORE_DECLARE_JSON_BINDER(RpcSecurityMethodJsonBinder,
                                RpcSecurityMethod::Ptr,
                                JsonSerializationOptions,
                                JsonSerializationOptions);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_RPC_SECURITY_REGISTRY_H_
