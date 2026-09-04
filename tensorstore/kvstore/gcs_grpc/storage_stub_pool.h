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

#ifndef TENSORSTORE_KVSTORE_GCS_GRPC_STORAGE_STUB_POOL_H_
#define TENSORSTORE_KVSTORE_GCS_GRPC_STORAGE_STUB_POOL_H_

#include <stdint.h>

#include <memory>
#include <string>

#include "google/storage/v2/storage.grpc.pb.h"
#include "absl/time/time.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/grpc/stub_pool.h"

namespace tensorstore {
namespace internal_gcs_grpc {

// A gRPC ConnectionPool for Storage stubs.
using StorageStubPool = ::tensorstore::internal_grpc::StubPool<
    ::google::storage::v2::Storage::StubInterface>;

// Returns a shared_pointer to the shared StubPool. Care must be taken
// to use the same credentials for the same address.
std::shared_ptr<StorageStubPool> GetSharedStorageStubPool(
    std::string address, uint32_t size,
    std::shared_ptr<internal_grpc::GrpcAuthenticationStrategy> auth_strategy,
    absl::Duration wait_for_connected);

}  // namespace internal_gcs_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_GRPC_STORAGE_STUB_POOL_H_
