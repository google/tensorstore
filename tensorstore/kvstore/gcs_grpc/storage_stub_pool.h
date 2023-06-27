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

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "google/storage/v2/storage.grpc.pb.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party

namespace tensorstore {
namespace internal_gcs_grpc {

// A gRPC ConnectionPool for Storage stubs.
class StorageStubPool {
  using Storage = ::google::storage::v2::Storage;

 public:
  StorageStubPool(std::string address, uint32_t size,
                  std::shared_ptr<::grpc::ChannelCredentials> creds);

  // Accessors
  const std::string& address() const { return address_; }
  size_t size() const { return stubs_.size(); }

  // Round-robin stub acquisition.
  std::shared_ptr<Storage::StubInterface> get_next_stub() const {
    size_t id = (stubs_.size() > 1)
                    ? (next_channel_index_.fetch_add(1) % stubs_.size())
                    : 0;
    return stubs_[id];
  }

  // Wait for the channels to resolve to the Connected state.
  void WaitForConnected(absl::Duration duration);

 private:
  std::string address_;
  std::vector<std::shared_ptr<Storage::StubInterface>> stubs_;
  std::vector<std::shared_ptr<grpc::Channel>> channels_;
  mutable std::atomic<size_t> next_channel_index_ = 0;
};

// Returns a shared_pointer to the shared StubPool. Care must be taken
// to use the same credentials for the same address.
std::shared_ptr<StorageStubPool> GetSharedStorageStubPool(
    std::string address, uint32_t size,
    std::shared_ptr<::grpc::ChannelCredentials> creds);

}  // namespace internal_gcs_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_GRPC_STORAGE_STUB_POOL_H_
