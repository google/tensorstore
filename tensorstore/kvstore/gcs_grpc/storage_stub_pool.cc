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

#include "tensorstore/kvstore/gcs_grpc/storage_stub_pool.h"

#include <stdint.h>

#include <memory>
#include <optional>
#include <string>

#include "google/storage/v2/storage.grpc.pb.h"
#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/grpc/channel_options.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/grpc/stub_pool.h"
#include "tensorstore/internal/log/verbose_flag.h"

ABSL_FLAG(std::optional<uint32_t>, tensorstore_gcs_grpc_channels, std::nullopt,
          "Default (and maximum) channels to use in gcs_grpc driver. "
          "Overrides TENSORSTORE_GCS_GRPC_CHANNELS.");

using ::tensorstore::internal::GetFlagOrEnvValue;
using Storage = ::google::storage::v2::Storage;

namespace tensorstore {
namespace internal_gcs_grpc {
namespace {

ABSL_CONST_INIT absl::Mutex global_mu(absl::kConstInit);

ABSL_CONST_INIT internal_log::VerboseFlag gcs_grpc_logging("gcs_grpc");

}  // namespace

std::shared_ptr<StorageStubPool> GetSharedStorageStubPool(
    std::string address, uint32_t size,
    std::shared_ptr<internal_grpc::GrpcAuthenticationStrategy> auth_strategy,
    absl::Duration wait_for_connected) {
  static absl::NoDestructor<
      absl::flat_hash_map<std::string, std::shared_ptr<StorageStubPool>>>
      shared_pool;

  auto opt = GetFlagOrEnvValue(FLAGS_tensorstore_gcs_grpc_channels,
                               "TENSORSTORE_GCS_GRPC_CHANNELS");
  size = internal_grpc::ResolveChannelCount(address, size, opt);
  std::string key = absl::StrFormat("%d/%s", size, address);

  absl::MutexLock lock(global_mu);
  auto& pool = (*shared_pool)[key];
  if (pool == nullptr) {
    ABSL_LOG_IF(INFO, gcs_grpc_logging)
        << "Connecting to " << address << " with " << size << " channels";
    pool = internal_grpc::CreateStubPool<Storage, Storage::StubInterface>(
        address, size, *auth_strategy, wait_for_connected);
    if (!pool->channels().empty()) {
      ABSL_LOG_IF(INFO, gcs_grpc_logging)
          << "Connection established to " << address << " in state "
          << pool->channels()[0]->GetState(false);
    }
  }
  return pool;
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
