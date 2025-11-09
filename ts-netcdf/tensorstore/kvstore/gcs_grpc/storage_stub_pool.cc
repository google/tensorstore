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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpc/grpc.h"
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "grpcpp/support/client_interceptor.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/grpc/clientauth/create_channel.h"
#include "tensorstore/internal/log/verbose_flag.h"

// Protos
#include "google/protobuf/empty.pb.h"
#include "google/storage/v2/storage.pb.h"

ABSL_FLAG(std::optional<uint32_t>, tensorstore_gcs_grpc_channels, std::nullopt,
          "Default (and maximum) channels to use in gcs_grpc driver. "
          "Overrides TENSORSTORE_GCS_GRPC_CHANNELS.");

using ::tensorstore::internal::GetFlagOrEnvValue;

namespace tensorstore {
namespace internal_gcs_grpc {
namespace {

ABSL_CONST_INIT absl::Mutex global_mu(absl::kConstInit);

ABSL_CONST_INIT internal_log::VerboseFlag gcs_grpc_logging("gcs_grpc");

bool IsDirectPathAddress(std::string_view address) {
  return absl::StartsWith(address, "google:///") ||
         absl::StartsWith(address, "google-c2p:///") ||
         absl::StartsWith(address, "google-c2p-experimental:///");
}

/// Returns the number of channels to use.
/// See also the channel construction in googleapis/google-cloud-cpp repository:
/// https://github.com/googleapis/google-cloud-cpp/blob/main/google/cloud/storage/internal/grpc_client.cc#L188
uint32_t ChannelsForAddress(std::string_view address, uint32_t num_channels) {
  // If num_channels was explicitly requested, use that.
  if (num_channels != 0) {
    return num_channels;
  }

  if (IsDirectPathAddress(address)) {
    // google-c2p are direct-path addresses; multiple channels are handled
    // internally in gRPC.
    return 1;
  }

  // Use the flag --tensorstore_gcs_grpc_channels if present.
  // Use the environment variable TENSORSTORE_GCS_GRPC_CHANNELS if present.
  auto opt = GetFlagOrEnvValue(FLAGS_tensorstore_gcs_grpc_channels,
                               "TENSORSTORE_GCS_GRPC_CHANNELS");
  if (opt && *opt > 0) {
    return *opt;
  }

  // Otherwise multiplex over a multiple channels.
  return std::max(4u, std::thread::hardware_concurrency());
}

// Create a gRPC channel. See google cloud storage client in:
// https://github.com/googleapis/google-cloud-cpp/blob/main/google/cloud/storage/internal/storage_stub_factory.cc
std::shared_ptr<grpc::Channel> CreateChannel(
    const std::string& address,
    internal_grpc::GrpcAuthenticationStrategy& auth_strategy, int channel_id) {
  grpc::ChannelArguments args;

  if (!IsDirectPathAddress(address)) {
    args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, 1);
    args.SetInt(GRPC_ARG_CHANNEL_ID, channel_id);
    args.SetInt(GRPC_ARG_DNS_ENABLE_SRV_QUERIES, 0);

    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS,
                absl::ToInt64Milliseconds(absl::Minutes(5)));
  }

  // TODO: Consider adding the following gcp API flags:
  // args.SetInt(GRPC_ARG_TCP_TX_ZEROCOPY_ENABLED, 1);
  // args.SetCompressionAlgorithm(GRPC_COMPRESS_NONE);

  return internal_grpc::CreateChannel(auth_strategy, address, args);
}

}  // namespace

StorageStubPool::StorageStubPool(
    std::string address, std::vector<std::shared_ptr<grpc::Channel>> channels)
    : address_(std::move(address)), channels_(std::move(channels)) {
  stubs_.reserve(channels_.size());
  for (auto& channel : channels_) {
    // See google cloud storage client in:
    // https://github.com/googleapis/google-cloud-cpp/blob/main/google/cloud/storage/internal/storage_stub_factory.cc
    stubs_.push_back(Storage::NewStub(channel));
  }
}

void StorageStubPool::WaitForConnected(absl::Duration duration) {
  for (auto& channel : channels_) {
    channel->GetState(true);
  }
  if (duration > absl::ZeroDuration()) {
    auto timeout = absl::ToChronoTime(absl::Now() + duration);
    for (auto& channel : channels_) {
      channel->WaitForConnected(timeout);
    }
  }
  ABSL_LOG_IF(INFO, gcs_grpc_logging)
      << "Connection established to " << address_ << " in state "
      << channels_[0]->GetState(false);
}

std::shared_ptr<StorageStubPool> GetSharedStorageStubPool(
    std::string address, uint32_t size,
    std::shared_ptr<internal_grpc::GrpcAuthenticationStrategy> auth_strategy,
    absl::Duration wait_for_connected) {
  static absl::NoDestructor<
      absl::flat_hash_map<std::string, std::shared_ptr<StorageStubPool>>>
      shared_pool;
  size = ChannelsForAddress(address, size);
  std::string key = absl::StrFormat("%d/%s", size, address);

  absl::MutexLock lock(&global_mu);
  auto& pool = (*shared_pool)[key];
  if (pool == nullptr) {
    ABSL_LOG_IF(INFO, gcs_grpc_logging)
        << "Connecting to " << address << " with " << size << " channels";

    std::vector<std::shared_ptr<grpc::Channel>> channels;
    channels.reserve(size);
    for (int id = 0; id < size; id++) {
      // See google cloud storage client in:
      // https://github.com/googleapis/google-cloud-cpp/blob/main/google/cloud/storage/internal/storage_stub_factory.cc
      channels.push_back(CreateChannel(address, *auth_strategy, id));
    }
    pool = std::make_shared<StorageStubPool>(std::move(address),
                                             std::move(channels));
    pool->WaitForConnected(wait_for_connected);
  }
  return pool;
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
