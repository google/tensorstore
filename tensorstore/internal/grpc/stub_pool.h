// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_GRPC_STUB_POOL_H_
#define TENSORSTORE_INTERNAL_GRPC_STUB_POOL_H_

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpc/grpc.h"
#include "grpcpp/channel.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "tensorstore/internal/grpc/channel_options.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/grpc/clientauth/create_channel.h"

namespace tensorstore {
namespace internal_grpc {

/// Thread-safe round-robin stub pool multiplexed across multiple gRPC channels.
template <typename StubInterface>
class StubPool {
 public:
  StubPool(std::string address,
           std::vector<std::shared_ptr<grpc::Channel>> channels,
           std::vector<std::shared_ptr<StubInterface>> stubs)
      : address_(std::move(address)),
        channels_(std::move(channels)),
        stubs_(std::move(stubs)) {}

  const std::string& address() const { return address_; }
  size_t size() const { return stubs_.size(); }

  absl::Span<const std::shared_ptr<grpc::Channel>> channels() const {
    return channels_;
  }
  absl::Span<const std::shared_ptr<StubInterface>> stubs() const {
    return stubs_;
  }

  std::shared_ptr<StubInterface> get_next_stub() const {
    if (stubs_.empty()) return nullptr;
    size_t id =
        (stubs_.size() > 1)
            ? (next_channel_index_.fetch_add(1, std::memory_order_relaxed) %
               stubs_.size())
            : 0;
    return stubs_[id];
  }

  void WaitForConnected(absl::Duration duration) const {
    WaitForChannelsConnected(channels_, duration);
  }

 private:
  std::string address_;
  std::vector<std::shared_ptr<grpc::Channel>> channels_;
  std::vector<std::shared_ptr<StubInterface>> stubs_;
  mutable std::atomic<size_t> next_channel_index_ = 0;
};

/// Factory function to create a `StubPool` from a service class `ServiceType`.
template <typename ServiceType,
          typename StubInterface = typename ServiceType::StubInterface>
std::shared_ptr<StubPool<StubInterface>> CreateStubPool(
    const std::string& address, uint32_t num_channels,
    GrpcAuthenticationStrategy& auth_strategy,
    absl::Duration wait_for_connected = absl::ZeroDuration(),
    const DefaultChannelArgumentsOptions& channel_options = {}) {
  bool is_direct_path = IsDirectPathAddress(address);
  num_channels = ResolveChannelCount(address, num_channels, std::nullopt);

  grpc::ChannelArguments args;
  if (is_direct_path) {
    ApplyDirectPathClientChannelArguments(args, channel_options);
  } else {
    ApplyDefaultClientChannelArguments(args, channel_options);
  }

  if (!is_direct_path && num_channels > 1) {
    args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, 1);
  }

  std::vector<std::shared_ptr<grpc::Channel>> channels;
  std::vector<std::shared_ptr<StubInterface>> stubs;
  channels.reserve(num_channels);
  stubs.reserve(num_channels);
  for (uint32_t id = 0; id < num_channels; ++id) {
    if (!is_direct_path && num_channels > 1) {
      args.SetInt(GRPC_ARG_CHANNEL_ID, id);
    }
    auto channel = CreateChannel(auth_strategy, address, args);
    stubs.push_back(ServiceType::NewStub(channel));
    channels.push_back(std::move(channel));
  }

  auto pool = std::make_shared<StubPool<StubInterface>>(
      address, std::move(channels), std::move(stubs));
  if (wait_for_connected > absl::ZeroDuration()) {
    pool->WaitForConnected(wait_for_connected);
  }
  return pool;
}

}  // namespace internal_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_STUB_POOL_H_
