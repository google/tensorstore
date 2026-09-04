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

#ifndef TENSORSTORE_INTERNAL_GRPC_CHANNEL_OPTIONS_H_
#define TENSORSTORE_INTERNAL_GRPC_CHANNEL_OPTIONS_H_

#include <stdint.h>

#include <memory>
#include <optional>
#include <string_view>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party

namespace tensorstore {
namespace internal_grpc {

struct DefaultChannelArgumentsOptions {
  absl::Duration keepalive_time = absl::Minutes(5);
  absl::Duration keepalive_timeout = absl::Seconds(20);
  int max_receive_message_length = -1;
  int max_send_message_length = -1;
  bool enable_http2_bdp_probe = true;
};

/// Returns whether the target address uses gRPC DirectPath / C2P.
bool IsDirectPathAddress(std::string_view address);

/// Determines the number of subchannels to use for an address.
///
/// When `num_channels` i is greater than 0, it is used directly.
/// Otherwise, DirectPath uses 1, then override_channels (e.g. env/flag),
/// then reasonable default values depending on the address.
uint32_t ResolveChannelCount(std::string_view address, uint32_t num_channels,
                             std::optional<uint32_t> override_channels);

/// Configures channel arguments for DirectPath / C2P endpoints.
void ApplyDirectPathClientChannelArguments(
    grpc::ChannelArguments& args,
    const DefaultChannelArgumentsOptions& options);

/// Configures high-performance default channel arguments.
void ApplyDefaultClientChannelArguments(
    grpc::ChannelArguments& args,
    const DefaultChannelArgumentsOptions& options);

/// Waits for channels to transition to the connected state up to timeout.
void WaitForChannelsConnected(
    absl::Span<const std::shared_ptr<grpc::Channel>> channels,
    absl::Duration timeout);

/// Returns whether the absl::Status represents a retriable gRPC error.
/// Retriable codes: DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED, UNAVAILABLE,
/// INTERNAL.
bool IsRetriable(const absl::Status& status);

}  // namespace internal_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_CHANNEL_OPTIONS_H_
