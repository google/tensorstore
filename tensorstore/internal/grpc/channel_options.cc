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

#include "tensorstore/internal/grpc/channel_options.h"

#include <stdint.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string_view>
#include <thread>  // NOLINT

#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpc/grpc.h"
#include "grpcpp/channel.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party

namespace tensorstore {
namespace internal_grpc {

bool IsDirectPathAddress(std::string_view address) {
  return absl::StartsWith(address, "google:///") ||
         absl::StartsWith(address, "google-c2p:///") ||
         absl::StartsWith(address, "google-c2p-experimental:///");
}

/// Determines the number of subchannels to use for an address.
///
/// See also the channel construction in googleapis/google-cloud-cpp:
/// https://github.com/googleapis/google-cloud-cpp/blob/main/google/cloud/storage/internal/grpc_client.cc#L188
uint32_t ResolveChannelCount(std::string_view address, uint32_t num_channels,
                             std::optional<uint32_t> override_channels) {
  if (num_channels != 0) {
    return num_channels;
  }
  // DirectPath addresses manage multiple subchannels internally
  // in gRPC via xDS; use a single logical channel.
  if (IsDirectPathAddress(address)) {
    return 1;
  }
  // If an override is provided (e.g. flag or env var), use it.
  if (override_channels && *override_channels > 0) {
    return *override_channels;
  }
  // "localhost" is typically used in tests; limit to a small
  // number of channels to aid debugging, since
  // hardware_concurrency may be large.
  if (absl::StartsWith(address, "localhost:") ||
      absl::StartsWith(address, "127.0.0.1") ||
      absl::StartsWith(address, "[::1]")) {
    return 4;
  }
  // Otherwise multiplex over multiple channels.
  return std::max(4u, std::thread::hardware_concurrency());
}

void ApplyDirectPathClientChannelArguments(
    grpc::ChannelArguments& args,
    const DefaultChannelArgumentsOptions& options) {
  // Intentionally empty: DirectPath addresses manage multiple subchannels
  // and load balancing internally through xDS. Subchannel pool isolation
  // and custom keepalives are skipped.
}

void ApplyDefaultClientChannelArguments(
    grpc::ChannelArguments& args,
    const DefaultChannelArgumentsOptions& options) {
  args.SetInt(GRPC_ARG_DNS_ENABLE_SRV_QUERIES, 0);

  if (options.keepalive_time > absl::ZeroDuration()) {
    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS,
                absl::ToInt64Milliseconds(options.keepalive_time));
  }
  if (options.keepalive_timeout > absl::ZeroDuration()) {
    args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS,
                absl::ToInt64Milliseconds(options.keepalive_timeout));
  }
  args.SetInt(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);
  args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH,
              options.max_receive_message_length);
  args.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH,
              options.max_send_message_length);
  if (options.enable_http2_bdp_probe) {
    args.SetInt(GRPC_ARG_HTTP2_BDP_PROBE, 1);
  }

  // TODO: Consider adding the following flags:
  // args.SetInt(GRPC_ARG_TCP_TX_ZEROCOPY_ENABLED, 1);
  // args.SetCompressionAlgorithm(GRPC_COMPRESS_NONE);
}

void WaitForChannelsConnected(
    absl::Span<const std::shared_ptr<grpc::Channel>> channels,
    absl::Duration timeout) {
  for (const auto& channel : channels) {
    if (channel) channel->GetState(true);
  }
  if (timeout > absl::ZeroDuration()) {
    // Shared deadline: total wait across all channels is
    // bounded by timeout.
    auto deadline = absl::ToChronoTime(absl::Now() + timeout);
    for (const auto& channel : channels) {
      if (channel) channel->WaitForConnected(deadline);
    }
  }
}

bool IsRetriable(const absl::Status& status) {
  return (status.code() == absl::StatusCode::kDeadlineExceeded ||
          status.code() == absl::StatusCode::kResourceExhausted ||
          status.code() == absl::StatusCode::kUnavailable ||
          status.code() == absl::StatusCode::kInternal);
}

}  // namespace internal_grpc
}  // namespace tensorstore
