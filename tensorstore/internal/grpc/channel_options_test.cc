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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

#include "tensorstore/internal/grpc/channel_options.h"

#include <stdint.h>

#include <optional>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "grpcpp/support/channel_arguments.h"  // third_party

namespace {

using ::tensorstore::internal_grpc::ApplyDefaultClientChannelArguments;
using ::tensorstore::internal_grpc::ApplyDirectPathClientChannelArguments;
using ::tensorstore::internal_grpc::DefaultChannelArgumentsOptions;
using ::tensorstore::internal_grpc::IsDirectPathAddress;
using ::tensorstore::internal_grpc::IsRetriable;
using ::tensorstore::internal_grpc::ResolveChannelCount;

TEST(ChannelOptionsTest, IsDirectPathAddress) {
  EXPECT_TRUE(IsDirectPathAddress("google:///storage.googleapis.com"));
  EXPECT_TRUE(IsDirectPathAddress("google-c2p:///storage.googleapis.com"));
  EXPECT_TRUE(
      IsDirectPathAddress("google-c2p-experimental:///storage.googleapis.com"));
  EXPECT_FALSE(IsDirectPathAddress("dns:///localhost:8080"));
  EXPECT_FALSE(IsDirectPathAddress("127.0.0.1:9090"));
}

TEST(ChannelOptionsTest, ResolveChannelCount) {
  // num_channels > 0 wins unconditionally.
  EXPECT_EQ(ResolveChannelCount("localhost:8080", 8, std::nullopt), 8);
  EXPECT_EQ(ResolveChannelCount("localhost:8080", 8, 16), 8);
  // override_channels used when num_channels == 0.
  EXPECT_EQ(ResolveChannelCount("localhost:8080", 0, 16), 16);
  // DirectPath forces 1.
  EXPECT_EQ(ResolveChannelCount("google-c2p:///storage", 0, std::nullopt), 1);
  // Localhost defaults to 4.
  EXPECT_EQ(ResolveChannelCount("localhost:8080", 0, std::nullopt), 4);
  EXPECT_EQ(ResolveChannelCount("127.0.0.1:8080", 0, std::nullopt), 4);
  EXPECT_EQ(ResolveChannelCount("[::1]:8080", 0, std::nullopt), 4);
  // override_channels=0 is treated as absent (falls through).
  EXPECT_EQ(ResolveChannelCount("localhost:8080", 0, 0), 4);
  // Remote address: max(4, hardware_concurrency).
  uint32_t remote = ResolveChannelCount("example.com:443", 0, std::nullopt);
  EXPECT_GE(remote, 4u);
}

TEST(ChannelOptionsTest, ApplyDefaultClientChannelArguments) {
  grpc::ChannelArguments args;
  DefaultChannelArgumentsOptions options;
  options.keepalive_time = absl::Seconds(30);
  ApplyDefaultClientChannelArguments(args, options);

  grpc::ChannelArguments direct_args;
  ApplyDirectPathClientChannelArguments(direct_args, options);
}

TEST(IsRetriableTest, RetriableCodes) {
  EXPECT_TRUE(IsRetriable(absl::DeadlineExceededError("timeout")));
  EXPECT_TRUE(IsRetriable(absl::ResourceExhaustedError("quota")));
  EXPECT_TRUE(IsRetriable(absl::UnavailableError("connection reset")));
  EXPECT_TRUE(IsRetriable(absl::InternalError("rst_stream")));
}

TEST(IsRetriableTest, NonRetriableCodes) {
  EXPECT_FALSE(IsRetriable(absl::OkStatus()));
  EXPECT_FALSE(IsRetriable(absl::NotFoundError("missing")));
  EXPECT_FALSE(IsRetriable(absl::InvalidArgumentError("bad key")));
  EXPECT_FALSE(IsRetriable(absl::PermissionDeniedError("denied")));
  EXPECT_FALSE(IsRetriable(absl::UnauthenticatedError("token expired")));
  EXPECT_FALSE(IsRetriable(absl::CancelledError("cancelled")));
  EXPECT_FALSE(IsRetriable(absl::AlreadyExistsError("dup")));
  EXPECT_FALSE(IsRetriable(absl::FailedPreconditionError("precondition")));
  EXPECT_FALSE(IsRetriable(absl::UnimplementedError("unsupported")));
}

}  // namespace
