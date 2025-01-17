// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/kvstore/ocdbt/distributed/cooperator.h"
// Part of the Cooperator interface

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/time/clock.h"
#include "grpcpp/server_builder.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "tensorstore/internal/grpc/clientauth/create_channel.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/ocdbt/distributed/cooperator_impl.h"
#include "tensorstore/kvstore/ocdbt/distributed/coordinator.grpc.pb.h"
#include "tensorstore/kvstore/ocdbt/distributed/lease_cache_for_cooperator.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt_cooperator {

Result<CooperatorPtr> Start(Options&& options) {
  auto impl = internal::MakeIntrusivePtr<Cooperator>();
  if (options.clock) {
    impl->clock_ = std::move(options.clock);
  } else {
    impl->clock_ = [] { return absl::Now(); };
  }
  impl->io_handle_ = std::move(options.io_handle);
  impl->security_ = options.security;
  impl->storage_identifier_ = std::move(options.storage_identifier);

  auto server_auth = impl->security_->GetServerAuthenticationStrategy();

  grpc::ServerBuilder builder;
  builder.RegisterService(impl.get());
  server_auth->AddBuilderParameters(builder);
  auto creds = server_auth->GetServerCredentials();
  const auto add_listening_port = [&](const std::string& address) {
    builder.AddListeningPort(address, creds, &impl->listening_port_);
  };
  if (options.bind_addresses.empty()) {
    add_listening_port("[::]:0");
  } else {
    for (const auto& bind_address : options.bind_addresses) {
      add_listening_port(bind_address);
    }
  }
  impl->server_ = builder.BuildAndStart();

  auto auth_strategy = impl->security_->GetClientAuthenticationStrategy();

  grpc::ChannelArguments args;
  auto channel = internal_grpc::CreateChannel(
      *auth_strategy, options.coordinator_address, args);

  // Create the lease cache
  {
    LeaseCacheForCooperator::Options cache_options;
    cache_options.clock = impl->clock_;
    cache_options.coordinator_stub =
        tensorstore::internal_ocdbt::grpc_gen::Coordinator::NewStub(channel);
    cache_options.auth_strategy = std::move(auth_strategy);
    cache_options.cooperator_port = impl->listening_port_;
    cache_options.lease_duration = options.lease_duration;
    impl->lease_cache_ = LeaseCacheForCooperator(std::move(cache_options));
    impl->lease_cache_ptr_ = &impl->lease_cache_;
  }

  return impl;
}

}  // namespace internal_ocdbt_cooperator
}  // namespace tensorstore
