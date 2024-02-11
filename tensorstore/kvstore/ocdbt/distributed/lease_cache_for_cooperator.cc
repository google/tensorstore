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

#include "tensorstore/kvstore/ocdbt/distributed/lease_cache_for_cooperator.h"

#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/meta/type_traits.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grpc/grpc.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/create_channel.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/distributed/btree_node_identifier.h"
#include "tensorstore/kvstore/ocdbt/distributed/cooperator.grpc.pb.h"
#include "tensorstore/kvstore/ocdbt/distributed/coordinator.grpc.pb.h"
#include "tensorstore/kvstore/ocdbt/distributed/coordinator.pb.h"
#include "tensorstore/proto/encode_time.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt_cooperator {
namespace {
ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");
}

class LeaseCacheForCooperator::Impl
    : public internal::AtomicReferenceCount<LeaseCacheForCooperator::Impl> {
 public:
  std::shared_ptr<grpc_gen::Cooperator::StubInterface> GetCooperatorStub(
      const std::string& address);

  Clock clock_;
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, Future<const LeaseNode::Ptr>> leases_by_key_
      ABSL_GUARDED_BY(mutex_);

  // FIXME: need to support eviction from `peer_stub_cache_`
  absl::flat_hash_map<std::string,
                      std::shared_ptr<grpc_gen::Cooperator::StubInterface>>
      peer_stub_cache_ ABSL_GUARDED_BY(mutex_);

  // FIXME: manage expiration times
  // LeaseTree leases_by_expiration_time_ ABSL_GUARDED_BY(mutex_);

  std::shared_ptr<grpc_gen::Coordinator::StubInterface> coordinator_stub_;
  RpcSecurityMethod::Ptr security_;
  int32_t cooperator_port_;
  absl::Duration lease_duration_;
};

std::shared_ptr<grpc_gen::Cooperator::StubInterface>
LeaseCacheForCooperator::Impl::GetCooperatorStub(const std::string& address) {
  absl::MutexLock lock(&mutex_);
  auto& stub = peer_stub_cache_[address];
  if (stub) return stub;
  // Disable gRPC automatic retries, since cooperators are expected to go down
  // and in that case we must invalidate the lease rather than retry.
  grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_ENABLE_RETRIES, 0);
  stub = tensorstore::internal_ocdbt::grpc_gen::Cooperator::NewStub(
      grpc::CreateCustomChannel(address, security_->GetClientCredentials(),
                                args));
  return stub;
}

Future<const LeaseCacheForCooperator::LeaseNode::Ptr>
LeaseCacheForCooperator::FindLease(std::string_view key) const {
  assert(impl_);
  absl::MutexLock lock(&impl_->mutex_);
  auto it = impl_->leases_by_key_.find(key);
  if (it != impl_->leases_by_key_.end()) {
    return it->second;
  }
  return {};
}

namespace {

struct LeaseRequestState
    : public internal::AtomicReferenceCount<LeaseRequestState> {
  internal::IntrusivePtr<LeaseCacheForCooperator::Impl> owner;
  grpc::ClientContext client_context;
  BtreeNodeIdentifier node_identifier;
  Promise<LeaseCacheForCooperator::LeaseNode::Ptr> promise;
  grpc_gen::LeaseRequest request;
  grpc_gen::LeaseResponse response;
};

}  // namespace

Future<const LeaseCacheForCooperator::LeaseNode::Ptr>
LeaseCacheForCooperator::GetLease(std::string_view key,
                                  const BtreeNodeIdentifier& node_identifier,
                                  const LeaseNode* uncooperative_lease) const {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "GetLease: " << node_identifier
      << (uncooperative_lease
              ? tensorstore::StrCat(", uncooperative_lease_id=",
                                    uncooperative_lease->lease_id)
              : "");
  assert(impl_);

  // Holds stale future removed from `leases_by_key_` to be destroyed after the
  // mutex is released.
  Future<const LeaseNode::Ptr> stale_future;

  PromiseFuturePair<LeaseNode::Ptr> promise_future;
  {
    absl::MutexLock lock(&impl_->mutex_);
    auto it = impl_->leases_by_key_.find(key);
    if (it != impl_->leases_by_key_.end()) {
      auto& future = it->second;
      if (!future.ready()) return future;
      auto& result = future.result();
      if (result.ok()) {
        auto& node = **result;
        if ((!uncooperative_lease ||
             uncooperative_lease->lease_id != node.lease_id) &&
            node.expiration_time >= impl_->clock_()) {
          ABSL_LOG_IF(INFO, ocdbt_logging)
              << "GetLease: " << node_identifier
              << ": returning existing lease future";
          return future;
        }
      }
      // Existing lease request failed, or lease expired.  New request is
      // needed.
      stale_future = std::move(future);
    } else {
      // No cached lease, new request needed.
    }
    promise_future = PromiseFuturePair<LeaseNode::Ptr>::Make();
    if (it != impl_->leases_by_key_.end()) {
      it->second = promise_future.future;
    } else {
      impl_->leases_by_key_.emplace(key, promise_future.future);
    }
  }

  // Initiate the request.
  auto state = internal::MakeIntrusivePtr<LeaseRequestState>();
  state->request.set_key(key.data(), key.size());
  state->node_identifier = node_identifier;
  if (node_identifier.range.full()) {
    state->node_identifier.height = 0;
  }
  if (uncooperative_lease) {
    state->request.set_uncooperative_lease_id(uncooperative_lease->lease_id);
  }
  state->request.set_cooperator_port(impl_->cooperator_port_);
  internal::AbslDurationToProto(impl_->lease_duration_,
                                state->request.mutable_lease_duration());
  state->promise = std::move(promise_future.promise);
  state->owner = impl_;
  auto* state_ptr = state.get();
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "GetLease: " << node_identifier << ": requesting lease";
  impl_->coordinator_stub_->async()->RequestLease(
      &state_ptr->client_context, &state_ptr->request, &state_ptr->response,
      [state = std::move(state)](::grpc::Status s) {
        auto status = internal::GrpcStatusToAbslStatus(std::move(s));
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "GetLease: " << state->node_identifier
            << ": got lease: " << status;
        absl::Time expiration_time;
        if (status.ok()) {
          auto expiration_time_result =
              internal::ProtoToAbslTime(state->response.expiration_time());
          if (expiration_time_result.ok()) {
            expiration_time = *expiration_time_result;
          } else {
            status = tensorstore::MaybeAnnotateStatus(
                expiration_time_result.status(), "Invalid expiration_time");
          }
        }
        if (!status.ok()) {
          state->promise.SetResult(status);
          // Remove from cache due to error.
          auto& owner = *state->owner;
          absl::MutexLock lock(&owner.mutex_);
          auto it = owner.leases_by_key_.find(state->request.key());
          if (it != owner.leases_by_key_.end() &&
              HaveSameSharedState(state->promise, it->second)) {
            owner.leases_by_key_.erase(it);
          }
          return;
        }
        auto lease_node = internal::MakeIntrusivePtr<LeaseNode>();
        lease_node->key = std::move(*state->request.mutable_key());
        lease_node->node_identifier = std::move(state->node_identifier);
        lease_node->lease_id = state->response.lease_id();
        lease_node->peer_address = state->response.owner();
        if (!state->response.is_owner()) {
          lease_node->peer_stub =
              state->owner->GetCooperatorStub(state->response.owner());
          ABSL_CHECK(lease_node->peer_stub);
          ABSL_LOG_IF(INFO, ocdbt_logging)
              << "GetLease: " << state->node_identifier << ": owner is "
              << state->response.owner();
        } else {
          ABSL_LOG_IF(INFO, ocdbt_logging)
              << "GetLease: " << state->node_identifier
              << ": current cooperator is owner";
        }

        lease_node->expiration_time = expiration_time;
        state->promise.SetResult(std::move(lease_node));
      });

  return std::move(promise_future.future);
}

LeaseCacheForCooperator::LeaseCacheForCooperator() = default;
LeaseCacheForCooperator::LeaseCacheForCooperator(
    const LeaseCacheForCooperator& other) = default;
LeaseCacheForCooperator& LeaseCacheForCooperator::operator=(
    const LeaseCacheForCooperator& other) = default;
LeaseCacheForCooperator& LeaseCacheForCooperator::operator=(
    LeaseCacheForCooperator&& other) = default;
LeaseCacheForCooperator::~LeaseCacheForCooperator() = default;

LeaseCacheForCooperator::LeaseCacheForCooperator(Options&& options) {
  impl_.reset(new Impl);
  impl_->clock_ = std::move(options.clock);
  impl_->coordinator_stub_ = std::move(options.coordinator_stub);
  impl_->security_ = std::move(options.security);
  impl_->cooperator_port_ = options.cooperator_port;
  impl_->lease_duration_ = options.lease_duration;
}

}  // namespace internal_ocdbt_cooperator
}  // namespace tensorstore
