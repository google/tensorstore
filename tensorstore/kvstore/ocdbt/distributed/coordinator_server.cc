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

#include "tensorstore/kvstore/ocdbt/distributed/coordinator_server.h"

#include <stdint.h>

#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/security/server_credentials.h"  // third_party
#include "grpcpp/server.h"  // third_party
#include "grpcpp/server_builder.h"  // third_party
#include "grpcpp/server_context.h"  // third_party
#include "grpcpp/support/server_callback.h"  // third_party
#include "tensorstore/internal/container/heterogeneous_container.h"
#include "tensorstore/internal/container/intrusive_red_black_tree.h"
#include "tensorstore/internal/grpc/peer_address.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/ocdbt/distributed/coordinator.grpc.pb.h"
#include "tensorstore/kvstore/ocdbt/distributed/coordinator.pb.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security_registry.h"
#include "tensorstore/proto/encode_time.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace ocdbt {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");

struct LeaseNode;

using LeaseTree = internal::intrusive_red_black_tree::Tree<LeaseNode>;
struct LeaseNode : public LeaseTree::NodeBase {
  std::string key;
  std::string owner;
  absl::Time expiration_time;
  uint64_t lease_id;
};
}  // namespace

namespace jb = ::tensorstore::internal_json_binding;

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    CoordinatorServer::Spec,
    jb::Object(
        jb::Member("security",
                   jb::Projection<&CoordinatorServer::Spec::security>(
                       internal_ocdbt::RpcSecurityMethodJsonBinder)),
        jb::Member("bind_addresses",
                   jb::Projection<&CoordinatorServer::Spec::bind_addresses>(
                       jb::DefaultInitializedValue()))));

CoordinatorServer::CoordinatorServer() = default;
CoordinatorServer::~CoordinatorServer() = default;
CoordinatorServer::CoordinatorServer(CoordinatorServer&&) = default;
CoordinatorServer& CoordinatorServer::operator=(CoordinatorServer&&) = default;

class CoordinatorServer::Impl
    : public internal_ocdbt::grpc_gen::Coordinator::CallbackService {
 public:
  std::vector<int> listening_ports_;
  std::unique_ptr<grpc::Server> server_;
  internal_ocdbt::RpcSecurityMethod::Ptr security_;
  Clock clock_;

  grpc::ServerUnaryReactor* RequestLease(
      grpc::CallbackServerContext* context,
      const internal_ocdbt::grpc_gen::LeaseRequest* request,
      internal_ocdbt::grpc_gen::LeaseResponse* response) override;

  void PurgeExpiredLeases() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  absl::Mutex mutex_;
  LeaseTree leases_by_expiration_time_ ABSL_GUARDED_BY(mutex_);
  using LeaseSet =
      internal::HeterogeneousHashSet<std::unique_ptr<LeaseNode>,
                                     std::string_view, &LeaseNode::key>;
  LeaseSet leases_by_key_ ABSL_GUARDED_BY(mutex_);
};

span<const int> CoordinatorServer::ports() const {
  return impl_->listening_ports_;
}
int CoordinatorServer::port() const { return impl_->listening_ports_.front(); }

void CoordinatorServer::Impl::PurgeExpiredLeases() {
  auto now = clock_();
  for (LeaseTree::iterator it = leases_by_expiration_time_.begin(), next;
       it != leases_by_expiration_time_.end() && it->expiration_time < now;
       it = next) {
    next = std::next(it);
    LeaseNode& node = *it;
    leases_by_expiration_time_.Remove(node);
    leases_by_key_.erase(node.key);
  }
}

grpc::ServerUnaryReactor* CoordinatorServer::Impl::RequestLease(
    grpc::CallbackServerContext* context,
    const internal_ocdbt::grpc_gen::LeaseRequest* request,
    internal_ocdbt::grpc_gen::LeaseResponse* response) {
  auto* reactor = context->DefaultReactor();
  if (auto status = security_->ValidateServerRequest(context); !status.ok()) {
    reactor->Finish(internal::AbslStatusToGrpcStatus(status));
    return reactor;
  }
  auto peer_address = internal::GetGrpcPeerAddressAndPort(context);
  if (!peer_address.ok()) {
    reactor->Finish(grpc::Status(grpc::StatusCode::INTERNAL,
                                 std::string(peer_address.status().message())));
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "Coordinator: internal error: request=" << request->DebugString();
    return reactor;
  }

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto lease_duration,
      internal::ProtoToAbslDuration(request->lease_duration()),
      (reactor->Finish(grpc::Status(
           grpc::StatusCode::INVALID_ARGUMENT,
           tensorstore::StrCat("Invalid lease duration: ", _.message()))),
       reactor));

  // Lookup lease.
  {
    absl::MutexLock lock(&mutex_);
    PurgeExpiredLeases();
    LeaseNode* node;
    bool assign_new_lease = false;
    bool renew_lease = false;
    if (auto it = leases_by_key_.find(request->key());
        it != leases_by_key_.end()) {
      node = it->get();
      if (request->has_renew_lease_id() &&
          request->renew_lease_id() == node->lease_id) {
        // Extend existing lease.
        leases_by_expiration_time_.Remove(*node);
        renew_lease = true;
      } else if (request->has_uncooperative_lease_id() &&
                 request->uncooperative_lease_id() == node->lease_id) {
        // Terminate existing lease, and grant lease to requesting client.
        leases_by_expiration_time_.Remove(*node);
        assign_new_lease = true;
      }
    } else {
      auto new_node = std::make_unique<LeaseNode>();
      new_node->key = request->key();
      node = new_node.get();
      leases_by_key_.insert(std::move(new_node));
      assign_new_lease = true;
    }

    if (assign_new_lease || renew_lease) {
      auto cur_time = clock_();
      node->expiration_time = cur_time + lease_duration;
      if (assign_new_lease) {
        node->lease_id = static_cast<uint64_t>(
            absl::ToInt64Nanoseconds(cur_time - absl::UnixEpoch()));
        node->owner = tensorstore::StrCat(peer_address->first, ":",
                                          request->cooperator_port());
      }
      response->set_is_owner(true);
      leases_by_expiration_time_.FindOrInsert(
          [&](LeaseNode& other) {
            return node->expiration_time > other.expiration_time ? 1 : -1;
          },
          [&] { return node; });
    }
    response->set_owner(node->owner);
    internal::AbslTimeToProto(node->expiration_time,
                              response->mutable_expiration_time());
    response->set_lease_id(node->lease_id);
  }
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Coordinator: request=" << request->DebugString()
      << ", response=" << response->DebugString();
  reactor->Finish(grpc::Status());
  return reactor;
}

Result<CoordinatorServer> CoordinatorServer::Start(Options options) {
  auto impl = std::make_unique<Impl>();
  if (options.clock) {
    impl->clock_ = std::move(options.clock);
  } else {
    impl->clock_ = [] { return absl::Now(); };
  }
  impl->security_ = options.spec.security;
  if (!impl->security_) {
    impl->security_ = internal_ocdbt::GetInsecureRpcSecurityMethod();
  }
  grpc::ServerBuilder builder;
  builder.RegisterService(impl.get());
  auto creds = impl->security_->GetServerCredentials();
  if (options.spec.bind_addresses.empty()) {
    options.spec.bind_addresses.push_back("[::]:0");
  }
  impl->listening_ports_.resize(options.spec.bind_addresses.size());
  for (size_t i = 0; i < options.spec.bind_addresses.size(); ++i) {
    builder.AddListeningPort(options.spec.bind_addresses[i], creds,
                             &impl->listening_ports_[i]);
  }
  impl->server_ = builder.BuildAndStart();
  CoordinatorServer server;
  server.impl_ = std::move(impl);
  return server;
}

}  // namespace ocdbt
}  // namespace tensorstore
