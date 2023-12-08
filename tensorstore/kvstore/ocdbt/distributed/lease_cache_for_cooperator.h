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

#ifndef TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_LEASE_CACHE_FOR_COOPERATOR_H_
#define TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_LEASE_CACHE_FOR_COOPERATOR_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "absl/time/time.h"
#include "tensorstore/internal/container/intrusive_red_black_tree.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/ocdbt/distributed/btree_node_identifier.h"
#include "tensorstore/kvstore/ocdbt/distributed/cooperator.grpc.pb.h"
#include "tensorstore/kvstore/ocdbt/distributed/coordinator.grpc.pb.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_ocdbt_cooperator {

using namespace ::tensorstore::internal_ocdbt;  // NOLINT

// Manages the node lease state for a single cooperator.
//
// Tracks which leases are owned by the local cooperator, and a cache of leases
// and corresponding gRPC stubs for nodes owned by peers.
class LeaseCacheForCooperator {
 public:
  using Clock = std::function<absl::Time()>;

  using LeaseId = uint64_t;
  class LeaseNode;

  using LeaseTree = internal::intrusive_red_black_tree::Tree<LeaseNode>;
  class LeaseNode : public internal::AtomicReferenceCount<LeaseNode>,
                    public LeaseTree::NodeBase {
   public:
    using Ptr = internal::IntrusivePtr<const LeaseNode>;
    std::string key;
    BtreeNodeIdentifier node_identifier;
    absl::Time expiration_time;
    std::string peer_address;
    LeaseId lease_id;

    // Non-null if another cooperator owns the lease on this node.
    //
    // Null if this cooperator owns the lease on this node.
    std::shared_ptr<grpc_gen::Cooperator::StubInterface> peer_stub;
  };

  LeaseCacheForCooperator();

  struct Options {
    Clock clock;
    std::shared_ptr<grpc_gen::Coordinator::StubInterface> coordinator_stub;
    RpcSecurityMethod::Ptr security;
    int32_t cooperator_port;
    absl::Duration lease_duration;
  };

  explicit LeaseCacheForCooperator(Options&& options);

  ~LeaseCacheForCooperator();

  LeaseCacheForCooperator(const LeaseCacheForCooperator& other);
  LeaseCacheForCooperator(LeaseCacheForCooperator&& other) = default;

  LeaseCacheForCooperator& operator=(const LeaseCacheForCooperator& other);
  LeaseCacheForCooperator& operator=(LeaseCacheForCooperator&& other);

  // Requests a lease for the given key.
  //
  // Args:
  //   key: Lease key, must be obtained by calling `node_identifier.GetKey`.
  //   node_identifier: Node identifier corresponding to `key`.
  //   uncooperative_lease: If not `nullptr`, specifies an existing lease for
  //     `key` that should be revoked.
  //
  // Returns:
  //   Future that resolves to the lease node once the query completes
  //   successfully.
  Future<const LeaseNode::Ptr> GetLease(
      std::string_view key, const BtreeNodeIdentifier& node_identifier,
      const LeaseNode* uncooperative_lease = nullptr) const;

  Future<const LeaseNode::Ptr> FindLease(std::string_view key) const;

  // Treat as private:
  class Impl;

  internal::IntrusivePtr<Impl> impl_;
};

}  // namespace internal_ocdbt_cooperator
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_LEASE_CACHE_FOR_COOPERATOR_H_
