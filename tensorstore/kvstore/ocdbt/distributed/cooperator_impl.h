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

#ifndef TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_COOPERATOR_IMPL_H_
#define TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_COOPERATOR_IMPL_H_

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grpcpp/server.h"  // third_party
#include "grpcpp/server_context.h"  // third_party
#include "grpcpp/support/server_callback.h"  // third_party
#include "tensorstore/internal/container/heterogeneous_container.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/ocdbt/distributed/btree_node_identifier.h"
#include "tensorstore/kvstore/ocdbt/distributed/btree_node_write_mutation.h"
#include "tensorstore/kvstore/ocdbt/distributed/cooperator.grpc.pb.h"
#include "tensorstore/kvstore/ocdbt/distributed/cooperator.h"
#include "tensorstore/kvstore/ocdbt/distributed/cooperator.pb.h"
#include "tensorstore/kvstore/ocdbt/distributed/lease_cache_for_cooperator.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_ocdbt_cooperator {

namespace grpc_gen = ::tensorstore::internal_ocdbt::grpc_gen;
using internal_ocdbt::BtreeNodeHeight;

struct PendingRequest {
  Promise<MutationBatchResponse> batch_promise;
  size_t index_within_batch;
  Future<const void> flush_future;
  internal_ocdbt::BtreeNodeWriteMutation::Ptr mutation;
};

struct PendingRequests {
  std::vector<PendingRequest> requests;
  internal_ocdbt::GenerationNumber latest_root_generation = 0;
  StorageGeneration node_generation_at_latest_root_generation;
  absl::Time latest_manifest_time = absl::InfinitePast();

  void Append(PendingRequests&& other);
};

struct Cooperator : public grpc_gen::Cooperator::CallbackService,
                    public internal::AtomicReferenceCount<Cooperator> {
  ~Cooperator();

  struct NodeMutationRequests
      : public internal::AtomicReferenceCount<NodeMutationRequests> {
    using NodeKey = std::pair<std::string_view, BtreeNodeHeight>;
    LeaseCacheForCooperator::LeaseNode::Ptr lease_node;
    internal_ocdbt::BtreeNodeIdentifier node_identifier;
    absl::Mutex mutex;
    PendingRequests pending;
    bool commit_in_progress = false;

    NodeKey node_key() const {
      return {lease_node->key, node_identifier.height};
    }
  };

  grpc::ServerUnaryReactor* GetOrCreateManifest(
      grpc::CallbackServerContext* context,
      const grpc_gen::GetOrCreateManifestRequest* request,
      grpc_gen::GetOrCreateManifestResponse* response) override;

  grpc::ServerUnaryReactor* Write(grpc::CallbackServerContext* context,
                                  const grpc_gen::WriteRequest* request,
                                  grpc_gen::WriteResponse* response) override;

  internal::IntrusivePtr<NodeMutationRequests> GetNodeMutationRequests(
      const LeaseCacheForCooperator::LeaseNode& lease_node,
      internal_ocdbt::BtreeNodeHeight node_height);

  // Returns the lease cache and coordinator client.
  //
  // Returns `nullptr` if the lease cache has not yet been initialized (which
  // can only occur if `Start` has not yet returned).
  LeaseCacheForCooperator* lease_cache();

  int listening_port_;
  std::unique_ptr<grpc::Server> server_;
  internal_ocdbt::RpcSecurityMethod::Ptr security_;
  Clock clock_;

  internal_ocdbt::IoHandle::Ptr io_handle_;

  // Lease cache, initialized on first access.  This deferred initialization is
  // necessary because we need to know the cooperator port in order to
  // initialize the lease cache, but the cooperator port is not known until
  // after starting the gRPC server (gRPC does not offer an API to bind the
  // socket and determine the port number before actually beginning to process
  // requests).  If we initialize `lease_cache_` after
  LeaseCacheForCooperator lease_cache_;

  // Points to `lease_cache_` once it is initialized (immediately after the
  // cooperator server starts).  This additional atomic pointer is needed
  // because we need to know the cooperator port in order to initialize the
  // lease cache, but the cooperator port is not known until after starting the
  // gRPC server (gRPC does not offer an API to bind the socket and determine
  // the port number before actually beginning to process requests).  A gRPC
  // request may be received immediately after starting the cooperator, and the
  // lease cache must be queried to validate requests.
  //
  // Note that any request that arrives before the lease cache has been
  // initialized is necessarily invalid, since this cooperator cannot own the
  // lease in that case, but such requests still must be handled correctly.
  std::atomic<LeaseCacheForCooperator*> lease_cache_ptr_{nullptr};

  // Storage identifier used for computing lease keys.
  std::string storage_identifier_;

  absl::Mutex mutex_;
  Future<const absl::Time> manifest_available_;

  using NodeMutationMap = internal::HeterogeneousHashSet<
      internal::IntrusivePtr<NodeMutationRequests>,
      NodeMutationRequests::NodeKey, &NodeMutationRequests::node_key>;
  NodeMutationMap node_mutation_map_ ABSL_GUARDED_BY(mutex_);
};

void NoLeaseError(grpc::ServerUnaryReactor* reactor);

void MaybeCommit(
    Cooperator& server,
    internal::IntrusivePtr<Cooperator::NodeMutationRequests> mutation_requests,
    UniqueWriterLock<absl::Mutex>&& lock);

// Returns `true` if `status` returned from a gRPC request to another cooperator
// indicates that the lease should be revoked.
//
// The following error codes result in a `true` return:
//
// - `absl::StatusCode::kUnavailable`: indicates that the cooperator server is
// unreachable, which
//   may occur if the process exits or the OCDBT database was closed.
//
// - `absl::StatusCode::kFailedPrecondition`: specifically sent by the
//   cooperator to indicate an invalid lease.
//
// - `CANCELLED`: indicates that the cooperator server was exiting.
bool ShouldRevokeLeaseAndRetryAfterError(const absl::Status& status);

// Returns an error for the case that the manifest was deleted after it was
// previously successfully read.
absl::Status ManifestUnexpectedlyDeletedError(Cooperator& server);

}  // namespace internal_ocdbt_cooperator
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_COOPERATOR_IMPL_H_
