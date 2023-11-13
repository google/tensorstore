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

#include "tensorstore/kvstore/ocdbt/distributed/cooperator_impl.h"

#include <algorithm>
#include <atomic>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grpcpp/server.h"  // third_party
#include "grpcpp/support/server_callback.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/ocdbt/distributed/btree_node_identifier.h"
#include "tensorstore/kvstore/ocdbt/distributed/lease_cache_for_cooperator.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"

namespace tensorstore {
namespace internal_ocdbt_cooperator {
namespace {
ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");
}

void intrusive_ptr_increment(Cooperator* p) {
  intrusive_ptr_increment(
      static_cast<internal::AtomicReferenceCount<Cooperator>*>(p));
}

void intrusive_ptr_decrement(Cooperator* p) {
  intrusive_ptr_decrement(
      static_cast<internal::AtomicReferenceCount<Cooperator>*>(p));
}

void PendingRequests::Append(PendingRequests&& other) {
  if (requests.empty()) {
    requests = std::move(other.requests);
  } else {
    requests.insert(requests.end(),
                    std::make_move_iterator(other.requests.begin()),
                    std::make_move_iterator(other.requests.end()));
    other.requests.clear();
  }
  if (other.latest_root_generation > latest_root_generation) {
    latest_root_generation = other.latest_root_generation;
    node_generation_at_latest_root_generation =
        std::move(other.node_generation_at_latest_root_generation);
    latest_manifest_time =
        std::max(latest_manifest_time, other.latest_manifest_time);
  }
  other.latest_root_generation = 0;
  other.node_generation_at_latest_root_generation.value.clear();
  other.latest_manifest_time = absl::InfinitePast();
}

Cooperator::~Cooperator() {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "[Port=" << listening_port_ << "] ~Cooperator";
  server_->Shutdown();
  server_->Wait();
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "[Port=" << listening_port_ << "] shutdown complete";
}

LeaseCacheForCooperator* Cooperator::lease_cache() {
  return lease_cache_ptr_.load(std::memory_order_acquire);
}

internal::IntrusivePtr<Cooperator::NodeMutationRequests>
Cooperator::GetNodeMutationRequests(
    const LeaseCacheForCooperator::LeaseNode& lease_node,
    internal_ocdbt::BtreeNodeHeight node_height) {
  internal::IntrusivePtr<NodeMutationRequests> mutation_requests;
  {
    absl::MutexLock lock(&mutex_);
    auto it = node_mutation_map_.find(
        NodeMutationRequests::NodeKey(lease_node.key, node_height));
    if (it == node_mutation_map_.end()) {
      mutation_requests = internal::MakeIntrusivePtr<NodeMutationRequests>();
      mutation_requests->node_identifier = lease_node.node_identifier;
      mutation_requests->node_identifier.height = node_height;
      mutation_requests->lease_node.reset(&lease_node);
      node_mutation_map_.insert(mutation_requests);
    } else {
      mutation_requests = *it;
    }
  }
  return mutation_requests;
}

void NoLeaseError(grpc::ServerUnaryReactor* reactor) {
  reactor->Finish(
      grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "Lease not held"));
}

bool ShouldRevokeLeaseAndRetryAfterError(const absl::Status& status) {
  return absl::IsUnavailable(status) || absl::IsFailedPrecondition(status) ||
         absl::IsCancelled(status);
}

absl::Status ManifestUnexpectedlyDeletedError(Cooperator& server) {
  return kvstore::Driver::AnnotateErrorWithKeyDescription(
      server.io_handle_->DescribeLocation(), "reading",
      absl::FailedPreconditionError("Manifest unexpectedly deleted"));
}

}  // namespace internal_ocdbt_cooperator
}  // namespace tensorstore
