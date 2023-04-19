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

#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/ocdbt/distributed/cooperator_impl.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/create_new_manifest.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_ocdbt_cooperator {
namespace {

void StartGetManifestForWriting(
    Promise<absl::Time> promise, internal::IntrusivePtr<Cooperator> server,
    LeaseCacheForCooperator::LeaseNode::Ptr uncooperative_lease);

void GetManifestForWritingFromPeer(
    Promise<absl::Time> promise, internal::IntrusivePtr<Cooperator> server,
    LeaseCacheForCooperator::LeaseNode::Ptr lease) {
  struct RequestState : public internal::AtomicReferenceCount<RequestState> {
    grpc::ClientContext client_context;
    internal::IntrusivePtr<Cooperator> server;
    LeaseCacheForCooperator::LeaseNode::Ptr lease;
    Promise<absl::Time> promise;
    grpc_gen::GetOrCreateManifestRequest request;
    grpc_gen::GetOrCreateManifestResponse response;
  };
  auto executor = server->io_handle_->executor;

  auto state = internal::MakeIntrusivePtr<RequestState>();
  state->promise = std::move(promise);
  state->server = std::move(server);
  state->lease = std::move(lease);
  auto* state_ptr = state.get();

  state_ptr->lease->peer_stub->async()->GetOrCreateManifest(
      &state_ptr->client_context, &state_ptr->request, &state_ptr->response,
      WithExecutor(std::move(executor),
                   [state = std::move(state)](::grpc::Status s) {
                     auto status = internal::GrpcStatusToAbslStatus(s);
                     if (ShouldRevokeLeaseAndRetryAfterError(status)) {
                       StartGetManifestForWriting(std::move(state->promise),
                                                  std::move(state->server),
                                                  std::move(state->lease));
                     } else if (!status.ok()) {
                       state->promise.SetResult(std::move(status));
                     } else {
                       state->promise.SetResult(state->server->clock_());
                     }
                   }));
}

Future<const absl::Time> GetManifestAvailableFuture(
    internal::IntrusivePtr<Cooperator> server) {
  Future<const absl::Time> manifest_available_future;
  Promise<absl::Time> manifest_available_promise;
  {
    absl::MutexLock lock(&server->mutex_);
    manifest_available_future = server->manifest_available_;
    if (manifest_available_future.null()) {
      auto [promise, future] = PromiseFuturePair<absl::Time>::Make();
      server->manifest_available_ = manifest_available_future =
          std::move(future);
      manifest_available_promise = std::move(promise);
    }
  }

  if (!manifest_available_promise.null()) {
    StartGetManifestForWriting(std::move(manifest_available_promise),
                               std::move(server), /*uncooperative_lease=*/{});
  }
  return manifest_available_future;
}

void StartGetManifestForWriting(
    Promise<absl::Time> promise, internal::IntrusivePtr<Cooperator> server,
    LeaseCacheForCooperator::LeaseNode::Ptr uncooperative_lease) {
  // FIXME: Need to take into account lease expiration
  auto root_node_identifier = BtreeNodeIdentifier::Root();
  auto key = root_node_identifier.GetKey(server->storage_identifier_);
  auto lease_future = server->lease_cache()->GetLease(
      key, std::move(root_node_identifier), uncooperative_lease.get());
  LinkValue(
      [server = std::move(server)](
          Promise<absl::Time> promise,
          ReadyFuture<const LeaseCacheForCooperator::LeaseNode::Ptr>
              future) mutable {
        auto lease_node = *future.result();
        if (lease_node->peer_stub) {
          // Wait for remote cooperator to create initial manifest.
          GetManifestForWritingFromPeer(std::move(promise), std::move(server),
                                        std::move(lease_node));
        } else {
          // Create initial manifest locally.
          LinkResult(std::move(promise), internal_ocdbt::EnsureExistingManifest(
                                             server->io_handle_));
        }
      },
      std::move(promise), std::move(lease_future));
}

Future<ManifestWithTime> GetManifestForWriting(
    internal::IntrusivePtr<Cooperator> server, absl::Time staleness_bound) {
  // Check if a manifest has already been written.
  auto read_future = server->io_handle_->GetManifest(staleness_bound);
  auto [promise, future] = PromiseFuturePair<ManifestWithTime>::Make();
  LinkValue(
      [server = std::move(server)](
          Promise<ManifestWithTime> promise,
          ReadyFuture<const ManifestWithTime> future) mutable {
        auto manifest_with_time = future.value();
        if (manifest_with_time.manifest) {
          // Manifest already exists.
          promise.SetResult(std::move(manifest_with_time));
          return;
        }
        // Manifest does not already exist.
        auto manifest_available_future = GetManifestAvailableFuture(server);

        LinkValue(
            [server = std::move(server)](
                Promise<ManifestWithTime> promise,
                ReadyFuture<const absl::Time> future) mutable {
              auto read_future =
                  server->io_handle_->GetManifest(*future.result());
              LinkValue(
                  [server = std::move(server)](
                      Promise<ManifestWithTime> promise,
                      ReadyFuture<const ManifestWithTime> future) mutable {
                    auto manifest_with_time = future.value();
                    if (manifest_with_time.manifest) {
                      // Manifest already exists.
                      promise.SetResult(std::move(manifest_with_time));
                      return;
                    }
                    promise.SetResult(
                        internal_ocdbt_cooperator::
                            ManifestUnexpectedlyDeletedError(*server));
                  },
                  std::move(promise), std::move(read_future));
            },
            std::move(promise), std::move(manifest_available_future));
      },
      std::move(promise), std::move(read_future));
  return std::move(future);
}
}  // namespace

Future<const ManifestWithTime> GetManifestForWriting(
    Cooperator& coop, absl::Time staleness_bound) {
  return GetManifestForWriting(CooperatorPtr(&coop), staleness_bound);
}

grpc::ServerUnaryReactor* Cooperator::GetOrCreateManifest(
    grpc::CallbackServerContext* context,
    const grpc_gen::GetOrCreateManifestRequest* request,
    grpc_gen::GetOrCreateManifestResponse* response) {
  auto* reactor = context->DefaultReactor();
  if (auto status = security_->ValidateServerRequest(context); !status.ok()) {
    reactor->Finish(internal::AbslStatusToGrpcStatus(status));
    return reactor;
  }
  if (!internal::IncrementReferenceCountIfNonZero(*this)) {
    // Shutting down
    reactor->Finish(
        grpc::Status(grpc::StatusCode::CANCELLED, "Cooperator shutting down"));
    return reactor;
  }
  internal::IntrusivePtr<Cooperator> self(this, internal::adopt_object_ref);
  // Before handling request, check that this cooperator owns a lease on the
  // manifest.
  auto root_node_identifier = BtreeNodeIdentifier::Root();
  auto key = root_node_identifier.GetKey(storage_identifier_);
  auto* lease_cache = this->lease_cache();
  if (!lease_cache) {
    NoLeaseError(reactor);
    return reactor;
  }
  auto lease_node_future = lease_cache->FindLease(key);
  if (lease_node_future.null()) {
    // No lease was queried, which means this cooperator cannot possibly own it.
    NoLeaseError(reactor);
    return reactor;
  }
  lease_node_future.ExecuteWhenReady(
      [server = std::move(self),
       reactor](ReadyFuture<const LeaseCacheForCooperator::LeaseNode::Ptr>
                    lease_node_future) mutable {
        auto& r = lease_node_future.result();
        if (!r.ok() || (*r)->peer_stub) {
          // Either the lease query failed (which implies this cooperator does
          // not own the lease), or the query was successful but there is a
          // different owner.
          NoLeaseError(reactor);
          return;
        }
        auto future = GetManifestAvailableFuture(std::move(server));
        future.Force();
        future.ExecuteWhenReady([reactor](
                                    ReadyFuture<const absl::Time> future) {
          reactor->Finish(internal::AbslStatusToGrpcStatus(future.status()));
        });
      });
  return reactor;
}

}  // namespace internal_ocdbt_cooperator
}  // namespace tensorstore
