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

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/ocdbt/distributed/cooperator_impl.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace internal_ocdbt_cooperator {
namespace {
ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");
}

// Asynchronous state for submitting a mutation batch for a particular B+tree
// node (`SubmitMutationBatch`).
//
// This operation proceeds as follows:
//
// 1. This cooperator queries the coordinator to determine which cooperator owns
//    a lease on the node.  (`QueryLease`)
//
// 2. If the lease is owned by this cooperator, the batch request is enqueued
//    directly.  (`HandleRequestLocally`)
//
// 3. If the lease is owned by another cooperator, any indirect writes
//    (referenced from `PendingRequest::flush_future`) are flushed and then the
//    request is sent to the lease owner.  (`SendToPeer`)
//
//    a. If the response indicates that the node no longer exists or the lease
//       is invalid, the batch is retried starting at step 1.
struct SubmitMutationBatchOperation
    : public internal::AtomicReferenceCount<SubmitMutationBatchOperation> {
  using Ptr = internal::IntrusivePtr<SubmitMutationBatchOperation>;

  CooperatorPtr server;
  Promise<MutationBatchResponse> promise;
  BtreeNodeIdentifier node_identifier;
  MutationBatchRequest batch_request;
  LeaseCacheForCooperator::LeaseNode::Ptr lease_node;
  std::optional<grpc::ClientContext> client_context;
  grpc_gen::WriteRequest request;
  grpc_gen::WriteResponse response;
  absl::Time submit_time;

  // Starts the asynchronous operation.
  static Future<MutationBatchResponse> Start(
      Cooperator& server, BtreeNodeIdentifier&& node_identifier,
      MutationBatchRequest&& batch_request) {
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "[Port=" << server.listening_port_
        << "] SubmitMutationBatch: node_identifier=" << node_identifier;
    auto [promise, future] = PromiseFuturePair<MutationBatchResponse>::Make(
        MutationBatchResponse{0, BitVec<>(batch_request.mutations.size())});
    auto state = internal::MakeIntrusivePtr<SubmitMutationBatchOperation>();
    state->node_identifier = std::move(node_identifier);
    state->batch_request = std::move(batch_request);
    state->server.reset(&server);
    state->promise = std::move(promise);
    QueryLease(std::move(state));
    return future;
  }

  // Initiate a query to the coordinator to determine which cooperator owns the
  // lease.
  static void QueryLease(Ptr state) {
    auto* state_ptr = state.get();
    auto key =
        state->node_identifier.GetKey(state->server->storage_identifier_);
    auto future = state_ptr->server->lease_cache()->GetLease(
        key, state_ptr->node_identifier,
        /*uncooperative_lease=*/state_ptr->lease_node.get());
    Link(
        [state = std::move(state)](
            Promise<MutationBatchResponse> promise,
            ReadyFuture<const LeaseCacheForCooperator::LeaseNode::Ptr>
                future) mutable {
          ABSL_LOG_IF(INFO, ocdbt_logging)
              << "SubmitMutationBatch: " << state->node_identifier
              << ": got lease: " << future.status();
          TENSORSTORE_ASSIGN_OR_RETURN(state->lease_node, future.result(),
                                       static_cast<void>(promise.SetResult(_)));
          LeaseNodeReady(std::move(state));
        },
        state_ptr->promise, std::move(future));
  }

  // Called once the lease owner is known.
  static void LeaseNodeReady(Ptr state) {
    if (!state->lease_node->peer_stub) {
      HandleRequestLocally(std::move(state));
    } else {
      HandleRequestRemotely(std::move(state));
    }
  }

  // Called if the lease is owned by this cooperator.
  static void HandleRequestLocally(Ptr state) {
    // FIXME(jbms): Maybe handle lease expiration for local requests as well.
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "SubmitMutationBatch: HandleRequestLocally: "
        << state->node_identifier;
    auto& mutation_requests = state->batch_request.mutations;
    std::vector<PendingRequest> pending_requests(mutation_requests.size());
    for (size_t i = 0; i < pending_requests.size(); ++i) {
      auto& mutation_request = mutation_requests[i];
      auto& pending_request = pending_requests[i];
      pending_request.batch_promise = state->promise;
      pending_request.index_within_batch = i;
      pending_request.mutation = std::move(mutation_request.mutation);
      pending_request.flush_future = std::move(mutation_request.flush_future);
    }

    auto node_mutation_requests = state->server->GetNodeMutationRequests(
        *state->lease_node, state->node_identifier.height);
    UniqueWriterLock lock(node_mutation_requests->mutex);
    PendingRequests new_pending;
    new_pending.requests = std::move(pending_requests);
    node_mutation_requests->pending.Append(std::move(new_pending));
    MaybeCommit(*state->server, std::move(node_mutation_requests),
                std::move(lock));
  }

  // Called if the lease is owned by another cooperator.
  static void HandleRequestRemotely(Ptr state) {
    FlushPromise flush_promise;
    for (auto& request : state->batch_request.mutations) {
      flush_promise.Link(request.flush_future);
    }

    auto flush_future = std::move(flush_promise).future();

    if (!flush_future.null()) {
      // Must flush data before sending request to peer.
      flush_future.Force();
      auto* state_ptr = state.get();
      Link(
          [state = std::move(state)](Promise<MutationBatchResponse> promise,
                                     ReadyFuture<const void> future) mutable {
            ABSL_LOG_IF(INFO, ocdbt_logging)
                << "SubmitMutationBatch: " << state->node_identifier
                << ": Flushed indirect writes: " << future.status();
            TENSORSTORE_RETURN_IF_ERROR(
                future.status(), static_cast<void>(promise.SetResult(_)));
            SendToPeerOnExecutor(std::move(state));
          },
          state_ptr->promise, flush_future);
      return;
    }
    SendToPeerOnExecutor(std::move(state));
  }

  static void SendToPeerOnExecutor(Ptr state) {
    auto& executor = state->server->io_handle_->executor;
    executor(
        [state = std::move(state)]() mutable { SendToPeer(std::move(state)); });
  }

  static void SendToPeer(Ptr state) {
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "[Port=" << state->server->listening_port_
        << "] SendToPeer: " << state->node_identifier;
    auto* state_ptr = state.get();
    // Construct a new `grpc::ClientContext` for this RPC request.  It is not
    // permitted to re-use a `ClientContext` for multiple RPC requests.
    state->client_context.emplace();
    state->request.Clear();
    state->request.set_lease_key(state->lease_node->key);
    state->request.set_root_generation(state->batch_request.root_generation);
    state->request.set_node_generation(
        state->batch_request.node_generation.value);
    state->request.set_node_height(state->node_identifier.height);
    for (auto& mutation_request : state->batch_request.mutations) {
      TENSORSTORE_CHECK_OK(mutation_request.mutation->EncodeTo(
          riegeli::StringWriter{state->request.add_mutations()}));
    }
    state->submit_time = state->server->clock_();
    auto executor = state->server->io_handle_->executor;
    state_ptr->lease_node->peer_stub->async()->Write(
        &*state_ptr->client_context, &state_ptr->request, &state_ptr->response,
        WithExecutor(std::move(executor),
                     [state = std::move(state)](::grpc::Status s) {
                       OnPeerWriteResponse(std::move(state),
                                           internal::GrpcStatusToAbslStatus(s));
                     }));
  }

  static void OnPeerWriteResponse(Ptr state, absl::Status status) {
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "[Port=" << state->server->listening_port_
        << "] SendToPeer: " << state->node_identifier << ", status=" << status;
    if (!status.ok()) {
      if (absl::IsUnavailable(status) || absl::IsFailedPrecondition(status) ||
          absl::IsCancelled(status)) {
        QueryLease(std::move(state));
      } else {
        state->promise.SetResult(status);
      }
      return;
    }
    GenerationNumber new_generation = state->response.root_generation();
    if (!IsValidGenerationNumber(new_generation)) {
      state->promise.SetResult(absl::InternalError(tensorstore::StrCat(
          "Invalid root_generation (", new_generation,
          ") in response from cooperator: ",
          tensorstore::QuoteString(state->lease_node->peer_address))));
      return;
    }
    BitVec<> conditions(state->batch_request.mutations.size());
    assert(conditions.size() == state->request.mutations().size());
    std::string_view response_conditions_matched =
        state->response.conditions_matched();
    const size_t expected_conditions_matched_bytes =
        tensorstore::CeilOfRatio<size_t>(conditions.size(), 8);
    if (response_conditions_matched.size() !=
        expected_conditions_matched_bytes) {
      state->promise.SetResult(absl::InternalError(tensorstore::StrCat(
          "Invalid conditions_matched response from cooperator ",
          tensorstore::QuoteString(state->lease_node->peer_address),
          ": batch_size=", state->batch_request.mutations.size(),
          ", expected_bytes=", expected_conditions_matched_bytes,
          ", actual_bytes=", response_conditions_matched.size())));
      return;
    }
    conditions.bit_span().DeepAssign(
        BitSpan<const unsigned char>(reinterpret_cast<const unsigned char*>(
                                         response_conditions_matched.data()),
                                     0, state->batch_request.mutations.size()));
    state->promise.SetResult(
        MutationBatchResponse{state->response.root_generation(),
                              std::move(conditions), state->submit_time});
  }
};

Future<MutationBatchResponse> SubmitMutationBatch(
    Cooperator& coop, BtreeNodeIdentifier&& node_identifier,
    MutationBatchRequest&& batch_request) {
  return SubmitMutationBatchOperation::Start(coop, std::move(node_identifier),
                                             std::move(batch_request));
}

void EnqueueWriteRequest(Cooperator& server,
                         const LeaseCacheForCooperator::LeaseNode& lease_node,
                         absl::Time request_time,
                         grpc::ServerUnaryReactor* reactor,
                         const grpc_gen::WriteRequest* request,
                         grpc_gen::WriteResponse* response) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "[Port=" << server.listening_port_ << "] EnqueueWriteRequest";
  PendingRequests batch;
  batch.requests.resize(request->mutations().size());
  auto [promise, future] = PromiseFuturePair<MutationBatchResponse>::Make(
      MutationBatchResponse{0, BitVec<>(batch.requests.size())});
  internal::IntrusivePtr<BtreeNodeWriteMutation> write_mutation;
  BtreeNodeHeight node_height =
      static_cast<BtreeNodeHeight>(request->node_height());
  assert(lease_node.node_identifier.range.full() ||
         node_height == lease_node.node_identifier.height);
  for (size_t i = 0; i < batch.requests.size(); ++i) {
    auto& local_request = batch.requests[i];
    local_request.batch_promise = promise;
    local_request.index_within_batch = i;
    riegeli::StringReader<std::string_view> reader{request->mutations(i)};
    absl::Status status;
    if (node_height == 0) {
      auto m = internal::MakeIntrusivePtr<BtreeLeafNodeWriteMutation>();
      status = m->DecodeFrom(reader);
      local_request.mutation = std::move(m);
    } else {
      auto m = internal::MakeIntrusivePtr<BtreeInteriorNodeWriteMutation>();
      status = m->DecodeFrom(reader);
      local_request.mutation = std::move(m);
    }
    TENSORSTORE_RETURN_IF_ERROR(
        status,
        static_cast<void>(reactor->Finish(grpc::Status(
            grpc::StatusCode::INTERNAL,
            tensorstore::StrCat("Failed to decode write request: ", _)))));
  }
  auto mutation_requests =
      server.GetNodeMutationRequests(lease_node, node_height);
  future.ExecuteWhenReady([reactor, response](
                              ReadyFuture<MutationBatchResponse> future) {
    auto& result = future.result();
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "WriteRequest: completed: " << result.status();
    if (!result.ok()) {
      reactor->Finish(internal::AbslStatusToGrpcStatus(result.status()));
      return;
    }
    auto& conditions_matched = result->conditions_matched;
    response->set_root_generation(result->root_generation);
    auto& response_conditions_matched = *response->mutable_conditions_matched();
    response_conditions_matched.resize(
        tensorstore::CeilOfRatio<size_t>(conditions_matched.size(), 8));
    BitSpan<unsigned char>(
        reinterpret_cast<unsigned char*>(response_conditions_matched.data()), 0,
        conditions_matched.size())
        .DeepAssign(conditions_matched.bit_span());
    reactor->Finish(grpc::Status());
  });

  batch.latest_root_generation = request->root_generation();
  batch.latest_manifest_time = request_time;
  batch.node_generation_at_latest_root_generation.value =
      request->node_generation();

  UniqueWriterLock lock(mutation_requests->mutex);
  mutation_requests->pending.Append(std::move(batch));
  MaybeCommit(server, std::move(mutation_requests), std::move(lock));
}

grpc::ServerUnaryReactor* Cooperator::Write(
    grpc::CallbackServerContext* context, const grpc_gen::WriteRequest* request,
    grpc_gen::WriteResponse* response) {
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
  absl::Time request_time = clock_();
  // Before handling the request, check that this cooperator owns the lease.
  auto* lease_cache = this->lease_cache();
  if (!lease_cache) {
    // No lease cache means `Cooperator::Start` has not yet returned, and
    // therefore this cooperator can't possibly own the lease.
    NoLeaseError(reactor);
    return reactor;
  }
  auto lease_node_future = lease_cache->FindLease(request->lease_key());
  if (lease_node_future.null()) {
    // No lease was queried, which means this cooperator cannot possibly own it.
    NoLeaseError(reactor);
    return reactor;
  }
  lease_node_future.ExecuteWhenReady(
      [server = std::move(self), reactor, request, response,
       request_time](ReadyFuture<const LeaseCacheForCooperator::LeaseNode::Ptr>
                         lease_node_future) {
        auto& r = lease_node_future.result();
        if (!r.ok() || (*r)->peer_stub) {
          // Either the lease query failed (which implies this cooperator does
          // not own the lease), or the query was successful but there is a
          // different owner.
          NoLeaseError(reactor);
          return;
        }
        EnqueueWriteRequest(*server, **r, request_time, reactor, request,
                            response);
      });
  return reactor;
}

}  // namespace internal_ocdbt_cooperator
}  // namespace tensorstore
