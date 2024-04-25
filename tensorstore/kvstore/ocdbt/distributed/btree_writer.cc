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

// This module implements distributed write operations for the OCDBT database.
//
// For simplicity of the initial implementation, currently only single-key
// write/delete operations are implemented directly.  `DeleteRange` operations
// are supported but simply use the non-distributed implementation, meaning that
// there will be high contention if `DeleteRange` operations are attempted
// concurrently with other write operations.
//
// Distributed writing involves two gRPC services:
//
// - The *coordinator* service runs on a single fixed machine and manages the
//   assignment of leases on "node identifiers" (B+tree key ranges along with
//   the node height) to cooperator servers.  It also manages the lease for
//   writing the initial manifest of a new database.  Note that the coordinator
//   service does not itself read or write the B+tree.  The same coordinator
//   service may be used for more than one OCDBT database.
//
//   See `coordinator_server.h` for the server implementation and
//   `btree_node_lease_cache.h` for the client implementation.
//
// - The *cooperator* service runs on every machine that writes to the database.
//   It receives write requests for the node identifiers for which it owns a
//   lease.  The cooperator only deals with a single OCDBT database, but there
//   may be multiple cooperator instances running in the same process.
//
//   See `cooperator.h` for the combined server/client implementation.
//
// Write operations are handled as follows:
//
// 1. [Buffering requests] Write requests are buffered as "pending" requests.
//    As soon as there is at least one pending write operation, a commit is
//    initiated.  See `DistributedBtreeWriter::Write` and
//    `WriterCommitOperation::MaybeStart`.
//
// 2. [Submitting requests to appropriate cooperator] The buffered requests are
//    committed by submitting each request to the cooperator that owns a lease
//    on the corresponding node identifier.
//
// 2a. [Obtaining manifest] As the first step of committing writes, the existing
//     manifest is read, or a new manifest is written.  If there is no existing
//     manifest, the coordinator server is queried to determine which cooperator
//     has a lease for creating the initial manifest.  That cooperator is then
//     contacted in order to wait until the initial manifest is written.  See
//     `WriterCommitOperation::StartCommit`.
//
// 2b. [Staging writes] Once the manifest has been read, any pending writes
//     (including additional writes issued while obtaining the manifest) are
//     moved to a new staging list.  After this point, no additional writes will
//     be added to this "commit", but another commit will be started as soon as
//     step 1 is reached again.  See `WriterCommitOperation::Pending`.
//
// 2b. [Traversal to map writes to nodes] An asynchronous downward traversal of
//     the B+tree starting at the root is used to determine which leaf node key
//     range contains each write operation.  Note that this traversal stops at
//     height 1 nodes; leaf nodes are not read.  See
//     `WriterCommitOperation::TraverseBtreeStartingFromRoot`.
//
// 2c. [Submitting write requests to local cooperator] Each write request is
//     submitted along with its corresponding node identifier to the local
//     in-process cooperator instance to complete.  See
//     `WriterCommitOperation::SubmitRequests`.
//
// 3. [Processing of locally-originated requests by cooperators]
//    (cooperator.h)
//
// 3a. [Querying leases] For each write request that originates locally, the
//     cooperator queries the coordinator to determine which cooperator owns a
//     lease on the node identifier.  The lease information is cached.
//
// 3b. [Forwarding requests] Once the lease is determined, locally-originating
//     requests are forwarded via gRPC to the cooperator that owns the lease
//     (which may be the local cooperator instance).
//
//     - If the gRPC request fails with CANCELLED, UNAVAILABLE, or
//       FAILED_PRECONDITION, that indicates that the cooperator no longer owns
//       the lease.  Control continues at step 3a, and when re-querying the
//       coordinator, the existing lease that led to the error is specified as
//       the "uncooperative lease" that is to be terminated.
//
//     - If the gRPC request fails with ABORTED, that indicates that the tree
//       structure has changed and the node identifier is no longer valid.
//       Control continues at step 1.
//
// 4. [Processing of incoming requests by cooperators]
//    (cooperator_submit_mutation_batch.cc)
//
// 4a. [Buffering requests] For node identifiers owned by the local cooperator,
//     both locally-originating and remote-originating requests are buffered as
//     "pending" requests.  Once there is at least one buffered request for a
//     given node, a commit of the updates to that node is initiated, if one is
//     not already in progress.
//
// 4b. [Starting the commit] To start committing updates to a node, the
//     cooperator first reads the existing node, by following the path from the
//     root.  If there is no existing node corresponding to the node identifier,
//     all pending requests are completed with an error of ABORTED to indicate
//     that the node identifier is no longer valid (presumably because the node
//     identifier was based on an older generation of the tree).
//
// 4c. [Staging requests] Once the existing node has been read, any pending
//     requests for the key range are moved to a "staged request" list.
//
// 4d. [Writing new node] The staged requests are applied to the existing node,
//     producing zero or more new nodes.
//
// 4e. [Updating parent/root]
//
//     - If node is not the root node, the cooperator submits for the parent
//       node identifier a request to replace the reference to the current node
//       with references to the new nodes (via step 3a).  If this results in an
//       ABORTED error, or the existing node is no longer current, it indicates
//       a failure of the lease mechanism or a concurrent non-cooperative write.
//       Control continues at step 4b, ensuring that a more recent manifest is
//       used.
//
//     - If the node is the root node, the cooperator writes any necessary
//       version tree nodes and then updates the manifest.
//
// 4f. [Sending responses] Once the parent has indicated success, the staged
//     requests are completed.

#include "tensorstore/kvstore/ocdbt/distributed/btree_writer.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include <blake3.h>
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/btree_writer.h"
#include "tensorstore/kvstore/ocdbt/distributed/btree_node_identifier.h"
#include "tensorstore/kvstore/ocdbt/distributed/btree_node_write_mutation.h"
#include "tensorstore/kvstore/ocdbt/distributed/cooperator.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/btree_writer.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/storage_generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/bit_vec.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");

struct PendingDistributedRequests {
  struct WriteRequest {
    internal::IntrusivePtr<BtreeLeafNodeWriteMutation> mutation;
    Future<const void> flush_future;
    Promise<TimestampedStorageGeneration> promise;
  };
  std::vector<WriteRequest> write_requests;
  bool needs_inline_value_pass = false;
};

struct StagedDistributedRequests {
  std::vector<PendingDistributedRequests::WriteRequest> write_requests;
};

class DistributedBtreeWriter : public BtreeWriter {
 public:
  Future<TimestampedStorageGeneration> Write(
      std::string key, std::optional<absl::Cord> value,
      kvstore::WriteOptions options) override;
  Future<const void> DeleteRange(KeyRange range) override;
  Future<const void> CopySubtree(CopySubtreeOptions&& options) override;

  // Non-distributed writer instance used to handle `DeleteRange` requests.
  BtreeWriterPtr non_distributed_writer_;

  IoHandle::Ptr io_handle_;

  // Protects access to `pending_` and `commit_in_progress_`.
  absl::Mutex mutex_;

  // Pending write requests not yet submitted to a cooperator.
  PendingDistributedRequests pending_;

  // Set to `true` to indicate that pending requests are being committed, but
  // have not yet been staged.
  bool commit_in_progress_ = false;

  // Address of the coordinator server.
  std::string coordinator_address_;

  // Security method to use with coordinator and cooperators.
  RpcSecurityMethod::Ptr security_;

  // Lease duration to use.
  absl::Duration lease_duration_;

  // The cooperator server, initialized once a commit starts.  Only written by
  // the single thread responsible for the current commit in progress.
  internal_ocdbt_cooperator::CooperatorPtr cooperator_;

  // Unique identifier of base kvstore, used in making requests to the
  // coordinator server.  Currently defined as SHA256 hash of the base kvstore
  // JSON spec.
  std::string storage_identifier_;
};

struct WriterCommitOperation
    : public internal::AtomicReferenceCount<WriterCommitOperation> {
  using Ptr = internal::IntrusivePtr<WriterCommitOperation>;
  internal::IntrusivePtr<DistributedBtreeWriter> writer_;
  std::shared_ptr<const Manifest> existing_manifest_;
  absl::Time existing_manifest_time_;

  const Config& existing_config() {
    auto* config = writer_->io_handle_->config_state->GetExistingConfig();
    assert(config);
    return *config;
  }

  StagedDistributedRequests staged_;

  // Starts an asynchronous commit operation (by calling `StartCommit`) if one
  // is not already in progress.
  //
  // Args:
  //   writer: B+tree writer for which to commit pending mutations.
  //   manifest_staleness_bound: Staleness bound on manifest.
  //   lock: Handle to lock on `writer.mutex_`.
  static void MaybeStart(DistributedBtreeWriter& writer,
                         absl::Time manifest_staleness_bound,
                         UniqueWriterLock<absl::Mutex> lock);

  // Starts an asynchronous commit operation.
  static void StartCommit(DistributedBtreeWriter& writer,
                          absl::Time manifest_staleness_bound);

  // Called to indicate a failure to start the commit and traverse the B+tree.
  //
  // Once this function completes, another commit of pending writes is allowed
  // to start, if `StagePending` was not previously called for this commit.
  void CommitFailed(const absl::Status& error);

  // "Stage" all currently pending requests to be processed by this commit.
  //
  // Any new write operations issued after this is called will be handled by
  // another commit.  Once this function completes, another commit is allowed to
  // be started.
  void StagePending();

  // Begins asynchronously traversing the B+tree in order to determine the node
  // identifier for each staged write operation.
  //
  // The traversal stops at height 1 nodes (leaf nodes are not visited).
  //
  // Only subtrees corresponding to key ranges that contain staged write
  // requests are traversed.
  static void TraverseBtreeStartingFromRoot(
      WriterCommitOperation::Ptr commit_op);

  // State used for the B+tree traversal initiated by
  // `TraverseBtreeStartingFromRoot`.
  struct VisitNodeParameters {
    // Commit operation for which this traversal is being performed.
    WriterCommitOperation::Ptr commit_op;

    // Range of indices within `commit_op->staed_.write_requests` that
    // corresponds to this subtree.
    size_t begin_i, end_i;

    // Node identifier of the current subtree being visited.
    BtreeNodeIdentifier node_identifier;

    // Length of the prefix of `inclusive_min_key` that is a common prefix of
    // all keys within the current subtree.
    KeyLength subtree_common_prefix_length;

    // Inclusive min key within the current non-root subtree being visited
    // (equal to the empty string when visiting the root node).
    std::string inclusive_min_key;

    std::string_view key_prefix() const {
      return std::string_view(inclusive_min_key)
          .substr(0, subtree_common_prefix_length);
    }

    [[maybe_unused]] friend std::ostream& operator<<(
        std::ostream& os, const VisitNodeParameters& x) {
      return os << "{request_range=[" << x.begin_i << ", " << x.end_i
                << "), node_identifier=" << x.node_identifier
                << ", inclusive_min_key="
                << tensorstore::QuoteString(x.inclusive_min_key)
                << ", subtree_common_prefix_length="
                << x.subtree_common_prefix_length << "}";
    }

    void SetError(const absl::Status& error) {
      assert(!error.ok());
      const auto& requests = commit_op->staged_.write_requests;
      for (size_t i = begin_i; i < end_i; ++i) {
        const auto& request = requests[i];
        request.promise.SetResult(error);
      }
    }
  };

  // Asynchronously traverse the subtree rooted at the specified `node_ref`.
  static void VisitNodeReference(VisitNodeParameters&& state,
                                 const BtreeNodeReference& node_ref);

  // Asynchronously traverse the subtree rooted at the specified non-leaf
  // `node`.
  static void VisitNode(VisitNodeParameters&& state,
                        std::shared_ptr<const BtreeNode> node);

  // Submit write requests to the owner of the corresponding node identifier.
  static void SubmitRequests(
      WriterCommitOperation::Ptr commit_op, BtreeNodeIdentifier identifier,
      StorageGeneration node_generation,
      span<const PendingDistributedRequests::WriteRequest> write_requests);
};

void WriterCommitOperation::MaybeStart(DistributedBtreeWriter& writer,
                                       absl::Time manifest_staleness_bound,
                                       UniqueWriterLock<absl::Mutex> lock) {
  if (writer.commit_in_progress_) return;
  // FIXME: maybe have a delay

  // Start commit
  ABSL_LOG_IF(INFO, ocdbt_logging) << "Starting commit";
  writer.commit_in_progress_ = true;
  lock.unlock();

  StartCommit(writer, manifest_staleness_bound);
}

void WriterCommitOperation::StartCommit(DistributedBtreeWriter& writer,
                                        absl::Time manifest_staleness_bound) {
  auto commit_op = internal::MakeIntrusivePtr<WriterCommitOperation>();
  commit_op->writer_.reset(&writer);
  if (!writer.cooperator_) {
    // This is the first commit, start the local cooperator server.  Note that
    // we can defer starting the cooperator until here because we currently
    // don't use the cooperator for read operations.
    //
    // It is safe to modify the `writer->cooperator_` pointer because
    // only a single commit can start at a time, and the pointer is only read
    // after a commit has started.
    internal_ocdbt_cooperator::Options cooperator_options;
    cooperator_options.io_handle = writer.io_handle_;
    cooperator_options.coordinator_address = writer.coordinator_address_;
    cooperator_options.security = writer.security_;
    cooperator_options.lease_duration = writer.lease_duration_;
    cooperator_options.storage_identifier = writer.storage_identifier_;
    TENSORSTORE_ASSIGN_OR_RETURN(
        writer.cooperator_,
        internal_ocdbt_cooperator::Start(std::move(cooperator_options)),
        commit_op->CommitFailed(_));
  }

  internal_ocdbt_cooperator::GetManifestForWriting(*writer.cooperator_,
                                                   manifest_staleness_bound)
      .ExecuteWhenReady(WithExecutor(
          writer.io_handle_->executor,
          [commit_op = std::move(commit_op)](
              ReadyFuture<const ManifestWithTime> future) mutable {
            ABSL_LOG_IF(INFO, ocdbt_logging)
                << "StartCommit: Got manifest for writing: " << future.status();
            TENSORSTORE_ASSIGN_OR_RETURN(auto existing_manifest_with_time,
                                         future.result(),
                                         commit_op->CommitFailed(_));
            ABSL_LOG_IF(INFO, ocdbt_logging)
                << "StartCommit: manifest latest_version="
                << existing_manifest_with_time.manifest->latest_version()
                << ", time=" << existing_manifest_with_time.time;
            commit_op->existing_manifest_ =
                std::move(existing_manifest_with_time.manifest);
            commit_op->existing_manifest_time_ =
                std::move(existing_manifest_with_time.time);
            auto& config_state = *commit_op->writer_->io_handle_->config_state;
            TENSORSTORE_RETURN_IF_ERROR(
                config_state.ValidateNewConfig(
                    commit_op->existing_manifest_->config),
                commit_op->CommitFailed(_));
            commit_op->StagePending();
            TraverseBtreeStartingFromRoot(std::move(commit_op));
          }));
}

void WriterCommitOperation::CommitFailed(const absl::Status& error) {
  ABSL_LOG_IF(INFO, ocdbt_logging) << "Commit failed: " << error;
  assert(!error.ok());
  if (staged_.write_requests.empty()) {
    // No requests have been staged yet.
    //
    // In this case, `error` almost surely relates to reading or writing the
    // manifest, which will likely be a persistent error.
    //
    // Fail all pending requests as well, as otherwise they will be retried,
    // which assuming the error persists, would result in an infinite loop.
    PendingDistributedRequests pending;
    {
      absl::MutexLock lock(&writer_->mutex_);
      std::swap(pending, writer_->pending_);
      writer_->commit_in_progress_ = false;
    }
    staged_.write_requests = std::move(pending.write_requests);
  }
  for (auto& write_request : staged_.write_requests) {
    write_request.promise.SetResult(error);
  }
}

void WriterCommitOperation::StagePending() {
  PendingDistributedRequests pending;
  {
    absl::MutexLock lock(&writer_->mutex_);
    std::swap(pending, writer_->pending_);
    writer_->commit_in_progress_ = false;
  }
  staged_.write_requests = std::move(pending.write_requests);
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Staged write requests: " << staged_.write_requests.size();
  auto config = existing_config();
  const auto max_inline_value_bytes = config.max_inline_value_bytes;
  for (auto& write_request : staged_.write_requests) {
    auto& mutation = *write_request.mutation;
    if (mutation.mode == BtreeNodeWriteMutation::kAddNew) {
      auto& value_ref = mutation.new_entry.value_reference;
      if (auto* value = std::get_if<absl::Cord>(&value_ref); value) {
        if (value->size() > max_inline_value_bytes) {
          auto v = std::move(*value);
          write_request.flush_future = writer_->io_handle_->WriteData(
              std::move(v), value_ref.emplace<IndirectDataReference>());
        }
      }
    }
  }
  std::sort(staged_.write_requests.begin(), staged_.write_requests.end(),
            [](const PendingDistributedRequests::WriteRequest& a,
               const PendingDistributedRequests::WriteRequest& b) {
              return a.mutation->key < b.mutation->key;
            });
}

void WriterCommitOperation::TraverseBtreeStartingFromRoot(
    WriterCommitOperation::Ptr commit_op) {
  auto* existing_manifest = commit_op->existing_manifest_.get();
  ABSL_LOG_IF(INFO, ocdbt_logging) << "TraverseBtreeStartingFromRoot: root="
                                   << existing_manifest->latest_version().root;
  VisitNodeParameters state;
  state.begin_i = 0;
  state.end_i = commit_op->staged_.write_requests.size();
  state.node_identifier.height =
      existing_manifest->latest_version().root_height;
  state.subtree_common_prefix_length = 0;
  state.commit_op = std::move(commit_op);
  VisitNodeReference(std::move(state),
                     existing_manifest->latest_version().root);
}

void WriterCommitOperation::VisitNodeReference(
    VisitNodeParameters&& state, const BtreeNodeReference& node_ref) {
  if (state.node_identifier.height == 0) {
    // Stop traversing because we have reached a leaf node.  This means all
    // staged writes in the range `[state.begin_i, state.end_i)` map to this
    // leaf node.
    auto* commit_op_ptr = state.commit_op.get();
    SubmitRequests(std::move(state.commit_op), std::move(state.node_identifier),
                   internal_ocdbt::ComputeStorageGeneration(node_ref.location,
                                                            state.key_prefix()),
                   span(commit_op_ptr->staged_.write_requests)
                       .subspan(state.begin_i, state.end_i - state.begin_i));
    return;
  }
  // Read the interior node referenced by `node_ref` to continue traversing
  // the subtree.
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "VisitNodeReference: " << state << ", node_ref=" << node_ref;
  auto read_future =
      state.commit_op->writer_->io_handle_->GetBtreeNode(node_ref.location);
  auto executor = state.commit_op->writer_->io_handle_->executor;
  read_future.Force();
  read_future.ExecuteWhenReady(
      [state =
           std::move(state)](ReadyFuture<const std::shared_ptr<const BtreeNode>>
                                 read_future) mutable {
        TENSORSTORE_ASSIGN_OR_RETURN(auto node, read_future.result(),
                                     state.SetError(_));
        auto executor = state.commit_op->writer_->io_handle_->executor;
        executor([state = std::move(state), node = std::move(node)]() mutable {
          VisitNode(std::move(state), std::move(node));
        });
      });
}

void WriterCommitOperation::VisitNode(VisitNodeParameters&& state,
                                      std::shared_ptr<const BtreeNode> node) {
  ABSL_LOG_IF(INFO, ocdbt_logging) << "VisitNode: " << state;
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateBtreeNodeReference(
          *node, state.node_identifier.height,
          std::string_view(state.inclusive_min_key)
              .substr(state.subtree_common_prefix_length)),
      state.SetError(_));

  span<const PendingDistributedRequests::WriteRequest> write_requests =
      state.commit_op->staged_.write_requests;

  std::string existing_key_prefix =
      tensorstore::StrCat(state.key_prefix(), node->key_prefix);

  ComparePrefixedKeyToUnprefixedKey compare_existing_and_new_keys{
      existing_key_prefix};

  span<const InteriorNodeEntry> existing_entries =
      std::get<BtreeNode::InteriorNodeEntries>(node->entries);

  assert(!existing_entries.empty());
  auto existing_it = existing_entries.begin() + 1;

  // Partition the range of write requests for this subtree into the ranges of
  // write requests for each child subtree.
  for (size_t req_end_i = state.end_i, begin_i = state.begin_i;
       begin_i < req_end_i;) {
    size_t req_i = begin_i;
    for (; req_i != req_end_i; ++req_i) {
      auto& req = write_requests[req_i];
      if (existing_it != existing_entries.end() &&
          compare_existing_and_new_keys(existing_it->key, req.mutation->key) <=
              0) {
        break;
      }
    }

    if (req_i != begin_i) {
      VisitNodeParameters sub_state;
      sub_state.commit_op = state.commit_op;
      sub_state.begin_i = begin_i;
      sub_state.end_i = req_i;
      sub_state.node_identifier.height = state.node_identifier.height - 1;
      auto& existing_entry = *(existing_it - 1);
      sub_state.inclusive_min_key =
          tensorstore::StrCat(existing_key_prefix, existing_entry.key);
      if (&existing_entry == &existing_entries.front()) {
        sub_state.node_identifier.range.inclusive_min =
            state.node_identifier.range.inclusive_min;
      } else {
        sub_state.node_identifier.range.inclusive_min =
            sub_state.inclusive_min_key;
      }

      if (existing_it == existing_entries.end()) {
        sub_state.node_identifier.range.exclusive_max =
            state.node_identifier.range.exclusive_max;
      } else {
        sub_state.node_identifier.range.exclusive_max =
            tensorstore::StrCat(existing_key_prefix, existing_it->key);
      }
      size_t subtree_common_prefix_length =
          existing_key_prefix.size() +
          existing_entry.subtree_common_prefix_length;
      if (subtree_common_prefix_length >
          std::numeric_limits<KeyLength>::max()) {
        // FIXME: improve error message
        sub_state.SetError(absl::DataLossError(
            "subtree_common_prefix_length exceeds maximum"));
      } else {
        sub_state.subtree_common_prefix_length =
            static_cast<KeyLength>(subtree_common_prefix_length);
        VisitNodeReference(std::move(sub_state), existing_entry.node);
      }
      begin_i = req_i;

      if (existing_it != existing_entries.end()) {
        ++existing_it;
      }
      continue;
    }
    assert(existing_it != existing_entries.end());

    // No mutations are in the subtree ending at `existing_it`.  Perform a
    // binary search to locate the next partition point.
    auto& req = write_requests[req_i];
    existing_it =
        std::upper_bound(existing_it + 1, existing_entries.end(),
                         std::string_view(req.mutation->key),
                         [&](std::string_view mutation_key,
                             const InteriorNodeEntry& existing_entry) {
                           return compare_existing_and_new_keys(
                                      existing_entry.key, mutation_key) > 0;
                         });
  }
}

void WriterCommitOperation::SubmitRequests(
    WriterCommitOperation::Ptr commit_op, BtreeNodeIdentifier identifier,
    StorageGeneration node_generation,
    span<const PendingDistributedRequests::WriteRequest> write_requests) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "SubmitRequests: node_identifier=" << identifier
      << ", num_requests=" << write_requests.size();
  internal_ocdbt_cooperator::MutationBatchRequest batch_request;
  batch_request.root_generation =
      commit_op->existing_manifest_->latest_generation();
  batch_request.node_generation = std::move(node_generation);
  batch_request.mutations.resize(write_requests.size());
  std::vector write_requests_copy(write_requests.begin(), write_requests.end());
  for (size_t i = 0; i < write_requests_copy.size(); ++i) {
    auto& write_request = write_requests_copy[i];
    auto& mutation_request = batch_request.mutations[i];
    mutation_request.mutation = write_request.mutation;
    mutation_request.flush_future = write_request.flush_future;
  }
  auto future = internal_ocdbt_cooperator::SubmitMutationBatch(
      *commit_op->writer_->cooperator_, std::move(identifier),
      std::move(batch_request));
  std::move(future).ExecuteWhenReady(WithExecutor(
      commit_op->writer_->io_handle_->executor,
      [writer = commit_op->writer_,
       existing_manifest_time = commit_op->existing_manifest_time_,
       write_requests = std::move(write_requests_copy)](
          ReadyFuture<internal_ocdbt_cooperator::MutationBatchResponse>
              future) mutable {
        auto& r = future.result();
        if (r.ok()) {
          auto conditions_matched = r->conditions_matched.bit_span();
          for (size_t i = 0; i < write_requests.size(); ++i) {
            auto& request = write_requests[i];
            TimestampedStorageGeneration stamp;
            stamp.time = r->time;
            if (conditions_matched[i]) {
              auto& mutation = *request.mutation;
              switch (mutation.mode) {
                case BtreeNodeWriteMutation::kRetainExisting:
                  stamp.generation = mutation.existing_generation;
                  break;
                case BtreeNodeWriteMutation::kDeleteExisting:
                  stamp.generation = StorageGeneration::NoValue();
                  break;
                case BtreeNodeWriteMutation::kAddNew:
                  stamp.generation = internal_ocdbt::ComputeStorageGeneration(
                      mutation.new_entry.value_reference);
                  break;
              }
            }
            request.promise.SetResult(std::move(stamp));
          }
          return;
        }
        if (!absl::IsAborted(r.status())) {
          for (auto& request : write_requests) {
            request.promise.SetResult(r.status());
          }
          return;
        }
        // Retry
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "Retrying mutation batch: " << r.status();
        UniqueWriterLock lock{writer->mutex_};
        auto& pending = writer->pending_.write_requests;
        pending.insert(pending.end(), write_requests.begin(),
                       write_requests.end());
        auto new_staleness_bound =
            existing_manifest_time + absl::Nanoseconds(1);
        MaybeStart(*writer, new_staleness_bound, std::move(lock));
      }));
}

Future<TimestampedStorageGeneration> DistributedBtreeWriter::Write(
    std::string key, std::optional<absl::Cord> value,
    kvstore::WriteOptions options) {
  auto& writer = *this;
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Write: " << tensorstore::QuoteString(key) << " " << value.has_value();
  PendingDistributedRequests::WriteRequest request;
  request.mutation = internal::MakeIntrusivePtr<BtreeLeafNodeWriteMutation>();
  request.mutation->key = std::move(key);
  request.mutation->existing_generation =
      std::move(options.generation_conditions.if_equal);
  auto [promise, future] =
      PromiseFuturePair<TimestampedStorageGeneration>::Make(std::in_place);
  request.promise = std::move(promise);

  bool needs_inline_value_pass = false;
  request.mutation->mode = value ? BtreeNodeWriteMutation::kAddNew
                                 : BtreeNodeWriteMutation::kDeleteExisting;
  if (value) {
    auto& new_entry = request.mutation->new_entry;
    auto& value_ref = new_entry.value_reference;
    if (auto* config = writer.io_handle_->config_state->GetExistingConfig();
        !config || value->size() <= config->max_inline_value_bytes) {
      if (!config && !value->empty()) {
        needs_inline_value_pass = true;
      }
      // Config not yet known or value to be written inline.
      value_ref = std::move(*value);
    } else {
      request.flush_future = writer.io_handle_->WriteData(
          std::move(*value), value_ref.emplace<IndirectDataReference>());
    }
  }
  UniqueWriterLock lock{writer.mutex_};
  writer.pending_.write_requests.emplace_back(std::move(request));
  if (needs_inline_value_pass) {
    writer.pending_.needs_inline_value_pass = true;
  }
  WriterCommitOperation::MaybeStart(
      writer, /*manifest_staleness_bound=*/absl::InfinitePast(),
      std::move(lock));
  return std::move(future);
}

Future<const void> DistributedBtreeWriter::DeleteRange(KeyRange range) {
  // TODO(jbms): Implement cooperative write support for `DeleteRange`.
  return non_distributed_writer_->DeleteRange(range);
}

Future<const void> DistributedBtreeWriter::CopySubtree(
    CopySubtreeOptions&& options) {
  return non_distributed_writer_->CopySubtree(std::move(options));
}
}  // namespace

BtreeWriterPtr MakeDistributedBtreeWriter(
    DistributedBtreeWriterOptions&& options) {
  auto writer = internal::MakeIntrusivePtr<DistributedBtreeWriter>();
  writer->io_handle_ = std::move(options.io_handle);

  // Hash to storage identifier so that it is fixed length and can be used more
  // efficiently to compute `BtreeNodeIdentifier` keys.
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  writer->storage_identifier_.resize(32);
  blake3_hasher_update(&hasher, options.storage_identifier.data(),
                       options.storage_identifier.size());
  blake3_hasher_finalize(
      &hasher, reinterpret_cast<uint8_t*>(writer->storage_identifier_.data()),
      writer->storage_identifier_.size());

  // Used for DeleteRange currently.
  writer->non_distributed_writer_ =
      MakeNonDistributedBtreeWriter(writer->io_handle_);
  writer->coordinator_address_ = std::move(options.coordinator_address);
  writer->security_ = std::move(options.security);
  assert(writer->security_);
  writer->lease_duration_ = options.lease_duration;
  writer->storage_identifier_ = std::move(options.storage_identifier);
  return writer;
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
