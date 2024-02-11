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

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/ocdbt/distributed/cooperator_impl.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/create_new_manifest.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/storage_generation.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/write_nodes.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace internal_ocdbt_cooperator {
namespace {
ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");
}

using NodeMutationRequests = Cooperator::NodeMutationRequests;

// Asynchronous operation state for committing a batch of updates to a B+tree
// node.
//
// The commit operation proceeds as follows:
//
// 1. Obtain a manifest (create if necessary).
//
// 2. Asynchronously follow the path from the root to the node, retrieving nodes
// as needed.
//
// 2a. If the node is not found, fail all mutations with ABORTED to indicate
//     that a newer manifest must be used to correctly map those mutations to a
//     node.
//
// 3. Once the target node has been retrieved, stage all pending mutations.
//
// 4. Verify that the node is sufficiently recent for all staged requests.  If
//    not, return to step 1.
//
// 5. Apply mutations to the existing node, writing new nodes to replace it.
//
// 6. If existing node is not the root, submit update to parent node.
//
// 7. If existing node is the root, write new manifest.
struct NodeCommitOperation
    : internal::AtomicReferenceCount<NodeCommitOperation> {
  using Ptr = internal::IntrusivePtr<NodeCommitOperation>;
  internal::IntrusivePtr<Cooperator> server;
  internal::IntrusivePtr<NodeMutationRequests> mutation_requests;
  PendingRequests staged;

  std::shared_ptr<const Manifest> existing_manifest;
  std::shared_ptr<const Manifest> new_manifest;
  absl::Time existing_manifest_time;

  // Key prefix that applies to the current node being processed.
  std::string key_prefix;

  // Key range that bounds the current node being processed.
  KeyRange key_range;

  // Key range that identifies the parent of the current node.
  KeyRange parent_key_range;

  // Generation derived from location of parent of the current node.
  StorageGeneration parent_node_generation;

  // Height of the current node being processed.
  size_t height;

  // Inclusive min key of next child node (after `key_prefix`).
  std::string child_inclusive_min_key;

  // Generation derived from the current node's location.
  StorageGeneration node_generation;

  // Linked to all local writes that must be flushed before the updated
  // node/nodes can be referenced.
  FlushPromise flush_promise;

  // Starts or restarts the commit operation.
  //
  // Args:
  //   commit_op: Commit operation state.
  //   manifest_staleness_bound: Staleness bound on the manifest.
  static void StartCommit(NodeCommitOperation::Ptr commit_op,
                          absl::Time manifest_staleness_bound);

  // Called when the existing manifest has been successfully retrieved and saved
  // in `commit_op`.
  //
  // Begins an asynchronous traversal down from the root along the path to the
  // node corresponding to `commit_op->mutation_requests`.
  static void ExistingManifestReady(NodeCommitOperation::Ptr commit_op);

  // Called to asynchronously traverse down the path to the node corresponding
  // to `commit_op->mutation_requests`.
  //
  // Asynchronously retrieves the node indicated by `node_ref` and then calls
  // `VisitNode`.
  static void VisitNodeReference(NodeCommitOperation::Ptr commit_op,
                                 const BtreeNodeReference& node_ref);

  // Called to traverse down from `node` along the path to the node
  // corresponding to `commit_op->mutation_requests`.
  static void VisitNode(NodeCommitOperation::Ptr commit_op,
                        const BtreeNode& node);

  // Applies the mutations to existing node `node` which must correspond to
  // `commit_op->mutation_requests`.
  //
  // If there is no existing node, `node` is nullptr.
  static void ApplyMutations(NodeCommitOperation::Ptr commit_op,
                             const BtreeNode* node);

  // Called once the node updates have been applied successfully or an error has
  // occurred.
  //
  // If there are any new pending updates in `mutation_requests`, this results
  // in another commit starting.
  void Done();

  // Stages all pending requests in `mutation_requests` for inclusion in this
  // commit operation.
  void StagePending();

  // Called to indicate that the commit failed.
  //
  // The error status is propagated to all staged requests.
  void SetError(const absl::Status& status);

  // Called to indicate that the commit succeeded.
  //
  // The `root_generation` and `time` are propagated to all staged requests.
  void SetSuccess(GenerationNumber root_generation, absl::Time time);

  // Called to indicate that there is no node corresponding to
  // `mutation_requests`, presumably due to a concurrent modification.
  void LeasedNodeGone();

  // Called to do the work of `ApplyMutations` for either interior or leaf
  // nodes.
  //
  // Only two combinations of template arguments are allowed:
  //
  // 1. `Mutation = BtreeLeafNodeWriteMutation`, `Entry = LeafNodeEntry`
  // 2. `Mutation = BtreeInteriorNodeWriteMutation`, `Entry = InteriorNodeEntry`
  template <typename Mutation, typename Entry>
  static void ApplyMutationsForEntry(NodeCommitOperation::Ptr commit_op,
                                     const BtreeNode* node);

  // Computes the result of applying one or more mutation requests for a given
  // key.
  //
  // Args:
  //   existing_entry: The existing entry for the key, if present.
  //   key_prefix: The key prefix for `existing_entry`, if present.
  //   staged_requests: Remaining mutation requests, the first `k >= 1` requests
  //     correspond to the key.  Note: This function must determine `k`, as it
  //     is not yet known to the caller.
  //
  // Returns:
  //
  //   Pair where:
  //
  //   - The first element is the first entry in `staged_requests` that does not
  //     correspond to `existing_entry`, i.e. `staged_requests.begin() + k`.
  //
  //   - The second element is the request corresponding to `existing_entry`, if
  //     any, that should replace it.  A value of `nullptr` indicates to delete
  //     `existing_entry` without replacing it.  A value of `std::nullopt`
  //     indicates to leave `existing_entry` unchanged.
  template <typename Mutation, typename Entry>
  static std::pair<const PendingRequest*, std::optional<const PendingRequest*>>
  ResolveMutationsForKey(const Entry* existing_entry,
                         std::string_view key_prefix,
                         span<const PendingRequest> staged_requests);

  // Called if the existing node corresponding to `commit_op` is not the root
  // node.
  //
  // Sends a mutation request to the parent.
  //
  // If `new_entries` is `std::nullopt`, merely ensures that the existing entry
  // for the node corresponding to `commit_op` is unchanged.
  static void UpdateParent(
      NodeCommitOperation::Ptr commit_op,
      std::optional<std::vector<InteriorNodeEntryData<std::string>>>
          new_entries);

  // Called if the existing node corresponding to `commit_op` is the root node.
  //
  // Calls `WriteNewManifest` to update the manifest directly.
  //
  // If `new_entries` is `std::nullopt`, merely ensures that the existing
  // manifest is unchanged.
  static void UpdateRoot(
      NodeCommitOperation::Ptr commit_op,
      std::optional<std::vector<InteriorNodeEntryData<std::string>>>
          new_entries);

  // Called asynchronously by `UpdateRoot` to create the new manifest.
  static void CreateNewManifest(
      NodeCommitOperation::Ptr commit_op,
      std::optional<BtreeGenerationReference> new_generation);

  // Called asynchronously by `CreateNewManifest` to write the new manifest.
  static void WriteNewManifest(NodeCommitOperation::Ptr commit_op);

  // Restarts the commit, in the case that the manifest was concurrently
  // modified.
  static void RetryCommit(NodeCommitOperation::Ptr commit_op);
};

void NodeCommitOperation::StartCommit(NodeCommitOperation::Ptr commit_op,
                                      absl::Time manifest_staleness_bound) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "[Port=" << commit_op->server->listening_port_ << "] StartCommit";
  auto manifest_future =
      GetManifestForWriting(*commit_op->server, manifest_staleness_bound);
  manifest_future.Force();
  manifest_future.ExecuteWhenReady(
      [commit_op = std::move(commit_op)](
          ReadyFuture<const ManifestWithTime> future) mutable {
        TENSORSTORE_ASSIGN_OR_RETURN(const auto& manifest_with_time,
                                     future.result(), commit_op->SetError(_));
        assert(manifest_with_time.manifest);
        commit_op->existing_manifest = manifest_with_time.manifest;
        commit_op->existing_manifest_time = manifest_with_time.time;
        ExistingManifestReady(std::move(commit_op));
      });
}

void NodeCommitOperation::ExistingManifestReady(
    NodeCommitOperation::Ptr commit_op) {
  auto& latest_version = commit_op->existing_manifest->latest_version();
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "[Port=" << commit_op->server->listening_port_
      << "] ExistingManifestReady: root=" << latest_version.root
      << ", root_height=" << static_cast<int>(latest_version.root_height);
  // FIXME: maybe check height/depth compatibility
  commit_op->height = latest_version.root_height;
  commit_op->key_prefix.clear();
  commit_op->key_range = {};
  commit_op->parent_key_range = {};
  commit_op->child_inclusive_min_key = {};
  if (commit_op->existing_manifest->latest_version()
          .root.location.IsMissing()) {
    commit_op->node_generation.value.clear();
    ApplyMutations(std::move(commit_op), /*node=*/nullptr);
    return;
  }
  VisitNodeReference(std::move(commit_op), latest_version.root);
}

void NodeCommitOperation::VisitNodeReference(
    NodeCommitOperation::Ptr commit_op, const BtreeNodeReference& node_ref) {
  commit_op->node_generation = internal_ocdbt::ComputeStorageGeneration(
      node_ref.location, commit_op->key_prefix);

  auto read_future =
      commit_op->server->io_handle_->GetBtreeNode(node_ref.location);
  auto executor = commit_op->server->io_handle_->executor;
  std::move(read_future)
      .ExecuteWhenReady(WithExecutor(
          std::move(executor),
          [commit_op = std::move(commit_op)](
              ReadyFuture<const std::shared_ptr<const BtreeNode>> future) {
            TENSORSTORE_ASSIGN_OR_RETURN(auto node, future.result(),
                                         commit_op->SetError(_));
            TENSORSTORE_RETURN_IF_ERROR(
                ValidateBtreeNodeReference(*node, commit_op->height,
                                           commit_op->child_inclusive_min_key),
                commit_op->SetError(_));
            VisitNode(std::move(commit_op), *node);
          }));
}

void NodeCommitOperation::VisitNode(NodeCommitOperation::Ptr commit_op,
                                    const BtreeNode& node) {
  commit_op->key_prefix += node.key_prefix;
  auto& node_identifier = commit_op->mutation_requests->node_identifier;
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "[Port=" << commit_op->server->listening_port_
      << "] VisitNode: node_identifier=" << node_identifier
      << ", key_range=" << commit_op->key_range
      << ", parent_key_range=" << commit_op->parent_key_range
      << ", key_prefix=" << tensorstore::QuoteString(commit_op->key_prefix);
  if (commit_op->height == node_identifier.height) {
    ApplyMutations(std::move(commit_op), &node);
    return;
  }

  // Find child node.
  assert(node.height > 0);
  auto& entries = std::get<BtreeNode::InteriorNodeEntries>(node.entries);

  ComparePrefixedKeyToUnprefixedKey compare_existing_and_new_keys{
      commit_op->key_prefix};

  std::string_view inclusive_min_key = node_identifier.range.inclusive_min;

  auto it = std::upper_bound(
      entries.begin(), entries.end(), inclusive_min_key,
      [&](std::string_view inclusive_min_key, const InteriorNodeEntry& entry) {
        return compare_existing_and_new_keys(entry.key, inclusive_min_key) > 0;
      });
  KeyRange child_key_range;
  if (it != entries.begin()) {
    --it;
  }
  if (it == entries.begin()) {
    child_key_range.inclusive_min = commit_op->key_range.inclusive_min;
  } else {
    child_key_range.inclusive_min =
        tensorstore::StrCat(commit_op->key_prefix, it->key);
    if (child_key_range.inclusive_min > node_identifier.range.inclusive_min) {
      // Current inclusive_min bound is already more constrained than the target
      // node.  It is therefore impossible for a child node to have a range of
      // `node_identifier.range`.
      ABSL_LOG_IF(INFO, ocdbt_logging)
          << "[Port=" << commit_op->server->listening_port_
          << "] VisitNode: node_identifier=" << node_identifier
          << ", child_key_range.inclusive_min="
          << child_key_range.inclusive_min;
      commit_op->LeasedNodeGone();
      return;
    }
  }

  if (it + 1 == entries.end()) {
    child_key_range.exclusive_max = commit_op->key_range.exclusive_max;
  } else {
    child_key_range.exclusive_max =
        tensorstore::StrCat(commit_op->key_prefix, (it + 1)->key);
    if (KeyRange::CompareExclusiveMax(child_key_range.exclusive_max,
                                      node_identifier.range.exclusive_max) <
        0) {
      // Current exclusive_max bound is already more constrained than the
      // target node.
      ABSL_LOG_IF(INFO, ocdbt_logging)
          << "[Port=" << commit_op->server->listening_port_
          << "] VisitNode: node_identifier=" << node_identifier
          << ", child_key_range.exclusive_max="
          << child_key_range.exclusive_max;
      commit_op->LeasedNodeGone();
      return;
    }
  }

  if (commit_op->height == node_identifier.height + 1 &&
      node_identifier.range != child_key_range) {
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "[Port=" << commit_op->server->listening_port_
        << "] VisitNode: node_identifier=" << node_identifier
        << ", child_key_range=" << child_key_range
        << ", height=" << static_cast<int>(commit_op->height);
    commit_op->LeasedNodeGone();
    return;
  }

  std::string_view subtree_common_prefix =
      std::string_view(it->key).substr(0, it->subtree_common_prefix_length);

  commit_op->key_prefix += subtree_common_prefix;

  commit_op->child_inclusive_min_key =
      std::string_view(it->key).substr(it->subtree_common_prefix_length);
  --commit_op->height;
  commit_op->parent_key_range = std::move(commit_op->key_range);
  commit_op->parent_node_generation = std::move(commit_op->node_generation);
  commit_op->key_range = std::move(child_key_range);
  VisitNodeReference(std::move(commit_op), it->node);
}

void NodeCommitOperation::ApplyMutations(NodeCommitOperation::Ptr commit_op,
                                         const BtreeNode* node) {
  commit_op->StagePending();
  auto& staged = commit_op->staged;
  auto& manifest = *commit_op->existing_manifest;
  if (staged.latest_root_generation > manifest.latest_generation() &&
      staged.node_generation_at_latest_root_generation !=
          commit_op->node_generation &&
      commit_op->existing_manifest_time <= staged.latest_manifest_time) {
    // Node has been modified, must attempt to read newer node.
    //
    // This indicates a failure of the leasing mechanism.
    RetryCommit(std::move(commit_op));
    return;
  }
  BtreeNodeHeight height = commit_op->height;
  if (height == 0) {
    ApplyMutationsForEntry<BtreeLeafNodeWriteMutation, LeafNodeEntry>(
        std::move(commit_op), node);
  } else {
    ApplyMutationsForEntry<BtreeInteriorNodeWriteMutation, InteriorNodeEntry>(
        std::move(commit_op), node);
  }
}

void NodeCommitOperation::Done() {
  UniqueWriterLock lock{mutation_requests->mutex};
  mutation_requests->commit_in_progress = false;
  MaybeCommit(*server, std::move(mutation_requests), std::move(lock));
}

void NodeCommitOperation::StagePending() {
  absl::MutexLock lock(&mutation_requests->mutex);
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "[Port=" << server->listening_port_
      << "] StagePending: initial staged=" << staged.requests.size()
      << ", pending=" << mutation_requests->pending.requests.size();
  staged.Append(std::move(mutation_requests->pending));
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "[Port=" << server->listening_port_
      << "] StagePending: final staged=" << staged.requests.size()
      << ", pending=" << mutation_requests->pending.requests.size();
}

void NodeCommitOperation::SetError(const absl::Status& status) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "[Port=" << server->listening_port_ << "] SetError: " << status;
  if (staged.requests.empty()) {
    // Stage all pending requests to avoid an infinite retry loop.
    StagePending();
  }
  for (auto& request : staged.requests) {
    if (request.index_within_batch != 0) continue;
    request.batch_promise.SetResult(status);
  }
  Done();
}

void NodeCommitOperation::SetSuccess(GenerationNumber root_generation,
                                     absl::Time time) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "[Port=" << server->listening_port_
      << "] SetSuccess: root_generation=" << root_generation
      << ", time=" << time;
  for (auto& request : staged.requests) {
    if (request.index_within_batch != 0) continue;
    auto& p = request.batch_promise;
    auto& response = *p.raw_result();
    response.root_generation = root_generation;
    response.time = time;
  }
  Done();
}

void NodeCommitOperation::LeasedNodeGone() {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "[Port=" << server->listening_port_
      << "] LeasedNodeGone: node_identifier="
      << mutation_requests->node_identifier
      << ", root=" << existing_manifest->latest_version().root;
  SetError(absl::AbortedError("Leased node no longer exists"));
}

template <typename Mutation, typename Entry>
void NodeCommitOperation::ApplyMutationsForEntry(
    NodeCommitOperation::Ptr commit_op, const BtreeNode* node) {
  static_assert((std::is_same_v<Mutation, BtreeLeafNodeWriteMutation> &&
                 std::is_same_v<Entry, LeafNodeEntry>) ||
                (std::is_same_v<Mutation, BtreeInteriorNodeWriteMutation> &&
                 std::is_same_v<Entry, InteriorNodeEntry>));
  // Use stable sort to ensure multiple writes to the same key are resolved
  // consistently.
  std::stable_sort(
      commit_op->staged.requests.begin(), commit_op->staged.requests.end(),
      [](const PendingRequest& a, const PendingRequest& b) {
        return static_cast<const Mutation&>(*a.mutation).inclusive_min() <
               static_cast<const Mutation&>(*b.mutation).inclusive_min();
      });
  BtreeNodeHeight height = commit_op->height;
  const std::string key_prefix = commit_op->key_prefix;
  BtreeNodeEncoder<Entry> node_encoder(commit_op->existing_manifest->config,
                                       height, key_prefix);
  ComparePrefixedKeyToUnprefixedKey compare_existing_and_new_keys{key_prefix};
  bool modified = false;
  span<const Entry> existing_entries;
  if (node) {
    existing_entries = std::get<std::vector<Entry>>(node->entries);
  }
  auto existing_it = existing_entries.begin();
  span<const PendingRequest> staged_requests = commit_op->staged.requests;
  for (auto mutation_it = staged_requests.begin();
       mutation_it != staged_requests.end();) {
    // 3-way comparison result between inclusive_min of current existing child
    // entry (`existing_it`) and current mutation entry (`mutation_it`).
    int c = 0;
    {
      auto& mutation = static_cast<const Mutation&>(*mutation_it->mutation);
      if (existing_it == existing_entries.end()) {
        c = 1;
      } else {
        if (std::is_same_v<Entry, InteriorNodeEntry> &&
            existing_it == existing_entries.begin()) {
          // Note: Redundant `if constexpr` condition needed since
          // `mutation.existing_range` is only valid for interior nodes.
          if constexpr (std::is_same_v<Entry, InteriorNodeEntry>) {
            c = commit_op->key_range.inclusive_min.compare(
                mutation.existing_range.inclusive_min);
          }
        } else {
          c = compare_existing_and_new_keys(existing_it->key,
                                            mutation.inclusive_min());
        }
      }

      if (c < 0) {
        // Existing entry comes before mutation.
        node_encoder.AddEntry(/*existing=*/true, Entry(*existing_it));
        ++existing_it;
        continue;
      }

      if constexpr (std::is_same_v<Entry, InteriorNodeEntry>) {
        if (c == 0) {
          // Check that exclusive_max matches
          bool exclusive_max_equal;
          if (existing_it + 1 == existing_entries.end()) {
            exclusive_max_equal = (mutation.existing_range.exclusive_max ==
                                   commit_op->key_range.exclusive_max);
          } else {
            exclusive_max_equal =
                compare_existing_and_new_keys(
                    (existing_it + 1)->key,
                    mutation.existing_range.exclusive_max) == 0;
          }
          if (!exclusive_max_equal) {
            c = 1;
          }
        }
      }
    }

    // Existing key is >= mutation key.

    // There may be more than one consecutive mutation with the same key.
    auto [new_mutation_it, effective_request] =
        ResolveMutationsForKey<Mutation, Entry>(
            c == 0 ? existing_it : nullptr, key_prefix,
            {&*mutation_it, staged_requests.end()});
    mutation_it = new_mutation_it;

    if (effective_request.has_value()) {
      const PendingRequest* req = *effective_request;
      if (req != nullptr) {
        if (AddNewEntries(node_encoder,
                          static_cast<const Mutation&>(*req->mutation))) {
          commit_op->flush_promise.Link(req->flush_future);
          modified = true;
        }
      }
      if (c == 0) {
        // Even if no new entry is added, node is modified by removal of
        // existing entry.
        modified = true;
      }
    } else {
      // No changes.
      if (c == 0) {
        node_encoder.AddEntry(/*existing=*/true, Entry(*existing_it));
      }
    }

    if (c == 0) {
      ++existing_it;
    }
  }

  // Emit any remaining existing entries that are ordered after all of the
  // mutations.
  for (; existing_it != existing_entries.end(); ++existing_it) {
    node_encoder.AddEntry(/*existing=*/true, Entry(*existing_it));
  }

  std::optional<std::vector<InteriorNodeEntryData<std::string>>> new_entries;
  bool may_be_root =
      commit_op->mutation_requests->lease_node->node_identifier.range.full();

  if (modified) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto encoded_nodes,
                                 node_encoder.Finalize(may_be_root),
                                 commit_op->SetError(_));
    new_entries = internal_ocdbt::WriteNodes(*commit_op->server->io_handle_,
                                             commit_op->flush_promise,
                                             std::move(encoded_nodes));
  }

  if (may_be_root) {
    UpdateRoot(std::move(commit_op), std::move(new_entries));
  } else {
    UpdateParent(std::move(commit_op), std::move(new_entries));
  }
}

template <typename Mutation, typename Entry>
std::pair<const PendingRequest*, std::optional<const PendingRequest*>>
NodeCommitOperation::ResolveMutationsForKey(
    const Entry* existing_entry, std::string_view key_prefix,
    span<const PendingRequest> staged_requests) {
  // There may be more than one consecutive mutation with the same key.
  auto existing_generation = existing_entry
                                 ? internal_ocdbt::ComputeStorageGeneration(
                                       *existing_entry, key_prefix)
                                 : StorageGeneration::NoValue();
  std::optional<const PendingRequest*> effective_request;
  auto mutation_it = staged_requests.begin();
  const auto& mutation_key =
      static_cast<const Mutation&>(*mutation_it->mutation).key_or_range();
  do {
    auto& mutation = static_cast<const Mutation&>(*mutation_it->mutation);
    auto bit_ref = mutation_it->batch_promise.raw_result()
                       ->conditions_matched[mutation_it->index_within_batch];

    if (StorageGeneration::EqualOrUnspecified(existing_generation,
                                              mutation.existing_generation)) {
      // `if_equal` condition was satisfied, write will be marked as having
      // completed successfully.
      switch (mutation.mode) {
        case BtreeNodeWriteMutation::kAddNew:
          effective_request = &*mutation_it;
          existing_generation = StorageGeneration::Unknown();
          break;
        case BtreeNodeWriteMutation::kDeleteExisting:
          effective_request = nullptr;
          existing_generation = StorageGeneration::NoValue();
          break;
        case BtreeNodeWriteMutation::kRetainExisting:
          break;
      }
      bit_ref = true;
    } else {
      // `if_equal` condition was not satisfied, write will be marked as
      // having failed due to precondition.
      bit_ref = false;
    }
  } while (
      ++mutation_it != staged_requests.end() &&
      static_cast<const Mutation&>(*mutation_it->mutation).key_or_range() ==
          mutation_key);

  return {mutation_it, effective_request};
}

void NodeCommitOperation::UpdateParent(
    NodeCommitOperation::Ptr commit_op,
    std::optional<std::vector<InteriorNodeEntryData<std::string>>>
        new_entries) {
  auto mutation = internal::MakeIntrusivePtr<BtreeInteriorNodeWriteMutation>();
  mutation->existing_range = commit_op->key_range;
  mutation->existing_generation = commit_op->node_generation;
  if (new_entries.has_value()) {
    mutation->mode = new_entries->empty()
                         ? BtreeNodeWriteMutation::kDeleteExisting
                         : BtreeNodeWriteMutation::kAddNew;
    mutation->new_entries = std::move(*new_entries);
  } else {
    mutation->mode = BtreeNodeWriteMutation::kRetainExisting;
  }
  MutationBatchRequest batch_request;
  batch_request.root_generation =
      commit_op->existing_manifest->latest_generation();
  batch_request.node_generation = std::move(commit_op->parent_node_generation);
  batch_request.mutations.resize(1);
  auto& mutation_request = batch_request.mutations[0];
  mutation_request.mutation = std::move(mutation);
  mutation_request.flush_future = std::move(commit_op->flush_promise).future();
  auto future = SubmitMutationBatch(
      *commit_op->server,
      BtreeNodeIdentifier{static_cast<BtreeNodeHeight>(commit_op->height + 1),
                          commit_op->parent_key_range},
      std::move(batch_request));
  future.Force();
  future.ExecuteWhenReady(
      [commit_op = std::move(commit_op)](
          ReadyFuture<MutationBatchResponse> future) mutable {
        auto& r = future.result();
        if (!r.ok() || !r->conditions_matched[0]) {
          if (r.ok() || absl::IsAborted(r.status())) {
            if (r.ok()) {
              ABSL_LOG_IF(INFO, ocdbt_logging)
                  << "[Port=" << commit_op->server->listening_port_
                  << "] Retrying commit because conditions_matched="
                  << static_cast<bool>(r->conditions_matched[0]);
            } else {
              ABSL_LOG_IF(INFO, ocdbt_logging)
                  << "[Port=" << commit_op->server->listening_port_
                  << "] Retrying commit because: " << r.status();
            }
            // Retry
            auto new_staleness_bound =
                commit_op->existing_manifest_time + absl::Nanoseconds(1);
            StartCommit(std::move(commit_op), new_staleness_bound);
            return;
          }
          commit_op->SetError(r.status());
          return;
        }
        commit_op->SetSuccess(r->root_generation, r->time);
      });
}

void NodeCommitOperation::UpdateRoot(
    NodeCommitOperation::Ptr commit_op,
    std::optional<std::vector<InteriorNodeEntryData<std::string>>>
        new_entries) {
  std::optional<BtreeGenerationReference> new_generation;
  if (new_entries) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        new_generation,
        internal_ocdbt::WriteRootNode(
            *commit_op->server->io_handle_, commit_op->flush_promise,
            commit_op->height, std::move(*new_entries)),
        commit_op->SetError(_));
  }
  CreateNewManifest(std::move(commit_op), std::move(new_generation));
}

void NodeCommitOperation::CreateNewManifest(
    NodeCommitOperation::Ptr commit_op,
    std::optional<BtreeGenerationReference> new_generation) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "[Port=" << commit_op->server->listening_port_
      << "] WriteNewManifest: Initiate";

  if (!new_generation) {
    // Leave manifest unmodified.
    commit_op->new_manifest = commit_op->existing_manifest;
    WriteNewManifest(std::move(commit_op));
    return;
  }

  auto new_manifest_future = internal_ocdbt::CreateNewManifest(
      commit_op->server->io_handle_, commit_op->existing_manifest,
      *new_generation);
  new_manifest_future.Force();
  new_manifest_future.ExecuteWhenReady(
      [commit_op = std::move(commit_op)](
          ReadyFuture<std::pair<std::shared_ptr<Manifest>, Future<const void>>>
              future) mutable {
        auto [manifest, manifest_flush_future] = future.value();
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "[Port=" << commit_op->server->listening_port_
            << "] WriteNewManifest: New manifest generated.  root="
            << manifest->latest_version().root << ", root_height="
            << static_cast<int>(manifest->latest_version().root_height);
        commit_op->new_manifest = manifest;
        commit_op->flush_promise.Link(std::move(manifest_flush_future));

        auto flush_future = std::move(commit_op->flush_promise).future();
        if (flush_future.null()) {
          WriteNewManifest(std::move(commit_op));
          return;
        }
        flush_future.Force();
        flush_future.ExecuteWhenReady(
            [commit_op =
                 std::move(commit_op)](ReadyFuture<const void> future) mutable {
              ABSL_LOG_IF(INFO, ocdbt_logging)
                  << "WriteNewManifest: Flushed indirect writes";
              WriteNewManifest(std::move(commit_op));
            });
      });
}

void NodeCommitOperation::WriteNewManifest(NodeCommitOperation::Ptr commit_op) {
  auto update_future = commit_op->server->io_handle_->TryUpdateManifest(
      commit_op->existing_manifest, commit_op->new_manifest,
      /*time=*/absl::Now());
  update_future.Force();
  update_future.ExecuteWhenReady(
      [commit_op =
           std::move(commit_op)](ReadyFuture<TryUpdateManifestResult> future) {
        auto& r = future.result();
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "[Port=" << commit_op->server->listening_port_
            << "] WriteNewManifest: New manifest flushed: " << r.status()
            << ", success=" << (r.ok() ? r->success : false);
        if (!r.ok()) {
          commit_op->SetError(r.status());
          return;
        }
        if (!r->success) {
          RetryCommit(std::move(commit_op));
          return;
        }
        commit_op->SetSuccess(commit_op->new_manifest->latest_generation(),
                              r->time);
      });
}

void NodeCommitOperation::RetryCommit(NodeCommitOperation::Ptr commit_op) {
  auto new_staleness_bound =
      commit_op->existing_manifest_time + absl::Nanoseconds(1);
  StartCommit(std::move(commit_op), new_staleness_bound);
}

// Starts a commit operation if one is not already in progress.
void MaybeCommit(Cooperator& server,
                 internal::IntrusivePtr<NodeMutationRequests> mutation_requests,
                 UniqueWriterLock<absl::Mutex>&& lock) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "[Port=" << server.listening_port_ << "] MaybeCommit: node_identifier="
      << mutation_requests->lease_node->node_identifier
      << ", pending_requests=" << mutation_requests->pending.requests.size();
  while (mutation_requests->pending.requests.empty()) {
    // Attempt to remove.
    lock.unlock();
    absl::MutexLock server_lock(&server.mutex_);
    if (mutation_requests->use_count() == 2) {
      // Remove from map.
      server.node_mutation_map_.erase(mutation_requests->node_key());
      return;
    }
    lock = UniqueWriterLock{mutation_requests->mutex};
  }
  if (mutation_requests->commit_in_progress) return;
  mutation_requests->commit_in_progress = true;
  lock.unlock();
  auto commit_op = internal::MakeIntrusivePtr<NodeCommitOperation>();
  commit_op->server.reset(&server);
  commit_op->mutation_requests = std::move(mutation_requests);
  NodeCommitOperation::StartCommit(
      std::move(commit_op), /*manifest_staleness_bound=*/absl::InfinitePast());
}

}  // namespace internal_ocdbt_cooperator
}  // namespace tensorstore
