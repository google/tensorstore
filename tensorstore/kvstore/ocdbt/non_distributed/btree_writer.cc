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

// This module implements committing of changes to the B+tree.
//
// It makes use of the `PendingRequests` and `StagedMutations` data structures
// defined by the `staged_mutations` module.
//
// As `Write` and `DeleteRange` requests are made, they are simply appended to
// the queue contained within a `PendingRequests` data structure.  Out-of-line
// writes of large values are initiated immediately (without checking
// conditions), but not flushed.  Once the request queue is non-empty, a commit
// operation begins.  Once the manifest is available, the commit can proceed.
//
// The commit is ultimately an atomic read-modify-write operation on the
// manifest, and uses the following commit loop:
//
// 1. An empty `StagedMutations` data structure is created.
//
// 2. The existing manifest is read (possibly using a cached value).
//
// 3. The current `PendingRequests` are merged into the normalized
//    `StagedMutations` representation.
//
// 4. We traverse the B+tree top-down (starting from the root), recursively
//    partitioning the ordered list of staged mutations according to the B+tree
//    node structure.  B+tree nodes are fetched as required to perform the
//    partitioning.  Write conditions are checked during this traversal.
//
// 5. Nodes are re-written (and split as required) in a bottom-up fashion.
//    Non-leaf nodes are not rewritten until any child nodes that need to be
//    modified have been re-written.  Note: This step happens concurrently with
//    the traversal described in the previous step.
//
// 6. Once the root B+tree node has been written, a new manifest is created.
//    If all of the inline version slots in the manifest are full, new version
//    tree nodes must be written (which may require reading existing version
//    tree nodes) in order to make space.
//
// 7. The new manifest is written, conditioned on the existing manifest
//    matching what was obtained in step 2.
//
// 8. If the manifest is written successfully, then the commit is done.
//    Otherwise, return to step 2.
//
// TODO(jbms): Currently, B+tree nodes are never merged in response to delete
// operations, which means that lookups are `O(log M)`, where `M` is the total
// number of keys inserted, rather than `O(log N)`, where `N` is the current
// number of keys.  It is expected that for most use cases, there won't be many
// deletes, if any, and for such uses cases merging nodes is unnecessary.  To
// avoid writing too-small nodes, any remaining elements of too-small nodes
// could be propagated back to the parent `NodeTraversalState` and then inserted
// into a sibling (which would require reading or re-reading the sibling first).
//
// TODO(jbms): Currently the asynchronous traversal of the tree is not bounded
// in its memory usage.  That needs to be addressed, e.g. by limiting the number
// of in-flight nodes.

#include "tensorstore/kvstore/ocdbt/non_distributed/btree_writer.h"

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/container/intrusive_red_black_tree.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/btree_writer.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/btree_node_encoder.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/create_new_manifest.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/list.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/staged_mutations.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/storage_generation.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/write_nodes.h"
#include "tensorstore/kvstore/operations.h"
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

class NonDistributedBtreeWriter : public BtreeWriter {
 public:
  using Ptr = internal::IntrusivePtr<NonDistributedBtreeWriter>;
  Future<TimestampedStorageGeneration> Write(
      std::string key, std::optional<absl::Cord> value,
      kvstore::WriteOptions options) override;
  Future<const void> DeleteRange(KeyRange range) override;
  Future<const void> CopySubtree(CopySubtreeOptions&& options) override;

  IoHandle::Ptr io_handle_;

  // Guards access to `pending_` and `commit_in_progress_`.
  absl::Mutex mutex_;

  // Requested write operations that are not yet being committed.  A commit is
  // always started as soon as there are pending requests, but if additional
  // requests are made after the commit operation has already started, those
  // requests are enqueued here.
  PendingRequests pending_;

  // Indicates whether a commit operation is in progress.  Currently guaranteed
  // to be `true` if `pending_` is not empty.
  bool commit_in_progress_;
};

struct CommitOperation
    : public internal::AtomicReferenceCount<CommitOperation> {
  using Ptr = internal::IntrusivePtr<CommitOperation>;
  NonDistributedBtreeWriter::Ptr writer_;
  Future<void> future_;
  std::shared_ptr<const Manifest> existing_manifest_;
  std::shared_ptr<const Manifest> new_manifest_;

  const Config& existing_config() {
    auto* config = writer_->io_handle_->config_state->GetExistingConfig();
    assert(config);
    return *config;
  }

  FlushPromise flush_promise_;
  StagedMutations staged_;

  // Starts a commit operation (by calling `Start`) if one is not already in
  // progress.
  //
  // Args:
  //   writer: Btree writer for which to commit pending mutations.
  //   lock: Handle to lock on `writer.mutex_`.
  static void MaybeStart(NonDistributedBtreeWriter& writer,
                         UniqueWriterLock<absl::Mutex> lock);

  // Starts an asynchronous commit operation.
  //
  // This begins by requesting the existing manifest, and calls `StagePending`
  // and then `TraverseBtreeStartingFromRoot` when the manifest has been
  // retrieved.
  //
  // Precondition:
  //   No commit operation already in progress.
  //
  // Args:
  //   writer: Writer with pending mutations to commit.
  static void Start(NonDistributedBtreeWriter& writer);

  // Reads the existing manifest.
  static void ReadManifest(CommitOperation::Ptr commit_op,
                           absl::Time staleness_bound);

  // Fails the commit operation.
  static void Fail(CommitOperation::Ptr commit_op, const absl::Status& error);

  // "Stages" all pending mutations by merging them into the `StagedMutations`
  // data structure.
  //
  // Args:
  //   promise: Promise
  static void StagePending(CommitOperation& commit_op);

  // Initiates an asynchronous traversal of the B+tree, starting from the root,
  // that visits nodes with a key range that intersects the key range of a
  // staged mutation.
  static void TraverseBtreeStartingFromRoot(CommitOperation::Ptr commit_op,
                                            Promise<void> promise);

  // Specifies a mutation to an interior node.
  //
  // `NodeTraversalState` uses this type to represent the collected mutations
  // from child nodes.
  struct InteriorNodeMutation {
    // Specifies the new entry if `add == true`.  If `add == false`, only
    // `entry.key` is used.
    InteriorNodeEntryData<std::string> entry;

    // Indicates whether to add or replace an existing entry with the specified
    // `key`.  If `true`, the specified `entry` should be added, replacing any
    // existing entry with the same key of `entry.key`.  If `false`, the
    // existing entry with a key of `entry.key` should be removed.
    bool add;
  };

  // Collects mutations to a node during the asynchronous traversal of the
  // B+tree.
  //
  // During the traversal, a tree of `NodeTraversalState` objects is
  // constructed, where each child holds a reference to the parent's
  // `NodeTraversalState` object, to which it sends mutations.  Once the last
  // reference to a `NodeTraversalState` object is released, the asynchronous
  // traversal of all child nodes is known to have completed, and the collected
  // mutations are applied to the node.
  //
  // There are two subtypes:
  //
  // - `RootNodeTraversalState` represents the *parent* of the root node
  //   (essentially corresponding to the manifest).
  //
  // - `InteriorNodeTraversalState` represents an interior node of the B+tree.
  //
  // Note that there is no `NodeTraversalState` type for leaf nodes; leaf nodes
  // are not part of the `NodeTraversalState` tree, since they can be updated
  // synchronously without waiting on any additional I/O.
  struct NodeTraversalState
      : public internal::AtomicReferenceCount<NodeTraversalState> {
    using Ptr = internal::IntrusivePtr<NodeTraversalState>;

    virtual bool is_root_parent() { return false; }

    // Applies collected mutations to this node.
    //
    // Called after the traversal of all children has completed.
    virtual void ApplyMutations() = 0;

    virtual ~NodeTraversalState() = default;

    friend void intrusive_ptr_decrement(NodeTraversalState* p) {
      if (internal::DecrementReferenceCount(*p)) {
        if (p->promise_.result_needed()) {
          p->ApplyMutations();
        }
        delete p;
      }
    }

    CommitOperation::Ptr commit_op_;
    Promise<void> promise_;
    absl::Mutex mutex_;
    std::vector<InteriorNodeMutation> mutations_;
    // Full implicit key prefix of existing subtree.
    std::string existing_subtree_key_prefix_;
    // 1 + the height of the child nodes.
    BtreeNodeHeight height_;
  };

  // Special `NodeTraversalState` type for the *parent* of the root node.
  //
  // This is responsible for increasing the height of the tree as needed and
  // creating the updated manifest.
  struct RootNodeTraversalState : public NodeTraversalState {
    bool is_root_parent() final { return true; }

    void ApplyMutations() final {
      ABSL_LOG_IF(INFO, ocdbt_logging)
          << "ApplyMutations: height=" << static_cast<int>(height_)
          << ", num_mutations=" << mutations_.size();
      if (mutations_.empty()) {
        if (!commit_op_->existing_manifest_) {
          // Since there is no existing manifest, write an initial manifest.
          BtreeGenerationReference ref;
          ref.root_height = 0;
          ref.root.statistics = {};
          ref.root.location = IndirectDataReference::Missing();
          CreateNewManifest(std::move(promise_), std::move(commit_op_), ref);
          return;
        }
        // Leave manifest unchanged.
        commit_op_->new_manifest_ = commit_op_->existing_manifest_;
        NewManifestReady(std::move(promise_), std::move(commit_op_));
        return;
      }

      while (true) {
        // Mutations must be of the form: delete "", add ...
        [[maybe_unused]] auto& deletion_entry = mutations_.front();
        assert(!deletion_entry.add);
        assert(deletion_entry.entry.key.empty());

        if (mutations_.size() == 1) {
          // Root node is empty.
          BtreeGenerationReference ref;
          ref.root_height = 0;
          ref.root.statistics = {};
          ref.root.location = IndirectDataReference::Missing();
          CreateNewManifest(std::move(promise_), std::move(commit_op_), ref);
          return;
        }

        if (mutations_.size() == 2) {
          // Exactly one root node, no need to increase height.
          auto& new_root_mutation = mutations_[1];
          assert(new_root_mutation.add);
          assert(new_root_mutation.entry.key.empty());
          assert(new_root_mutation.entry.subtree_common_prefix_length == 0);
          assert(height_ > 0);
          BtreeGenerationReference ref;
          ref.root_height = height_ - 1;
          ref.root = new_root_mutation.entry.node;
          CreateNewManifest(std::move(promise_), std::move(commit_op_), ref);
          return;
        }

        // Need to add a level to the tree.
        auto mutations = std::exchange(mutations_, {});
        UpdateParent(
            *this, /*existing_relative_child_key=*/{},
            EncodeUpdatedInteriorNodes(commit_op_->existing_config(), height_,
                                       /*existing_prefix=*/{},
                                       /*existing_entries=*/{}, mutations,
                                       /*may_be_root=*/true));
        ++height_;
      }
    }
  };

  // `NodeTraversalState` type for interior nodes of the B+tree.
  struct InteriorNodeTraversalState : public NodeTraversalState {
    NodeTraversalState::Ptr parent_state_;
    std::shared_ptr<const BtreeNode> existing_node_;
    std::string existing_relative_child_key_;

    void ApplyMutations() final {
      ABSL_LOG_IF(INFO, ocdbt_logging)
          << "ApplyMutations: existing inclusive_min="
          << tensorstore::QuoteString(tensorstore::StrCat(
                 parent_state_->existing_subtree_key_prefix_,
                 existing_relative_child_key_))
          << ", height=" << static_cast<int>(height_)
          << ", num_mutations=" << mutations_.size();
      if (mutations_.empty()) {
        // There are no mutations to the key range of this node.  Therefore,
        // this node can remain referenced unmodified from its parent.
        return;
      }

      UpdateParent(
          *parent_state_, existing_relative_child_key_,
          EncodeUpdatedInteriorNodes(
              commit_op_->existing_config(), height_,
              existing_subtree_key_prefix_,
              std::get<BtreeNode::InteriorNodeEntries>(existing_node_->entries),
              mutations_,
              /*may_be_root=*/parent_state_->is_root_parent()));
    }
  };

  // Parameters needed to traverse a subtree rooted at a given requested node.
  struct VisitNodeReferenceParameters {
    // Traversal state of the *parent* of the requested node.
    NodeTraversalState::Ptr parent_state;

    // The key suffix (lower bound) used to identify the requested node in the
    // parent node, or the empty string if the requested node is the root.  The
    // full key is given by
    // `parent_state->existing_subtree_key_prefix + inclusive_min_key_suffix`.
    std::string inclusive_min_key_suffix;

    // The prefix of `inclusive_min_key_suffix` that is a common prefix of all
    // keys stored in the subtree rooted at the requested node.
    KeyLength subtree_common_prefix_length;

    // Key range corresponding to the requested node.
    //
    // For nodes that are *not* along the left-most path of the B+tree,
    // `key_range.inclusive_min` is equal to
    // `parent_state->existing_subtree_key_prefix + inclusive_min_key_suffix`.
    // For nodes along the left-most path of the B+tree,
    // `key_range.inclusive_min` is always the empty string, because for
    // mutation purposes all keys must map to an existing node.
    KeyRange key_range;

    // Range of mutations that intersect `key_range`.
    MutationEntryTree::Range entry_range;
  };

  // Asynchronously traverses the subtree rooted at the specified `node_ref`,
  // propagating to `parent_state` the updated node references that should
  // replace `node_ref` in the updated B+tree.
  //
  // This fetches `node_ref` and then calls `VisitNode`.
  //
  // Args:
  //   params: Parameters for traversing the subtree.
  //   node_ref: Node reference to lookup.
  static void VisitNodeReference(VisitNodeReferenceParameters&& params,
                                 const BtreeNodeReference& node_ref);

  // Parameters needed to traverse a subtree rooted at a node.
  struct VisitNodeParameters {
    // Traversal state for the *parent* of `node`.
    NodeTraversalState::Ptr parent_state;

    // Node to visit.  May be null if visiting a missing (empty) root node.
    std::shared_ptr<const BtreeNode> node;

    // The key suffix (lower bound) used to identify the requested node in the
    // parent node, or the empty string if the requested node is the root.  The
    // full key is given by
    // `parent_state->existing_subtree_key_prefix + inclusive_min_key_suffix`.
    std::string inclusive_min_key_suffix;

    // Full key prefix that applies to all children of `node`.  Note that this
    // already includes `node->key_prefix`.
    std::string full_prefix;

    // Key range corresponding to `node`.
    //
    // For nodes that are *not* along the left-most path of the B+tree,
    // `key_range.inclusive_min` is equal to
    // `parent_state->existing_subtree_key_prefix + inclusive_min_key_suffix`.
    // For nodes along the left-most path of the B+tree,
    // `key_range.inclusive_min` is always the empty string, because for
    // mutation purposes all keys must map to an existing node.
    KeyRange key_range;

    // Range of mutations that intersect `key_range`.
    MutationEntryTree::Range entry_range;
  };

  // Callback invoked once an existing btree node has been retrieved as part of
  // the traversal.
  struct NodeReadyCallback {
    VisitNodeReferenceParameters params;

    void operator()(
        Promise<void> promise,
        ReadyFuture<const std::shared_ptr<const BtreeNode>> read_future) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto node, read_future.result(),
          static_cast<void>(SetDeferredResult(promise, _)));
      TENSORSTORE_RETURN_IF_ERROR(
          ValidateBtreeNodeReference(
              *node, params.parent_state->height_ - 1,
              std::string_view(params.inclusive_min_key_suffix)
                  .substr(params.subtree_common_prefix_length)),
          static_cast<void>(SetDeferredResult(promise, _)));
      auto full_prefix = tensorstore::StrCat(
          params.parent_state->existing_subtree_key_prefix_,
          std::string_view(params.inclusive_min_key_suffix)
              .substr(0, params.subtree_common_prefix_length),
          node->key_prefix);
      VisitNode(VisitNodeParameters{
          std::move(params.parent_state), std::move(node),
          std::move(params.inclusive_min_key_suffix), std::move(full_prefix),
          std::move(params.key_range), params.entry_range});
    }
  };

  // Asynchronously traverses the subtree rooted at `node`, propagating to
  // `parent_state` the new node reference/references that should replace the
  // existing reference to `node` in the updated B+tree.
  //
  // Calls either `VisitInteriorNode` or `VisitLeafNode` depending on
  // `node->height`.
  static void VisitNode(VisitNodeParameters&& params);

  // Asynchronously traverses the subtree rooted at interior node `node`.
  static void VisitInteriorNode(VisitNodeParameters params);

  // Applies mutations to leaf node `node` and propagates to `parent_state` the
  // new node reference/references that should replace the existing reference to
  // `node` in the updated B+tree.
  static void VisitLeafNode(VisitNodeParameters params);

  // Adds mutations to `parent_state` to replace the child with key
  // `existing_relative_child_key` with the children in `encoded_nodes_result`.
  //
  // Args:
  //   parent_state: Parent to modify.
  //   existing_relative_child_key: Key of existing child to replace.
  //   encoded_nodes_result: New children, or error.
  static void UpdateParent(
      NodeTraversalState& parent_state,
      std::string_view existing_relative_child_key,
      Result<std::vector<EncodedNode>>&& encoded_nodes_result);

  // Applies mutations to an interior node.
  //
  // Args:
  //   config: Configuration to use.
  //   height: Height of the node.
  //   existing_prefix: Key prefix that applies to `existing_entries`.
  //   existing_entries: Existing children of the node.
  //   mutations: Mutations to apply.
  //   may_be_root: Whether the updated node will be the root node, if applying
  //     the mutations results in just a single node.
  static Result<std::vector<EncodedNode>> EncodeUpdatedInteriorNodes(
      const Config& config, BtreeNodeHeight height,
      std::string_view existing_prefix,
      span<const InteriorNodeEntry> existing_entries,
      span<InteriorNodeMutation> mutations, bool may_be_root);

  // Adds a new B+tree generation to the manifest.
  //
  // Called once a new root node has been generated.
  //
  // Writes new version tree nodes as needed.
  static void CreateNewManifest(Promise<void> promise,
                                CommitOperation::Ptr commit_op,
                                const BtreeGenerationReference& new_generation);

  // Called once a new manifest has been generated and any necessary version
  // tree nodes have been written.
  //
  // Ensures all indirect writes are flushed, and then marks `promise` as ready.
  static void NewManifestReady(Promise<void> promise,
                               CommitOperation::Ptr commit_op);

  // Attempts to write the new manifest.
  static void WriteNewManifest(CommitOperation::Ptr commit_op);
};

void CommitOperation::MaybeStart(NonDistributedBtreeWriter& writer,
                                 UniqueWriterLock<absl::Mutex> lock) {
  if (writer.commit_in_progress_) return;
  // TODO(jbms): Consider adding a delay here, using `ScheduleAt`.

  // Start commit
  ABSL_LOG_IF(INFO, ocdbt_logging) << "Starting commit";
  writer.commit_in_progress_ = true;
  lock.unlock();

  CommitOperation::Start(writer);
}

void CommitOperation::Start(NonDistributedBtreeWriter& writer) {
  auto commit_op = internal::MakeIntrusivePtr<CommitOperation>();
  commit_op->writer_.reset(&writer);
  ReadManifest(std::move(commit_op), absl::InfinitePast());
}

void CommitOperation::ReadManifest(CommitOperation::Ptr commit_op,
                                   absl::Time staleness_bound) {
  auto read_future =
      commit_op->writer_->io_handle_->GetManifest(staleness_bound);
  read_future.Force();
  read_future.ExecuteWhenReady(
      [commit_op = std::move(commit_op)](
          ReadyFuture<const ManifestWithTime> future) mutable {
        auto& r = future.result();
        if (!r.ok()) {
          Fail(std::move(commit_op), r.status());
          return;
        }
        commit_op->existing_manifest_ = r->manifest;
        auto& executor = commit_op->writer_->io_handle_->executor;
        executor([commit_op]() mutable {
          StagePending(*commit_op);
          auto [promise, future] =
              PromiseFuturePair<void>::Make(absl::OkStatus());
          TraverseBtreeStartingFromRoot(commit_op, std::move(promise));
          future.Force();
          future.ExecuteWhenReady([commit_op = std::move(commit_op)](
                                      ReadyFuture<void> future) mutable {
            auto& r = future.result();
            if (!r.ok()) {
              Fail(std::move(commit_op), r.status());
              return;
            }
            WriteNewManifest(std::move(commit_op));
          });
        });
      });
}

void CommitOperation::Fail(CommitOperation::Ptr commit_op,
                           const absl::Status& error) {
  ABSL_LOG_IF(INFO, ocdbt_logging) << "Commit failed: " << error;
  CommitFailed(commit_op->staged_, error);
  auto& writer = *commit_op->writer_;
  PendingRequests pending;
  {
    UniqueWriterLock lock(writer.mutex_);
    writer.commit_in_progress_ = false;
    std::swap(pending, writer.pending_);
  }
  // Normally, if an error occurs while committing, the error should
  // only propagate to the requests included in the commit; any
  // subsequent requests should be retried.  Although it is likely they
  // will fail as well, there isn't a risk of an infinite loop.

  // However, when there is no existing manifest, we first perform an
  // initial commit leaving all of the requests as pending (since we do
  // not yet know the final configuration).  In this case, if an error
  // occurs, we must fail the pending requests that led to the commit,
  // as otherwise we may get stuck in an infinite loop.

  // FIXME: ideally only abort requests that were added before this
  // commit started.
  AbortPendingRequestsWithError(pending, error);
}

void CommitOperation::StagePending(CommitOperation& commit_op) {
  auto& writer = *commit_op.writer_;
  auto* config = writer.io_handle_->config_state->GetExistingConfig();
  PendingRequests pending;
  // If the config is not yet known, don't actually stage any mutations.
  // Instead, execute the commit operation without any mutations, in order to
  // create a new manifest.
  if (config) {
    absl::MutexLock lock(&writer.mutex_);
    pending = std::exchange(writer.pending_, {});
  }

  if (config) {
    const auto max_inline_value_bytes = config->max_inline_value_bytes;
    for (auto& request : pending.requests) {
      if (request->kind != MutationEntry::kWrite) continue;
      auto& write_request = static_cast<WriteEntry&>(*request);
      if (!write_request.value ||
          !std::holds_alternative<absl::Cord>(*write_request.value) ||
          std::get<absl::Cord>(*write_request.value).size() <=
              max_inline_value_bytes) {
        continue;
      }
      auto value = std::move(std::get<absl::Cord>(*write_request.value));
      auto value_future = writer.io_handle_->WriteData(
          std::move(value),
          write_request.value->emplace<IndirectDataReference>());
      pending.flush_promise.Link(std::move(value_future));
    }
  }
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Stage requests: " << pending.requests.size();
  commit_op.flush_promise_.Link(std::move(pending.flush_promise));
  StageMutations(commit_op.staged_, std::move(pending));
}

void CommitOperation::TraverseBtreeStartingFromRoot(
    CommitOperation::Ptr commit_op, Promise<void> promise) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Manifest ready: generation_number="
      << GetLatestGeneration(commit_op->existing_manifest_.get());
  auto* commit_op_ptr = commit_op.get();
  auto parent_state = internal::MakeIntrusivePtr<RootNodeTraversalState>();
  parent_state->promise_ = std::move(promise);
  parent_state->commit_op_ = std::move(commit_op);
  MutationEntryTree::Range entry_range = commit_op_ptr->staged_.entries;
  if (entry_range.empty()) return;
  auto* existing_manifest = commit_op_ptr->existing_manifest_.get();
  if (!existing_manifest ||
      existing_manifest->latest_version().root.location.IsMissing()) {
    // Manifest does not yet exist or tree is empty.  No need to actually read
    // the root node; just begin the traversal with an empty node.
    parent_state->height_ = 1;
    VisitNode(VisitNodeParameters{std::move(parent_state),
                                  /*node=*/{},
                                  /*inclusive_min_key_suffix=*/{},
                                  /*full_prefix=*/{},
                                  /*key_range=*/{}, entry_range});
    return;
  }
  // Read the root node to begin the traversal.
  auto& latest_version = existing_manifest->latest_version();
  parent_state->height_ = latest_version.root_height + 1;
  VisitNodeReference(
      VisitNodeReferenceParameters{std::move(parent_state),
                                   /*inclusive_min_key_suffix=*/{},
                                   /*subtree_common_prefix_length=*/0,
                                   /*key_range=*/{}, entry_range},
      latest_version.root);
}

void CommitOperation::VisitNodeReference(VisitNodeReferenceParameters&& params,
                                         const BtreeNodeReference& node_ref) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Process node reference: " << params.key_range
      << ", height=" << (params.parent_state->height_ - 1);
  auto read_future =
      params.parent_state->commit_op_->writer_->io_handle_->GetBtreeNode(
          node_ref.location);
  auto executor =
      params.parent_state->commit_op_->writer_->io_handle_->executor;
  auto promise = params.parent_state->promise_;
  Link(WithExecutor(std::move(executor), NodeReadyCallback{std::move(params)}),
       std::move(promise), std::move(read_future));
}

void CommitOperation::VisitNode(VisitNodeParameters&& params) {
  assert(!params.entry_range.empty());
  BtreeNodeHeight height = params.node ? params.node->height : 0;
  if (!params.node) {
    assert(params.inclusive_min_key_suffix.empty());
  }
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "VisitNode: " << params.key_range
      << ", height=" << static_cast<int>(height)
      << ", inclusive_min_key_suffix="
      << tensorstore::QuoteString(params.inclusive_min_key_suffix)
      << ", full_prefix=" << tensorstore::QuoteString(params.full_prefix);
  if (height == 0) {
    // Leaf node.
    VisitLeafNode(std::move(params));
  } else {
    VisitInteriorNode(std::move(params));
  }
}

void CommitOperation::VisitInteriorNode(VisitNodeParameters params) {
  span<const InteriorNodeEntry> existing_entries =
      std::get<BtreeNode::InteriorNodeEntries>(params.node->entries);

  if (existing_entries.empty()) {
    SetDeferredResult(
        params.parent_state->promise_,
        absl::DataLossError("Empty non-root/non-leaf b-tree node found"));
    return;
  }

  auto self_state = internal::MakeIntrusivePtr<InteriorNodeTraversalState>();
  self_state->commit_op_ = params.parent_state->commit_op_;
  self_state->promise_ = params.parent_state->promise_;
  self_state->parent_state_ = std::move(params.parent_state);
  self_state->height_ = params.node->height;
  self_state->existing_node_ = std::move(params.node);
  self_state->existing_subtree_key_prefix_ = std::move(params.full_prefix);
  self_state->existing_relative_child_key_ =
      std::move(params.inclusive_min_key_suffix);

  PartitionInteriorNodeMutations(
      existing_entries, self_state->existing_subtree_key_prefix_,
      params.key_range, params.entry_range,
      [&](const InteriorNodeEntry& existing_entry, KeyRange key_range,
          MutationEntryTree::Range entry_range) {
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "VisitInteriorNode: Partition: existing_entry="
            << tensorstore::QuoteString(
                   self_state->existing_subtree_key_prefix_)
            << "+" << tensorstore::QuoteString(existing_entry.key)
            << ", key_range=" << key_range << ", entry_range="
            << tensorstore::QuoteString(entry_range.begin()->key);
        if (MustReadNodeToApplyMutations(key_range, entry_range)) {
          ABSL_LOG_IF(INFO, ocdbt_logging)
              << "VisitInteriorNode: Partition: existing_entry="
              << tensorstore::QuoteString(
                     self_state->existing_subtree_key_prefix_)
              << "+" << tensorstore::QuoteString(existing_entry.key)
              << ": must visit node";
          // Descend into the subtree to apply mutations.  This is the most
          // common case, and is needed for:
          //
          // - Regular write operations
          //
          // - `DeleteRange` operations that do not start or end precisely at a
          //   child boundary at this level.
          //
          // - Conditional write operation superseded by a `DeleteRange`
          //   operation.
          VisitNodeReference(
              VisitNodeReferenceParameters{
                  self_state, std::string(existing_entry.key),
                  existing_entry.subtree_common_prefix_length,
                  std::move(key_range), entry_range},
              existing_entry.node);
        } else {
          ABSL_LOG_IF(INFO, ocdbt_logging)
              << "VisitInteriorNode: Partition: existing_entry="
              << tensorstore::QuoteString(
                     self_state->existing_subtree_key_prefix_)
              << "+" << tensorstore::QuoteString(existing_entry.key)
              << ": deleting node";
          // Single DeleteRange operation that covers the full key range of
          // `existing_entry`.  Can just delete unconditionally at this level
          // without descending.
          absl::MutexLock lock(&self_state->mutex_);
          auto& mutation = self_state->mutations_.emplace_back();
          mutation.add = false;
          mutation.entry.key = tensorstore::StrCat(
              self_state->existing_subtree_key_prefix_, existing_entry.key);
        }
      });
}

void CommitOperation::VisitLeafNode(VisitNodeParameters params) {
  span<const LeafNodeEntry> existing_entries;
  if (params.node) {
    existing_entries =
        std::get<BtreeNode::LeafNodeEntries>(params.node->entries);
  }
  BtreeLeafNodeEncoder encoder(
      params.parent_state->commit_op_->existing_config(),
      /*height=*/0, params.full_prefix);
  ComparePrefixedKeyToUnprefixedKey compare_existing_and_new_keys{
      params.full_prefix};
  bool modified = false;
  auto existing_it = existing_entries.begin();
  const auto& key_range = params.key_range;
  for (auto entry_it = params.entry_range.begin(),
            entry_range_end = params.entry_range.end();
       entry_it != entry_range_end;) {
    auto* mutation = &*entry_it;
    int c = existing_it != existing_entries.end()
                ? compare_existing_and_new_keys(existing_it->key, mutation->key)
                : 1;
    if (c < 0) {
      // Existing key comes before mutation.
      encoder.AddEntry(/*existing=*/true, LeafNodeEntry(*existing_it));
      ++existing_it;
      continue;
    }

    // Existing key is >= mutation key.
    ++entry_it;
    if (mutation->kind == MutationEntry::kDeleteRange) {
      // Delete range mutation.
      auto& dr_entry = *static_cast<DeleteRangeEntry*>(mutation);

      // Determine the set of superseded writes to validate against this leaf
      // node.  Only validate superseded writes that are within `key_range`.
      // The `DeleteRangeEntry` may not be fully contained within the range of
      // this leaf node, in which case it will also be processed by other leaf
      // nodes.
      auto superseded_writes =
          GetWriteEntryInterval(dr_entry.superseded_writes, key_range);

      // Validate any superseded writes and skip past any existing entries
      // ordered <= superseded writes.
      auto new_existing_it = ValidateSupersededWriteEntries(
          superseded_writes, span(existing_it, existing_entries.end()),
          params.full_prefix);

      // Skip past any existing entries covered by the deletion range.
      std::string_view exclusive_max = dr_entry.exclusive_max;
      if (exclusive_max.empty()) {
        new_existing_it = existing_entries.end();
      } else {
        for (; new_existing_it != existing_entries.end(); ++new_existing_it) {
          if (compare_existing_and_new_keys(new_existing_it->key,
                                            exclusive_max) >= 0) {
            break;
          }
        }
      }

      if (new_existing_it != existing_it) {
        modified = true;
        existing_it = new_existing_it;
      }
      continue;
    }

    // Write mutation
    auto& write_entry = *static_cast<const WriteEntry*>(mutation);

    auto existing_generation = (c == 0)
                                   ? internal_ocdbt::ComputeStorageGeneration(
                                         existing_it->value_reference)
                                   : StorageGeneration::NoValue();
    auto new_value =
        ApplyWriteEntryChain(std::move(existing_generation), write_entry);
    if (new_value) {
      modified = true;
      if (*new_value) {
        LeafNodeEntry new_entry;
        new_entry.key = write_entry.key;
        new_entry.value_reference = **new_value;
        encoder.AddEntry(/*existing=*/false, std::move(new_entry));
      }
    } else if (c == 0) {
      encoder.AddEntry(/*existing=*/true, LeafNodeEntry(*existing_it));
    }
    if (c == 0) {
      ++existing_it;
    }
  }

  if (!modified) return;

  for (; existing_it != existing_entries.end(); ++existing_it) {
    encoder.AddEntry(/*existing=*/true, LeafNodeEntry(*existing_it));
  }

  UpdateParent(*params.parent_state, params.inclusive_min_key_suffix,
               encoder.Finalize(params.parent_state->is_root_parent()));
}

void CommitOperation::UpdateParent(
    NodeTraversalState& parent_state,
    std::string_view existing_relative_child_key,
    Result<std::vector<EncodedNode>>&& encoded_nodes_result) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto encoded_nodes, std::move(encoded_nodes_result),
      static_cast<void>(SetDeferredResult(parent_state.promise_, _)));

  auto new_entries = internal_ocdbt::WriteNodes(
      *parent_state.commit_op_->writer_->io_handle_,
      parent_state.commit_op_->flush_promise_, std::move(encoded_nodes));

  {
    absl::MutexLock lock(&parent_state.mutex_);

    // Remove `existing_relative_child_key` from the parent node.
    {
      auto& mutation = parent_state.mutations_.emplace_back();
      mutation.add = false;
      mutation.entry.key =
          tensorstore::StrCat(parent_state.existing_subtree_key_prefix_,
                              existing_relative_child_key);
    }

    // Add `new_entries` in its place.
    for (auto& new_entry : new_entries) {
      auto& mutation = parent_state.mutations_.emplace_back();
      mutation.add = true;
      mutation.entry = std::move(new_entry);
    }
  }
}

Result<std::vector<EncodedNode>> CommitOperation::EncodeUpdatedInteriorNodes(
    const Config& config, BtreeNodeHeight height,
    std::string_view existing_prefix,
    span<const InteriorNodeEntry> existing_entries,
    span<InteriorNodeMutation> mutations, bool may_be_root) {
  // Sort by key order, with deletions before additions, which allows the code
  // below to remove and add the same key without additional checks.
  std::sort(mutations.begin(), mutations.end(),
            [](const InteriorNodeMutation& a, const InteriorNodeMutation& b) {
              int c = a.entry.key.compare(b.entry.key);
              if (c != 0) return c < 0;
              return a.add < b.add;
            });

  BtreeInteriorNodeEncoder encoder(config, height, existing_prefix);
  auto existing_it = existing_entries.begin();
  auto mutation_it = mutations.begin();

  ComparePrefixedKeyToUnprefixedKey compare_existing_and_new_keys{
      existing_prefix};

  // Merge existing entries with mutations.
  while (existing_it != existing_entries.end() ||
         mutation_it != mutations.end()) {
    int c = existing_it == existing_entries.end() ? 1
            : mutation_it == mutations.end()
                ? -1
                : compare_existing_and_new_keys(existing_it->key,
                                                mutation_it->entry.key);
    if (c < 0) {
      // Existing key comes before mutation.
      encoder.AddEntry(/*existing=*/true, InteriorNodeEntry(*existing_it));
      ++existing_it;
      continue;
    }

    if (c == 0) {
      // Mutation replaces or deletes existing key.
      ++existing_it;
    }

    if (mutation_it->add) {
      internal_ocdbt::AddNewInteriorEntry(encoder, mutation_it->entry);
    }
    ++mutation_it;
  }

  return encoder.Finalize(may_be_root);
}

void CommitOperation::CreateNewManifest(
    Promise<void> promise, CommitOperation::Ptr commit_op,
    const BtreeGenerationReference& new_generation) {
  auto future = internal_ocdbt::CreateNewManifest(
      commit_op->writer_->io_handle_, commit_op->existing_manifest_,
      new_generation);
  LinkValue(
      [commit_op = std::move(commit_op)](
          Promise<void> promise,
          ReadyFuture<std::pair<std::shared_ptr<Manifest>, Future<const void>>>
              future) mutable {
        auto& create_result = future.value();
        commit_op->flush_promise_.Link(std::move(create_result.second));
        commit_op->new_manifest_ = std::move(create_result.first);
        auto executor = commit_op->writer_->io_handle_->executor;
        executor([commit_op = std::move(commit_op),
                  promise = std::move(promise)]() mutable {
          NewManifestReady(std::move(promise), std::move(commit_op));
        });
      },
      std::move(promise), std::move(future));
}

void CommitOperation::NewManifestReady(Promise<void> promise,
                                       CommitOperation::Ptr commit_op) {
  ABSL_LOG_IF(INFO, ocdbt_logging) << "NewManifestReady";
  auto flush_future = std::move(commit_op->flush_promise_).future();
  if (flush_future.null()) {
    return;
  }
  flush_future.Force();
  LinkError(std::move(promise), std::move(flush_future));
}

void CommitOperation::WriteNewManifest(CommitOperation::Ptr commit_op) {
  auto& writer = *commit_op->writer_;
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "WriteNewManifest: existing_generation="
      << GetLatestGeneration(commit_op->existing_manifest_.get())
      << ", new_generation="
      << GetLatestGeneration(commit_op->new_manifest_.get());
  auto update_future = writer.io_handle_->TryUpdateManifest(
      commit_op->existing_manifest_, commit_op->new_manifest_, absl::Now());
  update_future.Force();
  update_future.ExecuteWhenReady(WithExecutor(
      writer.io_handle_->executor,
      [commit_op =
           std::move(commit_op)](ReadyFuture<TryUpdateManifestResult> future) {
        auto& r = future.result();
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "Manifest written: " << r.status()
            << ", success=" << (r.ok() ? r->success : false);
        auto& writer = *commit_op->writer_;
        if (!r.ok()) {
          Fail(std::move(commit_op), r.status());
          return;
        }
        if (!r->success) {
          ReadManifest(std::move(commit_op), r->time);
          return;
        }
        CommitSuccessful(commit_op->staged_, r->time);
        UniqueWriterLock lock(writer.mutex_);
        writer.commit_in_progress_ = false;
        if (!writer.pending_.requests.empty()) {
          CommitOperation::MaybeStart(writer, std::move(lock));
        }
      }));
}

}  // namespace

Future<TimestampedStorageGeneration> NonDistributedBtreeWriter::Write(
    std::string key, std::optional<absl::Cord> value,
    kvstore::WriteOptions options) {
  auto& writer = *this;
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Write: " << tensorstore::QuoteString(key) << " " << value.has_value();
  auto request = std::make_unique<WriteEntry>();
  request->key = std::move(key);
  request->kind = MutationEntry::kWrite;
  request->if_equal = std::move(options.generation_conditions.if_equal);
  auto [promise, future] =
      PromiseFuturePair<TimestampedStorageGeneration>::Make(std::in_place);
  request->promise = std::move(promise);

  Future<const void> value_future;
  if (value) {
    auto& value_ref = request->value.emplace();
    if (auto* config = writer.io_handle_->config_state->GetExistingConfig();
        !config || value->size() <= config->max_inline_value_bytes) {
      // Config not yet known or value to be written inline.
      value_ref = std::move(*value);
    } else {
      value_future = writer.io_handle_->WriteData(
          std::move(*value), value_ref.emplace<IndirectDataReference>());
    }
  }
  UniqueWriterLock lock{writer.mutex_};
  writer.pending_.requests.emplace_back(
      MutationEntryUniquePtr(request.release()));
  if (!value_future.null()) {
    writer.pending_.flush_promise.Link(std::move(value_future));
  }
  CommitOperation::MaybeStart(writer, std::move(lock));
  return future;
}

Future<const void> NonDistributedBtreeWriter::DeleteRange(KeyRange range) {
  auto& writer = *this;
  ABSL_LOG_IF(INFO, ocdbt_logging) << "DeleteRange: " << range;
  auto request = std::make_unique<DeleteRangeEntry>();
  request->kind = MutationEntry::kDeleteRange;
  request->key = std::move(range.inclusive_min);
  request->exclusive_max = std::move(range.exclusive_max);
  UniqueWriterLock lock{writer.mutex_};
  writer.pending_.requests.emplace_back(
      MutationEntryUniquePtr(request.release()));
  Future<const void> future;
  if (writer.pending_.delete_range_promise.null() ||
      (future = writer.pending_.delete_range_promise.future()).null()) {
    auto p = PromiseFuturePair<void>::Make();
    writer.pending_.delete_range_promise = std::move(p.promise);
    future = std::move(p.future);
  }
  CommitOperation::MaybeStart(writer, std::move(lock));
  return future;
}

struct CopySubtreeListReceiver {
  NonDistributedBtreeWriter::Ptr writer;
  size_t strip_prefix_length;
  std::string add_prefix;
  Promise<void> promise;
  FutureCallbackRegistration cancel_registration;

  template <typename Cancel>
  void set_starting(Cancel cancel) {
    cancel_registration = promise.ExecuteWhenNotNeeded(std::move(cancel));
  }

  void set_stopping() { cancel_registration.Unregister(); }

  void set_error(absl::Status status) { promise.SetResult(std::move(status)); }

  void set_value(std::string_view key_prefix,
                 span<const LeafNodeEntry> entries) {
    if (entries.empty()) return;
    UniqueWriterLock lock{writer->mutex_};
    for (auto& entry : entries) {
      auto key = tensorstore::StrCat(
          add_prefix,
          std::string_view(key_prefix)
              .substr(std::min(key_prefix.size(), strip_prefix_length)),
          std::string_view(entry.key).substr(
              std::min(entry.key.size(),
                       strip_prefix_length -
                           std::min(strip_prefix_length, key_prefix.size()))));
      auto request = std::make_unique<WriteEntry>();
      request->key = std::move(key);
      request->kind = MutationEntry::kWrite;
      auto [promise, future] =
          PromiseFuturePair<TimestampedStorageGeneration>::Make(std::in_place);
      request->promise = std::move(promise);
      request->value = entry.value_reference;
      LinkError(this->promise, std::move(future));
      writer->pending_.requests.emplace_back(
          MutationEntryUniquePtr(request.release()));
    }
    CommitOperation::MaybeStart(*writer, std::move(lock));
  }

  void set_done() {}
};

Future<const void> NonDistributedBtreeWriter::CopySubtree(
    CopySubtreeOptions&& options) {
  // TODO(jbms): Currently this implementation avoids copying indirect values,
  // but never reuses B+tree nodes.  A more efficient implementation that
  // re-uses B+tree nodes in many cases is possible.
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "CopySubtree: " << options.node
      << ", height=" << static_cast<int>(options.node_height)
      << ", range=" << options.range << ", subtree_key_prefix="
      << tensorstore::QuoteString(options.subtree_key_prefix)
      << ", strip_prefix_length=" << options.strip_prefix_length
      << ", add_prefix=" << tensorstore::QuoteString(options.add_prefix);

  auto [promise, future] = PromiseFuturePair<void>::Make(absl::OkStatus());
  NonDistributedListSubtree(
      io_handle_, options.node, options.node_height,
      std::move(options.subtree_key_prefix), std::move(options.range),
      CopySubtreeListReceiver{
          NonDistributedBtreeWriter::Ptr(this), options.strip_prefix_length,
          std::move(options.add_prefix), std::move(promise)});
  return std::move(future);
}

BtreeWriterPtr MakeNonDistributedBtreeWriter(IoHandle::Ptr io_handle) {
  auto writer = internal::MakeIntrusivePtr<NonDistributedBtreeWriter>();
  writer->io_handle_ = std::move(io_handle);
  return writer;
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
