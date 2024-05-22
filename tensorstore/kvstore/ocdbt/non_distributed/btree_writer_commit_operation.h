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

#ifndef TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_BTREE_WRITER_COMMIT_OPERATION_H_
#define TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_BTREE_WRITER_COMMIT_OPERATION_H_

// This module implements non-distributed committing of mutations to the B+tree.
// It is templated on the actual mutation representation, and is used for both
// atomic transactional and non-transactional commits.
//
// The commit is ultimately an atomic read-modify-write operation on the
// manifest, and uses the following commit loop:
//
// 1. The existing manifest is read (possibly using a cached value), or created
//    if there is no existing manifest.
//
// 2. A mutation representation-specific "stage pending" operation is performed
//    once the configuration is fixed, which may pull in additional requests to
//    be committed, and adds any values that will be stored out-of-line to the
//    indirect data writer.
//
// 3. The B+tree is traversed top-down (starting from the root), recursively
//    partitioning the ordered list of staged mutations according to the B+tree
//    node structure. B+tree nodes are fetched as required to perform the
//    partitioning. Write conditions are checked during this traversal.
//
// 4. Nodes are re-written (and split as required) in a bottom-up fashion.
//    Non-leaf nodes are not rewritten until any child nodes that need to be
//    modified have been re-written.  Note: This step happens concurrently with
//    the traversal described in the previous step.
//
// 5. Once the root B+tree node has been written, a new manifest is created.
//    If all of the inline version slots in the manifest are full, new version
//    tree nodes must be written (which may require reading existing version
//    tree nodes) in order to make space.
//
// 6. The new manifest is written, conditioned on the existing manifest
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

#include <cassert>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/functional/function_ref.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/container/intrusive_red_black_tree.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/btree_node_encoder.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/storage_generation.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {

ABSL_CONST_INIT inline internal_log::VerboseFlag ocdbt_logging("ocdbt");

// Returns the interval of write entries within `tree` that intersect
// `[inclusive_min, exclusive_max)`.
template <typename Tree>
typename Tree::Range GetWriteEntryInterval(Tree& tree,
                                           std::string_view inclusive_min,
                                           std::string_view exclusive_max) {
  auto* lower_bound = tree.template FindBound<Tree::kLeft>([&](auto& entry) {
                            return entry.key_ < inclusive_min;
                          })
                          .found_node();
  auto* upper_bound =
      exclusive_max.empty()
          ? nullptr
          : tree.template FindBound<Tree::kRight>([&](auto& entry) {
                  return KeyRange::CompareKeyAndExclusiveMax(entry.key_,
                                                             exclusive_max) < 0;
                })
                .found_node();
  return typename Tree::Range(lower_bound, upper_bound);
}

// Returns the interval of write entries within `tree` that intersect
// `key_range`.
template <typename Tree>
typename Tree::Range GetWriteEntryInterval(Tree& tree,
                                           const KeyRange& key_range) {
  return GetWriteEntryInterval<Tree>(tree, key_range.inclusive_min,
                                     key_range.exclusive_max);
}

/// Checks if the root node of a subtree must be read in order to apply the
/// mutations specified in `entry_range`.
///
/// Normally mutations require reading the root node of the subtree, but that
/// can be skipped if the subtree is entirely deleted via a `DeleteRangeEntry`
/// and there are no superseded writes to validate.
template <typename MutationEntry>
bool MustReadNodeToApplyMutations(
    const KeyRange& key_range,
    internal::intrusive_red_black_tree::Range<MutationEntry> entry_range) {
  assert(!entry_range.empty());
  if (entry_range.end() != std::next(entry_range.begin())) {
    // More than one mutation, which means a single `DeleteRangeEntry` does not
    // cover the entire `key_range`.
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "MustReadNodeToApplyMutations: more than one mutation";
    return true;
  }
  MutationEntry* mutation = entry_range.begin().to_pointer();
  if (mutation->entry_type() != MutationEntry::kDeleteRange) {
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "MustReadNodeToApplyMutations: not delete range mutation";
    return true;
  }

  // `entry_range` consists of a single `DeleteRangeEntry`.
  auto& dr_entry =
      *static_cast<typename MutationEntry::DeleteRangeEntry*>(mutation);
  if (dr_entry.key_ > key_range.inclusive_min ||
      KeyRange::CompareExclusiveMax(dr_entry.exclusive_max_,
                                    key_range.exclusive_max) < 0) {
    // `DeleteRangeEntry` does not cover the entire key space.
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "MustReadNodeToApplyMutations: does not cover entire key space: "
           "dr_entry.key="
        << tensorstore::QuoteString(dr_entry.key_)
        << ", dr_entry.exclusive_max="
        << tensorstore::QuoteString(dr_entry.exclusive_max_)
        << ", key_range.exclusive_max="
        << tensorstore::QuoteString(key_range.exclusive_max);
    return true;
  }

  // `DeleteRangeEntry` covers the entire key space.

  auto writes = GetWriteEntryInterval(
      dr_entry.superseded_, key_range.inclusive_min, key_range.exclusive_max);
  if (!writes.empty()) {
    // There are superseded writes that need to be validated, in order to be
    // able to correctly indicate in the response whether the write "succeeded"
    // (before being overwritten) or was aborted due to a generation mismatch.
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "MustReadNodeToApplyMutations: superseded writes";
    return true;
  }

  // No superseded writes need to be validated.
  return false;
}

// Partitions a sub-range of a `MutationEntryTree` into the sub-ranges that
// intersect the key range of each child of an interior B+tree node.
//
// Args:
//   existing_entries: The children of the interior B+tree node.
//   existing_key_prefix: Key prefix that applies to `existing_entries`.
//   key_range: The full key range of the interior B+tree node.
//   entry_range: The sub-range to partition.
//   partition_callback: Callback invoked for each partition, where
//     `child_key_range` specifies the full key range of the child referenced
//     by `existing_child`, and `mutation_sub_range` is the corresponding
//     non-empty range of intersecting mutations.  The callback is not invoked
//     for existing entries with no intersecting mutations.  Note that the
//     `mutation_sub_range` values are not necessarily disjoint; the
//     intersecting mutation ranges for consecutive children may overlap by
//     exactly one `DeleteRange` mutation.
template <typename MutationEntry>
void PartitionInteriorNodeMutations(
    span<const InteriorNodeEntry> existing_entries,
    std::string_view existing_key_prefix, const KeyRange& key_range,
    internal::intrusive_red_black_tree::Range<MutationEntry> entry_range,
    absl::FunctionRef<
        void(const InteriorNodeEntry& existing_child, KeyRange child_key_range,
             internal::intrusive_red_black_tree::Range<MutationEntry>
                 mutation_sub_range)>
        partition_callback) {
  assert(!existing_entries.empty());

  ComparePrefixedKeyToUnprefixedKey compare_existing_and_new_keys{
      existing_key_prefix};

  // Next mutation entry to assign to a partition.
  auto entry_it = entry_range.begin();

  // Exclusive end point of current partition/child node.
  //
  // In non-leaf nodes, the keys for entries after the first specify partition
  // points.  The key for the first entry does not specify a partition point.
  auto existing_it = existing_entries.begin() + 1;

  // First mutation entry in the partition that ends at `existing_it`.
  MutationEntry* first_mutation_in_partition = entry_it.to_pointer();

  // Move to next partition/child node.  `end_mutation_in_partition` is one past
  // the last mutation assigned to the partition.
  const auto end_of_partition = [&](MutationEntry* end_mutation_in_partition) {
    if (first_mutation_in_partition != end_mutation_in_partition) {
      auto& existing_entry = *(existing_it - 1);
      KeyRange new_key_range;
      if (&existing_entry == &existing_entries.front()) {
        new_key_range.inclusive_min = key_range.inclusive_min;
      } else {
        new_key_range.inclusive_min =
            tensorstore::StrCat(existing_key_prefix, existing_entry.key);
      }
      if (existing_it == existing_entries.end()) {
        new_key_range.exclusive_max = key_range.exclusive_max;
      } else {
        new_key_range.exclusive_max =
            tensorstore::StrCat(existing_key_prefix, existing_it->key);
      }
      partition_callback(
          existing_entry, std::move(new_key_range),
          internal::intrusive_red_black_tree::Range<MutationEntry>(
              first_mutation_in_partition, end_mutation_in_partition));
      first_mutation_in_partition = entry_it.to_pointer();
    } else {
      ABSL_LOG_IF(INFO, ocdbt_logging)
          << "PartitionInteriorNodeMutations: existing child "
          << tensorstore::QuoteString(existing_key_prefix) << "+"
          << tensorstore::QuoteString((existing_it - 1)->key)
          << " has no mutations";
    }
    ++existing_it;
  };

  while (entry_it != entry_range.end()) {
    auto* mutation = &*entry_it;
    int c =
        existing_it != existing_entries.end()
            ? compare_existing_and_new_keys(existing_it->key, mutation->key_)
            : 1;
    if (c <= 0) {
      // Current partition ends before lower bound of mutation.
      end_of_partition(entry_it.to_pointer());
      continue;
    }

    if (mutation->entry_type() == MutationEntry::kDeleteRange) {
      auto& dr_entry =
          *static_cast<const typename MutationEntry::DeleteRangeEntry*>(
              mutation);

      // Compare `dr_entry.exclusive_max_` to the exclusive max of the current
      // partition.

      // Indicates the 3-way comparison between `dr_entry.exclusive_max_` and
      // the exclusive max of the current partition.
      int c_max;

      if (existing_it != existing_entries.end()) {
        // Current partition is not the last partition: compare against the
        // starting key of the next partition.
        c_max = dr_entry.exclusive_max_.empty()
                    ? 1
                    : -compare_existing_and_new_keys(existing_it->key,
                                                     dr_entry.exclusive_max_);
      } else {
        // Current partition is the last partition.
        c_max = -1;
      }

      if (c_max < 0) {
        // `dr_entry` is contained in current partition and not the next
        // partition, and does not end the current partition.
        ++entry_it;
        continue;
      }

      if (c_max == 0) {
        // `dr_entry` ends exactly at the end of the current partition.
        ++entry_it;
        end_of_partition(entry_it.to_pointer());
        continue;
      }

      assert(c_max > 0);
      // `dr_entry` extends past the current partition.
      end_of_partition(std::next(entry_it).to_pointer());
      continue;
    }

    assert(mutation->entry_type() == MutationEntry::kReadModifyWrite);
    ++entry_it;
  }

  end_of_partition(entry_it.to_pointer());
}

class BtreeWriterCommitOperationBase {
 protected:
  BtreeWriterCommitOperationBase(IoHandle::Ptr io_handle)
      : io_handle_(std::move(io_handle)) {}

  ~BtreeWriterCommitOperationBase() = default;

 public:
  IoHandle::Ptr io_handle_;

  std::shared_ptr<const Manifest> existing_manifest_;
  std::shared_ptr<const Manifest> new_manifest_;

  const Config& existing_config() {
    auto* config = io_handle_->config_state->GetAssumedOrExistingConfig();
    assert(config);
    return *config;
  }

  FlushPromise flush_promise_;
  absl::Time staleness_bound_ = absl::InfinitePast();

  // Starts a commit attempt, beginning by reading the existing manifest as of
  // `staleness_bound_`.
  //
  // This must be called initially to start the commit, and must also be called
  // again (possibly asynchronously) from `Retry` to retry a commit.
  void ReadManifest();

  // Fails the commit operation, and deletes `this`.
  virtual void Fail(const absl::Status& error) = 0;

  /// Retries the commit operation.
  virtual void Retry() = 0;

  class WriteStager {
   public:
    void Stage(LeafNodeValueReference& value_ref);

   private:
    friend class BtreeWriterCommitOperationBase;
    explicit WriteStager(BtreeWriterCommitOperationBase& op)
        : op(op), config(op.existing_config()) {}
    BtreeWriterCommitOperationBase& op;
    const Config& config;
  };

  // "Stages" all pending mutations by merging them into the `StagedMutations`
  // data structure.
  virtual void StagePending(WriteStager& stager) = 0;

  /// Completes the commit operation, and deletes `this`.
  virtual void CommitSuccessful(absl::Time time) = 0;

  // Initiates an asynchronous traversal of the B+tree, starting from the root,
  // that visits nodes with a key range that intersects the key range of a
  // staged mutation.
  virtual void TraverseBtreeStartingFromRoot(Promise<void> promise) = 0;

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

    void NotifyOutOfDate() { promise_.SetResult(absl::CancelledError()); }

    friend void intrusive_ptr_decrement(NodeTraversalState* p) {
      if (internal::DecrementReferenceCount(*p)) {
        if (p->promise_.result_needed()) {
          p->ApplyMutations();
        }
        delete p;
      }
    }

    BtreeWriterCommitOperationBase* writer_;
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
    void ApplyMutations() final;
  };

  // `NodeTraversalState` type for interior nodes of the B+tree.
  struct InteriorNodeTraversalState : public NodeTraversalState {
    NodeTraversalState::Ptr parent_state_;
    std::shared_ptr<const BtreeNode> existing_node_;
    std::string existing_relative_child_key_;

    void ApplyMutations() final;
  };

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
  void CreateNewManifest(Promise<void> promise,
                         const BtreeGenerationReference& new_generation);

  // Called once a new manifest has been generated and any necessary version
  // tree nodes have been written.
  //
  // Ensures all indirect writes are flushed, and then marks `promise` as ready.
  void NewManifestReady(Promise<void> promise);

  // Attempts to write the new manifest.
  void WriteNewManifest();
};

// Base class that implements the B+tree commit operation.
//
// \tparam MutationEntry The mutation entry type (either
//     `internal_kvstore::MutationEntry` or `internal_ocdbt::MutationEntry`),
//     must define the following members:
//
//     - `std::string key_`: key for write operations, inclusive min for delete
//       range operations.
//
//     - `entry_type()`, returns either `kDeleteRange` or `kReadModifyWrite`
//       (which must also be members).
//
//     - `DeleteRangeEntry`: type of derived entry corresponding to a "delete
//       range" operation. Must have `std::string exclusive_max_` and
//       `Tree<ReadModifyWriteEntry> superseded_` members.
//
//     - `ReadModifyWriteEntry`: type of derived entry corresponding to a write
//       operation.
template <typename MutationEntry>
class BtreeWriterCommitOperation : public BtreeWriterCommitOperationBase {
 public:
  using BtreeWriterCommitOperationBase::BtreeWriterCommitOperationBase;

 protected:
  /// Returns the red-black tree of mutation entries to be committed.
  virtual internal::intrusive_red_black_tree::Tree<MutationEntry>&
  GetEntries() = 0;

  // Determine the final result of a chain of write operations.
  //
  // Args:
  //   existing_generation: Generation of existing value.
  //   last_write_entry: The latest write entry in the chain.
  //   validated: Set to `false` if the commit needs to be retried due to a
  //     generation mismatch.
  //
  // Returns:
  //
  //   - `std::nullopt`, if the existing value should be retained; or
  //
  //   - a pointer to the new value to write within the chain starting at
  //     `last_write_entry`.
  virtual std::optional<const LeafNodeValueReference*> ApplyWriteEntryChain(
      StorageGeneration existing_generation,
      typename MutationEntry::ReadModifyWriteEntry& entry, bool& validated) = 0;

  // Validate superseded writes against leaf node entries.
  //
  // Stores the result of validation in the `Result` saved in each
  // `WriteEntry::promise`.
  //
  // Args:
  //   superseded_writes: Range of superseded writes that correspond to
  //     `existing_entries`.
  //   existing_entries: Entries in existing leaf node.
  //   existing_prefix: Key prefix that applies to `existing_entries`.
  span<const LeafNodeEntry>::iterator ValidateSupersededWriteEntries(
      internal::intrusive_red_black_tree::Range<
          typename MutationEntry::ReadModifyWriteEntry>
          superseded_writes,
      span<const LeafNodeEntry> existing_entries,
      std::string_view existing_prefix, bool& validated);

  // Initiates an asynchronous traversal of the B+tree, starting from the root,
  // that visits nodes with a key range that intersects the key range of a
  // staged mutation.
  void TraverseBtreeStartingFromRoot(Promise<void> promise) override;

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
    internal::intrusive_red_black_tree::Range<MutationEntry> entry_range;
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
    internal::intrusive_red_black_tree::Range<MutationEntry> entry_range;
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
};

template <typename MutationEntry>
span<const LeafNodeEntry>::iterator
BtreeWriterCommitOperation<MutationEntry>::ValidateSupersededWriteEntries(
    typename internal::intrusive_red_black_tree::Range<
        typename MutationEntry::ReadModifyWriteEntry>
        superseded_writes,
    span<const LeafNodeEntry> existing_entries,
    std::string_view existing_prefix, bool& validated) {
  ComparePrefixedKeyToUnprefixedKey compare_existing_and_new_keys{
      existing_prefix};

  auto existing_it = existing_entries.begin();
  auto superseded_it = superseded_writes.begin();
  while (superseded_it != superseded_writes.end() &&
         existing_it != existing_entries.end() && validated) {
    int c =
        compare_existing_and_new_keys(existing_it->key, superseded_it->key_);
    if (c < 0) {
      // Existing key comes before mutation.
      ++existing_it;
      continue;
    }
    if (c == 0) {
      // Existing key matches mutation key.
      ApplyWriteEntryChain(internal_ocdbt::ComputeStorageGeneration(
                               existing_it->value_reference),
                           *superseded_it, validated);
      ++existing_it;
      ++superseded_it;
      continue;
    }
    // Existing key comes after mutation.
    ApplyWriteEntryChain(StorageGeneration::NoValue(), *superseded_it,
                         validated);
    ++superseded_it;
  }
  for (; superseded_it != superseded_writes.end() && validated;
       ++superseded_it) {
    ApplyWriteEntryChain(StorageGeneration::NoValue(), *superseded_it,
                         validated);
  }
  return existing_it;
}

template <typename MutationEntry>
void BtreeWriterCommitOperation<MutationEntry>::TraverseBtreeStartingFromRoot(
    Promise<void> promise) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Manifest ready: generation_number="
      << GetLatestGeneration(existing_manifest_.get());
  auto parent_state = internal::MakeIntrusivePtr<RootNodeTraversalState>();
  parent_state->promise_ = std::move(promise);
  parent_state->writer_ = this;
  internal::intrusive_red_black_tree::Range<MutationEntry> entry_range =
      GetEntries();
  if (entry_range.empty()) return;
  auto* existing_manifest = existing_manifest_.get();
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

template <typename MutationEntry>
void BtreeWriterCommitOperation<MutationEntry>::VisitNodeReference(
    VisitNodeReferenceParameters&& params, const BtreeNodeReference& node_ref) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Process node reference: " << params.key_range
      << ", height=" << (params.parent_state->height_ - 1);
  auto read_future =
      params.parent_state->writer_->io_handle_->GetBtreeNode(node_ref.location);
  auto executor = params.parent_state->writer_->io_handle_->executor;
  auto promise = params.parent_state->promise_;
  Link(WithExecutor(std::move(executor), NodeReadyCallback{std::move(params)}),
       std::move(promise), std::move(read_future));
}

template <typename MutationEntry>
void BtreeWriterCommitOperation<MutationEntry>::VisitNode(
    VisitNodeParameters&& params) {
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

template <typename MutationEntry>
void BtreeWriterCommitOperation<MutationEntry>::VisitInteriorNode(
    VisitNodeParameters params) {
  span<const InteriorNodeEntry> existing_entries =
      std::get<BtreeNode::InteriorNodeEntries>(params.node->entries);

  if (existing_entries.empty()) {
    SetDeferredResult(
        params.parent_state->promise_,
        absl::DataLossError("Empty non-root/non-leaf b-tree node found"));
    return;
  }

  auto self_state = internal::MakeIntrusivePtr<InteriorNodeTraversalState>();
  self_state->writer_ = params.parent_state->writer_;
  self_state->promise_ = params.parent_state->promise_;
  self_state->parent_state_ = std::move(params.parent_state);
  self_state->height_ = params.node->height;
  self_state->existing_node_ = std::move(params.node);
  self_state->existing_subtree_key_prefix_ = std::move(params.full_prefix);
  self_state->existing_relative_child_key_ =
      std::move(params.inclusive_min_key_suffix);

  PartitionInteriorNodeMutations<MutationEntry>(
      existing_entries, self_state->existing_subtree_key_prefix_,
      params.key_range, params.entry_range,
      [&](const InteriorNodeEntry& existing_entry, KeyRange key_range,
          internal::intrusive_red_black_tree::Range<MutationEntry>
              entry_range) {
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "VisitInteriorNode: Partition: existing_entry="
            << tensorstore::QuoteString(
                   self_state->existing_subtree_key_prefix_)
            << "+" << tensorstore::QuoteString(existing_entry.key)
            << ", key_range=" << key_range << ", entry_range="
            << tensorstore::QuoteString(entry_range.begin()->key_);
        if (MustReadNodeToApplyMutations<MutationEntry>(key_range,
                                                        entry_range)) {
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

template <typename MutationEntry>
void BtreeWriterCommitOperation<MutationEntry>::VisitLeafNode(
    VisitNodeParameters params) {
  span<const LeafNodeEntry> existing_entries;
  if (params.node) {
    existing_entries =
        std::get<BtreeNode::LeafNodeEntries>(params.node->entries);
  }
  BtreeLeafNodeEncoder encoder(params.parent_state->writer_->existing_config(),
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
    int c =
        existing_it != existing_entries.end()
            ? compare_existing_and_new_keys(existing_it->key, mutation->key_)
            : 1;
    if (c < 0) {
      // Existing key comes before mutation.
      encoder.AddEntry(/*existing=*/true, LeafNodeEntry(*existing_it));
      ++existing_it;
      continue;
    }

    // Existing key is >= mutation key.
    ++entry_it;
    if (mutation->entry_type() == MutationEntry::kDeleteRange) {
      // Delete range mutation.
      auto& dr_entry =
          *static_cast<typename MutationEntry::DeleteRangeEntry*>(mutation);

      // Determine the set of superseded writes to validate against this leaf
      // node.  Only validate superseded writes that are within `key_range`.
      // The `DeleteRangeEntry` may not be fully contained within the range of
      // this leaf node, in which case it will also be processed by other leaf
      // nodes.
      auto superseded_writes =
          GetWriteEntryInterval(dr_entry.superseded_, key_range);

      // Validate any superseded writes and skip past any existing entries
      // ordered <= superseded writes.
      bool validated = true;
      auto new_existing_it =
          static_cast<BtreeWriterCommitOperation<MutationEntry>*>(
              params.parent_state->writer_)
              ->ValidateSupersededWriteEntries(
                  superseded_writes, span(existing_it, existing_entries.end()),
                  params.full_prefix, validated);
      if (!validated) {
        params.parent_state->NotifyOutOfDate();
        return;
      }

      // Skip past any existing entries covered by the deletion range.
      std::string_view exclusive_max = dr_entry.exclusive_max_;
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
    auto& write_entry =
        *static_cast<typename MutationEntry::ReadModifyWriteEntry*>(mutation);

    auto existing_generation = (c == 0)
                                   ? internal_ocdbt::ComputeStorageGeneration(
                                         existing_it->value_reference)
                                   : StorageGeneration::NoValue();
    bool validated = true;
    auto new_value = static_cast<BtreeWriterCommitOperation<MutationEntry>*>(
                         params.parent_state->writer_)
                         ->ApplyWriteEntryChain(std::move(existing_generation),
                                                write_entry, validated);
    if (!validated) {
      params.parent_state->NotifyOutOfDate();
      return;
    }
    if (new_value) {
      modified = true;
      if (*new_value) {
        LeafNodeEntry new_entry;
        new_entry.key = write_entry.key_;
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

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_BTREE_WRITER_COMMIT_OPERATION_H_
