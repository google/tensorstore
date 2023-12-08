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

#ifndef TENSORSTORE_KVSTORE_OCDBT_STAGED_MUTATIONS_H_
#define TENSORSTORE_KVSTORE_OCDBT_STAGED_MUTATIONS_H_

/// \file
///
/// Data structures used by `btree_writer` for tracking pending mutations to the
/// B+tree.
///
/// Currently, two mutation types (`MutationEntry::MutationKind`) are supported:
///
/// - Optionally-conditional write/delete of a single key
/// - Unconditional DeleteRange
///
/// There are two representations used for mutations:
///
/// - `PendingRequests`: simply a queue of mutations.  All mutation operations
///   are initially added to this queue.
///
/// - `StagedMutations`: a normalized red-black tree representation of mutations
///   ordered by key.  Just before the commit operation begins traversing the
///   B+tree, all pending requests are merged into this tree.
///
/// The reason for using a `PendingRequests` rather than just immediately
/// constructing a `StagedMutations` tree is that it avoids needing to implement
/// merging of two `StagedMutations` trees, as would be required if the commit
/// needs to restart (e.g. due to a concurrent modification by another process)
/// and additional mutation operations were enqueued since the commit operation
/// last began traversing the tree.
///
/// Refer to `btree_writer.cc` for details.

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/internal/container/intrusive_red_black_tree.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_ocdbt {

struct MutationEntry;
struct WriteEntry;

using MutationEntryTree =
    internal::intrusive_red_black_tree::Tree<MutationEntry>;

using WriteEntryTree = internal::intrusive_red_black_tree::Tree<WriteEntry>;

struct MutationEntry : public MutationEntryTree::NodeBase {
  std::string key;

  enum MutationKind {
    kWrite,
    kDeleteRange,
  };
  MutationKind kind;

 protected:
  ~MutationEntry() = default;
};

struct WriteEntry : public MutationEntry {
  // TODO: Consider changing this to
  // `std::variant<std::monostate, IndirectValueReference, absl::Cord>` to save
  // 8 bytes of memory.
  std::optional<LeafNodeValueReference> value;
  StorageGeneration if_equal;
  Promise<TimestampedStorageGeneration> promise;

  // Tag bit indicates that the indicated superseded entry was deleted by a
  // `DeleteRange` request before being superseded.
  using Supersedes = internal::TaggedPtr<WriteEntry, 1>;
  Supersedes supersedes{nullptr};
};

struct DeleteRangeEntry : public MutationEntry {
  std::string exclusive_max;
  WriteEntryTree superseded_writes;
};

struct MutationEntryDeleter {
  void operator()(MutationEntry* e) const {
    if (e->kind == MutationEntry::kWrite) {
      delete static_cast<WriteEntry*>(e);
    } else {
      delete static_cast<DeleteRangeEntry*>(e);
    }
  }
};

using MutationEntryUniquePtr =
    std::unique_ptr<MutationEntry, MutationEntryDeleter>;

struct PendingRequests {
  std::vector<MutationEntryUniquePtr> requests;

  Promise<void> delete_range_promise;
  FlushPromise flush_promise;
};

// Marks all mutations enqueued in `pending` as failed with the specified
// `error`.
void AbortPendingRequestsWithError(const PendingRequests& pending,
                                   const absl::Status& error);

struct StagedMutations {
  MutationEntryTree entries;
  Promise<void> delete_range_promise;
};

// Merges the mutations in `pending` into `staged`.
void StageMutations(StagedMutations& staged, PendingRequests&& pending);

// Updates the commit timestamp in every promise to `time`.
void CommitSuccessful(StagedMutations& staged, absl::Time time);

// Sets the result of all staged mutations to `error`.
void CommitFailed(StagedMutations& staged, absl::Status error);

// Returns the interval of write entries within `tree` that intersect
// `[inclusive_min, exclusive_max)`.
WriteEntryTree::Range GetWriteEntryInterval(WriteEntryTree& tree,
                                            std::string_view inclusive_min,
                                            std::string_view exclusive_max);

// Returns the interval of write entries within `tree` that intersect
// `key_range`.
WriteEntryTree::Range GetWriteEntryInterval(WriteEntryTree& tree,
                                            const KeyRange& key_range);

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
void PartitionInteriorNodeMutations(
    span<const InteriorNodeEntry> existing_entries,
    std::string_view existing_key_prefix, const KeyRange& key_range,
    MutationEntryTree::Range entry_range,
    absl::FunctionRef<void(const InteriorNodeEntry& existing_child,
                           KeyRange child_key_range,
                           MutationEntryTree::Range mutation_sub_range)>
        partition_callback);

// Determine the final result of a chain of write operations.
//
// Args:
//   existing_generation: Generation of existing value.
//   last_write_entry: The latest write entry in the chain.
//
// Returns:
//
//   - `std::nullopt`, if the existing value should be retained; or
//
//   - a pointer to the new value to write within the chain starting at
//     `last_write_entry`.
std::optional<const LeafNodeValueReference*> ApplyWriteEntryChain(
    StorageGeneration existing_generation, const WriteEntry& last_write_entry);

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
    WriteEntryTree::Range superseded_writes,
    span<const LeafNodeEntry> existing_entries,
    std::string_view existing_prefix);

/// Checks if the root node of a subtree must be read in order to apply the
/// mutations specified in `entry_range`.
///
/// Normally mutations require reading the root node of the subtree, but that
/// can be skipped if the subtree is entirely deleted via a `DeleteRangeEntry`
/// and there are no superseded writes to validate.
bool MustReadNodeToApplyMutations(const KeyRange& key_range,
                                  MutationEntryTree::Range entry_range);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_STAGED_MUTATIONS_H_
