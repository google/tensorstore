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

struct DeleteRangeEntry;

struct MutationEntry : public MutationEntryTree::NodeBase {
  std::string key_;

  enum MutationKind {
    kWrite = 0,
    // To match ``internal_kvstore::MutationEntry` API
    kReadModifyWrite = 0,
    kDeleteRange = 1,
  };
  MutationKind kind_;

  // To match ``internal_kvstore::MutationEntry` API
  MutationKind entry_type() { return kind_; }

  using DeleteRangeEntry = internal_ocdbt::DeleteRangeEntry;
  using ReadModifyWriteEntry = WriteEntry;

 protected:
  ~MutationEntry() = default;
};

struct WriteEntry : public MutationEntry {
  // TODO: Consider changing this to
  // `std::variant<std::monostate, IndirectValueReference, absl::Cord>` to save
  // 8 bytes of memory.
  std::optional<LeafNodeValueReference> value_;
  StorageGeneration if_equal_;
  Promise<TimestampedStorageGeneration> promise_;

  // Tag bit indicates that the indicated superseded entry was deleted by a
  // `DeleteRange` request before being superseded.
  using Supersedes = internal::TaggedPtr<WriteEntry, 1>;
  Supersedes supersedes_{nullptr};
};

struct DeleteRangeEntry : public MutationEntry {
  std::string exclusive_max_;
  WriteEntryTree superseded_;
};

struct MutationEntryDeleter {
  void operator()(MutationEntry* e) const {
    if (e->kind_ == MutationEntry::kWrite) {
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

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_STAGED_MUTATIONS_H_
