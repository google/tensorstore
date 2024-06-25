// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/kvstore/ocdbt/non_distributed/transactional_btree_writer.h"

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "absl/base/optimization.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/btree_writer.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/btree_writer_commit_operation.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/list.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/storage_generation.h"
#include "tensorstore/kvstore/read_modify_write.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/future_sender.h"  // IWYU pragma: keep
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {

using internal_kvstore::DeleteRangeEntry;
using internal_kvstore::kReadModifyWrite;
using internal_kvstore::MutationEntry;
using internal_kvstore::MutationEntryTree;

struct ReadModifyWriteEntry
    : public internal_kvstore::AtomicMultiPhaseMutationBase::
          ReadModifyWriteEntryWithStamp {
 public:
  kvstore::ReadResult::State value_state_;
  LeafNodeValueReference value_;
};

class BtreeWriterTransactionNode
    : public internal_kvstore::TransactionNodeBase<
          internal_kvstore::AtomicMultiPhaseMutationBase>,
      public BtreeWriterCommitOperation<MutationEntry> {
 public:
  using Base = internal_kvstore::TransactionNodeBase<
      internal_kvstore::AtomicMultiPhaseMutationBase>;
  using OperationBase = BtreeWriterCommitOperation<MutationEntry>;
  using ReadModifyWriteEntry = internal_ocdbt::ReadModifyWriteEntry;

  explicit BtreeWriterTransactionNode(kvstore::Driver* driver,
                                      const IoHandle& io_handle)
      : Base(driver), OperationBase(IoHandle::Ptr(&io_handle)) {}

  ReadModifyWriteEntry* AllocateReadModifyWriteEntry() override {
    return new ReadModifyWriteEntry;
  }
  void FreeReadModifyWriteEntry(
      internal_kvstore::ReadModifyWriteEntry* entry) override {
    delete static_cast<ReadModifyWriteEntry*>(entry);
  }

  void Retry() override { RetryAtomicWriteback(staleness_bound_); }

  internal_kvstore::MutationEntryTree& GetEntries() override {
    return GetCommittingPhase().entries_;
  }

  // Called once the individual mutations have all produced a writeback result.
  void AllEntriesDone(
      internal_kvstore::SinglePhaseMutation& single_phase_mutation) override;

  using Base::Writeback;

  // Overrides `MultiPhaseMutation::Writeback`.  Called with the writeback
  // result for an individual entry, before `AllEntriesDone` is called.`
  void Writeback(internal_kvstore::ReadModifyWriteEntry& base_entry,
                 internal_kvstore::ReadModifyWriteEntry& base_source_entry,
                 kvstore::ReadResult&& read_result) override;

  // Fails the commit operation.
  void Fail(const absl::Status& error) override;

  // "Stages" all pending mutations by merging them into the `StagedMutations`
  // data structure.
  void StagePending(WriteStager& stager) override;

  std::optional<const LeafNodeValueReference*> ApplyWriteEntryChain(
      StorageGeneration existing_generation,
      internal_kvstore::ReadModifyWriteEntry& base_entry,
      bool& validated) override;

  void CommitSuccessful(absl::Time time) override;
};

void BtreeWriterTransactionNode::AllEntriesDone(
    internal_kvstore::SinglePhaseMutation& single_phase_mutation) {
  if (single_phase_mutation.remaining_entries_.HasError()) {
    internal_kvstore::WritebackError(single_phase_mutation);
    internal_kvstore::MultiPhaseMutation::AllEntriesDone(single_phase_mutation);
    return;
  }

  // Begin B+tree writeback.
  ReadManifest();
}

void BtreeWriterTransactionNode::Fail(const absl::Status& error) {
  ABSL_LOG_IF(INFO, ocdbt_logging) << "Commit failed: " << error;
  SetError(error);
  auto& single_phase_mutation = GetCommittingPhase();
  internal_kvstore::WritebackError(single_phase_mutation);
  MultiPhaseMutation::AllEntriesDone(single_phase_mutation);
}

void BtreeWriterTransactionNode::Writeback(
    internal_kvstore::ReadModifyWriteEntry& base_entry,
    internal_kvstore::ReadModifyWriteEntry& base_source_entry,
    kvstore::ReadResult&& read_result) {
  assert(read_result.stamp.time != absl::InfinitePast());
  auto& entry = static_cast<ReadModifyWriteEntry&>(base_entry);
  entry.stamp_ = std::move(read_result.stamp);
  entry.value_state_ = read_result.state;
  if (auto* value_ref = base_source_entry.source_->IsSpecialSource()) {
    entry.value_ = *static_cast<const LeafNodeValueReference*>(value_ref);
  } else {
    entry.value_.emplace<absl::Cord>(std::move(read_result.value));
  }
  AtomicWritebackReady(base_entry);
}

void BtreeWriterTransactionNode::StagePending(WriteStager& stager) {
  for (auto& base_entry : GetCommittingPhase().entries_) {
    if (base_entry.entry_type() != kReadModifyWrite) continue;
    auto& entry = static_cast<ReadModifyWriteEntry&>(base_entry);
    if (entry.value_state_ != kvstore::ReadResult::kValue) continue;
    stager.Stage(entry.value_);
  }
}

std::optional<const LeafNodeValueReference*>
BtreeWriterTransactionNode::ApplyWriteEntryChain(
    StorageGeneration existing_generation,
    internal_kvstore::ReadModifyWriteEntry& base_entry, bool& validated) {
  auto& entry = static_cast<ReadModifyWriteEntry&>(base_entry);
  auto& stamp = entry.stamp();
  auto if_equal = StorageGeneration::Clean(stamp.generation);
  if (!StorageGeneration::EqualOrUnspecified(existing_generation, if_equal)) {
    validated = false;
    return std::nullopt;
  }

  switch (entry.value_state_) {
    case kvstore::ReadResult::kValue:
      return &entry.value_;
    case kvstore::ReadResult::kMissing:
      return nullptr;
    case kvstore::ReadResult::kUnspecified:
      return std::nullopt;
  }
  ABSL_UNREACHABLE();
}

void BtreeWriterTransactionNode::CommitSuccessful(absl::Time time) {
  auto& single_phase_mutation = GetCommittingPhase();
  for (auto& base_entry : single_phase_mutation.entries_) {
    if (base_entry.entry_type() != kReadModifyWrite) {
      internal_kvstore::WritebackSuccess(
          static_cast<DeleteRangeEntry&>(base_entry));
    } else {
      TimestampedStorageGeneration stamp;
      auto& entry = static_cast<ReadModifyWriteEntry&>(base_entry);
      stamp.time = time;
      switch (entry.value_state_) {
        case kvstore::ReadResult::kValue:
          stamp.generation =
              internal_ocdbt::ComputeStorageGeneration(entry.value_);
          break;
        case kvstore::ReadResult::kMissing:
          stamp.generation = StorageGeneration::NoValue();
          break;
        case kvstore::ReadResult::kUnspecified:
          stamp.generation = StorageGeneration::Unknown();
          break;
      }
      internal_kvstore::WritebackSuccess(
          static_cast<ReadModifyWriteEntry&>(entry), std::move(stamp));
    }
  }
  MultiPhaseMutation::AllEntriesDone(GetCommittingPhase());
}

}  // namespace

absl::Status AddReadModifyWrite(
    kvstore::Driver* driver, const IoHandle& io_handle,
    internal::OpenTransactionPtr& transaction, size_t& phase, kvstore::Key key,
    internal_kvstore::ReadModifyWriteSource& source) {
  return internal_kvstore::AddReadModifyWrite<BtreeWriterTransactionNode>(
      driver, transaction, phase, std::move(key), source, io_handle);
}

absl::Status AddDeleteRange(kvstore::Driver* driver, const IoHandle& io_handle,
                            const internal::OpenTransactionPtr& transaction,
                            KeyRange&& range) {
  return internal_kvstore::AddDeleteRange<BtreeWriterTransactionNode>(
      driver, transaction, std::move(range), io_handle);
}

namespace {

struct IndirectValueReadModifyWriteSource final
    : public kvstore::ReadModifyWriteSource {
  void KvsSetTarget(kvstore::ReadModifyWriteTarget& target) override {
    target_ =
        static_cast<BtreeWriterTransactionNode::ReadModifyWriteEntry*>(&target);
  }

  void KvsInvalidateReadState() override {}

  void KvsWriteback(WritebackOptions options,
                    WritebackReceiver receiver) override {
    if (options.writeback_mode == WritebackMode::kNormalWriteback ||
        options.writeback_mode == WritebackMode::kValidateOnly) {
      // Just return an empty string.  The actual value will be obtained by
      // calling `IsSpecialSource()`.
      TimestampedStorageGeneration stamp;
      stamp.time = absl::InfiniteFuture();
      stamp.generation.MarkDirty();
      execution::set_value(
          std::move(receiver),
          kvstore::ReadResult::Value(absl::Cord(), std::move(stamp)));
      return;
    }
    auto& writer =
        static_cast<BtreeWriterTransactionNode&>(target_->multi_phase());
    if (!StorageGeneration::IsUnknown(
            options.generation_conditions.if_not_equal)) {
      auto generation = internal_ocdbt::ComputeStorageGeneration(value_ref_);
      if (options.generation_conditions.if_not_equal == generation) {
        execution::set_value(
            receiver,
            kvstore::ReadResult::Unspecified(TimestampedStorageGeneration(
                std::move(generation), absl::InfiniteFuture())));
        return;
      }
    }
    if (auto* value_ptr = std::get_if<absl::Cord>(&value_ref_)) {
      auto generation = internal_ocdbt::ComputeStorageGeneration(value_ref_);

      execution::set_value(
          receiver,
          kvstore::ReadResult::Value(
              *value_ptr, TimestampedStorageGeneration(
                              std::move(generation), absl::InfiniteFuture())));
      return;
    }
    execution::submit(writer.io_handle_->ReadIndirectData(
                          std::get<IndirectDataReference>(value_ref_), {}),
                      std::move(receiver));
  }

  void KvsWritebackSuccess(TimestampedStorageGeneration new_stamp) override {
    delete this;
  }

  void KvsWritebackError() override { delete this; }

  void KvsRevoke() override {}

  void* IsSpecialSource() override { return &value_ref_; }

  LeafNodeValueReference value_ref_;
  BtreeWriterTransactionNode::ReadModifyWriteEntry* target_;
};

struct CopySubtreeListReceiver {
  internal::OpenTransactionNodePtr<BtreeWriterTransactionNode> writer;
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
      auto* source = new IndirectValueReadModifyWriteSource;
      source->value_ref_ = entry.value_reference;
      size_t phase;
      writer->ReadModifyWrite(phase, std::move(key), *source);
    }
  }

  void set_done() {}
};

}  // namespace

Future<const void> AddCopySubtree(
    kvstore::Driver* driver, const IoHandle& io_handle,
    const internal::OpenTransactionPtr& transaction,
    BtreeWriter::CopySubtreeOptions&& options) {
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

  auto transaction_copy = transaction;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node,
      internal_kvstore::GetTransactionNode<BtreeWriterTransactionNode>(
          driver, transaction_copy, io_handle));

  auto [promise, future] = PromiseFuturePair<void>::Make(absl::OkStatus());
  NonDistributedListSubtree(
      IoHandle::Ptr(&io_handle), options.node, options.node_height,
      std::move(options.subtree_key_prefix), std::move(options.range),
      CopySubtreeListReceiver{std::move(node), options.strip_prefix_length,
                              std::move(options.add_prefix),
                              std::move(promise)});
  return std::move(future);
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
