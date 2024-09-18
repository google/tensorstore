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
// operation begins.
//
// The actual commit logic is implemented in `btree_writer_commit_operation.h`.
//

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

#include "absl/container/inlined_vector.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/btree_writer.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/btree_writer_commit_operation.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/list.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/staged_mutations.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/storage_generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {

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

struct CommitOperation final
    : public BtreeWriterCommitOperation<MutationEntry> {
  using Base = BtreeWriterCommitOperation<MutationEntry>;

  using Base::Base;

  NonDistributedBtreeWriter::Ptr writer_;
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

  // Fails the commit operation.
  void Fail(const absl::Status& error) override;

  // "Stages" all pending mutations by merging them into the `StagedMutations`
  // data structure.
  void StagePending(WriteStager& stager) override;

  std::optional<const LeafNodeValueReference*> ApplyWriteEntryChain(
      StorageGeneration existing_generation, WriteEntry& last_write_entry,
      bool& validated) override;

  void Retry() override { ReadManifest(); }

  MutationEntryTree& GetEntries() override { return staged_.entries; }

  void CommitSuccessful(absl::Time time) override;
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
  auto commit_op = new CommitOperation(writer.io_handle_);
  // Will be deallocated when operation completes, either by `Fail` or
  // `CommitSuccessful`.
  commit_op->writer_.reset(&writer);
  commit_op->ReadManifest();
}

void CommitOperation::Fail(const absl::Status& error) {
  ABSL_LOG_IF(INFO, ocdbt_logging) << "Commit failed: " << error;
  CommitFailed(staged_, error);
  auto& writer = *writer_;
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

  delete this;
}

void CommitOperation::StagePending(WriteStager& stager) {
  auto& writer = *writer_;
  PendingRequests pending;
  {
    absl::MutexLock lock(&writer.mutex_);
    pending = std::exchange(writer.pending_, {});
  }

  for (auto& request : pending.requests) {
    if (request->kind_ != MutationEntry::kWrite) continue;
    auto& write_request = static_cast<WriteEntry&>(*request);
    if (!write_request.value_) continue;
    stager.Stage(*write_request.value_);
  }

  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Stage requests: " << pending.requests.size();
  flush_promise_.Link(std::move(pending.flush_promise));
  StageMutations(staged_, std::move(pending));
}

std::optional<const LeafNodeValueReference*>
CommitOperation::ApplyWriteEntryChain(StorageGeneration existing_generation,
                                      WriteEntry& last_write_entry,
                                      bool& validated) {
  std::optional<const LeafNodeValueReference*> new_value;
  absl::InlinedVector<const WriteEntry*, 8> write_entries;
  for (const WriteEntry* e = &last_write_entry;;) {
    write_entries.push_back(e);
    if (!(e = e->supersedes_)) break;
  }
  for (auto entry_it = write_entries.rbegin(); entry_it != write_entries.rend();
       ++entry_it) {
    const WriteEntry* e = *entry_it;
    if (e->supersedes_.tag()) {
      // Previous entry was deleted by a `DeleteRange` request before being
      // superseded.
      existing_generation = StorageGeneration::NoValue();
      new_value = nullptr;
    }
    if (StorageGeneration::EqualOrUnspecified(existing_generation,
                                              e->if_equal_)) {
      // `if_equal_` condition was satisfied, write will be marked as having
      // completed successfully.
      if (e->value_) {
        existing_generation =
            internal_ocdbt::ComputeStorageGeneration(*e->value_);
        e->promise_.raw_result()->generation = existing_generation;
        new_value = &*e->value_;
      } else {
        e->promise_.raw_result()->generation = StorageGeneration::NoValue();
        new_value = nullptr;
        existing_generation = StorageGeneration::NoValue();
      }
    } else {
      // `if_equal_` condition was not satisfied, write will be marked as
      // having failed.
      e->promise_.raw_result()->generation = StorageGeneration::Unknown();
    }
  }
  return new_value;
}

void CommitOperation::CommitSuccessful(absl::Time time) {
  internal_ocdbt::CommitSuccessful(staged_, time);
  auto writer = std::move(writer_);
  delete this;
  UniqueWriterLock lock(writer->mutex_);
  writer->commit_in_progress_ = false;
  if (!writer->pending_.requests.empty()) {
    CommitOperation::MaybeStart(*writer, std::move(lock));
  }
}

}  // namespace

Future<TimestampedStorageGeneration> NonDistributedBtreeWriter::Write(
    std::string key, std::optional<absl::Cord> value,
    kvstore::WriteOptions options) {
  auto& writer = *this;
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Write: " << tensorstore::QuoteString(key) << " " << value.has_value();
  auto request = std::make_unique<WriteEntry>();
  request->key_ = std::move(key);
  request->kind_ = MutationEntry::kWrite;
  request->if_equal_ = std::move(options.generation_conditions.if_equal);
  auto [promise, future] =
      PromiseFuturePair<TimestampedStorageGeneration>::Make(std::in_place);
  request->promise_ = std::move(promise);

  Future<const void> value_future;
  if (value) {
    auto& value_ref = request->value_.emplace();
    if (auto* config =
            writer.io_handle_->config_state->GetAssumedOrExistingConfig();
        !config || value->size() <= config->max_inline_value_bytes) {
      // Config not yet known or value to be written inline.
      value_ref = *std::move(value);
    } else {
      value_future = writer.io_handle_->WriteData(
          IndirectDataKind::kValue, *std::move(value),
          value_ref.emplace<IndirectDataReference>());
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
  request->kind_ = MutationEntry::kDeleteRange;
  request->key_ = std::move(range.inclusive_min);
  request->exclusive_max_ = std::move(range.exclusive_max);
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
      request->key_ = std::move(key);
      request->kind_ = MutationEntry::kWrite;
      auto [promise, future] =
          PromiseFuturePair<TimestampedStorageGeneration>::Make(std::in_place);
      request->promise_ = std::move(promise);
      request->value_ = entry.value_reference;
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
