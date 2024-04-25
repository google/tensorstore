// Copyright 2020 The TensorStore Authors
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

#include "tensorstore/internal/cache/async_cache.h"

#include <algorithm>
#include <cassert>
#include <mutex>  // NOLINT
#include <type_traits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/utility/utility.h"
#include "tensorstore/batch.h"
#include "tensorstore/batch_impl.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/container/intrusive_linked_list.h"
#include "tensorstore/internal/container/intrusive_red_black_tree.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

namespace {
using Entry = AsyncCache::Entry;
using ReadState = AsyncCache::ReadState;
using TransactionNode = AsyncCache::TransactionNode;
using TransactionTree = AsyncCache::Entry::TransactionTree;
using PendingWritebackQueueAccessor =
    TransactionNode::PendingWritebackQueueAccessor;
using PrepareForCommitState = TransactionNode::PrepareForCommitState;

// While the real epsilon is `absl::Nanoseconds(1) / 4`, `operator/` is not
// constexpr, and this value is sufficient for use here.
constexpr absl::Duration kEpsilonDuration = absl::Nanoseconds(1);

void AcquireReadRequestReference(Entry& entry) {
  // Prevent the entry from being destroyed while the read is in progress.
  internal::PinnedCacheEntry<AsyncCache>(&entry).release();
}

void ReleaseReadRequestReference(Entry& entry) {
  // Prevent the entry from being destroyed while the read is in progress.
  internal::PinnedCacheEntry<AsyncCache>(&entry, internal::adopt_object_ref);
}

void AcquireReadRequestReference(TransactionNode& node) {
  if (!node.transaction()->commit_started()) {
    // Prevent the transaction from being committed while the read is in
    // progress.
    node.transaction()->AcquireCommitBlock();
  }
  intrusive_ptr_increment(&node);
}

void ReleaseReadRequestReference(TransactionNode& node) {
  if (!node.transaction()->commit_started()) {
    ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
        << node << "Releasing commit block";
    node.transaction()->ReleaseCommitBlock();
  }
  intrusive_ptr_decrement(&node);
}

Future<const void> GetFuture(Promise<void>& promise) {
  if (!promise.null()) {
    auto future = promise.future();
    if (!future.null()) return future;
  }
  auto pair = PromiseFuturePair<void>::Make();
  promise = std::move(pair.promise);
  return std::move(pair.future);
}

const AsyncCache::ReadRequestState& GetEffectiveReadRequestState(Entry& entry) {
  return entry.read_request_state_;
}

const AsyncCache::ReadRequestState& GetEffectiveReadRequestState(
    TransactionNode& node) {
  return node.reads_committed_ ? GetOwningEntry(node).read_request_state_
                               : node.read_request_state_;
}

template <typename EntryOrNode>
void EntryOrNodeStartRead(EntryOrNode& entry_or_node,
                          UniqueWriterLock<Entry> lock, Batch::View batch) {
  static_assert(std::is_same_v<EntryOrNode, Entry> ||
                std::is_same_v<EntryOrNode, TransactionNode>);
  auto& request_state = entry_or_node.read_request_state_;
  if (request_state.queued_request_is_deferred) {
    ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
        << entry_or_node << "EntryOrNodeStartRead: no pending read request";
    return;
  }
  if (!request_state.queued.result_needed()) {
    ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
        << entry_or_node
        << "EntryOrNodeStartRead: pending read request was cancelled";
    request_state.queued = Promise<void>();
    request_state.queued_request_is_deferred = true;
    request_state.queued_time = absl::InfinitePast();
    return;
  }
  assert(request_state.issued.null());
  auto staleness_bound = request_state.issued_time =
      std::exchange(request_state.queued_time, absl::InfinitePast());
  request_state.issued = std::move(request_state.queued);
  request_state.queued_request_is_deferred = true;
  lock.unlock();
  AcquireReadRequestReference(entry_or_node);
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
      << entry_or_node << "EntryOrNodeStartRead: calling DoRead";
  AsyncCache::AsyncCacheReadRequest read_request;
  read_request.staleness_bound = staleness_bound;
  read_request.batch = batch;
  entry_or_node.DoRead(std::move(read_request));
}

/// Starts a previously-requested read or writeback operation.
///
/// This function is called when a read or writeback operation completes, or a
/// new writeback is requested.
void MaybeStartReadOrWriteback(Entry& entry, UniqueWriterLock<Entry> lock,
                               Batch::View read_batch) {
  auto& read_request_state = entry.read_request_state_;

  if (TransactionNode* committing_transaction_node =
          entry.committing_transaction_node_) {
    TransactionNode* next;
    while (true) {
      // It is safe to use `std::memory_order_relaxed` since writes are already
      // protected by the entry's mutex.
      const auto existing_prepare_for_commit_state =
          committing_transaction_node->prepare_for_commit_state_.load(
              std::memory_order_relaxed);
      const bool read_request_issued = !read_request_state.issued.null();
      // Determine the transitions we will make on `prepare_for_commit_state_`.
      // We need to update `prepare_for_commit_state_` while holding the `lock`,
      // but we can't actually perform the required actions until after
      // releasing the `lock`.  Therefore, we duplicate this switch logic below.
      PrepareForCommitState new_prepare_for_commit_state;
      switch (existing_prepare_for_commit_state) {
        case PrepareForCommitState::kNone:
        case PrepareForCommitState::kPrepareDoneCalled:
          new_prepare_for_commit_state =
              PrepareForCommitState::kPrepareDoneCalled;
          // Don't allow this `TransactionNode` to be committed until any
          // in-flight read request on the `Entry` completes.  This limitation
          // serves to avoid the possibility of a cache inconsistency that could
          // occur if both a read and write operation to the underlying storage
          // are in flight simultaneously, given that we don't assume an
          // ordering on `StorageGeneration` values.
          if (read_request_issued) break;
          [[fallthrough]];
        case PrepareForCommitState::kReadyForCommitCalled:
          new_prepare_for_commit_state =
              PrepareForCommitState::kReadyForCommitCalled;
      }
      committing_transaction_node->prepare_for_commit_state_ =
          new_prepare_for_commit_state;
      next =
          PendingWritebackQueueAccessor::GetNext(committing_transaction_node);
      if (next == committing_transaction_node ||
          next->transaction() != committing_transaction_node->transaction() ||
          next->prepare_for_commit_state_.load(std::memory_order_relaxed) ==
              PrepareForCommitState::kReadyForCommitCalled) {
        next = nullptr;
      }
      lock.unlock();
      switch (existing_prepare_for_commit_state) {
        case PrepareForCommitState::kNone:
          ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
              << *committing_transaction_node << "PrepareDone";
          committing_transaction_node->PrepareDone();
          [[fallthrough]];
        case PrepareForCommitState::kPrepareDoneCalled:
          if (read_request_issued) return;
          ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
              << *committing_transaction_node << "ReadyForCommit";
          committing_transaction_node->ReadyForCommit();
          break;
        case PrepareForCommitState::kReadyForCommitCalled:
          break;
      }
      if (!next) return;
      committing_transaction_node = next;
      lock = UniqueWriterLock<Entry>(entry);
    }
  }

  if (read_request_state.issued.null()) {
    // Issue a read if requested.
    EntryOrNodeStartRead(entry, std::move(lock), read_batch);
  }
}

void MaybeIssueRead(Entry& entry, UniqueWriterLock<Entry> lock,
                    Batch::View batch) {
  MaybeStartReadOrWriteback(entry, std::move(lock), batch);
}

void MaybeIssueRead(TransactionNode& node, UniqueWriterLock<Entry> lock,
                    Batch::View batch) {
  if (!node.read_request_state_.issued.null()) return;
  EntryOrNodeStartRead(node, std::move(lock), batch);
}

template <typename EntryOrNode>
void SetReadState(EntryOrNode& entry_or_node, ReadState&& read_state,
                  size_t read_state_size) {
  static_assert(std::is_same_v<EntryOrNode, Entry> ||
                std::is_same_v<EntryOrNode, TransactionNode>);
  if constexpr (std::is_same_v<EntryOrNode, TransactionNode>) {
    if (entry_or_node.reads_committed_) {
      assert(entry_or_node.prepare_for_commit_state_.load(
                 std::memory_order_relaxed) ==
             PrepareForCommitState::kReadyForCommitCalled);
      SetReadState(GetOwningEntry(entry_or_node), std::move(read_state),
                   read_state_size);
      return;
    }
  }
  entry_or_node.read_request_state_.known_to_be_stale = false;
  entry_or_node.read_request_state_.read_state = std::move(read_state);
  size_t change =
      read_state_size -
      std::exchange(entry_or_node.read_request_state_.read_state_size,
                    read_state_size);
  if (change != 0) {
    if constexpr (std::is_same_v<EntryOrNode, TransactionNode>) {
      entry_or_node.UpdateSizeInBytes(change);
    } else {
      entry_or_node.NotifySizeChanged();
    }
  }
}

template <typename EntryOrNode>
class AsyncCacheBatchEntry : public Batch::Impl::Entry {
 public:
  using EntryOrNodePtr =
      std::conditional_t<std::is_same_v<EntryOrNode, AsyncCache::Entry>,
                         PinnedCacheEntry<AsyncCache>,
                         OpenTransactionNodePtr<AsyncCache::TransactionNode>>;

  using KeyParam = internal_future::FutureStateBase*;
  explicit AsyncCacheBatchEntry(size_t nesting_depth,
                                EntryOrNode& entry_or_node,
                                Promise<void> promise)
      : Batch::Impl::Entry(nesting_depth),
        entry_or_node_(&entry_or_node),
        promise_(std::move(promise)) {}
  KeyParam key() const { return &internal_future::FutureAccess::rep(promise_); }

 private:
  void Submit(Batch::View batch) override {
    ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
        << *entry_or_node_ << "Submitting batch read";
    auto& entry = GetOwningEntry(*entry_or_node_);
    UniqueWriterLock lock(entry);
    auto& read_request_state = entry_or_node_->read_request_state_;
    if (!HaveSameSharedState(read_request_state.queued, promise_)) {
      // Read was cancelled or already issued, nothing to do.
      return;
    }
    read_request_state.queued_request_is_deferred = false;
    MaybeIssueRead(*entry_or_node_, std::move(lock), batch);
    delete this;
  }

  EntryOrNodePtr entry_or_node_;
  Promise<void> promise_;
};

template <typename EntryOrNode>
Future<const void> RequestRead(EntryOrNode& entry_or_node,
                               AsyncCache::AsyncCacheReadRequest options,
                               bool must_not_be_known_to_be_stale) {
  static_assert(std::is_same_v<EntryOrNode, Entry> ||
                std::is_same_v<EntryOrNode, TransactionNode>);
  auto& entry = GetOwningEntry(entry_or_node);
  UniqueWriterLock lock(entry);

  auto& effective_request_state = GetEffectiveReadRequestState(entry_or_node);
  const auto existing_time = effective_request_state.read_state.stamp.time;
  if (existing_time != absl::InfinitePast() &&
      existing_time >= options.staleness_bound) {
    if (must_not_be_known_to_be_stale &&
        effective_request_state.known_to_be_stale) {
      options.staleness_bound = existing_time + kEpsilonDuration;
    } else {
      // `staleness_bound` satisfied by current data.
      return MakeReadyFuture();
    }
  }

  auto& request_state = entry_or_node.read_request_state_;
  // `staleness_bound` not satisfied by current data.
  request_state.queued_time =
      std::max(request_state.queued_time,
               std::min(options.staleness_bound, absl::Now()));
  if (!request_state.issued.null() &&
      request_state.issued_time >= options.staleness_bound) {
    // Another read is in progress, and `staleness_bound` will be satisfied by
    // it when it completes.
    return GetFuture(request_state.issued);
  }

  auto future = GetFuture(request_state.queued);

  if (options.batch.deferred() && request_state.queued_request_is_deferred) {
    // Don't actually issue request, just ensure there is a batch node for
    // it.  The actual request will be issued when the batch is submitted.
    using BatchE = AsyncCacheBatchEntry<EntryOrNode>;
    auto& promise = request_state.queued;
    Batch::Impl::From(options.batch)
        ->GetEntry<BatchE>(&internal_future::FutureAccess::rep(promise), [&] {
          return std::make_unique<BatchE>(
              GetOwningCache(entry).BatchNestingDepth(), entry_or_node,
              promise);
        });
  } else {
    request_state.queued_request_is_deferred = false;
  }

  MaybeIssueRead(entry_or_node, std::move(lock), options.batch);
  return future;
}

/// Completes a queued read in the destructor if already satisfied by the read
/// state.
class QueuedReadHandler {
 public:
  // Must be invoked with the `Entry::mutex_` locked.
  explicit QueuedReadHandler(AsyncCache::ReadRequestState& request_state,
                             absl::Time time) {
    if (!request_state.queued.null() && time >= request_state.queued_time) {
      // Queued read is also satisfied.
      queued_ = std::move(request_state.queued);
      request_state.queued_time = absl::InfinitePast();
      request_state.queued_request_is_deferred = true;
    }
  }

  // Must be invoked with the `Entry::mutex_` unlocked.
  ~QueuedReadHandler() {
    if (!queued_.null()) {
      queued_.SetResult(tensorstore::MakeResult());
    }
  }

 private:
  Promise<void> queued_;
};

template <typename EntryOrNode>
void ResolveIssuedRead(EntryOrNode& entry_or_node, absl::Status status,
                       UniqueWriterLock<Entry> lock) {
  static_assert(std::is_same_v<EntryOrNode, Entry> ||
                std::is_same_v<EntryOrNode, TransactionNode>);
  auto& request_state = entry_or_node.read_request_state_;
  auto issued = std::move(request_state.issued);
  auto time = GetEffectiveReadRequestState(entry_or_node).read_state.stamp.time;
  assert(!issued.null());
  assert(!status.ok() || time >= request_state.issued_time);
  {
    QueuedReadHandler queued_read_handler(request_state, time);
    MaybeIssueRead(entry_or_node, std::move(lock), /*batch=*/{});
    // Resolve promises after locks are released, to avoid running Future
    // callbacks with locks held.  It is possible that `issued` was already
    // resolved by a prior `ReadUpdate` call, in which case the call to
    // `issued.SetResult` will have no effect.
    issued.SetResult(tensorstore::MakeResult(status));
  }
  ReleaseReadRequestReference(entry_or_node);
}

size_t GetReadStateSize(Entry& entry, const void* read_data) {
  if (!read_data) return 0;
  return entry.ComputeReadDataSizeInBytes(read_data);
}

template <typename EntryOrNode>
void EntryOrNodeReadSuccess(EntryOrNode& entry_or_node,
                            ReadState&& read_state) {
  static_assert(std::is_same_v<EntryOrNode, Entry> ||
                std::is_same_v<EntryOrNode, TransactionNode>);
  Entry& entry = GetOwningEntry(entry_or_node);
  const size_t read_state_size = GetReadStateSize(entry, read_state.data.get());
  UniqueWriterLock lock(entry);
  assert(read_state.stamp.time != absl::InfinitePast());
  assert(!StorageGeneration::IsUnknown(read_state.stamp.generation));
  SetReadState(entry_or_node, std::move(read_state), read_state_size);
  ResolveIssuedRead(entry_or_node, absl::OkStatus(), std::move(lock));
}

template <typename EntryOrNode>
void EntryOrNodeReadError(EntryOrNode& entry_or_node, absl::Status error) {
  static_assert(std::is_same_v<EntryOrNode, Entry> ||
                std::is_same_v<EntryOrNode, TransactionNode>);
  assert(!error.ok());
  ResolveIssuedRead(entry_or_node, std::move(error),
                    UniqueWriterLock{GetOwningEntry(entry_or_node)});
}

void RemoveTransactionFromMap(TransactionNode& node) {
  if (TransactionTree::IsDisconnected(node)) {
    return;
  }
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
      << node << "RemoveTransactionFromMap";
  GetOwningEntry(node).transactions_.Remove(node);
}

void ResolveIssuedWriteback(AsyncCache::TransactionNode& node,
                            UniqueWriterLock<Entry> lock) {
  auto& entry = GetOwningEntry(node);

  // Writeback of this transaction must have been in progress.
  assert(node.prepare_for_commit_state_.load(std::memory_order_relaxed) ==
         PrepareForCommitState::kReadyForCommitCalled);
  assert(entry.committing_transaction_node_ &&
         entry.committing_transaction_node_->transaction() ==
             node.transaction());

  // Read must not be in progress.
  assert(entry.read_request_state_.issued.null());

  if (entry.committing_transaction_node_ != &node) {
    intrusive_linked_list::Remove(PendingWritebackQueueAccessor{}, &node);
  } else {
    auto* next_node = PendingWritebackQueueAccessor::GetNext(&node);
    if (next_node != &node) {
      intrusive_linked_list::Remove(PendingWritebackQueueAccessor{}, &node);
      if (next_node->transaction() == node.transaction()) {
        entry.committing_transaction_node_ = next_node;
      } else {
        entry.committing_transaction_node_ = next_node;
      }
    } else {
      entry.committing_transaction_node_ = nullptr;
    }
  }
  RemoveTransactionFromMap(node);
  MaybeStartReadOrWriteback(entry, std::move(lock), /*read_batch=*/{});
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG) << node << "CommitDone";
  node.CommitDone();
}

}  // namespace

const ReadState& AsyncCache::ReadState::Unknown() {
  static const internal::NoDestructor<ReadState> read_state;
  return *read_state;
}

size_t AsyncCache::Entry::ComputeReadDataSizeInBytes(const void* data) {
  return 0;
}

size_t AsyncCache::DoGetFixedSizeInBytes(Cache::Entry* entry) {
  return this->Cache::DoGetSizeInBytes(entry);
}

size_t AsyncCache::DoGetSizeInBytes(Cache::Entry* base_entry) {
  auto* entry = static_cast<Entry*>(base_entry);
  return this->DoGetFixedSizeInBytes(entry) +
         entry->read_request_state_.read_state_size;
}

Future<const void> AsyncCache::Entry::Read(AsyncCacheReadRequest request,
                                           bool must_not_be_known_to_be_stale) {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
      << *this << "Read: staleness_bound=" << request.staleness_bound
      << ", must_not_be_known_to_be_stale=" << must_not_be_known_to_be_stale;
  return RequestRead(*this, request, must_not_be_known_to_be_stale);
}

void AsyncCache::Entry::ReadSuccess(ReadState&& read_state) {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
      << *this << "ReadSuccess: " << read_state.stamp
      << ", data=" << read_state.data.get();
  internal::EntryOrNodeReadSuccess(*this, std::move(read_state));
}

void AsyncCache::Entry::ReadError(absl::Status error) {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
      << *this << "ReadError: error=" << error;
  internal::EntryOrNodeReadError(*this, std::move(error));
}

AsyncCache::TransactionNode::TransactionNode(Entry& entry)
    : internal::TransactionState::Node(Cache::PinnedEntry(&entry).release()),
      reads_committed_(false),
      size_updated_(false) {}

Future<const void> AsyncCache::TransactionNode::Read(
    AsyncCacheReadRequest request, bool must_not_be_known_to_be_stale) {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
      << *this << "Read: staleness_bound=" << request.staleness_bound
      << ", must_not_be_known_to_be_stale=" << must_not_be_known_to_be_stale;
  if (reads_committed_ &&
      (prepare_for_commit_state_.load(std::memory_order_acquire) !=
       PrepareForCommitState::kReadyForCommitCalled)) {
    return RequestRead(GetOwningEntry(*this), request,
                       must_not_be_known_to_be_stale);
  }
  return RequestRead(*this, request, must_not_be_known_to_be_stale);
}

void AsyncCache::TransactionNode::ReadSuccess(ReadState&& read_state) {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
      << *this << "ReadSuccess: " << read_state.stamp
      << ", data=" << read_state.data.get();
  internal::EntryOrNodeReadSuccess(*this, std::move(read_state));
}

void AsyncCache::TransactionNode::ReadError(absl::Status error) {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
      << *this << "ReadError: error=" << error;
  internal::EntryOrNodeReadError(*this, std::move(error));
}

void AsyncCache::TransactionNode::PrepareForCommit() {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
      << *this << "PrepareForCommit";
  // Acquire reference to be released by `Commit`.
  intrusive_ptr_increment(this);
  auto& entry = GetOwningEntry(*this);
  UniqueWriterLock lock(entry);
  RemoveTransactionFromMap(*this);
  if (entry.committing_transaction_node_) {
    // Another node is already being committed.  Add this node to the end of the
    // queue.
    intrusive_linked_list::InsertBefore(PendingWritebackQueueAccessor{},
                                        entry.committing_transaction_node_,
                                        this);
    if (entry.committing_transaction_node_->transaction() != transaction()) {
      ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
          << *this << "Commit: enqueuing for writeback";
      return;
    }
    // PrepareDone on the prior node in this same transaction must have been
    // called already, as otherwise `PrepareForCommit` would not have been
    // invoked on this node yet.
    assert(entry.committing_transaction_node_->prepare_for_commit_state_.load(
               std::memory_order_relaxed) >=
           PrepareForCommitState::kPrepareDoneCalled);
  } else {
    // No node is already being committed.
    intrusive_linked_list::Initialize(PendingWritebackQueueAccessor{}, this);
  }
  // Can request writeback immediately (but it will still wait until any
  // previously issued read request completes).
  entry.committing_transaction_node_ = this;
  MaybeStartReadOrWriteback(entry, std::move(lock), /*read_batch=*/{});
}

void AsyncCache::TransactionNode::Abort() {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG) << *this << "Abort";
  auto& entry = GetOwningEntry(*this);
  UniqueWriterLock lock(entry);
  RemoveTransactionFromMap(*this);
  lock.unlock();
  AbortDone();
}

void AsyncCache::TransactionNode::WritebackSuccess(ReadState&& read_state) {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
      << *this << "WritebackSuccess: " << read_state.stamp
      << ", data=" << read_state.data.get();
  auto& entry = GetOwningEntry(*this);
  const size_t read_state_size = GetReadStateSize(entry, read_state.data.get());
  UniqueWriterLock lock{entry};
  auto& request_state = entry.read_request_state_;
  absl::Time read_state_time = read_state.stamp.time;
  if (!StorageGeneration::IsUnknown(read_state.stamp.generation)) {
    assert(read_state.stamp.generation != StorageGeneration::Invalid());
    assert(read_state_time != absl::InfinitePast());
    assert(read_state_time >= request_state.read_state.stamp.time);
    SetReadState(entry, std::move(read_state), read_state_size);
  } else if (read_state_time > request_state.read_state.stamp.time) {
    request_state.known_to_be_stale = true;
  }

  QueuedReadHandler queued_read_handler(request_state, read_state_time);
  ResolveIssuedWriteback(*this, std::move(lock));
}

void AsyncCache::TransactionNode::WritebackError() {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG) << *this << "WritebackError";
  ResolveIssuedWriteback(*this, UniqueWriterLock{GetOwningEntry(*this)});
}

Result<OpenTransactionNodePtr<AsyncCache::TransactionNode>>
AsyncCache::Entry::GetTransactionNodeImpl(OpenTransactionPtr& transaction) {
  // Initializes `node` if it has not already been initialized.  Returns `true`
  // if the current performed initialization.
  constexpr auto EnsureTransactionNodeInitialized =
      [](AsyncCache::TransactionNode& node,
         OpenTransactionPtr& transaction) -> bool {
    auto& entry = GetOwningEntry(node);
    // Ensure transaction node is initialized.
    bool initialized = false;
    absl::call_once(node.initialized_, [&] {
      const bool new_implicit_transaction = !transaction;
      node.initialized_status_ = node.DoInitialize(transaction);
      if (node.initialized_status_.ok()) {
        if (new_implicit_transaction) {
          node.SetTransaction(GetOrCreateOpenTransaction(transaction));
          UniqueWriterLock lock(entry);
          entry.transactions_.FindOrInsert(
              [&](TransactionNode& existing_node) {
                return internal::intrusive_red_black_tree::
                    ThreeWayFromLessThan<>()(transaction.get(),
                                             existing_node.transaction());
              },
              [&] { return &node; });
        }
        assert(node.transaction() == transaction.get());
        ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
            << node << "New node, new implicit=" << new_implicit_transaction
            << ", transaction=" << transaction.get();
        node.initialized_status_ = node.Register();
      } else if (!new_implicit_transaction) {
        // If initialization failed, remove reference in cache entry to node.
        UniqueWriterLock lock(entry);
        RemoveTransactionFromMap(node);
      }
      initialized = true;
    });
    return initialized;
  };

  WeakTransactionNodePtr<TransactionNode> node;
  if (!transaction) {
    // Create new implicit transaction.
    WeakTransactionNodePtr<TransactionNode> stale_node;
    while (true) {
      node.reset(GetOwningCache(*this).DoAllocateTransactionNode(*this));
      [[maybe_unused]] bool initialized =
          EnsureTransactionNodeInitialized(*node, transaction);
      TENSORSTORE_RETURN_IF_ERROR(node->initialized_status_);
      assert(initialized);
      if (node->IsRevoked()) {
        ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
            << *node << "Node is revoked";
        std::swap(stale_node, node);
        continue;
      }
      node->transaction()->RequestCommit();
      break;
    }
  } else {
    // Handle explicit transaction case.
    size_t min_phase = transaction->phase();
    WeakTransactionNodePtr<TransactionNode> stale_node;
    while (true) {
      UniqueWriterLock lock(*this);
      const auto MakeNode = [&] {
        auto* node = GetOwningCache(*this).DoAllocateTransactionNode(*this);
        node->SetTransaction(*transaction);
        ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
            << *node << "Adding transaction to map";
        return node;
      };
      auto* candidate_node =
          transactions_
              .FindOrInsert(
                  [transaction = transaction.get()](TransactionNode& node) {
                    return internal::intrusive_red_black_tree::
                        ThreeWayFromLessThan<>()(transaction,
                                                 node.transaction());
                  },
                  MakeNode)
              .first;
      if (candidate_node == stale_node.get()) {
        // Existing node is from a previous phase or has been revoked, and
        // must be replaced by a new node.
        auto* new_node = MakeNode();
        ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
            << *candidate_node << "Replacing in map";
        ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
            << *new_node << "Adding to map";
        transactions_.Replace(*candidate_node, *new_node);
        candidate_node = new_node;
      }
      // Note: `IntrusivePtr::reset` is safe to call on a moved-from `node`
      // object.
      node.reset(candidate_node);  // NOLINT
      lock.unlock();
      stale_node.reset();
      EnsureTransactionNodeInitialized(*node, transaction);
      TENSORSTORE_RETURN_IF_ERROR(node->initialized_status_);
      if (node->phase() >= min_phase && !node->IsRevoked()) {
        break;
      }
      stale_node = std::move(node);
    }
  }
  // Upgrade `WeakTransactionNodePtr` to `OpenTransactionNodePtr`.  It is safe
  // to acquire another open reference because one is already owned by
  // `transaction`.
  OpenTransactionPtr(node->transaction()).release();
  return OpenTransactionNodePtr<TransactionNode>(node.release(),
                                                 internal::adopt_object_ref);
}

void AsyncCache::TransactionNode::Commit() { intrusive_ptr_decrement(this); }

void AsyncCache::TransactionNode::WriterLock() { mutex_.WriterLock(); }

void AsyncCache::TransactionNode::WriterUnlock() {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG) << *this << "unlock";
  UniqueWriterLock lock(mutex_, std::adopt_lock);
  if (!size_updated_) return;
  size_updated_ = false;
  const size_t new_size = this->ComputeWriteStateSizeInBytes();
  const size_t change = new_size - std::exchange(write_state_size_, new_size);
  if (change == 0) return;
  this->UpdateSizeInBytes(change);
}

bool AsyncCache::TransactionNode::try_lock() {
  mutex_.WriterLock();
  if (!IsRevoked()) return true;
  mutex_.WriterUnlock();
  return false;
}

size_t AsyncCache::TransactionNode::ComputeWriteStateSizeInBytes() { return 0; }

absl::Status AsyncCache::TransactionNode::DoInitialize(
    internal::OpenTransactionPtr& transaction) {
  return absl::OkStatus();
}

void AsyncCache::TransactionNode::DoApply(ApplyOptions options,
                                          ApplyReceiver receiver) {
  ABSL_UNREACHABLE();  // COV_NF_LINE
}

void AsyncCache::TransactionNode::Revoke() {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG) << *this << "Revoke";
  revoked_.store(true, std::memory_order_release);
}

void AsyncCache::TransactionNode::InvalidateReadState() {
  assert(this->transaction()->commit_started());
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
      << *this << "InvalidateReadState";
  this->read_request_state_.read_state = ReadState{};
}

AsyncCache::TransactionNode::~TransactionNode() {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
      << *this << "~TransactionNode";
  Cache::PinnedEntry(static_cast<Cache::Entry*>(associated_data()),
                     adopt_object_ref);
}

#ifdef TENSORSTORE_ASYNC_CACHE_DEBUG
AsyncCache::Entry::~Entry() {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG) << *this << "~Entry";
}
#endif

}  // namespace internal
}  // namespace tensorstore
