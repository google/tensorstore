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

#include "tensorstore/transaction.h"

#include "absl/base/optimization.h"
#include "absl/functional/function_ref.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/transaction_impl.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

/// Ensure `kAtomic` remains consistent with `TransactionMode` definitions.
static_assert(TransactionState::kAtomic ==
              TransactionMode::atomic_isolated - TransactionMode::isolated);

namespace {
absl::Status GetCancelledError() {
  return absl::CancelledError("Transaction aborted");
}
}  // namespace

TransactionState::Node::Node(void* associated_data)
    : associated_data_(associated_data) {}

std::string TransactionState::Node::Describe() { return {}; }
TransactionState::Node::~Node() = default;

void TransactionState::NoMoreCommitReferences() {
  UniqueWriterLock lock(mutex_);
  const size_t count = commit_reference_count_.load(std::memory_order_relaxed);
  if (count >= kCommitReferenceIncrement) {
    // Another commit reference was created concurrently.
    return;
  }

  if (count == kFutureReferenceIncrement) {
    // The only remaining commit handle is the future handle.  Release the
    // future reference owned by the `TransactionState` itself to break the
    // reference cycle.
    //
    // Release the future reference without `mutex_` held, since it results in
    // promise callbacks being run (and may in turn result in a call to
    // `NoMoreCommitReferences`).
    auto future = std::move(future_);
    lock.unlock();
    return;
  }

  assert(count == 0);

  // Abort if still pending.
  if (commit_state_ != kOpen) return;
  this->RequestAbort(GetCancelledError(), std::move(lock));
}

TransactionState::OpenPtr TransactionState::AcquireImplicitOpenPtr() {
  assert(implicit_transaction_);
  absl::MutexLock lock(&mutex_);
  if (commit_state_ == kAborted || commit_state_ == kCommitStarted) {
    return {};
  }
  Future<const void> future;
  if (future_.null()) {
    // Future reference was already released.  Try to obtain another future
    // reference.
    future = promise_.future();
    if (future.null()) return {};
  }
  if (!future.null()) {
    future_ = std::move(future);
  }
  return OpenPtr(this);
}

TransactionState::TransactionState(TransactionMode mode,
                                   bool implicit_transaction)
    : mode_(mode),
      commit_reference_count_{kFutureReferenceIncrement +
                              kCommitReferenceIncrement},
      open_reference_count_{implicit_transaction ? 1u : 0u},
      // Two weak references initially, one owned by the promise callback
      // attached to `future_`, and one for the initial `Transaction` object.
      weak_reference_count_{2},
      total_bytes_{0},
      commit_state_{kOpen},
      implicit_transaction_(implicit_transaction) {
  if (IsAtomic(mode)) {
    existing_terminal_node_ = nullptr;
  } else {
    phase_ = 0;
  }
  auto [promise, future] = PromiseFuturePair<void>::Make(MakeResult());
  promise_callback_ = promise.ExecuteWhenForced(
      [self = IntrusivePtr<TransactionState,
                           CommitPtrTraits<kFutureReferenceIncrement>>(
           this, adopt_object_ref)](Promise<void> promise) {
        self->RequestCommit();
      });
  promise_ = std::move(promise);
  future_ = std::move(future);
}

Result<TransactionState::OpenPtr> TransactionState::AcquireOpenPtrOrError() {
  if (auto handle = AcquireOpenPtr()) return handle;
  return absl::InvalidArgumentError("Transaction not open");
}

TransactionState::OpenPtr TransactionState::AcquireOpenPtr() {
  absl::MutexLock lock(&mutex_);
  assert(commit_reference_count_.load() != 0);
  if (commit_state_ == kAborted || commit_state_ == kCommitStarted) {
    return {};
  }
  return TransactionState::OpenPtr(this);
}

void TransactionState::RequestCommit() {
  {
    absl::MutexLock lock(&mutex_);
    if (commit_state_ != kOpen) return;
    if (open_reference_count_.load(std::memory_order_relaxed) != 0) {
      // A thread is not permitted to increase `open_reference_count_` from 0
      // except while owning a lock on `mutex_`.  If another thread concurrently
      // decreases `open_reference_count_` to 0, `NoMoreOpenReferences` will
      // take care of calling `ExecuteCommit()`.
      commit_state_ = kOpenAndCommitRequested;
      return;
    } else {
      commit_state_ = kCommitStarted;
    }
  }
  ExecuteCommit();
}

void TransactionState::RequestAbort() { RequestAbort(GetCancelledError()); }

void TransactionState::RequestAbort(const absl::Status& error) {
  RequestAbort(error, UniqueWriterLock(mutex_));
}

void TransactionState::RequestAbort(const absl::Status& error,
                                    UniqueWriterLock<absl::Mutex> lock) {
  auto commit_state = commit_state_;
  if (commit_state > kOpenAndCommitRequested) return;
  SetDeferredResult(promise_, error);
  if (open_reference_count_.load(std::memory_order_relaxed) != 0) {
    // A thread is not permitted to increase `open_reference_count_` from 0
    // except while owning a lock on `mutex_`.  If another thread concurrently
    // decreases `open_reference_count_` to 0, `NoMoreOpenReferences` will take
    // care of calling `ExecuteAbort()`.
    commit_state_ = kAbortRequested;
    return;
  } else {
    commit_state_ = kAborted;
  }
  // Ensure `ExecuteAbort` is run with the lock released.
  lock.unlock();
  ExecuteAbort();
}

void TransactionState::ExecuteAbort() {
  // Release the promise callback to break the reference cycle.
  promise_callback_.Unregister();
  if (nodes_.empty()) {
    // Nothing to abort, just release `promise_` so that it becomes ready with
    // the error set previously by `SetDeferredResult`.
    promise_ = Promise<void>();
    return;
  }
  // Transaction nodes are allowed to abort asynchronously.  When they are
  // finished aborting, they call `AbortDone`, which invokes
  // `DecrementNodesPendingAbort`.
  //
  // Unlike in `DecrementNodesPendingReadyForCommit`, we do not need to hold an
  // additional weak reference to `this` while calling `node->Abort()`, because
  // the caller of `ExecuteAbort` must be holding a weak reference to `this`.
  nodes_pending_abort_.store(0, std::memory_order_relaxed);
  size_t count = 0;
  for (Node *next, *node = nodes_.ExtremeNode(Tree::kLeft); node; node = next) {
    // Save `next` node before removing the current node.
    next = Tree::Traverse(*node, Tree::kRight);
    nodes_.Remove(*node);
    node->Abort();
    ++count;
  }
  // Increment counter just once for the entire loop.  If some or all nodes call
  // `AbortDone` before the loop ends, the counter will just be "negative"
  // (actually a high positive value close to
  // `std::numeric_limits<size_t>::max()` since `nodes_pending_abort_` is
  // unsigned), but it is not possible for the count to be zero.
  DecrementNodesPendingAbort(-count);
}

void TransactionState::DecrementNodesPendingAbort(size_t count) {
  if (nodes_pending_abort_.fetch_sub(count, std::memory_order_acq_rel) !=
      count) {
    // Count hasn't reached zero, some nodes still aborting.
    return;
  }
  // All nodes aborted.  Release `promise_` so that it becomes ready with the
  // error set previously by `SetDeferredResult`.
  promise_ = Promise<void>();
}

void TransactionState::ExecuteCommit() {
  assert(commit_state_ == kCommitStarted);
  // Release the promise callback to break the reference cycle.
  promise_callback_.Unregister();
  ExecuteCommitPhase();
}

void TransactionState::ExecuteCommitPhase() {
  if (nodes_.empty()) {
    // All phases completed.
    promise_ = Promise<void>();
    return;
  }
  // Reset the `commit_start_time_` at the start of each phase, because for
  // "consistent read" validation, the read must be verified as of the start of
  // the current phase.
  commit_start_time_ = absl::Now();

  // Initialize the `nodes_pending_commit_` count to 1.  It will be incremented
  // for each node in the phase before calling `PrepareForCommit`.  The initial
  // count of 1 serves to indicate that `PrepaerForCommit` has not yet been
  // called on all nodes in the phase.  Once the end of the phase is reached,
  // the counter is decremented to account for that.
  nodes_pending_commit_.store(1, std::memory_order_relaxed);

  // Start at the left-most node in the tree, which is necessarily associated
  // with the earliest not-yet-committed phase.  (Nodes from already-committed
  // phases are destroyed once commit completes.)
  auto* node = nodes_.ExtremeNode(Tree::kLeft);
  ContinuePrepareForCommit(node, node->phase());
}

void TransactionState::ContinuePrepareForCommit(Node* node,
                                                size_t current_phase) {
  while (true) {
    if (!node || node->phase() != current_phase) {
      // End of phase.
      DecrementNodesPendingReadyForCommit();
      break;
    }
    waiting_for_prepare_done_.store(true, std::memory_order_relaxed);
    nodes_pending_commit_.fetch_add(1, std::memory_order_relaxed);
    node->PrepareForCommit();
    if (waiting_for_prepare_done_.exchange(false, std::memory_order_acq_rel)) {
      // `node->PrepareDone` has not yet been called.  `PrepareDone` will
      // continue the commit process when called.
      return;
    }
    node = Tree::Traverse(*node, Tree::kRight);
  }
}

void TransactionState::DecrementNodesPendingReadyForCommit() {
  if (nodes_pending_commit_.fetch_sub(1, std::memory_order_acq_rel) != 1) {
    // Not all nodes have called `ReadyForCommit` yet, or
    // `ContinuePrepareForCommit` is not done yet.
    return;
  }
  // Current phase ready to be committed.

  // Ensure the transaction state is not freed until this method completes.  The
  // call the `node->Commit()` below may cause `node` to be freed, which might
  // otherwise hold the last reference to `this`.
  WeakPtrTraits::increment(this);
  Node* node = nodes_.ExtremeNode(Tree::kLeft);
  const size_t current_phase = node->phase();
  // Reuse the `nodes_pending_commit_` counter (which is guaranteed to be 0) to
  // count the number of nodes that still need to call `CommitDone`.  As in
  // `ExecuteAbort`, we increment the counter just once after all the calls to
  // `Commit`.  If any of the derived node implementations call `CommitDone`
  // before the loop ends and our call to `DecrementNodesPendingCommit`, the
  // counter will just wrap around, but is still guaranteed not to equal 0 until
  // the call to `DecrementNodesPendingCommit` below.
  size_t count = 0;
  while (true) {
    // Save next node before removing `node`.
    Node* next = Tree::Traverse(*node, Tree::kRight);
    // Nodes destroy themselves when they finish committing; remove them before
    // commit starts since that avoids the need for locks.
    nodes_.Remove(*node);
    ++count;
    node->Commit();
    if (!next || next->phase() != current_phase) break;
    node = next;
  }
  DecrementNodesPendingCommit(-count);
  WeakPtrTraits::decrement(this);
}

void TransactionState::DecrementNodesPendingCommit(size_t count) {
  if (nodes_pending_commit_.fetch_sub(count, std::memory_order_acq_rel) !=
      count) {
    // Not all nodes have called `CommitDone`, or
    // `DecrementNodesPendingReadyForCommit` has not yet finished invoking
    // `Commit` on every node in the phase.
    return;
  }
  // Current phase completed.
  if (!nodes_.empty()) {
    if (promise_.raw_result().ok()) {
      // Commit next phase.
      ExecuteCommitPhase();
    } else {
      // An error occurred during commit of the last phase.  Abort remaining
      // phases.
      ExecuteAbort();
    }
  } else {
    // All phases completed.  Release the reference to `promise_` so that it
    // becomes ready either with success or with the error set previously by
    // `SetDeferredResult`.
    promise_ = Promise<void>();
  }
}

void TransactionState::Barrier() {
  if (atomic()) {
    return;
  }
  absl::MutexLock lock(&mutex_);
  if (commit_state_ == kCommitStarted || commit_state_ == kAborted) {
    return;
  }
  ++phase_;
}

void TransactionState::Node::SetTransaction(TransactionState& transaction) {
  transaction_.reset(&transaction);
}

void TransactionState::Node::SetPhase(size_t phase) { phase_ = phase; }

size_t TransactionState::phase() {
  if (atomic()) {
    return 0;
  }
  absl::MutexLock lock(&mutex_);
  assert(commit_state_ < kCommitStarted);
  return phase_;
}

absl::Status TransactionState::Node::MarkAsTerminal() {
  auto* transaction = transaction_.get();
  if (!transaction->atomic()) {
    return absl::OkStatus();
  }
  UniqueWriterLock lock(transaction->mutex_);
  if (transaction->existing_terminal_node_) {
    auto error = GetAtomicError(
        transaction->existing_terminal_node_->Describe(), this->Describe());
    transaction->RequestAbort(error, std::move(lock));
    return error;
  }
  transaction->existing_terminal_node_ = this;
  return absl::OkStatus();
}

absl::Status TransactionState::Node::GetAtomicError(
    std::string_view a_description, std::string_view b_description) {
  return absl::InvalidArgumentError(
      tensorstore::StrCat("Cannot ", a_description, " and ", b_description,
                          " as single atomic transaction"));
}

absl::Status TransactionState::Node::Register() {
  auto* transaction = transaction_.get();
  UniqueWriterLock lock(transaction->mutex_);
  switch (transaction->commit_state_) {
    case kOpen:
    case kOpenAndCommitRequested:
      break;
    case kAbortRequested:
      return GetCancelledError();
    default:
      ABSL_UNREACHABLE();  // COV_NF_LINE
  }
  if (phase_ == kInvalidPhase) {
    // Duplicate `Transaction::phase()` logic rather than call it, since we are
    // already holding the mutex.
    phase_ = transaction->atomic() ? 0 : transaction->phase_;
  }
  assert(phase_ <= (transaction->atomic() ? 0 : transaction->phase_));
  transaction->nodes_.FindOrInsert(
      [associated_data = associated_data_, phase = phase_](Node& node) {
        // Never return 0, because we allow duplicates.
        return NodeTreeCompare(phase, associated_data, node.phase_,
                               node.associated_data_) < 0
                   ? -1
                   : 1;
      },
      [&] { return this; });
  intrusive_ptr_increment(this);
  return absl::OkStatus();
}

Result<OpenTransactionNodePtr<TransactionState::Node>>
TransactionState::GetOrCreateMultiPhaseNode(
    void* associated_data, absl::FunctionRef<Node*()> make_node) {
  UniqueWriterLock lock(mutex_);
  switch (commit_state_) {
    case kOpen:
    case kOpenAndCommitRequested:
      break;
    case kAbortRequested:
      return GetCancelledError();
    default:
      ABSL_UNREACHABLE();  // COV_NF_LINE
  }
  return OpenTransactionNodePtr<Node>(
      nodes_
          .FindOrInsert(
              [associated_data](Node& node) {
                return NodeTreeCompare(0, associated_data, node.phase_,
                                       node.associated_data_);
              },
              [&] {
                Node* node = make_node();
                node->SetTransaction(*this);
                node->phase_ = 0;
                intrusive_ptr_increment(node);
                return node;
              })
          .first);
}

void TransactionState::NoMoreWeakReferences() { delete this; }

void TransactionState::NoMoreOpenReferences() {
  bool abort;
  {
    absl::MutexLock lock(&mutex_);
    if (open_reference_count_.load(std::memory_order_relaxed) != 0) {
      // Another open handle was created concurrently.
      return;
    }
    switch (commit_state_) {
      case kOpenAndCommitRequested:
        commit_state_ = kCommitStarted;
        abort = false;
        break;
      case kAbortRequested:
        commit_state_ = kAborted;
        abort = true;
        break;
      default:
        return;
    }
  }
  if (abort) {
    this->ExecuteAbort();
  } else {
    this->ExecuteCommit();
  }
}

TransactionState::~TransactionState() = default;

void TransactionState::Node::PrepareDone() {
  auto& transaction = *this->transaction();
  if (transaction.waiting_for_prepare_done_.exchange(
          false, std::memory_order_acq_rel)) {
    // Caller of `PrepareForCommit` will continue the commit.
    return;
  }
  transaction.ContinuePrepareForCommit(Tree::Traverse(*this, Tree::kRight),
                                       this->phase());
}

void TransactionState::Node::ReadyForCommit() {
  this->transaction()->DecrementNodesPendingReadyForCommit();
}

void TransactionState::Node::CommitDone(size_t next_phase) {
  if (next_phase) {
    auto& transaction = *this->transaction();
    assert(!transaction.atomic());
    assert(next_phase > this->phase_);
    phase_ = next_phase;
    // Node was previously removed from the `transaction.nodes_` tree by
    // `PrepareDone`.
    transaction.nodes_.FindOrInsert(
        [next_phase, associated_data = associated_data_](Node& node) {
          return NodeTreeCompare(next_phase, associated_data, node.phase_,
                                 node.associated_data_);
        },
        [&] { return this; });
  }
  this->transaction()->DecrementNodesPendingCommit(1);
  if (!next_phase) {
    intrusive_ptr_decrement(this);
  }
}

void TransactionState::Node::Abort() { AbortDone(); }

void TransactionState::Node::AbortDone() {
  transaction()->DecrementNodesPendingAbort(1);
  intrusive_ptr_decrement(this);
}

void TransactionState::Node::SetError(const absl::Status& error) {
  assert(!error.ok());
  auto& promise = transaction()->promise_;
  if (promise.null()) return;
  SetDeferredResult(promise, error);
}

TransactionState::OpenPtr TransactionState::MakeImplicit() {
  return TransactionState::OpenPtr(
      new internal::TransactionState(TransactionMode::isolated,
                                     /*implicit_transaction=*/true),
      adopt_object_ref);
}

TransactionState& GetOrCreateOpenTransaction(OpenTransactionPtr& transaction) {
  if (!transaction) transaction = TransactionState::MakeImplicit();
  return *transaction;
}

absl::Status ChangeTransaction(Transaction& transaction,
                               Transaction new_transaction) {
  if (transaction != no_transaction &&
      (!transaction.future().ready() || !transaction.future().result().ok())) {
    return absl::InvalidArgumentError(
        "Cannot rebind transaction when existing transaction is uncommitted");
  }
  transaction = std::move(new_transaction);
  return absl::OkStatus();
}

}  // namespace internal

void Transaction::Abort() const {
  if (!state_) return;
  state_->RequestAbort();
}

Future<const void> Transaction::CommitAsync() const {
  auto* state = state_.get();
  if (!state) return MakeReadyFuture();
  state->RequestCommit();
  return state->future_;
}

void Transaction::Barrier() const {
  auto* state = state_.get();
  if (!state) return;
  state->Barrier();
}

Transaction::Transaction(TransactionMode mode) {
  if (mode == TransactionMode::no_transaction_mode) return;
  state_.reset(new internal::TransactionState(mode,
                                              /*implicit_transaction=*/false),
               internal::adopt_object_ref);
}

std::ostream& operator<<(std::ostream& os, TransactionMode mode) {
  switch (mode) {
    case TransactionMode::no_transaction_mode:
      return os << "no_transaction_mode";
    case TransactionMode::isolated:
      return os << "isolated";
    case TransactionMode::atomic_isolated:
      return os << "atomic_isolated";
    default:
      return os << "unknown(" << static_cast<int>(mode) << ")";
  }
}

namespace serialization {
bool Serializer<Transaction>::Encode(EncodeSink& sink,
                                     const Transaction& value) {
  if (value != no_transaction) {
    sink.Fail(absl::InvalidArgumentError("Cannot serialize bound transaction"));
    return false;
  }
  return true;
}

bool Serializer<Transaction>::Decode(DecodeSource& sink, Transaction& value) {
  return true;
}
}  // namespace serialization

}  // namespace tensorstore
