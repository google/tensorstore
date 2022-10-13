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

#ifndef TENSORSTORE_TRANSACTION_IMPL_H_
#define TENSORSTORE_TRANSACTION_IMPL_H_

/// \file
///
/// Defines the internal API for implementing transaction-aware operations.
///
/// See `transaction.h` for the public API.

// IWYU pragma: private, include "third_party/tensorstore/transaction.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/intrusive_red_black_tree.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

enum TransactionMode : std::uint8_t;
class Transaction;

namespace internal {

// Uncomment the line below when debugging to verify that the unions are not the
// issue.
//
// #define TENSORSTORE_INTERNAL_TRANSACTION_DEBUG_UNIONS

/// Transactions are represented internally by a `TransactionState` object
/// managed by intrusive reference counting.
///
/// The actual modifications contained in the transaction are represented by
/// `TransactionState::Node` objects, which are also managed by intrusive
/// reference counting.  In the common case, a transaction may contain nodes
/// representing modifications to one or more chunks in a `ChunkCache`, as well
/// as nodes representing the downstream writeback of those chunks to an
/// underlying `kvstore::Driver`.  If this `kvstore::Driver` is itself backed by
/// a cache (e.g. in the case of the neuroglancer_precomputed
/// `Uint64ShardedKeyValueStore`), there may be additional downstream nodes.
/// Implicitly, the dependencies between nodes form a forest, where the roots
/// correspond to "terminal" nodes that are added via
/// `GetOrCreateMultiPhaseNode` and on which `MarkAsTerminal` has been called.
///
/// In an isolated, non-atomic transaction, there may be multiple terminal
/// nodes, each of which may also commit non-atomically.  For example, an
/// isolated, non-atomic transaction may affect multiple independent
/// `kvstore::Driver` objects.  In an isolated atomic transaction, there may
/// only be a single terminal node, and that node must commit atomically.
///
/// Note that `TransactionState` does not explicitly represent the links between
/// nodes; it merely contains a flat list of nodes.  It is up to the individual
/// `Node` implementations to keep track of any necessary dependency
/// relationships, e.g. via `kvstore::ReadModifyWriteSource` and
/// `kvstore::ReadModifyWriteTarget` objects.
///
/// As an example of a transactional operation, writing an in-memory array to a
/// non-sharded neuroglancer_precomputed volume with an explicit transaction
/// proceeds as follows:
///
/// 1. The user supplies a `Transaction` object that points to a non-null
///    `TransactionState`.
///
/// 2. The common `internal::DriverWrite` implementation calls
///    `AcquireOpenTransactionPtrOrError` to obtain an `OpenTransactionPtr` from
///    the `Transaction` that serves to block the transaction from being
///    committed while the write operation is in progress.  This fails if the
///    transaction has already been aborted by a prior error or call to
///    `Transaction::Abort`.  This allows the user to issue one or more
///    asynchronous operations on a transaction, then immediately call
///    `Transaction::Commit` to request the transaction be committed, and then
///    wait on the `Future` returned by `Transaction::Commit` to obtain the
///    combined status of all of the issued operations.
///
/// 3. The `neuroglancer_precomputed` `DataCache` class inherits its write
///    implementation from `ChunkCache`, which determines the relevant chunks,
///    and for each chunk, calls `GetTransactionNode` to obtain a transaction
///    node (a derived class of `TransactionState::Node`) in which to record the
///    writes to the chunk made within the given transaction.  These transaction
///    nodes are held via `OpenTransactionNodePtr` smart pointers to ensure the
///    transaction is not committed while they are in use, even if the initial
///    `OpenTransactionPtr` smart pointer is destroyed.
///
///    - If the chunk was already modified in the transaction by a prior write,
///      the existing transaction node will be re-used.  The existing
///      transaction node is located in the red-black tree of transaction nodes
///      (keyed by their associated `TransactionState` pointers) stored within
///      the `AsyncCache::Entry`.
///
///    - Otherwise, a new transaction node is created.  When the new transaction
///      node is initialized, the `neuroglancer_precomputed` `DataCache` class
///      inherits the behavior of `KvsBackedCache`, which calls
///      `kvstore::Driver::ReadModifyWrite` to create/update a transaction node
///      representing the modifications to the backing `kvstore::Driver`.
///
/// 4. After the copy stage of the write completes, all modifications are
///    recorded in transaction nodes, but no changes have been written back to
///    the backing `kvstore::Driver`.
///
/// 5. If the user calls `Transaction::Abort`, or an error occurs during
///    copying, the transaction is aborted, and the transaction nodes are
///    destroyed.
///
/// 6. Otherwise, if the user calls `Transaction::Commit`, or forces the future
///    returned from `Transaction::future()`, to request that the transaction is
///    committed.  Once all `OpenTransactionPtr` and `OpenTransactionNodePtr`
///    references are released, the commit proceeds by invoking the `Commit`
///    method of the nodes added to the transaction.
///
/// 7. The transaction node associated with the `kvstore::Driver` requests
///    writeback from all of the `ReadModifyWriteSource` objects that have been
///    added from `KvsBackedCache` via `ReadModifyWrite` operations, initially
///    with a `staleness_bound` of `absl::InfinitePast()`.  In turn, the
///    `KvsBackedCache` arranges for the `DataCache` to encode the modified
///    chunks (after reading and decoding the existing values from the
///    `kvstore::Driver`) and passes the updated values (along with the read
///    generations on which they are conditioned, if any) back to the
///    `kvstore::Driver` via the `ReadModifyWriteTarget` interface.  The
///    `kvstore::Driver` performs the writes (which may be conditioned on
///    existing generations).  If the generations don't match (due to concurrent
///    modifications), this step is repeated, with an updated `staleness_bound`
///    to ensure the updated values are read.
///
/// 8. Once all the `kvstore::Driver` writes complete (either successfully or
///    with an error), the transaction becomes ready.
///
/// In the case of a sharded neuroglancer_precomputed volume, everything
/// proceeds in the same way except:
///
/// - In step 3, when the `ChunkCache` creates a new transaction node, the
///   `Uint64ShardedKeyValueStore::ReadModifyWrite` implementation does not
///   create/update a transaction node directly on the backing
///   `kvstore::Driver`.  Instead, it obtains a new or existing transaction node
///   corresponding to the shard that contains the chunk.  (In general, many
///   chunks will be in the same shard.)  The
///   `Uint64ShardedKeyValueStoreWriteCache::TransactionNode`, in turn, also
///   inherits from `KvsBackedCache` which calls
///   `kvstore::Driver::ReadModifyWrite` to create/update a transaction node in
///   the backing `kvstore::Driver`.
///
/// - If all chunks are in the same shard, then the transaction will only affect
///   a single key in the backing `kvstore::Driver`, and the transaction can be
///   committed atomically even if the backing store does not support atomic
///   multi-key transactions.  Otherwise, each shard will be committed
///   separately, but there will still only be one write per shard.
///
/// In the case of a non-transactional write, fine-grained implicit transactions
/// are created automatically.  A non-transactional write to a non-sharded
/// neuroglancer_precomputed volume proceeds as follows:
///
/// 1. As in the transactional case, the `ChunkCache` determines the relevant
///    chunks, and for each chunk obtains an implicit transaction node, also by
///    calling `GetTransactionNode`.
///
///    - If the chunk has an existing implicit transaction node that is still
///      open (i.e. not already being committed), it is re-used.
///
///    - Otherwise, a new implicit transaction node is created, but not yet
///      associated with a transaction.  When the new transaction node is
///      initialized, `KvsBackedCache` calls `kvstore::Driver::ReadModifyWrite`.
///      The default `kvstore::Driver::ReadModifyWrite` implementation simply
///      creates a new unique implicit transaction.  Consequently, for each
///      modified chunk, there will be a separate implicit transaction
///      containing just two nodes, one node associated with the entry in the
///      `ChunkCache` and one node associated with the `kvstore::Driver`.
///
/// 2. If the user requests writeback of the complete write operation (by
///    forcing the `Future` returned from the write operation), commit is
///    requested for all of the implicit transactions.
///
/// 3. Alternatively, commit of implicit transactions may be triggered
///    automatically by memory pressure from the CachePool.
///
/// 4. In either case, each chunk will be written back independently; some may
///    remain in the cache while others are written back.
///
/// A non-transactional write to a sharded neuroglancer_precomputed volume
/// proceeds as follows:
///
/// 1. As in the non-sharded case above, the `ChunkCache` determine the relevant
///    chunks and obtains an implicit transaction node for each chunk.  When a
///    new implicit transaction node is created, `KvsBackedCache` calls
///    `Uint64ShardedKeyValueStore::ReadModifyWrite` to create a downstream
///    node.
///
///    - If there is already an open implicit transaction node associated with
///      the shard that contains the chunk, it is re-used, and the associated
///      implicit transaction is re-used by the upstream transaction node in the
///      `ChunkCache`.
///
///    - Otherwise, a new implicit transaction node is created for the shard,
///      and which in turn calls `kvstore::Driver::ReadModifyWrite` to obtain a
///      new transaction node in the backing `kvstore::Driver` and a new
///      implicit transaction.
///
///    Consequently, for each modified shard, there will be a separate implicit
///    transaction containing:
///
///    a. for each modified chunk in the shard, a `ChunkCache` transaction node;
///
///    b. a `Uint64ShardedKeyValueStoreWriteCache` transaction node
///       corresponding to the shard;
///
///    c. a transaction node corresponding to the writeback of the shard to the
///       backing `kvstore::Driver`.
///
/// 2. If the `CachePool` is sufficiently large that writeback is not triggered
///    automatically while the copying stage of the write operation is in
///    progress (and no manual writeback is requested), all writes to a given
///    shard will be optimally coalesced, just as in the case of an explicit
///    transaction.
///
/// 3. However, if writeback of a shard is triggered while the copying is still
///    in progress, there may be multiple writes to the same shard.
///
class TransactionState {
 public:
  class Node;

 private:
  /// IntrusivePtr traits representing the weak ownership of a
  /// `TransactionState` by a `Node`.
  ///
  /// A `Node` keeps its associated `TransactionState` from being destroyed, but
  /// does not keep it from being aborted.
  struct WeakPtrTraits {
    template <typename>
    using pointer = TransactionState*;

    static void increment(TransactionState* state) noexcept {
      state->weak_reference_count_.fetch_add(1, std::memory_order_relaxed);
    }
    static void decrement(TransactionState* state) noexcept {
      if (state->weak_reference_count_.fetch_sub(
              1, std::memory_order_acq_rel) == 1) {
        state->NoMoreWeakReferences();
      }
    }
  };

  /// Amount added to `commit_reference_count_` for each `CommitPtr` or
  /// `OpenPtr`.
  constexpr static size_t kCommitReferenceIncrement = 2;

  /// Amount added to `commit_reference_count_` until `future_.Force()` or
  /// `promise_.result_needed()` becomes `false`.
  constexpr static size_t kFutureReferenceIncrement = 1;

  /// IntrusivePtr traits used by `CommitPtr`.
  template <std::size_t IncrementAmount = kCommitReferenceIncrement>
  struct CommitPtrTraits {
    template <typename>
    using pointer = TransactionState*;

    static void increment(TransactionState* state) {
      WeakPtrTraits::increment(state);
      state->commit_reference_count_.fetch_add(IncrementAmount,
                                               std::memory_order_relaxed);
    }

    static void decrement(TransactionState* state) {
      const size_t new_count = state->commit_reference_count_.fetch_sub(
                                   IncrementAmount, std::memory_order_acq_rel) -
                               IncrementAmount;
      if (new_count < IncrementAmount) {
        state->NoMoreCommitReferences();
      }
      WeakPtrTraits::decrement(state);
    }
  };

  struct BlockCommitPtrTraits {
    template <typename>
    using pointer = TransactionState*;

    static void increment(TransactionState* state) noexcept {
      state->open_reference_count_.fetch_add(1, std::memory_order_relaxed);
    }

    static void decrement(TransactionState* state) noexcept {
      if (state->open_reference_count_.fetch_sub(
              1, std::memory_order_acq_rel) == 1) {
        state->NoMoreOpenReferences();
      }
    }
  };

  /// IntrusivePtr traits used by `OpenPtr`.
  struct OpenPtrTraits {
    template <typename>
    using pointer = TransactionState*;

    static void increment(TransactionState* state) noexcept {
      CommitPtrTraits<>::increment(state);
      BlockCommitPtrTraits::increment(state);
    }

    static void decrement(TransactionState* state) noexcept {
      BlockCommitPtrTraits::decrement(state);
      // Decrement `commit_reference_count_` last since that may result in
      // `state` being destroyed.
      CommitPtrTraits<>::decrement(state);
    }
  };

  /// IntrusivePtr traits used by `OpenNodePtr`.
  struct OpenNodePtrTraits {
    template <typename T>
    using pointer = T*;

    static void increment(Node* node) {
      intrusive_ptr_increment(node);
      OpenPtrTraits::increment(node->transaction());
    }

    static void decrement(Node* node) {
      OpenPtrTraits::decrement(node->transaction());
      intrusive_ptr_decrement(node);
    }
  };

 public:
  /// Smart pointer that may be used to abort or commit the transaction, but not
  /// directly to perform read or write operations.
  ///
  /// Each such pointer increments `weak_reference_count_` by 1 and
  /// `commit_reference_count_` by `kCommitReferenceIncrement` (which equals 2).
  ///
  /// To perform a read or write operation, a `CommitPtr` may be upgraded to an
  /// `OpenPtr` by calling `AcquireOpenPtr` (fails if the "commit state" is not
  /// `kOpen` or `kOpenAndCommitRequested`).
  using CommitPtr = IntrusivePtr<TransactionState,
                                 CommitPtrTraits<kCommitReferenceIncrement>>;

  /// Smart pointer that keeps the `TransactionState` object alive but does not
  /// prevent the transaction from being aborted or committed.
  ///
  /// Each such pointer increments `weak_reference_count_` by 1.
  using WeakPtr = IntrusivePtr<TransactionState, WeakPtrTraits>;

  /// Smart pointer that may be used to read or write.
  ///
  /// Each such pointer increments `weak_reference_count_` by 1,
  /// `commit_reference_count_` by `kCommitReferenceIncrement`, and
  /// `open_reference_count_` by 1.
  ///
  /// Even if `Commit` is called, the transaction the commit won't start until
  /// there are no remaining `OpenPtr` references.
  using OpenPtr = IntrusivePtr<TransactionState, OpenPtrTraits>;

  constexpr static size_t kInvalidPhase = static_cast<size_t>(-1);

  /// Smart pointer that prevents the transaction from being committed.
  ///
  class Node : public AtomicReferenceCount<Node>,
               public intrusive_red_black_tree::NodeBase<Node> {
   public:
    Node(void* associated_data);

   private:
    /// When a particular phase of the transaction is committed, this is called
    /// sequentially on each node in the phase in order of the
    /// `associated_data()` pointer.
    ///
    /// The derived class implementation must call `PrepareDone` once any
    /// necessary locks have been acquired, and must call `ReadyForCommit` once
    /// the node is ready to be committed.
    ///
    /// Until `PrepareDone` is called, `PrepareForCommit` won't be called on the
    /// next node in the phase.
    ///
    /// The sequential invocation of `PrepareForCommit` in order of
    /// `associated_data()` pointer allows locks to be acquired in a consistent
    /// order to avoid deadlock.
    virtual void PrepareForCommit() = 0;

    /// After `PrepareForCommit` has been called on all nodes in a phase, and
    /// those nodes' implementations have all called `PrepareDone` and
    /// `ReadyForCommit`, `Commit` is called on each node in the phase.  Note
    /// that `Commit` is not called on any node in the phase until all nodes are
    /// ready.
    ///
    /// The derived class implementation must call `CommitDone` once the commit
    /// completes, either successfully or with an error.  If an error occurs,
    /// `SetError` should be called prior to calling `CommitDone`.
    virtual void Commit() = 0;

   public:
    /// Must be called (synchronously or asynchronously) by the derived class
    /// implementation of `PrepareForCommit` to indicate that any necessary
    /// locks have been acquired and it is safe to proceed with calling
    /// `PrepareForCommit` on the next node in the phase.
    void PrepareDone();

    /// Must be called (synchronously or asynchronously) by the derived class
    /// implementation of `PrepareForCommit` to indicate that `Commit` may be
    /// called.
    void ReadyForCommit();

    /// Must be called after the node finishes (either successfully or with an
    /// error) executing.
    ///
    /// If this is a multi-phase node and there is a subsequent phase, a
    /// non-zero `next_phase` must be specified, which will cause `Commit` to be
    /// invoked again when that phase is committed.
    void CommitDone(size_t next_phase = 0);

   private:
    /// Called when the transaction is aborted (once all `OpenTransactionPtr`
    /// and `OpenTransactionNodePtr` references have been released).  The
    /// derived node class must call `AbortDone` after the abort completes.
    ///
    /// The default implementation just calls `AbortDone` immediately.
    virtual void Abort();

   public:
    /// Must be called after `Abort` performs any necessary work to abort the
    /// transaction.
    void AbortDone();

    /// Adjusts the size in bytes accounted to the transaction by
    /// `new_minus_old`.
    void UpdateSizeInBytes(size_t new_minus_old) {
      transaction_->total_bytes_.fetch_add(new_minus_old,
                                           std::memory_order_relaxed);
    }

    /// Returns a string description of the node,
    /// e.g. `"write to local file xyz"`.
    virtual std::string Describe();

    /// Sets an error status on the entire transaction.
    ///
    /// \dchecks `!error.ok()`
    void SetError(const absl::Status& error);

    /// Returns the associated transaction.
    TransactionState* transaction() const { return transaction_.get(); }

    /// Prepares to register this node with `transaction`.  Must be called prior
    /// to `Register`.
    ///
    /// Must not be called more than once.
    ///
    /// This causes the node to acquire a weak reference to `transaction`.
    ///
    /// This two-step registration procedure allows `AsyncCache` to set
    /// `this->transaction()` in order to store the node in a map keyed by the
    /// `TransactionState` address before the node is fully initialized, and
    /// later call `Register` after fully initializing the node.
    void SetTransaction(TransactionState& transaction);

    /// Sets the phase.
    void SetPhase(size_t phase);

    /// Registers this node with the transaction set previously by a call to
    /// `SetTransaction`.
    ///
    /// If the transaction has already been committed or aborted, this returns
    /// an error and the node remains unregistered.
    ///
    /// On successful return, `transaction()` acquires a reference to this node.
    ///
    /// Must be called at most once.
    absl::Status Register();

    virtual ~Node();

    size_t phase() const { return phase_; }

    /// If the transaction requires atomic commit and this node is a terminal
    /// node that directly performs I/O, this function should be called.  If
    /// this returns an error, then there is already a terminal node added to
    /// the transaction.  The transaction will be aborted in this case.
    absl::Status MarkAsTerminal();

    static absl::Status GetAtomicError(std::string_view a_description,
                                       std::string_view b_description);

   protected:
    void* associated_data() const { return associated_data_; }

   private:
    friend class TransactionState;

    /// The transaction with which the node is associated.
    TransactionState::WeakPtr transaction_;

    /// The phase of the transaction with which this node is associated.  If
    /// `transaction_->atomic()`, will be set to 0.
    size_t phase_ = kInvalidPhase;

    /// The associated data of this node.  In the case of
    /// `AsyncCache::TransactionNode`, this is a pointer to the
    /// `AsyncCache::Entry`.
    void* associated_data_;
  };

  /// Constructs a new transaction state.
  explicit TransactionState(TransactionMode mode, bool implicit_transaction);

  /// Returns the future associated with this transaction.
  ///
  /// This is only guaranteed to be valid if the caller holds a commit reference
  /// to the transaction.
  Future<const void> const& future() const { return future_; }

  static TransactionState* get(const Transaction& t);

  /// Converts an open transaction pointer to a `Transaction` object.
  static Transaction ToTransaction(OpenPtr transaction);

  static Transaction ToTransaction(CommitPtr transaction);
  static CommitPtr ToCommitPtr(Transaction t);

  template <typename T>
  using WeakNodePtrT = IntrusivePtr<T>;

  template <typename T>
  using OpenNodePtrT = IntrusivePtr<T, OpenNodePtrTraits>;

  /// Creates a new implicit transaction.
  static OpenPtr MakeImplicit();

  /// Attempts to acquire an open transaction pointer.  The caller must own at
  /// least a commit pointer.  Returns `nullptr` if the transaction has already
  /// been aborted or commit has already started.
  OpenPtr AcquireOpenPtr();

  /// Same as `AcquireOpenPtr`, but returns an
  /// `absl::StatusCode::kInvalidArgument` error rather than a `nullptr` in the
  /// case of failure.
  Result<OpenPtr> AcquireOpenPtrOrError();

  /// Attempts to acquire another `OpenPtr`.  Returns `nullptr` if one cannot be
  /// acquired.
  ///
  /// Unlike `AcquireOpenPtr()`, the caller need only hold a weak reference, not
  /// necessarily a commit reference.
  ///
  /// \dchecks `implicit_transaction()`
  OpenPtr AcquireImplicitOpenPtr();

  /// Returns `true` if this is an implicit transaction.
  bool implicit_transaction() const { return implicit_transaction_; }

  /// Returns the sum of the memory occupied by the transaction nodes.  In the
  /// event of concurrent modifications, this should be treated as an
  /// approximation.
  std::size_t total_bytes() const {
    return total_bytes_.load(std::memory_order_relaxed);
  }

  /// Requests that the transaction be committed.  Has no effect if commit or
  /// abort has already been requested.
  void RequestCommit();

  /// Requests that the transaction be aborted.  Has no effect if commit or
  /// abort has already been requested.
  void RequestAbort();

  /// Requests that the transaction be aborted, with the specified error.  Has
  /// no effect if commit or abort has already been requested.
  void RequestAbort(const absl::Status& error);

  /// Ensure all previously registered nodes are committed before any
  /// subsequently registered nodes.
  ///
  /// This has no effect in the case of an atomic transaction.  Otherwise, it
  /// increments `phase()`.
  void Barrier();

  /// Returns the current phase number.
  ///
  /// The caller must hold an open reference.
  ///
  /// If `atomic()`, this always returns 0.
  size_t phase();

  /// Returns `true` if the transaction has been aborted.
  bool aborted() {
    absl::MutexLock lock(&mutex_);
    return commit_state_ >= kAbortRequested;
  }

  /// Bit in TransactionMode that indicates whether the transaction is atomic.
  ///
  /// Equal to `TransactionMode::atomic_isolated - TransactionMode::isolated`
  /// (checked in `transaction.cc`, since TransactionMode is an incomplete type
  /// here).
  static constexpr uint8_t kAtomic = 2;

  static constexpr bool IsAtomic(TransactionMode mode) {
    return static_cast<bool>(mode & kAtomic);
  }

  /// Returns `true` if this is an atomic transaction.
  bool atomic() { return IsAtomic(mode_); }

  /// Returns `true` if commit has started.
  bool commit_started() {
    absl::MutexLock lock(&mutex_);
    return commit_state_ == kCommitStarted;
  }

  TransactionMode mode() const { return mode_; }

  absl::Time commit_start_time() const { return commit_start_time_; }

  /// Acquires a lock that blocks commit from starting.
  ///
  /// This increments `weak_reference_count_` and `open_reference_count_`.  This
  /// is used by `AsyncStorageBackedCache` to defer committing an implicit
  /// transaction while another implicit transaction is already being committed.
  void AcquireCommitBlock() {
    WeakPtrTraits::increment(this);
    BlockCommitPtrTraits::increment(this);
  }

  /// Releases the lock acquired by `AcquireCommitBlock`.
  void ReleaseCommitBlock() {
    BlockCommitPtrTraits::decrement(this);
    WeakPtrTraits::decrement(this);
  }

  /// Returns the existing node for `associated_data`.  If there is no existing
  /// node, creates a new one by calling `make_node`.
  Result<OpenNodePtrT<Node>> GetOrCreateMultiPhaseNode(
      void* associated_data, absl::FunctionRef<Node*()> make_node);

 private:
  friend class tensorstore::Transaction;

  /// Begins the asynchronous commit process.
  ///
  /// This is invoked once `RequestCommit` has been called and all
  /// `OpenTransactionPtr`, `OpenNodePtr`, and "commit blocks" have been
  /// released.
  ///
  /// \pre `commit_state_ == kCommitStarted`
  void ExecuteCommit();

  /// Begins the commit of the next not-yet-committed phase.
  ///
  /// This is called initially by `ExecuteCommit` and is then called again after
  /// each phase commits successfully if there are still remaining phases.
  ///
  /// This invokes `PrepareForCommit` on each node in the phase, sequentially.
  void ExecuteCommitPhase();

  /// Asynchronously continues the sequential invocation of `PrepareForCommit`
  /// on each node in the phase started by `ExecuteCommitPhase`.
  ///
  /// If `node` is `nullptr` or `node->phase() != current_phase`, then all nodes
  /// in the phase have already been handled, and the commit process will
  /// continue only once all nodes in the phase have called `ReadyForCommit`.
  ///
  /// \param node The next node for which `PrepareForCommit` has not yet been
  ///     called.
  /// \param current_phase The current phase being committed.
  void ContinuePrepareForCommit(Node* node, size_t current_phase);

  /// Called when `ContinuePrepareForCommit` reaches the end of the phase, and
  /// also by `Node::ReadyForCommit`.  Decrements the `nodes_pending_commit_`
  /// counter.
  ///
  /// If the counter reaches 0, proceeds by calling `Commit` on every node in
  /// the phase.
  void DecrementNodesPendingReadyForCommit();

  /// Called when `DecrementNodesPendingReadyForCommit` finishes calling
  /// `Commit` on every node in the phase, and also by `Node::CommitDone`.
  /// Decrements the `nodes_pending_commit_` counter.
  ///
  /// If the counter reaches 0, handles the completion of the phase commit.
  void DecrementNodesPendingCommit(size_t count);

  /// Requests that the transaction be aborted with the specified error.
  ///
  /// \param error The error to abort with, if one has not already been set.
  /// \param lock Must be a lock on `mutex_`.
  void RequestAbort(const absl::Status& error,
                    UniqueWriterLock<absl::Mutex> lock);

  /// Begins the asynchronous abort process.
  ///
  /// This is invoked once `RequestAbort` has been called and all
  /// `OpenTransactionPtr` and `OpenNodePtr` references have been released.  It
  /// is also invoked directly by `DecrementNodesPendingCommit` to abort any
  /// remaining phases if an error occurred while committing the last phase.
  void ExecuteAbort();

  /// Called when `ExecuteAbort` finishes calling `Abort` on every node, and
  /// also by `Node::AbortDone`.
  ///
  /// Decrements the `nodes_pending_abort_` counter.
  ///
  /// If the counter reaches 0, handles the completion of the abort.
  void DecrementNodesPendingAbort(size_t count);

  /// Called when no more references of any kind remain.  Destroys the
  /// `TransactionState`.
  void NoMoreWeakReferences();

  /// Called when no more `OpenPtr` references remain.  This handles calling
  /// `ExecuteAbort` or `ExecuteCommit` as appropriate.
  void NoMoreOpenReferences();

  /// Called when there are no remaining `CommitPtr` references, or no remaining
  /// references via the `future_`.  If a commit has not already been requested,
  /// this aborts the transaction.
  void NoMoreCommitReferences();

  static int NodeTreeCompare(size_t phase_a, void* data_a, size_t phase_b,
                             void* data_b) {
    if (phase_a < phase_b) return -1;
    if (phase_a > phase_b) return 1;
    return intrusive_red_black_tree::ThreeWayFromLessThan<>()(data_a, data_b);
  }

  ~TransactionState();

  absl::Mutex mutex_;
  TransactionMode mode_;

  /// Red-black tree of nodes in the transaction.
  ///
  /// Nodes are ordered by `phase` and then `associated_data_`.  This provides a
  /// canonical order in which locks are acquired, in order to avoid deadlock.
  /// For example, `AsyncCache` only permits writeback of a single transaction
  /// at a time for a given `AsyncCache::Entry`, and using inconsistent orders
  /// could result in deadlock.
  using Tree = intrusive_red_black_tree::Tree<Node, Node>;
  Tree nodes_;

#ifndef TENSORSTORE_INTERNAL_TRANSACTION_DEBUG_UNIONS
  union {
#endif
    /// The current phase number.  Only valid when `commit_started() == false`
    /// and `(mode() & atomic_isolated) == isolated`.  A non-atomic transaction
    /// may contain multiple phases.  Each `Node` is associated with a
    /// particular phase.  When the transaction is being committed, a node at a
    /// given phase is not committed until all prior phase nodes are
    /// successfully committed.
    std::size_t phase_;

    /// Indicates whether there is already a terminal node in an atomic
    /// transaction.  Only valid when `commit_state_ <= kCommitStarted` and
    /// `(mode() & atomic_isolated) == atomic_isolated`.
    Node* existing_terminal_node_;

    /// Used when calling `Node::PrepareForCommit` to prevent unbounded
    /// recursion in the case that `Node::PrepareForCommit` synchronously
    /// invokes `Node::PrepareDone`.  Only valid when
    /// `commit_started() == true`.
    ///
    /// Set to `true` before invoking `Node::PrepareForCommit`, and set to
    /// `false` by the caller of `Node::PrepareForCommit` if
    /// `Node::PrepareForCommit` returns without `Node::PrepareDone` having been
    /// called; in this case, `PrepareDone` is itself responsible for continuing
    /// the commit process.  Otherwise, `PrepareDone` sets this to `false` and
    /// does nothing, and the caller of `PrepareForCommit` observes that this
    /// has already been set to `false`, which indicates that it is responsible
    /// for continuing the commit process.
    std::atomic<bool> waiting_for_prepare_done_;

    /// Number of nodes still being aborted.  Once this reaches 0, `promise_` is
    /// reset.  Only valid when `commit_state_ == kAborted`.
    std::atomic<size_t> nodes_pending_abort_;
#ifndef TENSORSTORE_INTERNAL_TRANSACTION_DEBUG_UNIONS
  };
#endif

  /// Time at which commit started.  This may be used by `kvstore::Driver` to
  /// verify that a given generation is up to date as of the start of the
  /// commit.
  absl::Time commit_start_time_;

  /// Registration of "force" callback on `promise_` that commits the
  /// transaction.  The callback holds a commit reference.  This is unregistered
  /// when all other commit references have been released, in order to break the
  /// reference cycle.
  FutureCallbackRegistration promise_callback_;

  /// Retained until all write handles are released.
  Promise<void> promise_;

  /// Retained until all commit handles are released.
  Future<const void> future_;

  /// Equal to `kCommitReferenceIncrement` times the number of `CommitPtr` and
  /// `OpenPtr` references, plus `kFutureReferenceIncrement` if the `future` is
  /// still referenced.  When all commit handles are released, the `future`
  /// reference owned by the `TransactionState` itself is also released to break
  /// the reference cycle.
  std::atomic<size_t> commit_reference_count_;

#ifndef TENSORSTORE_INTERNAL_TRANSACTION_DEBUG_UNIONS
  union {
#endif
    /// Equal to the number of write handles.  Commit is deferred until this
    /// count goes to 0.  Only valid when `commit_started() == false`.
    std::atomic<size_t> open_reference_count_;

    /// Number of nodes in the current phase still being committed.  Once this
    /// reaches 0, the next phase (if any) is committed, or if any error has
    /// occurred, all remaining nodes are aborted.  Only valid when
    /// `commit_started() == true`.
    std::atomic<size_t> nodes_pending_commit_;
#ifndef TENSORSTORE_INTERNAL_TRANSACTION_DEBUG_UNIONS
  };
#endif

  /// Number of weak references.  Each associated transaction node holds a weak
  /// reference.  There is also a weak reference for each `CommitPtr`, each
  /// `OpenPtr`, and each "commit block" lock.  The `TransactionState` is
  /// destroyed once `weak_reference_count_` becomes 0.
  std::atomic<size_t> weak_reference_count_;

  /// Estimated bytes of memory occupied by transaction.
  std::atomic<size_t> total_bytes_;

  /// Commit state values, indicating the current state of the transaction.
  enum CommitState {
    /// Additional reads or writes may be performed using the transaction.  No
    /// request has been made to commit or abort the transaction.
    kOpen = 0,

    /// Commit has been requested, but will be deferred until all `OpenPtr`,
    /// `OpenNodePtr`, and "commit block" references to the transaction are
    /// released.  Additional reads or writes may be performed.  The transaction
    /// may still be aborted, in which case the state changes to
    /// `kAbortRequested`.
    kOpenAndCommitRequested = 1,

    /// Commit has started due to a call to `ExecuteCommit`.  No additional
    /// reads or writes may be performed.  Once the state enters
    /// `kCommitStarted`, it does not transition out.  The commit completion and
    /// any errors are indicated by the `future`.
    kCommitStarted = 2,

    /// Abort has been requested, but will be deferred until all `OpenPtr` and
    /// `OpenNodePtr` references to the transaction are released (this avoids
    /// the need to separately prevent nodes from being destroyed while
    /// operations are in progress).
    kAbortRequested = 3,

    /// The transaction has been aborted.  The `TransactionState` itself will be
    /// destroyed once the last reference is released. Once the state enters
    /// `kAborted`, it does not transition out.
    kAborted = 4,
  };

  CommitState commit_state_;

  /// Set to `true` if this is an implicit transaction.
  bool implicit_transaction_;
};

/// Smart pointer that prevents a transaction from being committed and keeps it
/// open for additional reads and writes.
using OpenTransactionPtr = TransactionState::OpenPtr;

/// Smart pointer that holds a weak reference to a transaction node as well as
/// an `OpenTransactionPtr` to its associated transaction.  This should be held
/// while performing a read or write of the node prior to commit.
template <typename Node>
using OpenTransactionNodePtr = TransactionState::OpenNodePtrT<Node>;

/// Smart pointer that holds a weak reference to a transaction node.
template <typename Node>
using WeakTransactionNodePtr = TransactionState::WeakNodePtrT<Node>;

/// Converts an `OpenTransactionNodePtr` to a `WeakTransactionNodePtr`.
template <typename Node>
WeakTransactionNodePtr<Node> ToWeakTransactionNodePtr(
    OpenTransactionNodePtr<Node> node) {
  if (node) {
    OpenTransactionPtr::traits_type::decrement(node->transaction());
  }
  return WeakTransactionNodePtr<Node>(node.release(),
                                      internal::adopt_object_ref);
}

/// Returns either the existing transaction or a new implicit transaction.
///
/// If `transaction` is null, sets `transaction` to a new implicit transaction.
/// Otherwise, leaves `transaction` unchanged.
///
/// \param transaction[in,out] The open transaction reference, may be null.
/// \returns `*transaction`
TransactionState& GetOrCreateOpenTransaction(OpenTransactionPtr& transaction);

/// Attempts to set `transaction = new_transaction`, but fails if `transaction`
/// is non-null and uncommitted.
absl::Status ChangeTransaction(Transaction& transaction,
                               Transaction new_transaction);

}  // namespace internal

}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal::TransactionState)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(tensorstore::Transaction)

#endif  // TENSORSTORE_TRANSACTION_IMPL_H_
