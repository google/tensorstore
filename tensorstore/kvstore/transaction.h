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

#ifndef TENSORSTORE_KVSTORE_TRANSACTION_H_
#define TENSORSTORE_KVSTORE_TRANSACTION_H_

/// \file
///
/// Facilities for implementing the transactional KeyValueStore operations:
///
/// - `KeyValueStore::ReadModifyWrite`
///
/// - `KeyValueStore::TransactionalDeleteRange`
///
/// There are three types of KeyValueStore drivers:
///
/// - Terminal: The `KeyValueStore` driver directly accesses the underlying
///   storage without going through any lower-level `KeyValueStore`.  There are
///   two sub-types of terminal `KeyValueStore`:
///
///   - Non-atomic: implements `Read`, `Write` (and possibly) `DeleteRange` but
///     not any transactional operations.  The default
///     `KeyValueStore::ReadModifyWrite` and
///     `KeyValueStore::TransactionalDeleteRange` implementations using
///     `NonAtomicTransactionNode` are used, which are based on the driver's
///     non-transactional `Read`, `Write`, and `DeleteRange` methods.  When used
///     with an atomic-mode transaction, the transaction will fail if more than
///     a single key is affected.  The "file" and "gcs" drivers are in this
///     category.
///
///   - Atomic: The `KeyValueStore` implements its own `TransactionNode` class
///     that inherits from `AtomicTransactionNode` and defines `ReadModifyWrite`
///     based on `AddReadModifyWrite` and `TransactionalDeleteRange` based on
///     `AddDeleteRange`.  The driver implements non-transactional `Read` and
///     may either directly implement the non-transactional `Write`, or may use
///     `WriteViaTransaction` to implement it in terms of the transactional
///     `ReadModifyWrite`.  The "memory" driver is in this category.
///
/// - Non-terminal: The `KeyValueStore` driver is merely an adapter over an
///   underlying `KeyValueStore`; the connection to the underlying
///   `KeyValueStore` is provided by `KvsBackedCache`.  The
///   uint64_sharded_key_value_store driver is in this category.
///
/// For all three types of KeyValueStore driver, the `MultiPhaseMutation` class
/// tracks the uncommitted transactional operations in order to provide a
/// virtual view of the result after the transaction is committed:
///
/// In the simple case, the only transactional operations are `ReadModifyWrite`
/// operations to independent keys, all in a single phase (i.e. without any
/// interleaved `Transaction::Barrier` calls), and there is not much to track.
///
/// In general, though, there may be multiple `ReadModifyWrite` operations for
/// the same key, possibly with intervening `Transaction::Barrier` calls, and
/// there may be `TransctionalDeleteRange` operations before or after
/// `ReadModifyWrite` operations that affect some of the same keys.  (Note that
/// `Transaction::Barrier` calls are only relevant to terminal `KeyValueStore`
/// drivers, which use a multi-phase TransactionNode.  A TransactionNode for a
/// non-terminal driver only needs to track operations for a single phase; if
/// transactional operations are performed on a non-terminal KeyValueStore over
/// multiple phases, a separate TransactionNode object is used for each phase.)
///
/// Transactional operations are tracked as follows:
///
/// `MultiPhaseMutation` contains a circular linked list of
/// `SinglePhaseMutation` objects corresponding to each phase in which
/// transactional operations are performed, in order of phase number.
///
/// `SinglePhaseMutation` contains an interval tree (implemented as a red-black
/// tree) containing `ReadModifyWriteEntry` (corresponding to a single key) or
/// `DeleteRangeEntry` (corresponding to a range of keys) objects.  Before
/// commit starts, the entries in the interval tree of the last
/// `SinglePhaseMutation` represent the combined effect of all transactional
/// operations.  This makes it possible for subsequent `ReadModifyWrite`
/// operations to efficiently take into account the current modified state.

#include "tensorstore/internal/intrusive_red_black_tree.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace internal_kvs {

class ReadModifyWriteEntry;
class DeleteRangeEntry;
class MutationEntry;
class MultiPhaseMutation;
class SinglePhaseMutation;

/// The type of entry within the interval tree contained in a
/// `SinglePhaseMutation` object.
enum MutationEntryType {
  /// ReadModifyWriteEntry object corresponding to a ReadModifyWrite operation.
  kReadModifyWrite = 0,
  /// DeleteRangeEntry object corresponding to a TransactionalDeleteRange
  /// operation.
  kDeleteRange = 1,
  /// Same as `kDeleteRange`, except this indicates a placeholder entry created
  /// when a `DeleteRangeEntry` from a prior phase is split by a subsequent
  /// ReadModifyWrite or DeleteRange operation in a later phase.  The phase
  /// indicated by the entry's `single_phase_mutation()` must already contain a
  /// `kDeleteRange` entry that contains this entry's range.  Entries of this
  /// type serve solely to provide a view of the modified state; they are
  /// discarded during commit.
  kDeleteRangePlaceholder = 2,
};

using MutationEntryTree =
    internal::intrusive_red_black_tree::Tree<MutationEntry>;
using ReadModifyWriteEntryTree =
    internal::intrusive_red_black_tree::Tree<ReadModifyWriteEntry>;

/// Base class for interval tree entry types `ReadModifyWriteEntry` and
/// `DeleteRangeEntry`.
class MutationEntry : public MutationEntryTree::NodeBase {
 public:
  /// The single affected key for `ReadModifyWriteEntry`, or the `inclusive_min`
  /// bound of the range to be deleted for `DeleteRangeEntry`.
  std::string key_;

  /// Pointer to the `SinglePhaseMutation` object indicating the phase with
  /// which this entry is associated.  Prior to commit starting, all entries are
  /// added to the interval tree of the last phase, and remain there until they
  /// are superseded by another entry (in order to ensure the last phase
  /// provides a complete view of the modified state).  Therefore, the last
  /// phase interval tree may contain some entries for prior phases.  When
  /// commit starts, entries are moved to the interval tree of their appropriate
  /// phase using a single pass through the last phase's interval tree.
  ///
  /// The tag bits specify the `MutationEntryType`.
  internal::TaggedPtr<SinglePhaseMutation, 2> single_phase_mutation_;

  SinglePhaseMutation& single_phase_mutation() const {
    return *single_phase_mutation_;
  }

  MutationEntryType entry_type() const {
    return static_cast<MutationEntryType>(single_phase_mutation_.tag());
  }

  /// Returns the `MultiPhaseMutation` to which this entry is associated.
  MultiPhaseMutation& multi_phase() const;

  /// Returns the mutex of the `MultiPhaseMutation` object.
  absl::Mutex& mutex() const;

 protected:
  ~MutationEntry() = default;
};

/// Atomic counter with error indicator, used to track number of outstanding
/// entries.
class EntryCounter {
 public:
  /// Sets the error indicator (cannot be unset).
  void SetError() { value_.fetch_or(1, std::memory_order_relaxed); }

  /// Returns `true` if `SetError` has been called.
  bool HasError() const { return value_.load(std::memory_order_relaxed) & 1; }

  /// Increments the count by `amount`.
  void IncrementCount(size_t amount = 1) {
    value_.fetch_add(2 * amount, std::memory_order_relaxed);
  }

  /// Decrements the count by `amount`, returns `true` if the count becomes
  /// zero.  Wrapping is allowed.
  bool DecrementCount(size_t amount = 1) {
    return value_.fetch_sub(2 * amount, std::memory_order_acq_rel) -
               2 * amount <=
           1;
  }

  /// Returns `true` if the count is 0.
  bool IsDone() const { return value_ <= 1; }

 private:
  std::atomic<size_t> value_{0};
};

/// MutationEntry representing a `TransactionalDeleteRange` operation.
class DeleteRangeEntry final : public MutationEntry {
 public:
  /// The `exclusive_max` bound for the range to be deleted.
  std::string exclusive_max_;

  /// Tree of `ReadModifyWriteEntry` objects representing prior
  /// `ReadModifyWrite` operations in the same phase that were
  /// superseded/overwritten by this `TransacitonalDeleteRange` operation.
  ///
  /// These entries don't contribute to the writeback result, but they are still
  /// able to perform validation on the prior state, and may abort the
  /// transaction if the validation fails.
  ReadModifyWriteEntryTree superseded_;

  /// Counter used during writeback to track the number of entries in
  /// `superseded_` not yet completed.
  EntryCounter remaining_entries_;
};

/// MutationEntry representing a `ReadModifyWrite` operation.
class ReadModifyWriteEntry : public MutationEntry,
                             public KeyValueStore::ReadModifyWriteTarget {
 public:
  /// The `ReadModifyWriteSource` (normally associated with a
  /// `KvsBackedCache::TransactionNode`) provided to the `ReadModifyWrite`
  /// operation.
  KeyValueStore::ReadModifyWriteSource* source_;

  /// Pointer to prior `ReadModifyWrite` operation (possibly in a prior phase)
  /// that was superseded by this operation.
  ReadModifyWriteEntry* prev_ = nullptr;

  /// If `superseded_` is `false`, pointer to next `ReadModifyWrite` operation
  /// (possibly in a subsequent phase) that supersedes this operation.  If this
  /// entry is directly contained in a `SinglePhaseMutation::entries_` tree,
  /// `next_` is `nullptr`.  Otherwise, `next_` must not be `nullptr`.
  ///
  /// If `superseded_` is `true`, will be `nullptr` prior to writeback/commit
  /// starting; after writeback/commit starts, will be set to a pointer to the
  /// `DeleteRangeEntry` that directly contains this entry in its `superseded_`
  /// tree.  While transactional operations may still be added, we don't set
  /// `next_` to point to the `DeleteRangeEntry` that contains this entry,
  /// because (1) we don't need it; and (2) the `next_` pointers would have to
  /// be updated when a split happens, and that can't be done in amortized O(log
  /// N) time.
  MutationEntry* next_ = nullptr;

  /// Returns a pointer to the next `ReadModifyWriteEntry` that supersedes this
  /// entry, or `nullptr` if there is none.  This returns `nullptr` if, and only
  /// if, this entry is directly contained in either a
  /// `SinglePhaseMutation::entries_` tree or a `DeleteRangeEntry::supereseded_`
  /// tree.
  ReadModifyWriteEntry* next_read_modify_write() const {
    if (!next_ || (flags_ & kDeleted)) return nullptr;
    return static_cast<ReadModifyWriteEntry*>(next_);
  }

  // Bit vector of flags, see flag definitions below.
  using Flags = std::uint8_t;

  Flags flags_ = 0;

  /// Indicates whether `ReadModifyWriteSource::Writeback` was called (and has
  /// completed) on `source_`.
  constexpr static Flags kWritebackProvided = 1;

  /// Indicates whether a prior call to `RReadMOdifyWriteSource::Writeback` on
  /// this entry's `_source_` or that of a direct or indirect predecessor (via
  /// `prev_`) has returned an unconditional generation, which indicates that
  /// the writeback result is not conditional on the existing read state.  In
  /// this case, if the actual writeback result is not needed (because a
  /// successor entry is also unconditional), `ReadModifyWriteSource::Writeback`
  /// won't be called again merely to validate the existing read state.
  constexpr static Flags kTransitivelyUnconditional = 2;

  /// Only meaningful if `kWritebackProvided` is also set.  Indicates that the
  /// most recent writeback actually modified the input.
  constexpr static Flags kDirty = 4;

  /// Indicates that this entry supersedes a prior `TransactionalDeleteRange`
  /// operation that affected the same key, and therefore the input is always a
  /// `ReadResult` with a `state` of `kMissing`.  If this flag is specified,
  /// note that `prev_`, if non-null, is not a special entry corresponding to
  /// this deletion; rather, it is a regular `ReadModifyWriteEntry` for the same
  /// key in the same phase that was superseded by the
  /// `TransactionalDeleteRange` operation.
  constexpr static Flags kPrevDeleted = 8;

  /// WritebackError has already been called on this entry sequence.
  constexpr static Flags kError = 16;

  /// If set, this is a member of the supereseded_ list of a `DeleteRangeEntry`.
  /// After commit starts, `next_` will be updated to point to that
  /// DeleteRangeEntry, which also serves to signal that any necessary
  /// "writeback" can begin.
  ///
  /// We don't set `next_` prior to commit because we may have to modify it
  /// repeatedly due to splits and merges.
  constexpr static Flags kDeleted = 32;

  // Implementation of `KeyValueStore::ReadModifyWriteTarget` interface:

  /// Satisfies a read request by requesting a writeback of `prev_`, or by
  /// calling `MultiPhaseMutation::Read` if there is no previous operation.
  void KvsRead(TransactionalReadOptions options,
               ReadReceiver receiver) override;

  /// Returns `false` if `prev_` is not null, or if
  /// `MultiPhaseMutation::MultiPhaseReadsCommitted` returns `false`.
  /// Otherwise, returns `true`.
  bool KvsReadsCommitted() override;

  virtual ~ReadModifyWriteEntry() = default;
};

/// Represents the modifications made during a single phase.
class SinglePhaseMutation {
 public:
  SinglePhaseMutation() = default;
  SinglePhaseMutation(const SinglePhaseMutation&) = delete;

  /// The `MultiPhaseMutation` object that contains this `SinglePhaseMutation`
  /// object.
  MultiPhaseMutation* multi_phase_;

  /// The phase number to which this object is associated.  The special value of
  /// `std::numeric_limits<size_t>::max()` is used for the initial
  /// `SinglePhaseMutation` of a `MultiPhaseMutation` when it is first
  /// initialized before any operations are added.
  size_t phase_number_;

  /// The interval tree representing the operations.
  MutationEntryTree entries_;

  /// Pointer to next phase in circular linked list contained in `multi_phase_`.
  SinglePhaseMutation* next_;

  /// Pointer to previous phase in circular linked list contained in
  /// `multi_phase_`.
  SinglePhaseMutation* prev_;

  /// Counter used during writeback to track the number of entries in `entries_`
  /// not yet completed.
  EntryCounter remaining_entries_;
};

/// Destroys all entries backward-reachable from the interval tree contained in
/// `single_phase_mutation`, and removes any linked list references to the
/// destroyed nodes.
///
/// Entries in a later phase that supersede a destroyed node will end up with a
/// `prev_` pointer of `nullptr`.
///
/// The reachable entries include all entries directly contained in the interval
/// tree, as well as:
///
///   - in the case of a `ReadModifyWriteEntry`, any entry reachable by
///     following `prev_` pointers.
///
///   - in the case of a `DeleteRangeEntry` contained in the interval tree, all
///     `ReadModifyWriteEntry` object directly contained in its `superseded_`
///     tree, as well as any prior entries reachable by following `prev_`
///     pointers from any of those entries.
///
/// This should be called to abort, or after commit completes.
void DestroyPhaseEntries(SinglePhaseMutation& single_phase_mutation);

/// Notifies all `ReadModifyWriteEntry` objects backward-reachable from `entry`
/// of a writeback error.
///
/// Has no effect on entries that have already been notified.
void WritebackError(MutationEntry& entry);
void WritebackError(ReadModifyWriteEntry& entry);
void WritebackError(DeleteRangeEntry& entry);

/// Calls `WritebackError` on all entries in the interval tree contained in
/// `single_phase_mutation`.
void WritebackError(SinglePhaseMutation& single_phase_mutation);

void WritebackSuccess(DeleteRangeEntry& entry);
void WritebackSuccess(ReadModifyWriteEntry& entry,
                      TimestampedStorageGeneration new_stamp);
void InvalidateReadState(SinglePhaseMutation& single_phase_mutation);

class MultiPhaseMutation {
 public:
  MultiPhaseMutation();

  SinglePhaseMutation phases_;

  virtual internal::TransactionState::Node& GetTransactionNode() = 0;
  virtual std::string DescribeKey(std::string_view key) = 0;

  /// Allocates a new `ReadModifyWriteEntry`.
  ///
  /// By default returns `new ReadModifyWriteEntry`, but if a derived
  /// `MultiPhaseMutation` type uses a derived `ReadModifyWriteEntry` type, it
  /// must override this method to return a new derived entry object.
  virtual ReadModifyWriteEntry* AllocateReadModifyWriteEntry();

  /// Destroys and frees an entry returned by `AllocateReadModifyWriteEntry`.
  ///
  /// By default calls `delete entry`, but derived classes must override this if
  /// `AllocateReadModifyWriteEntry` is overridden.
  virtual void FreeReadModifyWriteEntry(ReadModifyWriteEntry* entry);

  /// Reads from the underlying storage.  This is called when a
  /// `ReadModifyWriteSource` requests a read that cannot be satisfied by a
  /// prior (superseded) entry.
  virtual void Read(
      ReadModifyWriteEntry& entry,
      KeyValueStore::ReadModifyWriteTarget::TransactionalReadOptions&& options,
      KeyValueStore::ReadModifyWriteTarget::ReadReceiver&& receiver) = 0;

  virtual void Writeback(ReadModifyWriteEntry& entry,
                         KeyValueStore::ReadResult&& read_result) = 0;

  virtual void Writeback(DeleteRangeEntry& entry) = 0;
  virtual bool MultiPhaseReadsCommitted() { return true; }
  virtual void PhaseCommitDone(size_t next_phase) = 0;

  virtual void AllEntriesDone(SinglePhaseMutation& single_phase_mutation);
  virtual void RecordEntryWritebackError(ReadModifyWriteEntry& entry,
                                         absl::Status error);

  void AbortRemainingPhases();
  void CommitNextPhase();

  enum class ReadModifyWriteStatus {
    // No change to the number of keys affected.
    kExisting,
    // Added first affected key.
    kAddedFirst,
    // Added subsequent affected key.
    kAddedSubsequent,
  };

  /// Registers a `ReadModifyWrite` operation for the specified `key`.
  ///
  /// This is normally called by implementations of
  /// `KeyValueStore::ReadModifyWrite`.
  ///
  /// \pre Must be called with `mutex()` held.
  /// \param phase[out] On return, set to the transaction phase to which the
  ///     operation was added.  Note that the transaction phase number may
  ///     change due to a concurrent call to `Transaction::Barrier`; therefore,
  ///     the current transaction phase may not equal `phase` after this call
  ///     completes.
  /// \param key The key affected by the operation.
  /// \param source The write source.
  /// \returns A status value that may be used to validate constraints in the
  ///     case that multi-key transactions are not supported.
  ReadModifyWriteStatus ReadModifyWrite(
      size_t& phase, KeyValueStore::Key key,
      KeyValueStore::ReadModifyWriteSource& source);

  /// Registers a `DeleteRange` operation for the specified `range`.
  ///
  /// This is normally called by implementations of
  /// `KeyValueStore::DeleteRange`.
  ///
  /// \pre Must be called with `mutex()` held.
  /// \param range The range to delete.
  void DeleteRange(KeyRange range);

  /// Returns a description of the first entry, used for error messages in the
  /// case that an atomic transaction is requested but is not supported.
  std::string DescribeFirstEntry();

  /// Returns the mutex used to protect access to the data structures.  Derived
  /// classes must implement this.  In the simple case, the derived class will
  /// simply declare an `absl::Mutex` data member, but the indirection through
  /// this virtual method allows a `MultiPhaseMutation` that is embedded within
  /// an `AsyncCache::TransactionNode` to share the mutex used by
  /// `AsyncCache::TransactionNode`.
  virtual absl::Mutex& mutex() = 0;

 protected:
  ~MultiPhaseMutation() = default;
};

inline MultiPhaseMutation& MutationEntry::multi_phase() const {
  return *single_phase_mutation().multi_phase_;
}
inline absl::Mutex& MutationEntry::mutex() const {
  return multi_phase().mutex();
}

class AtomicMultiPhaseMutation : public MultiPhaseMutation {
 public:
  class BufferedReadModifyWriteEntry : public ReadModifyWriteEntry {
   public:
    KeyValueStore::ReadResult read_result_;

    bool IsOutOfDate(absl::Time staleness_bound) {
      return read_result_.stamp.time == absl::InfinitePast() ||
             read_result_.stamp.time < staleness_bound;
    }
  };

  ReadModifyWriteEntry* AllocateReadModifyWriteEntry() override;
  void FreeReadModifyWriteEntry(ReadModifyWriteEntry* entry) override;
  void Writeback(ReadModifyWriteEntry& entry,
                 KeyValueStore::ReadResult&& read_result) override;
  void Writeback(DeleteRangeEntry& entry) override;
  void RevokeAllEntries();

 protected:
  ~AtomicMultiPhaseMutation() = default;
};

void RetryAtomicWriteback(SinglePhaseMutation& single_phase_mutation,
                          absl::Time staleness_bound);
void AtomicCommitWritebackSuccess(SinglePhaseMutation& single_phase_mutation);

void ReadDirectly(
    KeyValueStore* kvstore, ReadModifyWriteEntry& entry,
    KeyValueStore::ReadModifyWriteTarget::TransactionalReadOptions&& options,
    KeyValueStore::ReadModifyWriteTarget::ReadReceiver&& receiver);

void WritebackDirectly(KeyValueStore* kvstore, ReadModifyWriteEntry& entry,
                       KeyValueStore::ReadResult&& read_result);

void WritebackDirectly(KeyValueStore* kvstore, DeleteRangeEntry& entry);

template <typename DerivedMultiPhaseMutation = MultiPhaseMutation>
class TransactionNodeBase : public internal::TransactionState::Node,
                            public DerivedMultiPhaseMutation {
 public:
  TransactionNodeBase(KeyValueStore* kvstore)
      : internal::TransactionState::Node(kvstore) {
    intrusive_ptr_increment(kvstore);
  }

  ~TransactionNodeBase() { intrusive_ptr_decrement(this->kvstore()); }

  KeyValueStore* kvstore() {
    return static_cast<KeyValueStore*>(this->associated_data());
  }

  internal::TransactionState::Node& GetTransactionNode() override {
    return *this;
  }

  std::string DescribeKey(std::string_view key) override {
    return this->kvstore()->DescribeKey(key);
  }

  void Read(
      ReadModifyWriteEntry& entry,
      KeyValueStore::ReadModifyWriteTarget::TransactionalReadOptions&& options,
      KeyValueStore::ReadModifyWriteTarget::ReadReceiver&& receiver) override {
    internal_kvs::ReadDirectly(kvstore(), entry, std::move(options),
                               std::move(receiver));
  }

  void PhaseCommitDone(size_t next_phase) override {
    this->CommitDone(next_phase);
  }

  void PrepareForCommit() override {
    this->PrepareDone();
    this->ReadyForCommit();
  }

  void Commit() override { this->CommitNextPhase(); }

  absl::Mutex& mutex() override { return mutex_; }

  void Abort() override {
    this->AbortRemainingPhases();
    this->AbortDone();
  }

  std::string Describe() override {
    absl::MutexLock lock(&mutex_);
    return this->DescribeFirstEntry();
  }

  absl::Mutex mutex_;
};

class NonAtomicTransactionNode
    : public TransactionNodeBase<MultiPhaseMutation> {
 public:
  using TransactionNodeBase<MultiPhaseMutation>::TransactionNodeBase;

  void Writeback(ReadModifyWriteEntry& entry,
                 KeyValueStore::ReadResult&& read_result) override {
    internal_kvs::WritebackDirectly(this->kvstore(), entry,
                                    std::move(read_result));
  }

  void Writeback(DeleteRangeEntry& entry) override {
    internal_kvs::WritebackDirectly(kvstore(), entry);
  }
};

using AtomicTransactionNode = TransactionNodeBase<AtomicMultiPhaseMutation>;

template <typename TransactionNode>
Result<internal::OpenTransactionNodePtr<TransactionNode>> GetTransactionNode(
    KeyValueStore* kvstore, internal::OpenTransactionPtr& transaction) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node, internal::GetOrCreateOpenTransaction(transaction)
                     .GetOrCreateMultiPhaseNode(kvstore, [kvstore] {
                       return new TransactionNode(kvstore);
                     }));
  return internal::static_pointer_cast<TransactionNode>(std::move(node));
}

template <typename TransactionNode>
absl::Status AddReadModifyWrite(KeyValueStore* kvstore,
                                internal::OpenTransactionPtr& transaction,
                                size_t& phase, KeyValueStore::Key key,
                                KeyValueStore::ReadModifyWriteSource& source) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node,
      internal_kvs::GetTransactionNode<TransactionNode>(kvstore, transaction));
  absl::MutexLock lock(&node->mutex_);
  node->ReadModifyWrite(phase, std::move(key), source);
  return absl::OkStatus();
}

template <typename TransactionNode>
absl::Status AddDeleteRange(KeyValueStore* kvstore,
                            const internal::OpenTransactionPtr& transaction,
                            KeyRange&& range) {
  auto transaction_copy = transaction;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node, internal_kvs::GetTransactionNode<TransactionNode>(
                     kvstore, transaction_copy));
  absl::MutexLock lock(&node->mutex_);
  node->DeleteRange(std::move(range));
  return absl::OkStatus();
}

Future<TimestampedStorageGeneration> WriteViaTransaction(
    KeyValueStore* kvstore, KeyValueStore::Key key,
    std::optional<KeyValueStore::Value> value,
    KeyValueStore::WriteOptions options);

#ifdef TENSORSTORE_INTERNAL_KVSTORE_TRANSACTION_DEBUG

/// Logs a message associated with a `MutationEntry` when debugging is enabled.
///
/// The first parameter must be a `MutationEntry` reference.  The remaining
/// parameters must be compatible with `tensorstore::StrCat`.
///
/// Usage:
///
///     MutationEntry &entry = ...;
///     TENSORSTORE_KVSTORE_DEBUG_LOG(entry, "Information", to, "include");
#define TENSORSTORE_KVSTORE_DEBUG_LOG(...)                                    \
  do {                                                                        \
    tensorstore::internal_kvs::KvstoreDebugLog(TENSORSTORE_LOC, __VA_ARGS__); \
  } while (false)

template <typename... T>
void KvstoreDebugLog(tensorstore::SourceLocation loc, MutationEntry& entry,
                     const T&... arg) {
  std::string message;
  tensorstore::StrAppend(
      &message, "[", typeid(entry.multi_phase()).name(),
      ": multi_phase=", &entry.multi_phase(), ", entry=", &entry,
      ", phase=", entry.single_phase_mutation().phase_number_,
      ", key=", tensorstore::QuoteString(entry.key_));
  if (entry.entry_type() == kDeleteRange) {
    tensorstore::StrAppend(
        &message, ", exclusive_max=",
        tensorstore::QuoteString(
            static_cast<DeleteRangeEntry&>(entry).exclusive_max_));
  } else {
    size_t seq = 0;
    for (auto* e = static_cast<ReadModifyWriteEntry*>(&entry)->prev_; e;
         e = e->prev_) {
      ++seq;
    }
    tensorstore::StrAppend(&message, ", seq=", seq);
  }
  tensorstore::StrAppend(&message, "] ", arg...);
  tensorstore::internal::LogMessage(message.c_str(), loc);
}
#else
#define TENSORSTORE_KVSTORE_DEBUG_LOG(...) while (false)
#endif

}  // namespace internal_kvs
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TRANSACTION_H_
