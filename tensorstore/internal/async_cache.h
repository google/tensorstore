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

#ifndef TENSORSTORE_INTERNAL_ASYNC_CACHE_H_
#define TENSORSTORE_INTERNAL_ASYNC_CACHE_H_

/// \file Defines the abstract `AsyncCache` base class that extends the basic
/// `Cache` class with asynchronous read and read-modify-write functionality.

#include <atomic>
#include <cstddef>

#include "absl/base/thread_annotations.h"
#include "absl/time/time.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/intrusive_red_black_tree.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/sender.h"

namespace tensorstore {
namespace internal {

/// Abstract base class that extends `Cache` with asynchronous read and
/// read-modify-write functionality based on optimistic concurrency.
///
/// Each `Entry` in the cache stores a `ReadState` object, which contains a
/// representation of the current committed value as
/// `std::shared_ptr<const ReadData>`, along with a corresponding
/// `TimestampedStorageGeneration` indicating the generation and time as of
/// which it is known to be current.
///
/// If writes are supported, each `Entry` additionally may have associated
/// `TransactionNode` objects representing uncommitted modifications.  For
/// example, for an entry representing a chunk/region of an array, the
/// modifications may be represented by an `AsyncWriteArray` object specifying
/// new values for the chunk along with a mask indicating which elements have
/// been modified.
///
/// Transaction nodes are used to represent both *transactional* modifications
/// made using an explicit transaction, as well as *non-transactional*
/// modifications, which are automatically assigned to an implicit transaction.
///
///  - The memory required by a explicit transaction node is accounted for in
///    the `total_bytes` of the transaction, but is not accounted for by the
///    `CachePool`, and can neither cause, nor be affected by, automatic
///    writeback triggered by the `queued_for_writeback_bytes_limit` of the
///    `CachePool`.
///
///  - The memory required by an implicit transaction node is accounted for by
///    the `CachePool`, and implicit transactions associated with the
///    least-recently-used entries are committed automatically when the
///    `queued_for_writeback_bytes_limit` is reached.
///
/// A final, concrete `Derived` cache class should be defined as follows:
///
///     class Derived;
///     using DerivedBase = AsyncCacheBase<Derived, AsyncCache>;
///     class Derived : public DerivedBase {
///       using Base = DerivedBase;
///      public:
///       // Specifies the type representing the current committed data
///       // corresponding to an entry.
///       using ReadData = ...;
///
///       class Entry : public Base::Entry {
///        public:
///         using Cache = Derived;
///
///         void DoRead(absl::Time staleness_bound) override;
///         size_t ComputeReadDataSizeInBytes(const void *read_data) override;
///       };
///
///       class TransactionNode : public Base::TransactionNode {
///        public:
///         using Cache = Derived;
///         using Base::TransactionNode::TransactionNode;
///
///         void DoRead(absl::Time staleness_bound);
///         void DoWriteback() override;
///         void DoApply(absl::Time staleness_bound,
///                      ApplyReceiver receiver) override;
///         size_t ComputeWriteStateSizeInBytes() override;
///
///         // Additional data members representing the modification state...
///       };
///
/// If writes are not supported, the nested `Derived::TransactionNode` class
/// need not be defined.
///
/// Currently, this class is always used with the `KvsBackedCache` mixin, where
/// each entry corresponds to a single key in a `KeyValueStore`, and the
/// `KvsBackedCache` mixin defines the read and writeback behavior in terms of
/// "decode" and "encode" operations.
///
/// For each entry, the `AsyncCache` implementation keeps track of
/// read requests (made by calls to `Entry::Read`) and writeback requests (due
/// to either explicitly committing a transaction or for an implicit transaction
/// node, due to memory pressure in the `CachePool`), and calls `Entry::DoRead`
/// and `TransactionNode::DoWriteback` as needed to satisfy the requests.
/// Regardless of how many transactions are associated with an entry, at most a
/// single read operation or a single writeback operation may be in flight at
/// any given time.  Writeback operations are performed in the order in which
/// the transaction commit started, and take precedence over read requests (but
/// the read request will normally be satisfied by the completion of the
/// writeback operation).
///
/// `ChunkCache` extends `AsyncCache` to provide caching of chunked
/// array data.
///
class AsyncCache : public Cache {
 public:
  class Entry;
  class TransactionNode;
  struct ReadRequestState;

  /// Defines the representation type of the cached read.  Derived classes
  /// should override this.
  ///
  /// The "read data" is represented in the `Entry` and `TransactionNode` as a
  /// `std::shared_ptr<const void>`, which for a given `Derived` cache class is
  /// assumed to point to either `nullptr` or an object that inherits from
  /// `typename Derived::ReadData`.
  using ReadData = void;

  /// Pairs a "read data" value with an associated generation.  This represents
  /// the result of a successful read operation.
  struct ReadState {
    /// The read data.
    std::shared_ptr<const void> data;

    /// Generation corresponding to `read_data`, and time as of which it is
    /// known to be up to date.
    ///
    /// If `stamp.generation` equals `StorageGeneration::Unknown()`, then `data`
    /// is `nullptr` and not meaningful.
    ///
    /// After a successful read, the `stamp.generation` must not equal
    /// `StorageGeneration::Unknown()`, and `stamp.time.inclusive_lower` must
    /// not equal `absl::InfinitePast()`.
    TimestampedStorageGeneration stamp;

    /// Returns a reference to a default-constructed `ReadState`.
    static const ReadState& Unknown();
  };

  // Internal representation for tracking the most recent successful read and
  // queued/in-progress read operations.  Treat this as private.
  struct ReadRequestState {
    /// Promise corresponding to an in-progress read request, which was
    /// initialized at the local time `issued_time`.  If
    /// `issued.valid() == false`, there is no read request in progress.
    Promise<void> issued;

    /// Promise corresponding to a queued read request, to be issued after
    /// `issued` completes if the resultant `read_generation.time` is older than
    /// `queued_time`.  If `queued.valid() == false`, there is no queued read
    /// request.
    ///
    /// \invariant `!queued.valid() || issued.valid()`.
    Promise<void> queued;

    /// Only meaningful if `issued.valid()`.
    absl::Time issued_time;

    /// Only meaningful if `queued.valid()`.
    absl::Time queued_time = absl::InfinitePast();

    /// The most recently-cached read state.
    ReadState read_state;

    /// The size in bytes consumed by `read_state.read_data`.  This is the
    /// cached result of calling
    /// `Entry::ComputeReadDataSizeInBytes(read_state.read_data.get())`.
    size_t read_state_size = 0;
  };

  /// View of a `ReadState` with the "read data" cast to a derived `ReadData`
  /// type.
  template <typename ReadData>
  class ReadView {
   public:
    ReadView() : read_state_(&ReadState::Unknown()) {}

    explicit ReadView(const ReadState& read_state) : read_state_(&read_state) {}

    bool has_timestamp() const {
      return stamp().time.inclusive_lower != absl::InfinitePast();
    }

    bool has_generation() const {
      return !StorageGeneration::IsUnknown(stamp().generation);
    }

    std::shared_ptr<const ReadData> shared_data() const {
      return std::static_pointer_cast<const ReadData>(read_state_->data);
    }

    const ReadData* data() const {
      return static_cast<const ReadData*>(read_state_->data.get());
    }

    const ReadState& read_state() const { return *read_state_; }

    const TimestampedStorageGeneration& stamp() const {
      return read_state_->stamp;
    }

   private:
    const ReadState* read_state_;
  };

  /// RAII lock class that provides read access to the read state of an `Entry`
  /// or `TransactionNode`.
  ///
  /// The inherited methods of `ReadView` are used for accessing the read state
  /// with the read lock held.
  ///
  /// Example:
  ///
  ///     AsyncCache::ReadLock<Data> lock{entry};
  ///     // access lock.data() or lock.stamp()
  ///     // access lock.shared_Data()
  ///
  /// \tparam The type to which the `ReadData` will be cast.
  template <typename ReadData>
  class ReadLock : public ReadView<ReadData> {
   public:
    ReadLock() = default;

    template <typename DerivedEntryOrNode,
              typename = std::enable_if_t<
                  std::is_base_of_v<Entry, DerivedEntryOrNode> ||
                  std::is_base_of_v<TransactionNode, DerivedEntryOrNode>>>
    explicit ReadLock(DerivedEntryOrNode& entry_or_node)
        : ReadView<ReadData>(entry_or_node.LockReadState()),
          lock_(GetOwningEntry(entry_or_node).mutex_, std::adopt_lock) {
      static_assert(std::is_convertible_v<
                    const typename DerivedEntryOrNode::Cache::ReadData*,
                    const ReadData*>);
    }

   private:
    UniqueWriterLock<absl::Mutex> lock_;
  };

  // Class template argument deduction (CTAD) would make `ReadLock` more
  // convenient to use, but GCC doesn't support deduction guides for inner
  // classes.
#if 0
  template <typename DerivedEntryOrNode,
            typename = std::enable_if_t<
                std::is_base_of_v<Entry, DerivedEntryOrNode> ||
                std::is_base_of_v<TransactionNode, DerivedEntryOrNode>>>
  explicit ReadLock(DerivedEntryOrNode& entry_or_node)
      -> ReadLock<typename DerivedEntryOrNode::Cache::ReadData>;
#endif

  /// RAII lock class that provides write access to a `TransactionNode`.
  ///
  /// It holds an `OpenTransactionNodePtr` to the node, which prevents the
  /// transaction from being committed or aborted and prevents the node from
  /// being aborted.  It also holds a write lock on the node, which prevents
  /// concurrent access to the node's state.
  template <typename DerivedNode>
  class WriteLock {
    static_assert(std::is_base_of_v<TransactionNode, DerivedNode>);

   public:
    explicit WriteLock(internal::OpenTransactionNodePtr<DerivedNode> node,
                       std::adopt_lock_t)
        : node_(std::move(node)) {}

    WriteLock(WriteLock&&) = default;
    WriteLock(const WriteLock&) = delete;

    WriteLock& operator=(WriteLock&&) = default;
    WriteLock& operator=(const WriteLock&) = delete;

    DerivedNode* operator->() const { return node_.get(); }
    DerivedNode& operator*() const { return *node_; }

    /// Unlocks the write lock, and returns (releases) the open transaction node
    /// pointer.
    internal::OpenTransactionNodePtr<DerivedNode> unlock() {
      if (node_) {
        node_->WriterUnlock();
      }
      return std::exchange(node_, {});
    }

    ~WriteLock() {
      if (node_) node_->WriterUnlock();
    }

   private:
    internal::OpenTransactionNodePtr<DerivedNode> node_;
  };

  /// Base Entry class.  `Derived` classes must define a nested `Derived::Entry`
  /// class that extends this `Entry` class.
  ///
  /// Derived classes should define a nested `ReadState` type to represent the
  /// "read state", and a `read_state` data member of that type.  While
  /// `AsyncCache` does not itself rely on those members, some
  /// mixin types like `KvsBackedCache` do rely on those members.
  class ABSL_LOCKABLE Entry : public Cache::Entry {
   public:
    /// For convenience, a `Derived::Entry` type should define a nested `Cache`
    /// alias to the `Derived` cache type, in order for `GetOwningCache` to
    /// return a pointer cast to the derived cache type.
    using Cache = AsyncCache;

    Entry() = default;

    template <typename DerivedEntry>
    friend std::enable_if_t<std::is_base_of_v<Entry, DerivedEntry>,
                            DerivedEntry&>
    GetOwningEntry(DerivedEntry& entry) {
      return entry;
    }

    void WriterLock() ABSL_EXCLUSIVE_LOCK_FUNCTION();
    void WriterUnlock() ABSL_UNLOCK_FUNCTION();

    /// Requests data no older than `staleness_bound`.
    ///
    /// \param staleness_bound Limit on data staleness.
    /// \returns A future that resolves to a success state once data no older
    ///     than `staleness_bound` is available, or to an error state if the
    ///     request failed.
    Future<const void> Read(absl::Time staleness_bound);

    /// Obtains an existing or new transaction node for the specified entry and
    /// transaction.  May also be used to obtain an implicit transaction node.
    ///
    /// \param entry The associated cache entry for which to obtain a
    ///     transaction node.
    /// \param transaction[in,out] Transaction associated with the entry.  If
    ///     non-null, must specify an explicit transaction, and an associated
    ///     transaction node will be created if one does not already exist.  In
    ///     this case, the `tranaction` pointer itself will not be modified.  An
    ///     implicit transaction node is requested by specifying `transaction`
    ///     initially equally to `nullptr`.  If there is an existing implicit
    ///     transaction node that is still open, it will be returned.
    ///     Otherwise, a new implicit transaction node will be created.  Upon
    ///     return, `transaction` will hold an open transaction reference to the
    ///     associated implicit transaction.
    template <typename DerivedEntry>
    friend std::enable_if_t<std::is_base_of_v<Entry, DerivedEntry>,
                            Result<OpenTransactionNodePtr<
                                typename DerivedEntry::Cache::TransactionNode>>>
    GetTransactionNode(DerivedEntry& entry,
                       internal::OpenTransactionPtr& transaction) {
      TENSORSTORE_ASSIGN_OR_RETURN(auto node,
                                   entry.GetTransactionNodeImpl(transaction));
      return internal::static_pointer_cast<
          typename DerivedEntry::Cache::TransactionNode>(std::move(node));
    }

    template <typename DerivedEntry>
    friend std::enable_if_t<
        std::is_base_of_v<Entry, DerivedEntry>,
        Result<WriteLock<typename DerivedEntry::Cache::TransactionNode>>>
    GetWriteLockedTransactionNode(
        DerivedEntry& entry, const internal::OpenTransactionPtr& transaction) {
      using DerivedNode = typename DerivedEntry::Cache::TransactionNode;
      while (true) {
        auto transaction_copy = transaction;
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto node, entry.GetTransactionNodeImpl(transaction_copy));
        if (node->try_lock()) {
          return WriteLock<DerivedNode>(
              internal::static_pointer_cast<DerivedNode>(std::move(node)),
              std::adopt_lock);
        }
        // `node->Revoke()` was called by another thread after the call to
        // `GetTransactionNodeImpl` but before the call to `try_lock`.  (This is
        // expected to be rare.)
      }
    }

    /// Requests initial or updated data from persistent storage for a single
    /// `Entry`.
    ///
    /// This is called automatically by the `AsyncCache`
    /// implementation either due to a call to `Read` that cannot be satisfied
    /// by the existing cached data.
    ///
    /// Derived classes must implement this method, and implementations must
    /// call (either immediately or asynchronously) `ReadSuccess` or `ReadError`
    /// to signal completion.
    virtual void DoRead(absl::Time staleness_bound) = 0;

    /// Signals that the read request initiated by the most recent call to
    /// `DoRead` succeeded.
    ///
    /// Derived classes may override this method, but must ensure this base
    /// class method is called.
    virtual void ReadSuccess(ReadState&& read_state);

    /// Signals that the read request initiated by the most recent call to
    /// `DoRead` failed.
    ///
    /// Derived classes may override this method, but must ensure this base
    /// class method is called.
    virtual void ReadError(absl::Status error);

    /// Derived classes should override this to return the size of the "read
    /// state".
    ///
    /// This method is always called with at least a shared lock on the "read
    /// state".
    virtual size_t ComputeReadDataSizeInBytes(const void* data);

    /// Returns `true` if implicit transaction nodes may be shared.  If `false`,
    /// a separate implicit transaction node will be created each time one is
    /// requested.
    virtual bool ShareImplicitTransactionNodes() { return true; }

    // Below members should be treated as private:

    ReadState& LockReadState() ABSL_NO_THREAD_SAFETY_ANALYSIS {
      mutex_.WriterLock();
      return read_request_state_.read_state;
    }

    Result<OpenTransactionNodePtr<TransactionNode>> GetTransactionNodeImpl(
        OpenTransactionPtr& transaction);

    /// Protects access to all other members.  Also protects access to the
    /// `read_request_state_` of all associated `TransactionNode` objects.
    absl::Mutex mutex_;

    ReadRequestState read_request_state_;

    /// Sum of the size of all implicit transaction nodes associated with this
    /// entry.
    size_t write_state_size_ = 0;

    /// The current implicit transaction node used for non-transactional writes.
    /// There may be additional implicit transactions in flight.
    ///
    /// The tag bit is set to 1 if this entry has acquired a commit block on
    /// `implicit_transaction_node->transaction()`.
    internal::TaggedPtr<TransactionNode, 1> implicit_transaction_node_;

    using TransactionTree =
        internal::intrusive_red_black_tree::Tree<TransactionNode,
                                                 TransactionNode>;

    /// Associated transactions, ordered by `TransactionState` pointer. Implicit
    /// transactions are not included.
    TransactionTree transactions_;

    /// Pointer to transaction node for which writeback is currently being
    /// performed.  If not `nullptr`,
    /// committing_transaction_node_->pending_queue_`
    TransactionNode* committing_transaction_node_{nullptr};

    /// Number of implicit transaction nodes associated with this entry.  The
    /// entry is considered "clean" when this is 0.
    size_t num_implicit_transactions_ = 0;

    using Flags = uint8_t;

    /// Bit vector of flags, see below.
    Flags flags_ = 0;

    /// Set if `write_state_size_` or `read_request_state_.read_state_size` has
    /// been updated while `mutex_` is locked, and the cache should be notified
    /// the next time `mutex_` is unlocked.
    constexpr static Flags kSizeChanged = 1;

    /// Set if the entry state should be changed to `dirty` when `mutex_` is
    /// next unlocked.
    constexpr static Flags kStateChanged = 2;

    /// Set if the entry state should be changed to `writeback_requested` when
    /// `mutex_` is next unlocked.
    constexpr static Flags kMarkWritebackRequested = 4;
  };

  /// Base transaction node class.  Derived classes must define a nested
  /// `TransactionNode` type that inherits from this class and represents
  /// uncommitted modifications to an entry.
  class ABSL_LOCKABLE TransactionNode
      : public internal::TransactionState::Node,
        public internal::intrusive_red_black_tree::NodeBase<TransactionNode> {
   public:
    /// For convenience, a `Derived::TransactionNode` type should define a
    /// nested `Cache` alias to the `Derived` cache type, in order for
    /// `GetOwningEntry` and `GetOwningCache` to return a pointer cast to the
    /// derived cache type.
    using Cache = AsyncCache;

    /// A `Derived::TransactionNode` class should define either directly, or via
    /// inheritance, a constructor that takes a single `Derived::Entry&`
    /// argument specifying the associated cache entry.  This constructor will
    /// by invoked by the implementation of `DoAllocateTransactionNode` provided
    /// by `AsyncCacheBase`, from which concrete derived classes
    /// should inherit.
    explicit TransactionNode(Entry& entry);

    ~TransactionNode();

    /// Acquires an exclusive lock on the "modification state" (i.e. `mutex_`).
    /// When recording modifications, this lock must be held.  `TransactionNode`
    /// may be used with `UniqueWriterLock`.
    void WriterLock() ABSL_EXCLUSIVE_LOCK_FUNCTION();

    /// Updates the transaction or `CachePool` size accounting by calling
    /// `ComputeSizeInBytes()`, and then releases the lock acquired by
    /// `WriterLock()`.
    void WriterUnlock() ABSL_UNLOCK_FUNCTION();

    /// Attempts to acquire an exclusive lock while `IsRevoked() == false`.  If
    /// successful, returns `true` with a write lock held.  If `IsRevoked()` is
    /// set to `true`, before a lock can be acquired, returns `false`.
    ///
    /// Unlike `std::mutex::try_lock`, this method still blocks until a lock can
    /// be acquired.  It only returns `false` if `IsRevoked() == true`.
    bool try_lock() ABSL_EXCLUSIVE_TRYLOCK_FUNCTION(true);

    /// Computes the heap memory bytes consumed by the write state.
    ///
    /// Derived classes with any heap-allocated members must override this
    /// method to track memory usage properly.  The default implementation
    /// returns 0.
    virtual size_t ComputeWriteStateSizeInBytes();

    /// Returns the entry with which the node is associated.
    ///
    /// This is defined as a friend rather than a member to allow the derived
    /// entry type to be returned.
    template <typename DerivedNode>
    friend std::enable_if_t<std::is_base_of_v<TransactionNode, DerivedNode>,
                            typename DerivedNode::Cache::Entry&>
    GetOwningEntry(DerivedNode& node) {
      return static_cast<typename DerivedNode::Cache::Entry&>(
          *static_cast<Cache::Entry*>(node.associated_data()));
    }

    /// Returns the cache with which the node is associated.
    ///
    /// This is defined as a friend rather than a member to allow the derived
    /// cache type to be returned.
    template <typename DerivedNode>
    friend std::enable_if_t<std::is_base_of_v<TransactionNode, DerivedNode>,
                            typename DerivedNode::Cache&>
    GetOwningCache(DerivedNode& node) {
      return GetOwningCache(GetOwningEntry(node));
    }

    /// Called by `GetTransactionNode` (or `GetWriteLockedTransactionNode`) to
    /// initialize the transaction before it is otherwise used.
    ///
    /// \param transaction[in,out] Upon invocation, `transaction` will equal
    ///     `this->transaction()`.  If this node corresponds to an explicit
    ///     transaction, `transaction` will be non-null and specifies the
    ///     transaction to use for any downstream nodes (e.g. `KeyValueStore`
    ///     write nodes).  If this is an implicit transaction, `transaction`
    ///     will be `nullptr` upon invocation, and the derived `DoInitialize`
    ///     implementation must set `transaction` to a new or existing implicit
    ///     transaction (e.g. via calling `KeyValueStore::ReadModifyWrite`).
    virtual absl::Status DoInitialize(
        internal::OpenTransactionPtr& transaction);

    /// May be called by the derived `DoInitialize` implementation to indicate
    /// that reading from this transaction node is equivalent to reading from
    /// the `Entry`.
    ///
    /// This must only be called from `DoInitialize`, or immediately before a
    /// call to `InvalidateReadState`.
    void SetReadsCommitted() { reads_committed_ = true; }

    /// Must be called any time the write state is modified.
    void MarkSizeUpdated() { size_updated_ = true; }

    /// Marks this transaction node as revoked, meaning it may no longer be used
    /// for writes or up-to-date reads.
    ///
    /// Subsequent calls to `GetTransactionNode` or
    /// `GetWriteLockedTransactionNode` on the entry will return a new
    /// transaction node.
    ///
    /// This is called automatically by `KvsBackedCache` when another write
    /// source is added for the same key.
    ///
    /// Note that previously requested reads may still be completed using this
    /// transaction node (and must be completed using this transaction node to
    /// avoid potential livelock).
    ///
    /// The modifications made previously to a revoked transaction node will
    /// still be committed if the associated transaction is committed.
    virtual void Revoke();

    /// Returns true if this node has been revoked.
    bool IsRevoked() { return revoked_.load(std::memory_order_acquire); }

    /// Invalidates the read state of this transaction node.  This must only be
    /// called if `transaction->commit_started() == true` and `Commit()` has not
    /// yet been called.  This is called in the case of multi-phase
    /// transactions, where the read state of this node may depend on the local
    /// modifications made in a prior phase.  Once that prior phase is
    /// committed, those modifications are no longer guaranteed and this node
    /// must be "rebased" on the new committed state.
    ///
    /// Derived classes may override this method to also invalidate any other
    /// cached state dependent on a "dirty" read generation, but must also call
    /// this base implementation.
    ///
    /// Note that in the common case, `SetReadsCommitted` will have just been
    /// invoked and the owning `Entry` will contain a cached read state once
    /// writeback of the prior phase completes, before this transaction node
    /// actually needs a valid read state, and therefore no additional I/O will
    /// actually be required.
    virtual void InvalidateReadState();

    /// Requests a read state for this transaction node that is current as of
    /// the specified `staleness_bound`.
    Future<const void> Read(absl::Time staleness_bound);

    /// Requests initial or updated data from persistent storage for a single
    /// `Entry`.
    ///
    /// This is called automatically by the `AsyncCache`
    /// implementation either due to a call to `Read` that cannot be satisfied
    /// by the existing cached data.
    ///
    /// Derived classes must implement this method, and implementations must
    /// call (either immediately or asynchronously) `ReadSuccess` or `ReadError`
    /// to signal completion.
    virtual void DoRead(absl::Time staleness_bound) = 0;

    /// Signals that the read request initiated by the most recent call to
    /// `DoRead` succeeded.
    ///
    /// Derived classes may override this method, but must ensure this base
    /// class method is called.
    virtual void ReadSuccess(ReadState&& read_state);

    /// Signals that the read request initiated by the most recent call to
    /// `DoRead` failed.
    ///
    /// Derived classes may override this method, but must ensure this base
    /// class method is called.
    virtual void ReadError(absl::Status error);

    /// Initiates writeback to persistent storage of the modifications
    /// represented by this transaction node.
    ///
    /// This is called automatically by the `AsyncCache`
    /// implementation, after the transaction commit has started, once any prior
    /// read operations or transaction writebacks have completed.
    ///
    /// The writeback operation is considered to be in progress from the start
    /// of the call to `Commit` until either `WritebackSuccess` or
    /// `WritebackError` is invoked.  While the writeback is in progress, no
    /// other code may modify the entry "read state", and no other code may
    /// access the transaction node.
    ///
    /// The `AsyncCache` implementation holds a weak reference to the
    /// `TransactionNode` while writeback is in progress to ensure it is not
    /// destroyed.  It is therefore not necessary for the `Commit`
    /// implementation to hold a weak reference to this transaction node, but be
    /// aware that this transaction node may be destroyed inside the call to
    /// `WritebackSuccess` or `WritebackError` and therefore should not be
    /// referenced afterwards unless a weak reference has been separately
    /// acquired.
    ///
    /// Derived classes must implement this method.  The `KvsBackedCache`
    /// mixin provides an implementation of writeback in terms of simpler
    /// "encode" and "decode" operations.
    void Commit() override;

    /// Signals that the writeback request initiated by the most recent call to
    /// `Commit` has completed successfully.
    ///
    /// Derived classes may override this method (e.g. to perform some or all of
    /// the work of step 2b or 2c), but must ensure this base class method is
    /// called.
    virtual void WritebackSuccess(ReadState&& read_state);

    /// Signals that the writeback request initiated by the most recent call to
    /// `Commit` failed.
    ///
    /// Derived classes may override this method (e.g. to perform any necessary
    /// cleanup), but must ensure this base class method is called.
    virtual void WritebackError();

    using ApplyReceiver =
        AnyReceiver<absl::Status, ReadState, UniqueWriterLock<TransactionNode>>;

    struct ApplyOptions {
      /// Returned `ReadStateUpdate` must reflect an existing read state that is
      /// current as of `staleness_bound`.
      absl::Time staleness_bound;

      /// If `true`, the `data` returned in the `ReadState` will be ignored.
      /// This option is used if `DoApply` is called solely to validate the read
      /// state.  If no validation is necessary, the `stamp` field of the
      /// `ReadState` may be set to
      /// `TimestampedStorageGeneration::Unconditional()`.
      bool validate_only = false;
    };

    /// Requests an updated read state that reflects the modifications made by
    /// this transaction node applied to the existing read state that is current
    /// as of `staleness_bound`.
    ///
    /// Typically, if this transaction node completely overwrites the existing
    /// state, then the `receiver` can be invoked immediately.  Otherwise,
    /// `DoRead(staleness_bound)` should be called to obtain an updated read
    /// state to which the modifications indicated in this transaction node can
    /// be applied.
    ///
    /// This is not invoked directly by `AsyncCache` (and therefore does not
    /// necessarily need to be implemented), but is required by
    /// `KvsBackedCache`.
    virtual void DoApply(ApplyOptions option, ApplyReceiver receiver);

    /// Invoked by the `TransactionState` implementation to commit this node.
    ///
    /// Enqueues this transaction node for writeback once any previously-issued
    /// read request or previously issued/enqueued writeback request completes.
    /// (In order to maintain cache consistency, more than more than one
    /// concurrent read/writeback operation is not permitted.)  Calls
    /// `PrepareDone` once this node is at the front of the writeback queue.
    void PrepareForCommit() override;

    /// Invoked by the `TransactionState` implementation when the transaction is
    /// aborted.  This won't be invoked after `Commit` is called, or while other
    /// code still holds a reference to this object in order to read or write
    /// from it.
    void Abort() override;

    // Treat as private:

    ReadState& LockReadState() ABSL_NO_THREAD_SAFETY_ANALYSIS {
      auto& entry = GetOwningEntry(*this);
      entry.mutex_.WriterLock();
      if (reads_committed_) {
        return entry.read_request_state_.read_state;
      } else {
        return read_request_state_.read_state;
      }
    }

    // The `NodeBase<TransactionNode>` base serves two roles:
    //
    // If `transaction()->implicit_transaction() == false` and `Commit` has not
    // called, it serves as a red-black tree node in the `Entry::transactions_`
    // tree.
    //
    // After `PrepareForCommit` has been called, it is used via
    // `PendingWritebackQueueAccessor` as a circular doubly-linked list to
    // represent the queue of additional nodes pending writeback.  The node
    // `n0 = committing_transaction_node_` is the head of the queue:
    //
    //     n0(tx) -> n1(ty) -> n2(tz)
    //
    // In the common case, each node `n0`, `n1`, `n2` is associated with a
    // separate transaction `tx`, `ty`, `tz`, respectively.  `PrepareForCommit`
    // has been called on all nodes in the queue, but `PrepareDone` has only
    // been called on `n0`.  (`ReadyForCommit` may also have been called, as
    // reflected by `ready_for_commit_called_`, depending on whether there is a
    // read operation in progress.)  Once commit of `n0` completes (either
    // successfully or with an error), `PrepareDone` and `ReadyForCommit` is
    // called on `n1`, `committing_transaction_node_` is set to `n1`:
    //
    //     n1(ty) -> n2(tz)
    //
    // and the commit process continues.
    //
    // It is possible for a single transaction phase to contain multiple
    // transaction nodes for the same `AsyncCache::Entry` (e.g. due to `Revoke`
    // being called).  In this case, the front of the queue may contain
    // additional nodes associated with the same transaction as
    // `n0 = committing_transaction_node_`:
    //
    //     n0(tx) -> n1(tx) -> n2(tx) -> n3(ty) -> n4(tz)
    //
    // In this case, `PrepareDone` (and possibly `ReadyForCommit`) has been
    // called on `n0`, `n1`, and `n2`, but not on `n3` or `n4`.
    //
    // If `n1` (or `n2`) completes writeback before `n0`, it is removed from the
    // list:
    //
    //     n0(tx) -> n2(tx) -> n3(ty) -> n4(tz)
    //
    // After `n0`, `n1`, and `n2` all complete writeback, `PrepareDone` and
    // `ReadyForCommit` are called on `n3`:
    //
    //     n3(ty) -> n4(tz)

    using PendingWritebackQueueAccessor =
        intrusive_red_black_tree::LinkedListAccessor<TransactionNode,
                                                     TransactionNode>;

    absl::once_flag initialized_;
    absl::Status initialized_status_;

    /// Mutex that must be locked while operating on the "modification state".
    absl::Mutex mutex_;

    ReadRequestState read_request_state_;

    /// Cached write state size.  Protected by `mutex_`.
    size_t write_state_size_ = 0;

    /// Set to indicate that a transactional read will have the same result as a
    /// non-transactional read directly on the owning entry; in that case, a
    /// non-transactional read will be used except when committing, as that
    /// allows the result to be cached in the entry.
    bool reads_committed_;

    /// Set to indicate that the write state size has changed while `mutex_` is
    /// locked.
    bool size_updated_;

    /// Indicates whether this transaction node has been revoked.  This may be
    /// set to `true` without holding any locks, but a thread that already owns
    /// a lock on `mutex_` may continue writing.  No writes may be performed if
    /// `mutex_` is acquired after `revoked_` has been set to `true`.
    std::atomic<bool> revoked_{false};

    enum class PrepareForCommitState {
      /// Either `PrepareForCommit` has not yet been called, or
      /// `PrepareForCommit` has been called but this node is still enqueued in
      /// the `committing_transaction_node_` list.
      kNone,
      /// Indicates that `PrepareDone` has been called in response to
      /// `PrepareForCommit`.  `PrepareDone` is called once this transaction
      /// node (or another transaction node associated with the same
      /// transaction) is first in line to be committed, even if there is still
      /// a read operation in progress on the owning `Entry`.
      kPrepareDoneCalled,
      /// Indicates that `ReadyForCommit` has been called in response to
      /// `PrepareForCommit`.  `ReadyForCommit` is called after `PrepareDone` is
      /// called once there is no read operation in progress on the owning
      /// `Entry` and commit can actually start.
      kReadyForCommitCalled,
    };

    /// Tracks the current state after `PrepareForCommit` is called and this
    /// node is in the `committing_transaction_node_` list.
    PrepareForCommitState prepare_for_commit_state_ =
        PrepareForCommitState::kNone;
  };

  /// Derived classes should override this to return the size of any additional
  /// heap allocations that are unaffected by changes to the read state, write
  /// state, or writeback state.
  ///
  /// Derived implementations should include in the returned sum the result of
  /// calling this base implementation.
  virtual size_t DoGetFixedSizeInBytes(Cache::Entry* entry);

  /// The total size in bytes is equal to
  ///
  ///     DoGetFixedSizeInBytes() +
  ///     entry->ComputeReadStateSizeInBytes() +
  ///     entry->entry_data_.write_state_size
  size_t DoGetSizeInBytes(Cache::Entry* entry) final;

  /// Handles writeback requests triggered by memory pressure in the containing
  /// `CachePool`.
  void DoRequestWriteback(PinnedEntry base_entry) final;

  virtual TransactionNode& DoAllocateTransactionNode(Entry& entry);
};

/// CRTP base class for concrete derived classes of `AsyncCache`.
///
/// \tparam Derived Derived class type.
/// \tparam Parent The base class from which to inherit, must inherit from (or
///     equal) `AsyncCache`.
template <typename Derived, typename Parent>
class AsyncCacheBase : public CacheBase<Derived, Parent> {
  static_assert(std::is_base_of_v<AsyncCache, Parent>);

 public:
  using CacheBase<Derived, Parent>::CacheBase;
  typename AsyncCache::TransactionNode& DoAllocateTransactionNode(
      AsyncCache::Entry& entry) override {
    return *new typename Derived::TransactionNode(
        static_cast<typename Derived::Entry&>(entry));
  }
};

#ifdef TENSORSTORE_ASYNC_CACHE_DEBUG
template <typename EntryOrNode, typename... T>
void AsyncCacheDebugLog(tensorstore::SourceLocation loc,
                        EntryOrNode& entry_or_node, const T&... arg) {
  auto& entry = GetOwningEntry(entry_or_node);
  std::string node_desc;
  if constexpr (std::is_base_of_v<AsyncCache::TransactionNode, EntryOrNode>) {
    node_desc = tensorstore::StrCat(
        ", node=", &entry_or_node,
        ", transaction=", entry_or_node.transaction(),
        entry_or_node.transaction()->implicit_transaction() ? " (implicit)"
                                                            : " (explicit)",
        ", phase=", entry_or_node.phase());
  }
  tensorstore::internal::LogMessage(
      tensorstore::StrCat("[", typeid(GetOwningCache(entry)).name(),
                          ": entry=", &entry,
                          ", key=", tensorstore::QuoteString(entry.key()),
                          node_desc, "] ", arg...)
          .c_str(),
      loc);
}

/// Logs a debugging message associated with an `AsyncCache::Entry` or
/// `AsyncCache::TransactionNode`.
///
/// This macro is intended to be called like a function, where the first
/// parameter is an `AsyncCache::Entry&` or `AsyncCache::TransactionNode&`, and
/// the remaining parameters are passed to `tensorstore::StrCat` to generate a
/// message.
#define TENSORSTORE_ASYNC_CACHE_DEBUG_LOG(...)                                 \
  do {                                                                         \
    ::tensorstore::internal::AsyncCacheDebugLog(TENSORSTORE_LOC, __VA_ARGS__); \
  } while (false)
#else
#define TENSORSTORE_ASYNC_CACHE_DEBUG_LOG(...) while (false)
#endif

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ASYNC_CACHE_H_
