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

#ifndef TENSORSTORE_INTERNAL_CACHE_ASYNC_CACHE_H_
#define TENSORSTORE_INTERNAL_CACHE_ASYNC_CACHE_H_

/// \file Defines the abstract `AsyncCache` base class that extends the basic
/// `Cache` class with asynchronous read and read-modify-write functionality.

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/batch.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/container/intrusive_red_black_tree.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"

#ifndef TENSORSTORE_ASYNC_CACHE_DEBUG
#define TENSORSTORE_ASYNC_CACHE_DEBUG 0
#endif

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
/// A final, concrete `Derived` cache class should be defined as follows:
///
///     class Derived : public AsyncCache {
///       using Base = AsyncCache;
///      public:
///       // Specifies the type representing the current committed data
///       // corresponding to an entry.
///       using ReadData = ...;
///
///       class Entry : public Base::Entry {
///        public:
///         using OwningCache = Derived;
///
///         void DoRead(AsyncCacheReadRequest request) override;
///         size_t ComputeReadDataSizeInBytes(const void *read_data) override;
///       };
///
///       class TransactionNode : public Base::TransactionNode {
///        public:
///         using OwningCache = Derived;
///         using Base::TransactionNode::TransactionNode;
///
///         void DoRead(AsyncCacheReadRequest request) override;
///         void DoApply(ApplyOptions options,
///                      ApplyReceiver receiver) override;
///         size_t ComputeWriteStateSizeInBytes() override;
///
///         // Additional data members representing the modification state...
///       };
///
///     // Implement required virtual interfaces:
///
///     Entry* DoAllocateEntry() final { return new Entry; }
///     std::size_t DoGetSizeofEntry() final { return sizeof(Entry); }
///     TransactionNode* DoAllocateTransactionNode(
///            AsyncCache::Entry& entry) final {
///       return new TransactionNode(static_cast<Entry&>(entry));
///     }
///   };
///
/// If writes are not supported, the nested `Derived::TransactionNode` class
/// need not be defined.
///
/// Currently, this class is always used with the `KvsBackedCache` mixin, where
/// each entry corresponds to a single key in a `KeyValueStore`, and the
/// `KvsBackedCache` mixin defines the read and writeback behavior in terms of
/// "decode" and "encode" operations.
///
/// For each entry, the `AsyncCache` implementation keeps track of read requests
/// (made by calls to `Entry::Read`) and writeback requests and calls
/// `Entry::DoRead` and `TransactionNode::Commit` as needed to satisfy the
/// requests.  Regardless of how many transactions are associated with an entry,
/// at most a single read operation or a single writeback operation may be in
/// flight at any given time.  Writeback operations are performed in the order
/// in which the transaction commit started, and take precedence over read
/// requests (but the read request will normally be satisfied by the completion
/// of the writeback operation).
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

    /// `read_state` is known to be out-of-date, due to a local modification not
    /// reflected in the cache.
    bool known_to_be_stale = false;

    /// Indicates that the read request corresponding to `queued` should not be
    /// issued yet, because it was requested as part of a not-yet-submitted
    /// batch.  If a non-batch or submitted-batch request is subsequently made,
    /// it will be issued.
    bool queued_request_is_deferred = true;

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
          lock_(GetOwningEntry(entry_or_node).mutex(), std::adopt_lock) {
      static_assert(std::is_convertible_v<
                    const typename DerivedEntryOrNode::OwningCache::ReadData*,
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
      -> ReadLock<typename DerivedEntryOrNode::OwningCache::ReadData>;
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
    internal::OpenTransactionNodePtr<DerivedNode> unlock()
        ABSL_NO_THREAD_SAFETY_ANALYSIS {
      if (node_) {
        node_->WriterUnlock();
      }
      return std::exchange(node_, {});
    }

    ~WriteLock() ABSL_NO_THREAD_SAFETY_ANALYSIS {
      if (node_) node_->WriterUnlock();
    }

   private:
    internal::OpenTransactionNodePtr<DerivedNode> node_;
  };

  /// Specifies options for read operations.
  struct AsyncCacheReadRequest {
    /// Data that is read must not be older than `staleness_bound`.
    absl::Time staleness_bound = absl::InfiniteFuture();

    /// Batch to use.
    Batch::View batch;
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
    /// For convenience, a `Derived::Entry` type should define a nested
    /// `OwningCache` alias to the `Derived` cache type, in order for
    /// `GetOwningCache` to return a pointer cast to the derived cache type.
    using OwningCache = AsyncCache;

    Entry() = default;

    /// Returns the `entry`.
    ///
    /// This is defined to simplify access in generic methods which work on
    /// either an Entry or a TransactionNode.
    template <typename DerivedEntry>
    friend std::enable_if_t<std::is_base_of_v<Entry, DerivedEntry>,
                            DerivedEntry&>
    GetOwningEntry(DerivedEntry& entry) {
      return entry;
    }

    /// Requests data according to constraints specified by `request`.
    ///
    /// \param must_not_be_known_to_be_stale Requests newer data if the existing
    ///     data is known to be out-of-date (e.g. due to a local write not
    ///     reflected in the cache), even if the existing data satisfies
    ///     `options.staleness_bound`.
    Future<const void> Read(AsyncCacheReadRequest request,
                            bool must_not_be_known_to_be_stale = true);

    /// Obtains an existing or new transaction node for the specified entry and
    /// transaction.  May also be used to obtain an implicit transaction node.
    ///
    /// \param entry The associated cache entry for which to obtain a
    ///     transaction node.
    /// \param transaction[in,out] Transaction associated with the entry.  If
    ///     non-null, must specify an explicit transaction, and an associated
    ///     transaction node will be created if one does not already exist.  In
    ///     this case, the `tranaction` pointer itself will not be modified.  An
    ///     implicit transaction node associated with a new implicit transaction
    ///     is requested by specifying `transaction` initially equally to
    ///     `nullptr`.  Upon return, `transaction` will hold an open transaction
    ///     reference to the associated implicit transaction.
    template <typename DerivedEntry>
    friend std::enable_if_t<
        std::is_base_of_v<Entry, DerivedEntry>,
        Result<OpenTransactionNodePtr<
            typename DerivedEntry::OwningCache::TransactionNode>>>
    GetTransactionNode(DerivedEntry& entry,
                       internal::OpenTransactionPtr& transaction) {
      TENSORSTORE_ASSIGN_OR_RETURN(auto node,
                                   entry.GetTransactionNodeImpl(transaction));
      return internal::static_pointer_cast<
          typename DerivedEntry::OwningCache::TransactionNode>(std::move(node));
    }

    template <typename DerivedEntry>
    friend std::enable_if_t<
        std::is_base_of_v<Entry, DerivedEntry>,
        Result<WriteLock<typename DerivedEntry::OwningCache::TransactionNode>>>
    GetWriteLockedTransactionNode(
        DerivedEntry& entry, const internal::OpenTransactionPtr& transaction)
        ABSL_NO_THREAD_SAFETY_ANALYSIS {
      using DerivedNode = typename DerivedEntry::OwningCache::TransactionNode;
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
    virtual void DoRead(AsyncCacheReadRequest request) = 0;

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

    // Below members should be treated as private:

    ReadState& LockReadState() ABSL_NO_THREAD_SAFETY_ANALYSIS {
      mutex().WriterLock();
      return read_request_state_.read_state;
    }

    Result<OpenTransactionNodePtr<TransactionNode>> GetTransactionNodeImpl(
        OpenTransactionPtr& transaction);

#ifdef TENSORSTORE_ASYNC_CACHE_DEBUG
    ~Entry();
#endif

    ReadRequestState read_request_state_;

    using TransactionTree =
        internal::intrusive_red_black_tree::Tree<TransactionNode,
                                                 TransactionNode>;

    /// Associated transactions, ordered by `TransactionState` pointer.
    TransactionTree transactions_;

    /// Pointer to transaction node for which writeback is currently being
    /// performed.  If not `nullptr`,
    /// committing_transaction_node_->pending_queue_`
    TransactionNode* committing_transaction_node_{nullptr};

    // AbslStringify is used to dump the Entry to the ABSL_LOG sink.
    // Example:
    //   ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG) << *entry
    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Entry& entry) {
      auto& owning_cache = GetOwningCache(const_cast<Entry&>(entry));
      absl::Format(&sink, "[%s: entry=%p, key=%s] ",
                   typeid(owning_cache).name(), &entry,
                   tensorstore::QuoteString(entry.key()));
    }
  };

  /// Base transaction node class.  Derived classes must define a nested
  /// `TransactionNode` type that inherits from this class and represents
  /// uncommitted modifications to an entry.
  class ABSL_LOCKABLE TransactionNode
      : public internal::TransactionState::Node,
        public internal::intrusive_red_black_tree::NodeBase<TransactionNode> {
   public:
    /// For convenience, a `Derived::TransactionNode` type should define a
    /// nested `OwningCache` alias to the `Derived` cache type, in order for
    /// `GetOwningEntry` and `GetOwningCache` to return a pointer cast to the
    /// derived cache type.
    using OwningCache = AsyncCache;

    /// A `Derived::TransactionNode` class should define either directly, or via
    /// inheritance, a constructor that takes a single `Derived::Entry&`
    /// argument specifying the associated cache entry.  This constructor will
    /// typically be invoked by the implementation of
    /// `DoAllocateTransactionNode`.
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

    void DebugAssertMutexHeld() {
#ifndef NDEBUG
      mutex_.AssertHeld();
#endif
    }

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
                            typename DerivedNode::OwningCache::Entry&>
    GetOwningEntry(DerivedNode& node) {
      return static_cast<typename DerivedNode::OwningCache::Entry&>(
          *static_cast<Cache::Entry*>(node.associated_data()));
    }

    /// Returns the cache with which the node is associated.
    ///
    /// This is defined as a friend rather than a member to allow the derived
    /// cache type to be returned.
    template <typename DerivedNode>
    friend std::enable_if_t<std::is_base_of_v<TransactionNode, DerivedNode>,
                            typename DerivedNode::OwningCache&>
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

    /// Requests a read state for this transaction node with the constraints
    /// specified by `request`.
    Future<const void> Read(AsyncCacheReadRequest request,
                            bool must_not_be_known_to_be_stale = true);

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
    virtual void DoRead(AsyncCacheReadRequest request) = 0;

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

    using ApplyReceiver = AnyReceiver<absl::Status, ReadState>;

    struct ApplyOptions {
      // Returned `ReadStateUpdate` must reflect an existing read state that is
      // current as of `staleness_bound`.
      absl::Time staleness_bound;

      enum ApplyMode {
        // The `stamp` field of the `ReadState` may be set to
        // `TimestampedStorageGeneration::Unconditional()` to indicate that this
        // transaction node does nothing (e.g. read-only and does not
        // validation).  In this case the `data` returned in the `ReadState`
        // will be ignored.  Otherwise, the `stamp` must not be
        // `StorageGeneration::Unknown()`, and should be marked "dirty" if the
        // data has been modified by this transaction node.
        kNormal,

        // The `data` returned in the `ReadState` must be valid even if this
        // transaction node does not modify it.  The `stamp` field of the
        // `ReadState` must not be
        // `TimestampedStorageGeneration::Unconditional()`.
        kSpecifyUnchanged,

        // The `data` returned in the `ReadState` will be ignored.  This option
        // is used if `DoApply` is called solely to validate the read state.  If
        // no validation is necessary, the `stamp` field of the `ReadState` may
        // be set to `TimestampedStorageGeneration::Unconditional()`.
        kValidateOnly,
      };

      ApplyMode apply_mode = kNormal;
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
    ///
    /// Must not call `set_cancel`.
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
      entry.mutex().WriterLock();
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

    absl::Mutex& mutex() { return mutex_; }

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

    // Tracks the current state after `PrepareForCommit` is called and this
    // node is in the `committing_transaction_node_` list.
    //
    // Note: `std::atomic` is used by while writes are protected by the entry's
    // mutex, reads may occur without holding the entry's mutex.
    std::atomic<PrepareForCommitState> prepare_for_commit_state_{
        PrepareForCommitState::kNone};

    // AbslStringify is used to dump the TransactionNode to the ABSL_LOG sink.
    // Example:
    //   ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG) << *node
    template <typename Sink>
    friend void AbslStringify(Sink& sink, const TransactionNode& node) {
      auto& entry = GetOwningEntry(node);
      auto& owning_cache = GetOwningCache(entry);
      const auto* txn = node.transaction();
      absl::Format(
          &sink, "[%s: entry=%p, key=%s, node=%p, transaction=%p%s, phase=%d] ",
          typeid(owning_cache).name(), &entry,
          tensorstore::QuoteString(entry.key()), &node, txn,
          txn == nullptr                ? ""
          : txn->implicit_transaction() ? " (implicit)"
                                        : " (explicit)",
          node.phase());
    }
  };

  template <typename EntryOrNode>
  struct ReadReceiver {
    EntryOrNode* entry_or_node;
    void set_value(AsyncCache::ReadState&& read_state) {
      entry_or_node->ReadSuccess(std::move(read_state));
    }

    void set_error(absl::Status error) {
      entry_or_node->ReadError(std::move(error));
    }

    void set_cancel() {
      // Not normally used.
      set_error(absl::CancelledError(""));
    }
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

  size_t BatchNestingDepth() const { return batch_nesting_depth_; }
  void SetBatchNestingDepth(size_t value) { batch_nesting_depth_ = value; }

  /// Allocates a new `TransactionNode`.
  ///
  /// Usually this method can be defined as:
  ///
  /// TransactionNode* DoAllocateTransactionNode(
  ///       AsyncCache::Entry& entry) final {
  ///   return new TransactionNode(static_cast<Entry&>(entry));
  /// }
  virtual TransactionNode* DoAllocateTransactionNode(Entry& entry) = 0;

 private:
  size_t batch_nesting_depth_ = 0;
};

using AsyncCacheReadRequest = AsyncCache::AsyncCacheReadRequest;

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_ASYNC_CACHE_H_
