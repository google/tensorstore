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

#ifndef TENSORSTORE_TRANSACTION_H_
#define TENSORSTORE_TRANSACTION_H_

/// \file
///
/// Transactions are used to stage groups of modifications (e.g. writes to
/// `TensorStore` objects) in memory before being aborted or committed.

#include <cstddef>
#include <cstdint>
#include <iosfwd>

#include "absl/status/status.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/transaction_impl.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Specifies the transaction mode.
///
/// \relates Transaction
enum TransactionMode : std::uint8_t {
  /// Indicates non-transactional semantics.  This is the default for operations
  /// performed without an explicit transaction.
  ///
  /// Non-transactional operations have the following properties:
  ///
  /// - Atomicity is NOT guaranteed
  ///
  /// - Consistency is guaranteed for most operations, with a few exceptions:
  ///   resize under certain conditions, and creating a TensorStore with
  ///   `OpenMode::delete_existing` under certain conditions.
  ///
  /// - Read and write isolation are NOT guaranteed
  ///
  /// - Durability is guaranteed.
  no_transaction_mode = 0,

  /// Writes are isolated and will not be visible to other readers until the
  /// transaction is committed.
  ///
  /// Isolated transactions have the following properties:
  ///
  /// - Atomicity is NOT guaranteed
  ///
  /// - Consistency has the same guarantees as for `no_transaction_mode`.
  ///
  /// - Write isolation is guaranteed prior to commit.  Once commit starts,
  ///   isolation is not guaranteed.  Read isolation is not guaranteed in
  ///   general, but metadata operations do guarantee read
  ///   isolation/consistency.
  ///
  /// - Durability is guaranteed.
  isolated = 1,

  /// In addition to the properties of `isolated`, writes are guaranteed to be
  /// committed atomically.  If an operation cannot satisfy this guarantee, it
  /// returns an error (immediately while the transaction is being prepared).
  ///
  /// Atomic-isolated transactions have the following properties:
  ///
  /// - Atomicity is guaranteed
  ///
  /// - Consistency is guaranteed
  ///
  /// - Write isolation is guaranteed.  Read isolation is not guaranteed, but
  ///   metadata operations do guarantee read isolation/consistency.
  ///
  /// - Durability is guaranteed.
  atomic_isolated = 3,
};

/// Prints a string representation of the transaction mode to an `std::ostream`.
///
/// \id TransactionMode
/// \relates TransactionMode
std::ostream& operator<<(std::ostream& os, TransactionMode mode);

/// Shared handle to a transaction.
///
/// \ingroup core
class Transaction {
 public:
  /// Special type that indicates a null transaction.
  ///
  /// This is used via the `tensorstore::no_transaction_mode` constant.
  ///
  /// Implicitly converts to `TransactionMode::no_transaction_mode` and a null
  /// `Transaction`, and can be used for comparisons to either `TransactionMode`
  /// or `Transaction`.
  struct no_transaction_t {
    explicit no_transaction_t() = default;

    /// Implicitly converts to `TransactionMode::no_transaction_mode`.
    ///
    /// \id TransactionMode
    constexpr operator TransactionMode() const {
      return TransactionMode::no_transaction_mode;
    }

    // Forwards to `ApplyTensorStoreTransaction`, in order to support
    // `ChainResult` and "pipeline" ``operator|``.
    template <typename X>
    decltype(ApplyTensorStoreTransaction(std::declval<X&&>(),
                                         std::declval<Transaction>()))
    operator()(X&& x) const {
      return ApplyTensorStoreTransaction(static_cast<X&&>(x),
                                         Transaction(no_transaction_t{}));
    }
  };

  /// Creates a null transaction.
  ///
  /// This indicates non-transactional semantics.
  ///
  /// This allows the `tensorstore::no_transaction` constant to implicitly
  /// convert to a null transaction.
  ///
  /// \post `this-mode() == no_transaction`
  /// \id no_transaction
  constexpr Transaction(no_transaction_t) {}

  /// Creates a new transaction with the specified mode.
  ///
  /// \id mode
  explicit Transaction(TransactionMode mode);

  /// Returns the transaction mode.
  TransactionMode mode() const {
    return state_ ? state_->mode_ : TransactionMode::no_transaction_mode;
  }

  /// Returns true if this transaction guarantees atomicity.
  bool atomic() const { return state_ && state_->atomic(); }

  /// Returns `true` if the transaction has been aborted.
  ///
  /// Once this becomes `true`, it never becomes `false`.  This is mutually
  /// exclusive with `commit_started()`.
  bool aborted() const { return state_ && state_->aborted(); }

  /// Returns `true` if the transaction commit has started.
  ///
  /// Once this becomes `true`, it never becomes `false`.  This is mutually
  /// exclusive with `aborted()`.
  ///
  /// To determine whether the commit has completed, use `future()`.
  bool commit_started() const { return state_ && state_->commit_started(); }

  /// Aborts the transaction (has no effect if already aborted or committed).
  ///
  /// If there are outstanding operations on the transaction that are still in
  /// progress, the transaction won't actually abort until they complete.
  ///
  /// To wait until all outstanding operations have completed and the
  /// transaction is fully aborted, call `future().result()`.
  ///
  /// Once the transaction is fully aborted, `future()` becomes ready with an
  /// error status of `absl::StatusCode::kCancelled`.
  void Abort() const;

  /// Synchronously commits.  Blocks until the commit completes successfully or
  /// with an error.
  absl::Status Commit() const { return CommitAsync().status(); }

  /// Commits the transaction (has no effect if already committed or aborted).
  ///
  /// If there are outstanding operations on the transaction that are still in
  /// progress, commit won't actually start until they complete.
  ///
  /// This operation is asynchronous.  To synchronously commit and wait until
  /// the commit completes, use `Commit()`.  The returned result indicates
  /// whether the commit was successful.
  ///
  /// \returns `future()`, which becomes ready when the commit completes (either
  ///     successfully or with an error).
  Future<const void> CommitAsync() const;

  /// Creates a write barrier.  Guarantees that subsequent writes are not
  /// committed before any prior write.
  ///
  /// This can be used to ensure that metadata updates are committed before data
  /// updates that are dependent on the metadata.  In particular, this is used
  /// to implement certain resize operations and the `OpenMode::delete_existing`
  /// open mode when using `isolated` transactions.
  ///
  /// If `atomic()`, this has no effect since all writes are committed
  /// atomically.
  ///
  /// For example::
  ///
  ///     auto transaction = tensorstore::Transaction(tensorstore::isolated);
  ///     tensorstore::Write(data_array | transaction | ..., ...).value();
  ///     tensorstore::Write(data_array | transaction | ..., ...).value();
  ///     // Ensure writes to `data_array` complete successfully before the
  ///     // write to `done_indicator_array` below.
  ///     transaction.Barrier();
  ///     tensorstore::Write(done_indicator_array | transaction, ...).value();
  ///     transaction.Commit().value();
  void Barrier() const;

  /// Returns a `Future` that becomes ready when the transaction is committed or
  /// aborted.  Forcing the returned `Future` is equivalent to calling
  /// `CommitAsync`.
  ///
  /// If the transaction was aborted, the `Future` completes with an error
  /// status of `absl::StatusCode::kCancelled`.
  Future<const void> future() const {
    if (!state_) return MakeReadyFuture();
    return state_->future_;
  }

  /// Returns an estimate of the number of bytes of memory currently consumed by
  /// the transaction.
  std::size_t total_bytes() const {
    if (state_) return state_->total_bytes();
    return 0;
  }

  /// Checks if `a` and `b` refer to the same transaction state, or are both
  /// null.
  friend bool operator==(const Transaction& a, const Transaction& b) {
    return a.state_ == b.state_;
  }
  friend bool operator!=(const Transaction& a, const Transaction& b) {
    return !(a == b);
  }
  friend bool operator==(const Transaction& a, no_transaction_t b) {
    return !a.state_;
  }
  friend bool operator==(no_transaction_t a, const Transaction& b) {
    return !b.state_;
  }
  friend bool operator!=(const Transaction& a, no_transaction_t b) {
    return !(a == b);
  }
  friend bool operator!=(no_transaction_t a, const Transaction& b) {
    return !(a == b);
  }

  // Forwards to `ApplyTensorStoreTransaction`, in order to support
  // `ChainResult` and "pipeline" ``operator|``.
  template <typename X>
  decltype(ApplyTensorStoreTransaction(std::declval<X&&>(),
                                       std::declval<Transaction>()))
  operator()(X&& x) const& {
    return ApplyTensorStoreTransaction(static_cast<X&&>(x), *this);
  }
  template <typename X>
  decltype(ApplyTensorStoreTransaction(std::declval<X&&>(),
                                       std::declval<Transaction>()))
  operator()(X&& x) && {
    return ApplyTensorStoreTransaction(static_cast<X&&>(x), std::move(*this));
  }

 private:
  friend class internal::TransactionState;
  internal::TransactionState::CommitPtr state_;
};

/// Special value that indicates non-transactional semantics.
///
/// \relates Transaction
constexpr inline Transaction::no_transaction_t no_transaction{};

namespace internal {
inline TransactionState* TransactionState::get(const Transaction& t) {
  return t.state_.get();
}
inline Result<OpenTransactionPtr> AcquireOpenTransactionPtrOrError(
    const Transaction& t) {
  if (auto* state = TransactionState::get(t)) {
    return state->AcquireOpenPtrOrError();
  }
  return OpenTransactionPtr{};
}

inline Transaction TransactionState::ToTransaction(OpenPtr transaction) {
  if (transaction) {
    BlockCommitPtrTraits::decrement(transaction.get());
  }
  Transaction t(no_transaction);
  t.state_.reset(transaction.release(), adopt_object_ref);
  return t;
}

inline Transaction TransactionState::ToTransaction(CommitPtr transaction) {
  Transaction t(no_transaction);
  t.state_ = std::move(transaction);
  return t;
}

inline TransactionState::CommitPtr TransactionState::ToCommitPtr(
    Transaction t) {
  return std::move(t.state_);
}

}  // namespace internal

}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::Transaction)

#endif  // TENSORSTORE_TRANSACTION_H_
