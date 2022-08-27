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

#ifndef TENSORSTORE_KVSTORE_READ_MODIFY_WRITE_H_
#define TENSORSTORE_KVSTORE_READ_MODIFY_WRITE_H_

#include "absl/status/status.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/sender.h"

namespace tensorstore {
namespace kvstore {

/// `ReadModifyWriteTarget` and `ReadModifyWriteSource` serve as the two
/// halves of the bidirectional communication protocol used to implement
/// transactional read-modify-write operations.
///
/// A `ReadModifyWriteSource` is bound to a `ReadModifyWriteTarget` when it is
/// created by calling `KvsSetTarget`, which must be called exactly once.
/// Subsequently, the `ReadModifyWriteSource` may call methods of the
/// `ReadModifyWriteTarget`, and the `ReadModifyWriteTarget` may call methods
/// of the `ReadModifyWriteSource`.  Both the `ReadModifyWriteSource` and the
/// `ReadModifyWriteTarget` are implicitly associated with a particular
/// `Transaction` and phase.  They remain bound until the transaction phase is
/// either committed or aborted (and must remain valid until then).
///
/// `ReadModifyWriteSource` is implemented by
/// `KvsBackedCache::TransactionNode` and connects the `AsyncCache` interface
/// to the transactional `KeyValueStore` interface.
///
/// `ReadModifyWriteTarget` is implemented by `ReadModifyWriteEntry` defined
/// in `transaction.h`.
class ReadModifyWriteTarget {
 public:
  using TransactionalReadOptions =
      ::tensorstore::kvstore::TransactionalReadOptions;
  using ReadReceiver = AnyReceiver<absl::Status, ReadResult>;

  /// Reads from the KeyValueStore.
  ///
  /// The result should reflect the current committed state as of the
  /// specified `staleness_bound`, and should also reflect any previous
  /// read-modify-write operations made in the current transaction.
  virtual void KvsRead(TransactionalReadOptions options,
                       ReadReceiver receiver) = 0;

  /// Returns `true` if `Read` returns the same result as calling
  /// `KeyValueStore::Read` directly without a transaction.  If there are no
  /// prior read-modify-write operations of the key in the current
  /// transaction, it is safe for this to return `true`.  Otherwise, this must
  /// return `false`.  It is always safe for this to return `false`, but that
  /// may lead to unnecessary reads of cached data.
  virtual bool KvsReadsCommitted() = 0;

 protected:
  ~ReadModifyWriteTarget() = default;
};

class ReadModifyWriteSource {
 public:
  using WritebackResultReceiver =
      AnyReceiver<absl::Status, TimestampedStorageGeneration>;
  using WritebackReceiver = AnyReceiver<absl::Status, ReadResult>;

  enum WritebackMode {
    /// Request a value to writeback.  If this read-modify-write operation
    /// would leave the existing value unchanged, it is permitted to return a
    /// `ReadResult` with a state of `kUnspecified`.
    kNormalWriteback,

    /// Same as `kNormalWriteback`, but a known writeback value must always be
    /// specified.  It must not return a `ReadResult` in a
    /// state of `kUnspecified` even if this read-modify-write operation would
    /// leave the existing value unchanged.
    ///
    /// This is specified when the writeback was requested due to a read from
    /// a subsequent read-modify-write operation layered on top of this
    /// operation.
    kSpecifyUnchangedWriteback,

    /// Requests that any validation of the existing read state that would be
    /// done as part of writeback still be performed, but a writeback value
    /// need not be specified.  Any errors returned to the `WritebackReceiver`
    /// will be propagated, but a successful `ReadResult` will
    /// be ignored.  In particular, the implementation is free to provide a
    /// default-constructed `ReadResult`.
    ///
    /// This is specified when a subsequent read-modify-write operation
    /// layered on top specifies a new writeback value that is not conditioned
    /// on the value from this operation (e.g. an unconditional delete or
    /// overwrite).
    kValidateOnly,
  };

  /// Specifies options for requesting a writeback value.
  ///
  /// If `if_not_equal` is specified (i.e. not
  /// `StorageGeneration::Unknown()`), requests that a value to writeback need
  /// only be computed if it would be conditioned on a different storage
  /// generation than the one specified by `if_not_equal`.  If the writeback
  /// value would be conditioned on the `StorageGeneration` specified by
  /// `if_not_equal`, a `ReadResult` with a state of
  /// `kUnspecified` (but with an updated `stamp`) may be returned instead.
  ///
  /// If `staleness_bound < absl::Now()` is specified, the value to writeback
  /// may be conditioned on a cached existing read state, as long as it is not
  /// older than `staleness_bound`.
  struct WritebackOptions : public TransactionalReadOptions {
    /// Specifies additional constraints the writeback value that must be
    /// provided.
    WritebackMode writeback_mode;
  };

  /// Binds this to its associated `ReadModifyWriteTarget`.
  virtual void KvsSetTarget(ReadModifyWriteTarget& target) = 0;

  /// Indicates that the meaning of a `StorageGeneration` returned by a
  /// previous call to `ReadModifyWriteTarget::KvsRead` may have changed, and
  /// that the return value of `ReadModifyWriteTarget::KvsReadsCommitted` may
  /// have changed.  Any cached values should be invalidated.
  ///
  /// This will be called by the bound `ReadModifyWriteTarget` only after a
  /// prior phase of the transaction is committed.
  virtual void KvsInvalidateReadState() = 0;

  /// Requests the value to writeback.
  ///
  /// This is called by the `ReadModifyWriteTarget` either during commit, or
  /// in response to a read request by a subsequent read-modify-write
  /// operation layered on top of this operation.
  virtual void KvsWriteback(WritebackOptions options,
                            WritebackReceiver receiver) = 0;

  /// Indicates that the most recently-provided writeback value was
  /// successfully committed.
  ///
  /// If `new_stamp.generation` is not `StorageGeneration::Unknown()`, it
  /// indicates that the writeback value was committed with the specified
  /// generation, and may safely be cached.
  ///
  /// If `new_stamp.generation` is `StorageGeneration::Unknown()`, it
  /// indicates that a subsequent read-modify-write operation in the same
  /// transaction phase may have made further changes, and the writeback value
  /// provided by this `ReadModifyWriteSource` must not be cached.
  ///
  /// This is a terminal method: no further methods will be called after this
  /// one.
  virtual void KvsWritebackSuccess(TimestampedStorageGeneration new_stamp) = 0;

  /// Indicates that an error occurred during commit of this read-modify-write
  /// operation.  The actual error information is set as an error on the
  /// associated transaction.
  ///
  /// This is a terminal method: no further methods will be called after this
  /// one.
  virtual void KvsWritebackError() = 0;

  /// Indicates that another read-modify-write operation has been layered on
  /// top of this operation (in the same transaction).  If any further
  /// modifications are to be made, a new read-modify-write operation must be
  /// started.
  virtual void KvsRevoke() = 0;

 protected:
  ~ReadModifyWriteSource() = default;
};

}  // namespace kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_READ_MODIFY_WRITE_H_
