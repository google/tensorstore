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

#ifndef TENSORSTORE_KVSTORE_OPERATIONS_H_
#define TENSORSTORE_KVSTORE_OPERATIONS_H_

#include <stddef.h>
#include <stdint.h>

#include <limits>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/batch.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/any_sender.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace kvstore {

/// Specifies constraints on the generation for read operations.
///
/// \relates KvStore
struct ReadGenerationConditions {
  /// The read is aborted if the generation associated with the stored key
  /// matches `if_not_equal`.  The special values of
  /// `StorageGeneration::Unknown()` (the default) or
  /// `StorageGeneration::NoValue()` disable this condition.
  StorageGeneration if_not_equal;

  /// The read is aborted if the generation associated with the stored key does
  /// not match `if_equal`.  This is primarily useful in conjunction with a
  /// `ReadOptions::byte_range` request to ensure consistency.
  ///
  /// - The special value of `StorageGeneration::Unknown()` (the default)
  ///   disables this condition.
  ///
  /// - The special value of `StorageGeneration::NoValue()` specifies a
  ///   condition that the value not exist.  This condition is valid but of
  ///   limited use since the only possible read results are "not found" and
  ///   "aborted".
  StorageGeneration if_equal;

  /// Returns `true` if `generation` satisfies the constraints.
  bool Matches(const StorageGeneration& generation) const {
    assert(!StorageGeneration::IsUnknown(generation));
    return generation != if_not_equal &&
           (StorageGeneration::IsUnknown(if_equal) || generation == if_equal);
  }

  /// Indicates if any constraints are specified.
  explicit operator bool() const {
    return static_cast<bool>(if_not_equal) || static_cast<bool>(if_equal);
  }

  friend bool operator==(const ReadGenerationConditions& a,
                         const ReadGenerationConditions& b) {
    return a.if_not_equal == b.if_not_equal && a.if_equal == b.if_equal;
  }

  friend bool operator!=(const ReadGenerationConditions& a,
                         const ReadGenerationConditions& b) {
    return !(a == b);
  }

  template <typename H>
  friend H AbslHashValue(H h, const ReadGenerationConditions& x) {
    return H::combine(std::move(h), x.if_not_equal, x.if_equal);
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const ReadGenerationConditions& x);
};

/// Read options for non-transactional reads.
///
/// See also `TransactionalReadOptions`.
///
/// \relates KvStore
struct ReadOptions {
  /// Specifies conditions for the read.
  ReadGenerationConditions generation_conditions;

  /// Cached data may be used without validation if not older than
  /// `staleness_bound`.  Cached data older than `staleness_bound` must be
  /// validated before being returned.  A value of `absl::InfiniteFuture()` (the
  /// default) indicates that the result must be current as of the time the
  /// `Read` request was made, i.e. it is equivalent to specifying the value of
  /// `absl::Now()` just before invoking `Read`.
  absl::Time staleness_bound{absl::InfiniteFuture()};

  /// Specifies the byte range.
  OptionalByteRangeRequest byte_range;

  /// Optional batch to use.
  Batch batch{no_batch};
};

struct TransactionalReadGenerationConditions {
  /// The read is aborted if the generation associated with the stored ``key``
  /// matches `if_not_equal`.  The special values of
  /// `StorageGeneration::Unknown()` (the default) or
  /// `StorageGeneration::NoValue()` disable this condition.
  StorageGeneration if_not_equal;
};

/// Read options for transactional reads.
///
/// See also `ReadOptions`
///
/// \relates KvStore
struct TransactionalReadOptions {
  /// Specifies conditions for the read.
  TransactionalReadGenerationConditions generation_conditions;

  /// Cached data may be used without validation if not older than
  /// `staleness_bound`.  Cached data older than `staleness_bound` must be
  /// validated before being returned.  A value of `absl::InfiniteFuture()` (the
  /// default) indicates that the result must be current as of the time the
  /// `Read` request was made, i.e. it is equivalent to specifying the value of
  /// `absl::Now()` just before invoking `Read`.
  absl::Time staleness_bound{absl::InfiniteFuture()};

  /// Optional batch to use.
  Batch batch{no_batch};
};

struct WriteGenerationConditions {
  // Note: While it would be nice to use default member initializers to be
  // more explicit about what the default values are, doing so would trigger
  // Clang bug https://bugs.llvm.org/show_bug.cgi?id=36684.

  /// The write is aborted if the existing generation associated with the
  /// stored ``key`` does not match `if_equal`.
  ///
  /// - The special value of `StorageGeneration::Unknown()` (the default)
  ///   disables this condition.
  ///
  /// - The special value of `StorageGeneration::NoValue()` specifies a
  ///   condition that the ``key`` does not have an existing value.
  StorageGeneration if_equal;

  /// Returns `true` if `generation` matches the constraint.
  bool Matches(const StorageGeneration& generation) const {
    assert(!StorageGeneration::IsUnknown(generation));
    return StorageGeneration::IsUnknown(if_equal) || generation == if_equal;
  }

  /// Equivalent to `Matches(StorageGeneration::NoValue())`.
  bool MatchesNoValue() const {
    return StorageGeneration::IsUnknown(if_equal) ||
           StorageGeneration::IsNoValue(if_equal);
  }
};

/// Options for `Write`.
///
/// \relates KvStore
struct WriteOptions {
  /// Specifies conditions for the write.
  WriteGenerationConditions generation_conditions;
};

/// Options for `ListFuture`.
///
/// \relates KvStore
struct ListOptions {
  /// Only keys in this range are emitted.
  KeyRange range;

  /// Length of prefix to strip from keys.
  size_t strip_prefix_length = 0;

  /// Staleness bound on list results.
  absl::Time staleness_bound = absl::InfiniteFuture();
};

/// Return value for List operations
struct ListEntry {
  Key key;

  int64_t size;

  bool has_size() const { return size >= 0; }

  template <typename T>
  static int64_t checked_size(T checked_size) {
    return (checked_size >= 0 &&
            checked_size < std::numeric_limits<int64_t>::max())
               ? static_cast<int64_t>(checked_size)
               : -1;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ListEntry& entry) {
    absl::Format(&sink, "%s", entry.key);
  }
};

using ListReceiver = AnyFlowReceiver<absl::Status, ListEntry>;
using ListSender = AnyFlowSender<absl::Status, ListEntry>;

/// Options for `CopyRange`.
///
/// \relates KvStore
struct CopyRangeOptions {
  /// Only keys in this range are copied.
  KeyRange source_range;

  /// Staleness bound for reading from source.
  absl::Time source_staleness_bound = absl::InfiniteFuture();
};

/// Attempts to read the value for the key `store.path + key`.
///
/// .. note::
///
///    A missing value is not considered an error.
///
/// \param store `KvStore` from which to read.
/// \param key The key to read, interpreted as a suffix to be appended to
///     `store.path`.
/// \param options Specifies options for reading.
/// \returns A Future that resolves when the read completes successfully or with
///     an error.
/// \relates KvStore
Future<ReadResult> Read(const KvStore& store, std::string_view key,
                        ReadOptions options = {});

/// Performs an optionally-conditional write.
///
/// Atomically updates or deletes the value stored for `store.path + key`
/// subject to the conditions specified in `options`.
///
/// `WriteCommitted` behaves the same as `Write` for non-transactional writes.
/// For transactional writes, the future returned by `WriteCommitted` becomes
/// ready only once the transaction is committed or aborted, and if the write is
/// successful, contains the actual `TimestampedStorageGeneration`; in contrast,
/// the future returned by `Write` becomes ready immediately to reflect the fact
/// that the value can immediately be read back in the context of the
/// transaction.
///
/// \param store `KvStore` into which to perform the write operation.
/// \param key The key to write or delete, interpreted as a suffix to be
///     appended to `store.path`.
/// \param value The value to write, or `std::nullopt` to delete.
/// \param options Specifies options for writing.
/// \returns A Future that resolves to the generation corresponding to the new
///     value on success, or to `StorageGeneration::Unknown()` if the conditions
///     in `options` are not satisfied.
/// \relates KvStore
Future<TimestampedStorageGeneration> Write(const KvStore& store,
                                           std::string_view key,
                                           std::optional<Value> value,
                                           WriteOptions options = {});
Future<TimestampedStorageGeneration> WriteCommitted(const KvStore& store,
                                                    std::string_view key,
                                                    std::optional<Value> value,
                                                    WriteOptions options = {});

/// Performs an optionally-conditional delete.
///
/// Equivalent to `Write(store, key, std::nullopt, options)` or
/// `WriteCommitted(store, key, std::nullopt, options)`.
///
/// \param store `KvStore` from which to delete the key.
/// \param key Key to delete, interpreted as a suffix to be appended to
///     `store.path`.
/// \param options Specifies options for deleting.
/// \relates KvStore
Future<TimestampedStorageGeneration> Delete(const KvStore& store,
                                            std::string_view key,
                                            WriteOptions options = {});
Future<TimestampedStorageGeneration> DeleteCommitted(const KvStore& store,
                                                     std::string_view key,
                                                     WriteOptions options = {});

/// Deletes all keys in the specified range.
///
/// This operation is not guaranteed to be atomic with respect to other
/// operations affecting keys in `range`.  If there are concurrent writes to
/// keys in `range`, this operation may fail with an error or indicate success
/// despite not having removed the newly-added keys.
///
/// \param store `KvStore` from which to delete keys.
/// \param range Range of keys to delete, relative to `store.path`.
/// \returns A Future that becomes ready when the operation has completed
///     either successfully or with an error.
/// \relates KvStore
Future<const void> DeleteRange(const KvStore& store, KeyRange range);

Future<const void> DeleteRange(Driver* driver,
                               const internal::OpenTransactionPtr& transaction,
                               KeyRange range);

/// Copies a range from `source` to `target`.
///
/// \param source Source store.
/// \param target Target store.
/// \param options Specifies options for copying.
/// \relates KvStore
Future<const void> ExperimentalCopyRange(const KvStore& source,
                                         const KvStore& target,
                                         CopyRangeOptions options = {});

// Lists keys relative to `path`.
void List(const KvStore& store, ListOptions options, ListReceiver receiver);

ListSender List(const KvStore& store, ListOptions options);

/// Lists the keys in a kvstore.
///
/// \param store `KvStore` from which to list keys.
/// \param options List options.  The `options.range` is interpreted relative to
///     `store.path`.
/// \relates KvStore
Future<std::vector<ListEntry>> ListFuture(const KvStore& store,
                                          ListOptions options = {});

// Calls `List` and collects the results in an `std::vector`.
Future<std::vector<ListEntry>> ListFuture(Driver* driver,
                                          ListOptions options = {});

inline Future<std::vector<ListEntry>> ListFuture(const DriverPtr& driver,
                                                 ListOptions options = {}) {
  return ListFuture(driver.get(), options);
}

}  // namespace kvstore
}  // namespace tensorstore

#endif  //  TENSORSTORE_KVSTORE_OPERATIONS_H_
