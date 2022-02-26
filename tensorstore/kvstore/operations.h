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

#include <cstddef>
#include <optional>
#include <string_view>

#include "absl/time/time.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace kvstore {

/// Read options for non-transactional reads.
/// See also `TransactionalReadOptions`
struct ReadOptions {
  /// The read is aborted if the generation associated with the stored `key`
  /// matches `if_not_equal`.  The special values of
  /// `StorageGeneration::Unknown()` (the default) or
  /// `StorageGeneration::NoValue()` disable this condition.
  StorageGeneration if_not_equal;

  /// Cached data may be used without validation if not older than
  /// `staleness_bound`.  Cached data older than `staleness_bound` must be
  /// validated before being returned.  A value of `absl::InfiniteFuture()` (the
  /// default) indicates that the result must be current as of the time the
  /// `Read` request was made, i.e. it is equivalent to specifying the value of
  /// `absl::Now()` just before invoking `Read`.
  absl::Time staleness_bound{absl::InfiniteFuture()};

  /// The read is aborted if the generation associated with `key` does not
  /// match `if_equal`.  This is primarily useful in conjunction with a
  /// `byte_range` request to ensure consistency.
  ///
  /// - The special value of `StorageGeneration::Unknown()` (the default)
  ///   disables this condition.
  ///
  /// - The special value of `StorageGeneration::NoValue()` specifies a
  ///   condition that the value not exist.  This condition is valid but of
  ///   limited use since the only possible read results are "not found" and
  ///   "aborted".
  StorageGeneration if_equal;

  /// Specifies the byte range.
  OptionalByteRangeRequest byte_range;
};

/// Read options for transactional reads.
/// See also `ReadOptions`
struct TransactionalReadOptions {
  /// The read is aborted if the generation associated with the stored `key`
  /// matches `if_not_equal`.  The special values of
  /// `StorageGeneration::Unknown()` (the default) or
  /// `StorageGeneration::NoValue()` disable this condition.
  StorageGeneration if_not_equal;

  /// Cached data may be used without validation if not older than
  /// `staleness_bound`.  Cached data older than `staleness_bound` must be
  /// validated before being returned.  A value of `absl::InfiniteFuture()` (the
  /// default) indicates that the result must be current as of the time the
  /// `Read` request was made, i.e. it is equivalent to specifying the value of
  /// `absl::Now()` just before invoking `Read`.
  absl::Time staleness_bound{absl::InfiniteFuture()};
};

struct WriteOptions {
  // Note: While it would be nice to use default member initializers to be
  // more explicit about what the default values are, doing so would trigger
  // Clang bug https://bugs.llvm.org/show_bug.cgi?id=36684.

  /// The write is aborted if the existing generation associated with the
  /// stored `key` does not match `if_equal`.
  ///
  /// - The special value of `StorageGeneration::Unknown()` (the default)
  ///   disables this condition.
  ///
  /// - The special value of `StorageGeneration::NoValue()` specifies a
  ///   condition that the `key` does not have an existing value.
  StorageGeneration if_equal;
};

/// Options for `List`.
struct ListOptions {
  /// Only keys in this range are emitted.
  KeyRange range;

  /// Length of prefix to strip from keys.
  size_t strip_prefix_length = 0;
};

/// Reads `path + key`.
Future<ReadResult> Read(const KvStore& store, std::string_view key,
                        ReadOptions options = {});

/// Writes `path + key`.
Future<TimestampedStorageGeneration> Write(const KvStore& store,
                                           std::string_view key,
                                           std::optional<Value> value,
                                           WriteOptions options = {});

/// Deletes `path + key`.
Future<TimestampedStorageGeneration> Delete(const KvStore& store,
                                            std::string_view key,
                                            WriteOptions options = {});

/// Deletes a range relative to `path`.
Future<void> DeleteRange(const KvStore& store, KeyRange range);

/// Lists keys relative to `path`.
void List(const KvStore& store, ListOptions options,
          AnyFlowReceiver<Status, Key> receiver);

AnyFlowSender<absl::Status, Key> List(const KvStore& store,
                                      ListOptions options);

Future<std::vector<Key>> ListFuture(const KvStore& store,
                                    ListOptions options = {});

/// Calls `List` and collects the results in an `std::vector`.
Future<std::vector<Key>> ListFuture(Driver* driver, ListOptions options = {});

inline Future<std::vector<Key>> ListFuture(const DriverPtr& driver,
                                           ListOptions options = {}) {
  return ListFuture(driver.get(), options);
}

}  // namespace kvstore
}  // namespace tensorstore

#endif  //  TENSORSTORE_KVSTORE_OPERATIONS_H_
