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

#include "tensorstore/kvstore/operations.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/any_sender.h"
#include "tensorstore/util/execution/collecting_sender.h"
#include "tensorstore/util/execution/future_sender.h"  // IWYU pragma: keep
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/execution/sender_util.h"
#include "tensorstore/util/execution/sync_flow_sender.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace kvstore {

Future<std::vector<Key>> ListFuture(Driver* driver, ListOptions options) {
  return tensorstore::MakeSenderFuture<std::vector<Key>>(
      tensorstore::internal::MakeCollectingSender<std::vector<Key>>(
          tensorstore::MakeSyncFlowSender(driver->List(options))));
}

Future<std::vector<Key>> ListFuture(const KvStore& store, ListOptions options) {
  return tensorstore::MakeSenderFuture<std::vector<Key>>(
      tensorstore::internal::MakeCollectingSender<std::vector<Key>>(
          tensorstore::MakeSyncFlowSender(
              kvstore::List(store, std::move(options)))));
}

Future<ReadResult> Read(const KvStore& store, std::string_view key,
                        ReadOptions options) {
  auto full_key = tensorstore::StrCat(store.path, key);
  if (store.transaction == no_transaction) {
    // Regular non-transactional read.
    return store.driver->Read(std::move(full_key), std::move(options));
  }
  if (!StorageGeneration::IsUnknown(options.if_equal)) {
    return absl::UnimplementedError(
        "if_equal condition not supported for transactional reads");
  }
  if (options.byte_range.inclusive_min || options.byte_range.exclusive_max) {
    return absl::UnimplementedError(
        "byte_range restriction not supported for transactional reads");
  }
  TransactionalReadOptions transactional_read_options;
  transactional_read_options.if_not_equal = std::move(options.if_not_equal);
  transactional_read_options.staleness_bound = options.staleness_bound;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto open_transaction,
      internal::AcquireOpenTransactionPtrOrError(store.transaction));
  size_t phase;
  return internal_kvstore::ReadViaExistingTransaction(
      store.driver.get(), open_transaction, phase, std::move(full_key),
      std::move(transactional_read_options));
}

Future<TimestampedStorageGeneration> Write(const KvStore& store,
                                           std::string_view key,
                                           std::optional<Value> value,
                                           WriteOptions options) {
  auto full_key = tensorstore::StrCat(store.path, key);
  if (store.transaction == no_transaction) {
    // Regular non-transactional write.
    return store.driver->Write(std::move(full_key), std::move(value),
                               std::move(options));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto open_transaction,
      internal::AcquireOpenTransactionPtrOrError(store.transaction));
  size_t phase;
  // Drop the write future; the transactional write completes as soon as the
  // write is applied to the transaction.
  auto future = internal_kvstore::WriteViaExistingTransaction(
      store.driver.get(), open_transaction, phase, std::move(full_key),
      std::move(value), std::move(options));
  if (future.ready()) {
    // An error must have occurred, since a successful write can complete until
    // the transaction is committed, and the transaction cannot commit while we
    // hold an open transaction reference.
    assert(!future.result().ok());
    return future;
  }
  // Just return a dummy stamp; the actual write won't complete until the
  // transaction is committed.
  return TimestampedStorageGeneration();
}

Future<TimestampedStorageGeneration> Delete(const KvStore& store,
                                            std::string_view key,
                                            WriteOptions options) {
  return Write(store, key, std::nullopt, std::move(options));
}

Future<const void> DeleteRange(const KvStore& store, KeyRange range) {
  range = KeyRange::AddPrefix(store.path, std::move(range));
  if (store.transaction == no_transaction) {
    return store.driver->DeleteRange(std::move(range));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto open_transaction,
      internal::AcquireOpenTransactionPtrOrError(store.transaction));
  return store.driver->TransactionalDeleteRange(open_transaction,
                                                std::move(range));
}

namespace {
void AddListOptionsPrefix(ListOptions& options, std::string_view path) {
  options.range = KeyRange::AddPrefix(path, std::move(options.range));
  options.strip_prefix_length += path.size();
}
}  // namespace

void List(const KvStore& store, ListOptions options,
          AnyFlowReceiver<absl::Status, Key> receiver) {
  if (store.transaction != no_transaction) {
    execution::submit(ErrorSender{absl::UnimplementedError(
                          "transactional list not supported")},
                      FlowSingleReceiver{std::move(receiver)});
    return;
  }
  AddListOptionsPrefix(options, store.path);
  store.driver->ListImpl(std::move(options), std::move(receiver));
}

AnyFlowSender<absl::Status, Key> List(const KvStore& store,
                                      ListOptions options) {
  if (store.transaction != no_transaction) {
    return ErrorSender{
        absl::UnimplementedError("transactional list not supported")};
  }
  AddListOptionsPrefix(options, store.path);
  return store.driver->List(std::move(options));
}

}  // namespace kvstore
}  // namespace tensorstore
