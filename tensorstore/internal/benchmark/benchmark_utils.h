// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_BENCHMARK_BENCHMARK_UTILS_H_
#define TENSORSTORE_INTERNAL_BENCHMARK_BENCHMARK_UTILS_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <utility>

#include "tensorstore/batch.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_benchmark {

// Groups write operations into transactions of configurable size.
//
// Callers must invoke Next() for each write operation and Flush()
// after all writes are issued. Transactions are committed
// asynchronously via LinkError on the provided promise.
//
// tx_size semantics:
//   0  : no transactions — Next() always returns `base_`.
//  -1  : single transaction for all operations.
//  >0  : groups of tx_size operations per transaction.
//
// The template parameter `Store` must support transaction binding via the
// pipe operator `|` with `Transaction` (e.g. `KvStore` or `TensorStore`)
// and must be copyable/movable.
template <typename Store>
class TransactionBatcher {
 public:
  explicit TransactionBatcher(Store base, int64_t tx_size, size_t total_count)
      : base_(std::move(base)),
        tx_size_(tx_size),
        // When tx_size <= 0, set batch_limit_ = total_count so that the
        // modulo check triggers only once (creating a single
        // transaction). max(1, ...) guards against total_count == 0.
        batch_limit_((tx_size <= 0) ? std::max(size_t{1}, total_count)
                                    : static_cast<size_t>(tx_size)),
        current_(base_) {}

  // Returns the Store to use for the next write.
  // Commits the previous transaction if a new batch starts.
  Result<Store> Next(const Promise<void>& promise) {
    if (tx_size_ == 0) {
      return base_;
    }
    if (count_ % batch_limit_ == 0) {
      if (txn_ != no_transaction) {
        LinkError(promise, txn_.CommitAsync());
      }
      txn_ = Transaction(TransactionMode::isolated);
      transaction_count_++;
      auto result = base_ | txn_;
      if (!result.ok()) {
        count_++;
        return result.status();
      }
      current_ = *result;
    }
    count_++;
    return current_;
  }

  // Commits the final transaction.
  void Flush(const Promise<void>& promise) {
    if (txn_ != no_transaction) {
      LinkError(promise, txn_.CommitAsync());
    }
  }

  size_t num_transactions() const { return transaction_count_; }

 private:
  Store base_;
  int64_t tx_size_;
  size_t batch_limit_;
  size_t count_ = 0;
  size_t transaction_count_ = 0;
  Transaction txn_{no_transaction};
  Store current_;
};

// Groups read operations into batches of configurable size.
//
// Callers must invoke NextBatch() for each read and release the returned
// `batch_to_release` when non-null. Flush() returns the final batch, which
// the caller must release.
//
// batch_size semantics:
//   0  : no batching — returns Batch::no_batch.
//  -1  : single batch for all operations.
//  >0  : groups of batch_size operations per batch.
class ReadBatcher {
 public:
  explicit ReadBatcher(int64_t batch_size, size_t total_count)
      : batch_size_(batch_size),
        // When batch_size <= 0, set batch_limit_ = total_count so that the
        // modulo check triggers only once (creating a single batch).
        // max(1, ...) guards against total_count == 0.
        batch_limit_((batch_size <= 0) ? std::max(size_t{1}, total_count)
                                       : static_cast<size_t>(batch_size)) {}

  // Returns the Batch to use and optionally a batch to release.
  struct NextBatchResult {
    Batch batch_to_use = Batch::no_batch;
    Batch batch_to_release = Batch::no_batch;
  };

  NextBatchResult NextBatch() {
    NextBatchResult result;
    if (batch_size_ == 0) {
      // Note: count_ is not incremented when batch_size_ == 0 because
      // the modulo check is never reached.
      return result;
    }
    if (count_ % batch_limit_ == 0) {
      result.batch_to_release = std::move(batch_);
      batch_ = Batch::New();
      batch_count_++;
    }
    result.batch_to_use = batch_;
    count_++;
    return result;
  }

  // Releases the final batch.
  Batch Flush() { return std::move(batch_); }

  size_t num_batches() const { return batch_count_; }

 private:
  int64_t batch_size_;
  size_t batch_limit_;
  size_t count_ = 0;
  size_t batch_count_ = 0;
  Batch batch_ = Batch::no_batch;
};

}  // namespace internal_benchmark
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_BENCHMARK_BENCHMARK_UTILS_H_
