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

#ifndef TENSORSTORE_KVSTORE_TEST_UTIL_READ_OPS_H_
#define TENSORSTORE_KVSTORE_TEST_UTIL_READ_OPS_H_

#include <functional>
#include <string>
#include <string_view>

#include "absl/strings/cord.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

// Test read operations on `store`, where `key` is `expected_value`, and
// `missing_key` does not exist.
void TestKeyValueStoreReadOps(const KvStore& store, std::string key,
                              absl::Cord expected_value,
                              std::string missing_key);

void TestKeyValueStoreBatchReadOps(const KvStore& store, std::string key,
                                   absl::Cord expected_value);

void TestKeyValueStoreStalenessBoundOps(const KvStore& store, std::string key,
                                        absl::Cord value1, absl::Cord value2);

struct BatchReadGenericCoalescingTestOptions {
  internal_kvstore_batch::CoalescingOptions coalescing_options;
  std::string metric_prefix;
  bool has_file_open_metric = false;
};

void TestBatchReadGenericCoalescing(
    const KvStore& store, const BatchReadGenericCoalescingTestOptions& options);

struct TransactionalReadOpsParameters {
  KvStore store;
  std::string key;
  absl::Cord value1;
  absl::Cord value2;
  absl::Cord value3;
  bool write_outside_transaction;
  std::string_view write_operation_within_transaction;
  tensorstore::TransactionMode transaction_mode;

  // If set, `store` is an adapter kvstore. `write_to_other_node` writes the
  // specified key/value pair to the same backing storage as `store`, but using
  // a different write cache such that a separate `MultiPhase` instance will be
  // created.
  std::function<Result<TimestampedStorageGeneration>(std::string key,
                                                     absl::Cord value)>
      write_to_other_node;
};

void TestKeyValueStoreTransactionalReadOps(
    const TransactionalReadOpsParameters& p);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TEST_UTIL_READ_OPS_H_