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

#ifndef TENSORSTORE_KVSTORE_TEST_UTIL_LIST_OPS_H_
#define TENSORSTORE_KVSTORE_TEST_UTIL_LIST_OPS_H_

#include <stddef.h>

#include <functional>
#include <string>

#include "absl/functional/function_ref.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/transaction.h"

namespace tensorstore {
namespace internal {

/// Tests List on `store`, which should be empty.
void TestKeyValueStoreList(const KvStore& store, bool match_size = true);

struct TransactionalListOpsParameters {
  KvStore store;
  std::string keys[5];
  bool write_outside_transaction;
  tensorstore::TransactionMode transaction_mode;

  // If set, invokes callback with another store using the same backing storage
  // as `store`, but using a different write cache such that a separate
  // `MultiPhase` instance will be created.
  std::function<void(absl::FunctionRef<void(const KvStore& store)>)>
      get_other_store;

  bool match_size;
};

void TestKeyValueStoreTransactionalListOps(
    const TransactionalListOpsParameters& p);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TEST_UTIL_LIST_OPS_H_
