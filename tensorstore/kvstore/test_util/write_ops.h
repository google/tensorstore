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

#ifndef TENSORSTORE_KVSTORE_TEST_UTIL_WRITE_OPS_H_
#define TENSORSTORE_KVSTORE_TEST_UTIL_WRITE_OPS_H_

#include <stddef.h>

#include <array>
#include <functional>
#include <string>
#include <string_view>

#include "absl/strings/cord.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/transaction.h"

namespace tensorstore {
namespace internal {

void TestKeyValueStoreWriteOps(const KvStore& store,
                               std::array<std::string, 3> key,
                               absl::Cord expected_value,
                               absl::Cord other_value);

void TestKeyValueStoreTransactionalWriteOps(const KvStore& store,
                                            TransactionMode transaction_mode,
                                            std::string key,
                                            absl::Cord expected_value,
                                            std::string_view operation);

struct TestConcurrentWritesOptions {
  size_t num_iterations = 100;
  size_t num_threads = 4;
  std::string key = "test";
  std::function<KvStore()> get_store;
};

void TestConcurrentWrites(const TestConcurrentWritesOptions& options);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TEST_UTIL_WRITE_OPS_H_