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

#ifndef TENSORSTORE_KVSTORE_TEST_UTIL_DELETE_OPS_H_
#define TENSORSTORE_KVSTORE_TEST_UTIL_DELETE_OPS_H_

#include <stddef.h>

#include <array>
#include <string>

#include "absl/strings/cord.h"
#include "tensorstore/kvstore/kvstore.h"

namespace tensorstore {
namespace internal {
void TestKeyValueStoreDeleteOps(const KvStore& store,
                                std::array<std::string, 4> key,
                                absl::Cord expected_value);

void TestKeyValueStoreDeleteRange(const KvStore& store);

void TestKeyValueStoreDeletePrefix(const KvStore& store);

void TestKeyValueStoreDeleteRangeToEnd(const KvStore& store);

void TestKeyValueStoreDeleteRangeFromBeginning(const KvStore& store);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TEST_UTIL_DELETE_OPS_H_
