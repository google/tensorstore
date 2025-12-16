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

#ifndef TENSORSTORE_KVSTORE_TEST_UTIL_REGISTER_H_
#define TENSORSTORE_KVSTORE_TEST_UTIL_REGISTER_H_

#include <stddef.h>

#include <functional>
#include <string>

#include "absl/functional/function_ref.h"
#include "tensorstore/kvstore/kvstore.h"

namespace tensorstore {
namespace internal {

struct KeyValueStoreOpsTestParameters {
  // Name of test suite.
  std::string test_name;

  // Function that invokes a callback with the store.
  //
  // The `get_store` function can perform any necessary cleanup after the
  // callback returns.
  std::function<void(absl::FunctionRef<void(const KvStore& store)>)> get_store;

  // For kvstore adapters, returns an adapter on top of the base store.
  //
  // For non-kvstore adapters, should be left as a null function.
  std::function<void(const KvStore& base,
                     absl::FunctionRef<void(const KvStore& store)>)>
      get_store_adapter;

  // Minimum size of value to use for read/write tests.
  size_t value_size = 0;

  // Maps arbitrary strings (which are nonetheless valid file paths) to keys in
  // the format expected by `store`. For stores that support file paths as keys,
  // `get_key` can simply be the identity function. This function must ensure
  // that a given input key always maps to the same output key, and distinct
  // input keys always map to distinct output keys.
  std::function<std::string(std::string key)> get_key;

  // Perform transactional tests using an atomic_isolated transaction rather
  // than an isolated transaction.
  bool atomic_transaction = false;

  // Include DeleteRange tests.
  bool test_delete_range = true;

  // Include CopyRange tests.
  bool test_copy_range = false;

  // Include List tests.
  bool test_list = true;

  // If `test_list == true`, test list without an extra prefix. This fails if
  // keys remain across `get_store` calls.
  bool test_list_without_prefix = true;

  // Indicates if list is expected to return sizes.
  bool list_match_size = true;

  // If `test_list == true`, test listing with the specified prefix also.
  std::string test_list_prefix = "p/";

  // Test transactional list operations.
  bool test_transactional_list = true;

  // Test special characters in the key.
  bool test_special_characters = true;
};

// Registers a suite of tests according to `params`.
void RegisterKeyValueStoreOpsTests(KeyValueStoreOpsTestParameters params);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TEST_UTIL_REGISTER_H_