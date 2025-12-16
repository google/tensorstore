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

#include "tensorstore/kvstore/test_util/register.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorstore/internal/testing/dynamic.h"
#include "tensorstore/kvstore/driver.h"  // IWYU pragma: keep
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util/copy_ops.h"
#include "tensorstore/kvstore/test_util/delete_ops.h"
#include "tensorstore/kvstore/test_util/list_ops.h"
#include "tensorstore/kvstore/test_util/read_ops.h"
#include "tensorstore/kvstore/test_util/write_ops.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {
namespace {

using ::tensorstore::internal_testing::RegisterGoogleTestCaseDynamically;

// Functor object for TransactionalReadOps tests.
struct TransactionalReadOpsFunctor {
  std::function<void(absl::FunctionRef<void(const KvStore& store)>)> get_store;
  std::string key;
  std::function<void(const KvStore& base,
                     absl::FunctionRef<void(const KvStore& store)>)>
      get_store_adapter;
  bool write_outside_transaction;
  std::string_view write_operation_within_transaction;
  tensorstore::TransactionMode transaction_mode;
  absl::Cord value1;
  absl::Cord value2;
  absl::Cord value3;
  bool write_to_other_node;

  void operator()() const {
    get_store([this](const KvStore& store) {
      TransactionalReadOpsParameters p;
      p.store = store;
      p.key = key;
      p.write_outside_transaction = write_outside_transaction;
      p.write_operation_within_transaction = write_operation_within_transaction;
      p.transaction_mode = transaction_mode;
      p.value1 = value1;
      p.value2 = value2;
      p.value3 = value3;
      if (write_to_other_node) {
        TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base, store.base());
        p.write_to_other_node =
            [=, base = std::move(base)](
                std::string key,
                absl::Cord value) -> Result<TimestampedStorageGeneration> {
          Result<TimestampedStorageGeneration> result{};
          get_store_adapter(base, [&](const KvStore& other) {
            result = kvstore::Write(other, std::move(key), std::move(value))
                         .result();
          });
          return result;
        };
      }
      TestKeyValueStoreTransactionalReadOps(p);
    });
  }
};

// Functor object for TransactionalListOps tests.
struct TransactionalListOpsFunctor {
  std::function<void(absl::FunctionRef<void(const KvStore& store)>)> get_store;
  std::function<void(const KvStore& base,
                     absl::FunctionRef<void(const KvStore& store)>)>
      get_store_adapter;
  bool test_list_without_prefix;
  std::string test_list_prefix;
  std::function<std::string(std::string key)> get_key;
  bool write_outside_transaction;
  tensorstore::TransactionMode transaction_mode;
  bool write_to_other_node;
  bool list_match_size;

  void operator()() const {
    get_store([this](KvStore store) {
      TransactionalListOpsParameters p;
      if (!test_list_without_prefix) {
        store.path += test_list_prefix;
      }
      p.store = store;
      p.keys[0] = get_key("0");
      p.keys[1] = get_key("1");
      p.keys[2] = get_key("2");
      p.keys[3] = get_key("3");
      p.keys[4] = get_key("4");
      p.write_outside_transaction = write_outside_transaction;
      p.transaction_mode = transaction_mode;
      if (write_to_other_node) {
        TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base, store.base());
        p.get_other_store = [=, base = std::move(base)](auto callback) {
          get_store_adapter(base, std::move(callback));
        };
      }
      p.match_size = list_match_size;
      TestKeyValueStoreTransactionalListOps(p);
    });
  }
};

}  // namespace

void RegisterKeyValueStoreOpsTests(KeyValueStoreOpsTestParameters params) {
  if (!params.get_key) {
    params.get_key = [](std::string key) { return absl::StrCat("key_", key); };
  }
  absl::Cord expected_value("_kvstore_value_");
  if (params.value_size > expected_value.size()) {
    expected_value.Append(
        std::string(params.value_size - expected_value.size(), '*'));
  }
  absl::Cord other_value("._-=+=-_.");
  if (params.value_size > other_value.size()) {
    other_value.Append(
        std::string(params.value_size - other_value.size(), '-'));
  }
  absl::Cord other_value2("ABCDEFGHIJKLMNOP");
  if (params.value_size > other_value2.size()) {
    other_value2.Append(
        std::string(params.value_size - other_value2.size(), '+'));
  }

  std::vector<std::pair<std::string, tensorstore::TransactionMode>>
      transaction_modes;
  transaction_modes.emplace_back("NoTransaction",
                                 tensorstore::no_transaction_mode);
  transaction_modes.emplace_back("Isolated", tensorstore::isolated);
  transaction_modes.emplace_back(
      "IsolatedRepeatableRead",
      tensorstore::isolated | tensorstore::repeatable_read);
  if (params.atomic_transaction) {
    transaction_modes.emplace_back("AtomicIsolated",
                                   tensorstore::atomic_isolated);
    transaction_modes.emplace_back(
        "AtomicIsolatedRepeatableRead",
        tensorstore::atomic_isolated | tensorstore::repeatable_read);
  }

  for (const auto& txn_mode_info : transaction_modes) {
    const auto& transaction_mode_name = txn_mode_info.first;
    const auto transaction_mode = txn_mode_info.second;
    RegisterGoogleTestCaseDynamically(
        params.test_name,
        tensorstore::StrCat("ReadOps/", transaction_mode_name),
        [get_store = params.get_store, get_key = params.get_key,
         transaction_mode, expected_value] {
          get_store([&](const KvStore& store) {
            auto txn = tensorstore::Transaction(transaction_mode);
            TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto txn_store, store | txn);
            std::string missing_key = get_key("missing");
            kvstore::Delete(txn_store, missing_key)
                .result()
                .status()
                .IgnoreError();

            std::string key = get_key("read");
            auto write_result =
                kvstore::Write(txn_store, key, expected_value).result();
            ASSERT_THAT(write_result,
                        MatchesRegularTimestampedStorageGeneration());

            tensorstore::internal::TestKeyValueStoreReadOps(
                txn_store, key, expected_value, missing_key);

            kvstore::Delete(txn_store, key).result().status().IgnoreError();
          });
        });

    RegisterGoogleTestCaseDynamically(
        params.test_name,
        tensorstore::StrCat("BatchReadOps/", transaction_mode_name),
        [get_store = params.get_store, key = params.get_key("read"),
         transaction_mode] {
          get_store([&](const KvStore& store) {
            absl::Cord longer_expected_value;
            for (size_t i = 0; i < 4096; ++i) {
              char x = static_cast<char>(i);
              longer_expected_value.Append(std::string_view(&x, 1));
            }

            auto txn = tensorstore::Transaction(transaction_mode);
            TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto txn_store, store | txn);
            ASSERT_THAT(
                kvstore::Write(txn_store, key, longer_expected_value).result(),
                MatchesRegularTimestampedStorageGeneration());

            tensorstore::internal::TestKeyValueStoreBatchReadOps(
                txn_store, key, longer_expected_value);

            kvstore::Delete(txn_store, key).result().status().IgnoreError();
          });
        });

    if (transaction_mode != no_transaction_mode &&
        !(transaction_mode & repeatable_read)) {
      for (std::string_view operation :
           {"Unconditional", "MatchingCondition", "MatchingConditionAfterWrite",
            "NonMatchingCondition", "NonMatchingConditionAfterWrite"}) {
        RegisterGoogleTestCaseDynamically(
            params.test_name,
            tensorstore::StrCat("TransactionalWriteOps/", operation),
            [get_store = params.get_store, get_key = params.get_key,
             expected_value, transaction_mode, operation] {
              get_store([&](const KvStore& store) {
                TestKeyValueStoreTransactionalWriteOps(
                    store, transaction_mode, get_key("write1"), expected_value,
                    operation);
              });
            });
      }
    }
  }

  RegisterGoogleTestCaseDynamically(
      params.test_name, "WriteOps",
      [get_store = params.get_store, get_key = params.get_key, expected_value,
       other_value] {
        get_store([&](const KvStore& store) {
          TestKeyValueStoreWriteOps(
              store, {get_key("write1"), get_key("write2"), get_key("write3")},
              expected_value, other_value);
        });
      });

  RegisterGoogleTestCaseDynamically(
      params.test_name, "DeleteOps",
      [get_store = params.get_store, get_key = params.get_key, expected_value] {
        get_store([&](const KvStore& store) {
          TestKeyValueStoreDeleteOps(store,
                                     {get_key("del1"), get_key("del2"),
                                      get_key("del3"), get_key("del4")},
                                     expected_value);
        });
      });

  RegisterGoogleTestCaseDynamically(
      params.test_name, "StalenessBoundOps",
      [get_store = params.get_store, key = params.get_key("stale"),
       expected_value, other_value] {
        get_store([&](const KvStore& store) {
          TestKeyValueStoreStalenessBoundOps(store, key, expected_value,
                                             other_value);
        });
      });

  if (params.test_delete_range) {
    RegisterGoogleTestCaseDynamically(params.test_name, "DeleteRange",
                                      [get_store = params.get_store] {
                                        get_store([&](const KvStore& store) {
                                          TestKeyValueStoreDeleteRange(store);
                                        });
                                      });
    RegisterGoogleTestCaseDynamically(params.test_name, "DeletePrefix",
                                      [get_store = params.get_store] {
                                        get_store([&](const KvStore& store) {
                                          TestKeyValueStoreDeletePrefix(store);
                                        });
                                      });
    RegisterGoogleTestCaseDynamically(
        params.test_name, "DeleteRangeToEnd", [get_store = params.get_store] {
          get_store([&](const KvStore& store) {
            TestKeyValueStoreDeleteRangeToEnd(store);
          });
        });
    RegisterGoogleTestCaseDynamically(
        params.test_name, "DeleteRangFromBeginning",
        [get_store = params.get_store] {
          get_store([&](const KvStore& store) {
            TestKeyValueStoreDeleteRangeFromBeginning(store);
          });
        });
  }
  if (params.test_copy_range) {
    RegisterGoogleTestCaseDynamically(
        params.test_name, "CopyRange", [get_store = params.get_store] {
          get_store(
              [&](const KvStore& store) { TestKeyValueStoreCopyRange(store); });
        });
  }
  if (params.test_list) {
    if (params.test_list_without_prefix) {
      RegisterGoogleTestCaseDynamically(
          params.test_name, "List",
          [get_store = params.get_store,
           list_match_size = params.list_match_size] {
            get_store([&](const KvStore& store) {
              TestKeyValueStoreList(store, list_match_size);
            });
          });
    }
    if (!params.test_list_prefix.empty()) {
      RegisterGoogleTestCaseDynamically(
          params.test_name, "ListWithPrefix",
          [get_store = params.get_store,
           list_match_size = params.list_match_size,
           test_list_prefix = params.test_list_prefix] {
            get_store([&](KvStore store) {
              store.path += test_list_prefix;
              TestKeyValueStoreList(store, list_match_size);
            });
          });
    }
  }
  if (params.test_special_characters) {
    RegisterGoogleTestCaseDynamically(
        params.test_name, "SpecialCharacters",
        [get_store = params.get_store,
         special_key = params.get_key("subdir/a!b@c$d"), expected_value] {
          get_store([&](const KvStore& store) {
            kvstore::Delete(store, special_key).result().status().IgnoreError();

            auto write_result =
                kvstore::Write(store, special_key, expected_value).result();
            ASSERT_THAT(write_result,
                        MatchesRegularTimestampedStorageGeneration());

            auto read_result = kvstore::Read(store, special_key).result();
            EXPECT_THAT(read_result,
                        MatchesKvsReadResult(expected_value, testing::_));

            kvstore::Delete(store, special_key).result().status().IgnoreError();
          });
        });
  }
  // Transactional read ops tests
  {
    for (std::string_view write_operation_within_transaction : {
             "Unmodified",
             "DeleteRange",
             "Delete",
             "WriteUnconditionally",
             "WriteWithFalseCondition",
             "WriteWithTrueCondition",
         }) {
      for (const auto& txn_mode_info : transaction_modes) {
        const auto& transaction_mode_name = txn_mode_info.first;
        const auto transaction_mode = txn_mode_info.second;
        if (transaction_mode == no_transaction_mode) continue;

        for (bool write_outside_transaction : {false, true}) {
          auto register_with_write_to_other_node =
              [&](bool write_to_other_node) {
                RegisterGoogleTestCaseDynamically(
                    params.test_name,
                    tensorstore::StrCat(
                        "TransactionalReadOps/", transaction_mode_name, "/",
                        write_outside_transaction ? "WithCommittedValue"
                                                  : "WithoutCommittedValue",
                        "/", write_to_other_node ? "WriteToOtherNode/" : "",
                        write_operation_within_transaction),
                    TransactionalReadOpsFunctor{
                        params.get_store, params.get_key("read"),
                        params.get_store_adapter, write_outside_transaction,
                        write_operation_within_transaction, transaction_mode,
                        expected_value, other_value, other_value2,
                        write_to_other_node});
              };
          register_with_write_to_other_node(false);
          if (params.get_store_adapter) {
            register_with_write_to_other_node(true);
          }
        }
      }
    }
  }
  // Transactional list ops tests
  if (params.test_transactional_list) {
    for (const auto& txn_mode_info : transaction_modes) {
      const auto& transaction_mode_name = txn_mode_info.first;
      const auto transaction_mode = txn_mode_info.second;
      if (transaction_mode == no_transaction_mode) continue;
      if (transaction_mode & repeatable_read) continue;
      for (bool write_outside_transaction : {false, true}) {
        auto register_with_write_to_other_node = [&](bool write_to_other_node) {
          RegisterGoogleTestCaseDynamically(
              params.test_name,
              tensorstore::StrCat(
                  "TransactionalListOps/", transaction_mode_name, "/",
                  write_outside_transaction ? "WithCommittedValue"
                                            : "WithoutCommittedValue",
                  write_to_other_node ? "/WriteToOtherNode/" : ""),
              TransactionalListOpsFunctor{
                  params.get_store, params.get_store_adapter,
                  params.test_list_without_prefix, params.test_list_prefix,
                  params.get_key, write_outside_transaction, transaction_mode,
                  write_to_other_node, params.list_match_size});
        };
        register_with_write_to_other_node(false);
        if (params.get_store_adapter) {
          register_with_write_to_other_node(true);
        }
      }
    }
  }
}

}  // namespace internal
}  // namespace tensorstore
