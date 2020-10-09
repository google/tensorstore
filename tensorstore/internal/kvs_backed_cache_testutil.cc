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

#include "tensorstore/internal/kvs_backed_cache_testutil.h"

#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/async_cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/kvs_backed_cache.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/kvstore/key_value_store_testutil.h"
#include "tensorstore/transaction_impl.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/execution.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/sender.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

void KvsBackedTestCache::Entry::DoDecode(std::optional<absl::Cord> value,
                                         DecodeReceiver receiver) {
  if (value && value->Flatten().find('Z') != std::string_view::npos) {
    return execution::set_error(
        receiver, absl::FailedPreconditionError("existing value contains Z"));
  }
  execution::set_value(
      receiver, std::make_shared<absl::Cord>(value.value_or(absl::Cord())));
}

void KvsBackedTestCache::Entry::DoEncode(
    std::shared_ptr<const absl::Cord> data,
    UniqueWriterLock<AsyncCache::TransactionNode> lock,
    EncodeReceiver receiver) {
  lock.unlock();
  if (!data) {
    execution::set_value(receiver, std::nullopt);
  } else {
    execution::set_value(receiver, *data);
  }
}

Result<OpenTransactionNodePtr<KvsBackedTestCache::TransactionNode>>
KvsBackedTestCache::Entry::Modify(const OpenTransactionPtr& transaction,
                                  bool clear, std::string_view append_value) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node, GetWriteLockedTransactionNode(*this, transaction));
  if (clear) {
    node->cleared = true;
    node->value.clear();
  }
  node->value += append_value;
  return node.unlock();
}

Result<OpenTransactionNodePtr<KvsBackedTestCache::TransactionNode>>
KvsBackedTestCache::Entry::Validate(const OpenTransactionPtr& transaction,
                                    Validator validator) {
  while (true) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto node, GetWriteLockedTransactionNode(*this, transaction));
    if (node->cleared || !node->value.empty()) {
      // Can't add `validator` to node with existing modifications.
      node->Revoke();
      continue;
    }
    node->validators.push_back(std::move(validator));
    return node.unlock();
  }
}

Future<absl::Cord> KvsBackedTestCache::Entry::ReadValue(
    OpenTransactionPtr transaction, absl::Time staleness_bound) {
  struct Receiver {
    Promise<absl::Cord> promise_;

    void set_cancel() { TENSORSTORE_UNREACHABLE; }
    void set_error(absl::Status status) { promise_.SetResult(status); }
    void set_value(AsyncCache::ReadState update,
                   UniqueWriterLock<AsyncCache::TransactionNode> lock) {
      lock.unlock();
      promise_.SetResult(*static_cast<const ReadData*>(update.data.get()));
    }
  };
  auto [promise, future] = PromiseFuturePair<absl::Cord>::Make();
  TENSORSTORE_ASSIGN_OR_RETURN(auto node,
                               GetTransactionNode(*this, transaction));
  TransactionNode::ApplyOptions options;
  options.staleness_bound = staleness_bound;
  node->DoApply(options, Receiver{std::move(promise)});
  return future;
}

void KvsBackedTestCache::TransactionNode::DoApply(ApplyOptions options,
                                                  ApplyReceiver receiver) {
  if (options.validate_only && validators.empty()) {
    execution::set_value(
        receiver, ReadState{{}, TimestampedStorageGeneration::Unconditional()},
        UniqueWriterLock<TransactionNode>{});
    return;
  }
  auto continuation = [this, receiver = std::move(receiver)](
                          ReadyFuture<const void> future) mutable {
    auto& r = future.result();
    if (!r) {
      return execution::set_error(receiver, r.status());
    }
    if (value.find('Z') != std::string::npos) {
      return execution::set_error(
          receiver, absl::InvalidArgumentError("new value contains Z"));
    }
    AsyncCache::ReadState read_state;
    if (!IsUnconditional()) {
      read_state = AsyncCache::ReadLock<void>(*this).read_state();
    } else {
      read_state.stamp = TimestampedStorageGeneration::Unconditional();
    }
    absl::Cord encoded;
    if (read_state.data) {
      encoded = *static_cast<const absl::Cord*>(read_state.data.get());
    }
    for (const auto& validator : validators) {
      TENSORSTORE_RETURN_IF_ERROR(validator(encoded),
                                  execution::set_error(receiver, _));
    }
    if (cleared) {
      encoded = absl::Cord();
    }
    if (cleared || !value.empty()) {
      read_state.stamp.generation.MarkDirty();
    }
    encoded.Append(value);
    read_state.data = std::make_shared<absl::Cord>(encoded);
    return execution::set_value(receiver, std::move(read_state),
                                UniqueWriterLock<TransactionNode>{});
  };
  if ([&] {
        UniqueWriterLock lock(*this);
        return !IsUnconditional();
      }()) {
    this->Read(options.staleness_bound)
        .ExecuteWhenReady(std::move(continuation));
    return;
  }
  continuation(MakeReadyFuture());
}

CachePtr<KvsBackedTestCache> KvsBackedTestCache::Make(
    KeyValueStore::Ptr kvstore, CachePool::StrongPtr pool,
    std::string_view cache_identifier) {
  if (!pool) {
    pool = CachePool::Make(CachePool::Limits{});
  }
  return pool->GetCache<KvsBackedTestCache>(cache_identifier, [&] {
    return std::make_unique<KvsBackedTestCache>(kvstore);
  });
}

class RandomOperationTester {
 public:
  explicit RandomOperationTester(
      KeyValueStore::Ptr kvstore,
      std::function<std::string(std::string)> get_key)
      : kvstore(kvstore) {
    for (auto key : {"x", "y"}) {
      caches.push_back(KvsBackedTestCache::Make(kvstore, {}, key));
    }
    for (size_t i = 0; i < 10; ++i) {
      keys.push_back(get_key(std::string{static_cast<char>('a' + i)}));
    }
  }

  void SimulateDeleteRange(const KeyRange& range) {
    if (range.empty()) return;
    map.erase(map.lower_bound(range.inclusive_min),
              range.exclusive_max.empty()
                  ? map.end()
                  : map.lower_bound(range.exclusive_max));
  }

  void SimulateWrite(const std::string& key, bool clear,
                     const std::string& append) {
    auto& value = map[key];
    if (clear) value.clear();
    value += append;
  }

  using Map = std::map<std::string, std::string>;

  std::string SampleKey() {
    return keys[absl::Uniform(bitgen, 0u, keys.size())];
  }

  std::string SampleKeyOrEmpty() {
    size_t key_index = absl::Uniform(bitgen, 0u, keys.size() + 1);
    if (key_index == 0) return "";
    return keys[key_index - 1];
  }

  void PerformRandomAction(OpenTransactionPtr transaction) {
    if (absl::Bernoulli(bitgen, barrier_probability)) {
      transaction->Barrier();
      TENSORSTORE_LOG("Barrier");
    }
    if (absl::Bernoulli(bitgen, write_probability)) {
      const auto& key = SampleKey();
      const auto& cache = caches[absl::Uniform(bitgen, 0u, caches.size())];
      bool clear = absl::Bernoulli(bitgen, clear_probability);
      std::string append = tensorstore::StrCat(", ", ++write_number);
      SimulateWrite(key, clear, append);
      TENSORSTORE_LOG("Write: key=", QuoteString(key),
                      ", cache_key=", cache->cache_identifier(),
                      ", clear=", clear, ", append=\"", append, "\"");
      TENSORSTORE_EXPECT_OK(
          GetCacheEntry(cache, key)->Modify(transaction, clear, append));
    } else {
      KeyRange range{SampleKeyOrEmpty(), SampleKeyOrEmpty()};
      TENSORSTORE_LOG("DeleteRange: ", range);
      SimulateDeleteRange(range);
      TENSORSTORE_EXPECT_OK(
          kvstore->TransactionalDeleteRange(transaction, range));
    }
  }

  size_t num_actions = 100;

  void PerformRandomActions() {
    auto transaction = Transaction(tensorstore::isolated);
    {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto open_transaction,
          tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
      for (size_t i = 0; i < 100; ++i) {
        PerformRandomAction(open_transaction);
      }
    }
    transaction.CommitAsync().IgnoreFuture();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto kvstore_cord_map,
                                     tensorstore::internal::GetMap(kvstore));
    EXPECT_THAT(Map(kvstore_cord_map.begin(), kvstore_cord_map.end()),
                ::testing::ElementsAreArray(map));
  }

  KeyValueStore::Ptr kvstore;
  Map map;

  std::vector<std::string> keys;
  std::vector<CachePtr<KvsBackedTestCache>> caches;

  // TODO(jbms): Use absl::BitGen when it supports deterministic seeding.
  std::minstd_rand bitgen{internal::GetRandomSeedForTest(
      "TENSORSTORE_RANDOM_OPERATION_TESTER_SEED")};

  double write_probability = 0.8;
  double clear_probability = 0.5;
  double barrier_probability = 0.05;
  size_t write_number = 0;
};

void RegisterKvsBackedCacheBasicTransactionalTest(
    const KvsBackedCacheBasicTransactionalTestOptions& options) {
  using ::testing::Pointee;
  using ::testing::StrEq;
  auto suite_name = options.test_name + "/KvsBackedCacheBasicTransactionalTest";
  RegisterGoogleTestCaseDynamically(
      suite_name, "ReadCached",
      [=] {
        auto kvstore = options.get_store();
        auto cache = KvsBackedTestCache::Make(kvstore);
        auto get_key = options.get_key_getter();
        auto a_key = get_key("a");
        auto entry = GetCacheEntry(cache, a_key);

        // Read missing value.
        TENSORSTORE_EXPECT_OK(entry->Read(absl::InfinitePast()));
        EXPECT_THAT(AsyncCache::ReadLock<absl::Cord>(*entry).data(),
                    Pointee(absl::Cord("")));

        TENSORSTORE_EXPECT_OK(kvstore->Write(a_key, absl::Cord("abc")));

        // Read stale value from cache.
        TENSORSTORE_EXPECT_OK(entry->Read(absl::InfinitePast()));
        EXPECT_THAT(AsyncCache::ReadLock<absl::Cord>(*entry).data(),
                    Pointee(absl::Cord("")));

        // Read when there is new data.
        TENSORSTORE_EXPECT_OK(entry->Read(absl::Now()).result());
        EXPECT_THAT(AsyncCache::ReadLock<absl::Cord>(*entry).data(),
                    Pointee(absl::Cord("abc")));

        // Read when there is not new data.
        auto read_time = absl::Now();
        auto read_generation =
            AsyncCache::ReadLock<absl::Cord>(*entry).stamp().generation;
        TENSORSTORE_EXPECT_OK(entry->Read(absl::Now()).result());
        EXPECT_THAT(AsyncCache::ReadLock<absl::Cord>(*entry).data(),
                    Pointee(absl::Cord("abc")));
        EXPECT_THAT(AsyncCache::ReadLock<absl::Cord>(*entry).stamp(),
                    tensorstore::internal::MatchesTimestampedStorageGeneration(
                        read_generation, ::testing::Ge(read_time)));
      },
      TENSORSTORE_LOC);

  RegisterGoogleTestCaseDynamically(
      suite_name, "WriteAfterRead",
      [=] {
        auto kvstore = options.get_store();
        auto cache = KvsBackedTestCache::Make(kvstore);
        auto get_key = options.get_key_getter();
        auto a_key = get_key("a");
        TENSORSTORE_EXPECT_OK(kvstore->Write(a_key, absl::Cord("abc")));
        auto entry = GetCacheEntry(cache, a_key);
        TENSORSTORE_EXPECT_OK(entry->Read(absl::InfinitePast()).result());

        {
          auto transaction = Transaction(tensorstore::atomic_isolated);
          {
            TENSORSTORE_ASSERT_OK_AND_ASSIGN(
                auto open_transaction,
                tensorstore::internal::AcquireOpenTransactionPtrOrError(
                    transaction));
            TENSORSTORE_ASSERT_OK(
                entry->Modify(open_transaction, false, "def"));
          }
          TENSORSTORE_EXPECT_OK(transaction.CommitAsync().result());
        }
        EXPECT_THAT(AsyncCache::ReadLock<absl::Cord>(*entry).data(),
                    Pointee(absl::Cord("abcdef")));
        EXPECT_THAT(
            kvstore->Read(a_key).result(),
            MatchesKvsReadResult(
                absl::Cord("abcdef"),
                AsyncCache::ReadLock<absl::Cord>(*entry).stamp().generation));

        {
          TENSORSTORE_EXPECT_OK(kvstore->Delete(a_key));
          auto transaction = Transaction(tensorstore::atomic_isolated);
          {
            TENSORSTORE_ASSERT_OK_AND_ASSIGN(
                auto open_transaction,
                tensorstore::internal::AcquireOpenTransactionPtrOrError(
                    transaction));
            TENSORSTORE_ASSERT_OK(
                entry->Modify(open_transaction, false, "ghi"));
          }
          TENSORSTORE_EXPECT_OK(transaction.CommitAsync().result());
        }
        EXPECT_THAT(AsyncCache::ReadLock<absl::Cord>(*entry).data(),
                    Pointee(absl::Cord("ghi")));
        EXPECT_THAT(
            kvstore->Read(a_key).result(),
            MatchesKvsReadResult(
                absl::Cord("ghi"),
                AsyncCache::ReadLock<absl::Cord>(*entry).stamp().generation));

        {
          auto transaction = Transaction(tensorstore::atomic_isolated);
          {
            TENSORSTORE_ASSERT_OK_AND_ASSIGN(
                auto open_transaction,
                tensorstore::internal::AcquireOpenTransactionPtrOrError(
                    transaction));
            TENSORSTORE_ASSERT_OK(entry->Modify(open_transaction, false, ""));
          }
          TENSORSTORE_EXPECT_OK(transaction.CommitAsync().result());
          // No change.
          EXPECT_THAT(AsyncCache::ReadLock<absl::Cord>(*entry).data(),
                      Pointee(absl::Cord("ghi")));
          EXPECT_THAT(
              kvstore->Read(a_key).result(),
              MatchesKvsReadResult(
                  absl::Cord("ghi"),
                  AsyncCache::ReadLock<absl::Cord>(*entry).stamp().generation));
        }
      },
      TENSORSTORE_LOC);

  RegisterGoogleTestCaseDynamically(
      suite_name, "DecodeErrorDuringRead",
      [=] {
        auto kvstore = options.get_store();
        auto cache = KvsBackedTestCache::Make(kvstore);
        auto get_key = options.get_key_getter();
        auto a_key = get_key("a");
        TENSORSTORE_EXPECT_OK(kvstore->Write(a_key, absl::Cord("ghi")));
        auto entry = GetCacheEntry(cache, a_key);
        TENSORSTORE_EXPECT_OK(entry->Read(absl::InfinitePast()).result());

        {
          auto old_read_generation =
              AsyncCache::ReadLock<absl::Cord>(*entry).stamp();
          TENSORSTORE_EXPECT_OK(kvstore->Write(a_key, absl::Cord("Z")));
          EXPECT_THAT(entry->Read(absl::Now()).result(),
                      MatchesStatus(
                          absl::StatusCode::kFailedPrecondition,
                          StrCat("Error reading ", kvstore->DescribeKey(a_key),
                                 ": existing value contains Z")));
          // Read state is not modified.
          EXPECT_THAT(AsyncCache::ReadLock<absl::Cord>(*entry).data(),
                      Pointee(absl::Cord("ghi")));
          EXPECT_EQ(old_read_generation,
                    AsyncCache::ReadLock<absl::Cord>(*entry).stamp());

          TENSORSTORE_EXPECT_OK(kvstore->Write(a_key, absl::Cord("ccc")));
          TENSORSTORE_EXPECT_OK(entry->Read(absl::Now()));
          EXPECT_THAT(AsyncCache::ReadLock<absl::Cord>(*entry).data(),
                      Pointee(absl::Cord("ccc")));
        }
      },
      TENSORSTORE_LOC);

  RegisterGoogleTestCaseDynamically(
      suite_name, "DecodeErrorDuringWriteback",
      [=] {
        auto kvstore = options.get_store();
        auto cache = KvsBackedTestCache::Make(kvstore);
        auto get_key = options.get_key_getter();
        auto a_key = get_key("a");
        TENSORSTORE_EXPECT_OK(kvstore->Write(a_key, absl::Cord("Z")));
        auto entry = GetCacheEntry(cache, a_key);

        auto transaction = Transaction(tensorstore::atomic_isolated);
        {
          TENSORSTORE_ASSERT_OK_AND_ASSIGN(
              auto open_transaction,
              tensorstore::internal::AcquireOpenTransactionPtrOrError(
                  transaction));
          TENSORSTORE_ASSERT_OK(entry->Modify(open_transaction, false, "def"));
        }
        EXPECT_THAT(
            transaction.CommitAsync().result(),
            MatchesStatus(absl::StatusCode::kFailedPrecondition,
                          StrCat("Error reading ", kvstore->DescribeKey(a_key),
                                 ": existing value contains Z")));
        EXPECT_THAT(kvstore->Read(a_key).result(),
                    MatchesKvsReadResult(absl::Cord("Z")));
      },
      TENSORSTORE_LOC);

  RegisterGoogleTestCaseDynamically(
      suite_name, "UnconditionalWriteback",
      [=] {
        auto kvstore = options.get_store();
        auto cache = KvsBackedTestCache::Make(kvstore);
        auto get_key = options.get_key_getter();
        auto a_key = get_key("a");
        TENSORSTORE_EXPECT_OK(kvstore->Write(a_key, absl::Cord("ghi")));
        auto entry = GetCacheEntry(cache, a_key);
        TENSORSTORE_EXPECT_OK(entry->Read(absl::InfinitePast()).result());

        {
          auto transaction = Transaction(tensorstore::atomic_isolated);
          {
            TENSORSTORE_ASSERT_OK_AND_ASSIGN(
                auto open_transaction,
                tensorstore::internal::AcquireOpenTransactionPtrOrError(
                    transaction));
            TENSORSTORE_ASSERT_OK(entry->Modify(open_transaction, true, "def"));
          }
          TENSORSTORE_EXPECT_OK(transaction.CommitAsync().result());
        }

        EXPECT_THAT(AsyncCache::ReadLock<absl::Cord>(*entry).data(),
                    Pointee(absl::Cord("def")));
        EXPECT_THAT(
            kvstore->Read(a_key).result(),
            MatchesKvsReadResult(
                absl::Cord("def"),
                AsyncCache::ReadLock<absl::Cord>(*entry).stamp().generation));
      },
      TENSORSTORE_LOC);

  RegisterGoogleTestCaseDynamically(
      suite_name, "EncodeError",
      [=] {
        auto kvstore = options.get_store();
        auto cache = KvsBackedTestCache::Make(kvstore);
        auto get_key = options.get_key_getter();
        auto a_key = get_key("a");
        auto entry = GetCacheEntry(cache, a_key);
        TENSORSTORE_EXPECT_OK(kvstore->Write(a_key, absl::Cord("abc")));
        TENSORSTORE_EXPECT_OK(entry->Read(absl::InfinitePast()).result());

        {
          auto transaction = Transaction(tensorstore::atomic_isolated);
          {
            TENSORSTORE_ASSERT_OK_AND_ASSIGN(
                auto open_transaction,
                tensorstore::internal::AcquireOpenTransactionPtrOrError(
                    transaction));
            TENSORSTORE_ASSERT_OK(entry->Modify(open_transaction, true, "Z"));
          }
          EXPECT_THAT(transaction.CommitAsync().result(),
                      MatchesStatus(
                          absl::StatusCode::kInvalidArgument,
                          StrCat("Error writing ", kvstore->DescribeKey(a_key),
                                 ": new value contains Z")));
        }
        EXPECT_THAT(AsyncCache::ReadLock<absl::Cord>(*entry).data(),
                    Pointee(absl::Cord("abc")));
        EXPECT_THAT(
            kvstore->Read(a_key).result(),
            MatchesKvsReadResult(
                absl::Cord("abc"),
                AsyncCache::ReadLock<absl::Cord>(*entry).stamp().generation));
      },
      TENSORSTORE_LOC);

  if (!options.multi_key_atomic_supported) {
    RegisterGoogleTestCaseDynamically(
        suite_name, "AtomicError",
        [=] {
          auto kvstore = options.get_store();
          auto cache = KvsBackedTestCache::Make(kvstore);
          auto get_key = options.get_key_getter();
          auto a_key = get_key("a");
          auto b_key = get_key("b");
          auto entry_a = GetCacheEntry(cache, a_key);
          auto entry_b = GetCacheEntry(cache, b_key);
          {
            auto transaction = Transaction(tensorstore::atomic_isolated);
            {
              TENSORSTORE_ASSERT_OK_AND_ASSIGN(
                  auto open_transaction,
                  tensorstore::internal::AcquireOpenTransactionPtrOrError(
                      transaction));
              TENSORSTORE_ASSERT_OK(
                  entry_a->Modify(open_transaction, false, "abc"));
              EXPECT_THAT(
                  GetTransactionNode(*entry_b, open_transaction),
                  MatchesStatus(
                      absl::StatusCode::kInvalidArgument,
                      StrCat("Cannot read/write ", kvstore->DescribeKey(a_key),
                             " and read/write ", kvstore->DescribeKey(b_key),
                             " as single atomic transaction")));
            }
            EXPECT_THAT(
                transaction.future().result(),
                MatchesStatus(
                    absl::StatusCode::kInvalidArgument,
                    StrCat("Cannot read/write ", kvstore->DescribeKey(a_key),
                           " and read/write ", kvstore->DescribeKey(b_key),
                           " as single atomic transaction")));
          }
        },
        TENSORSTORE_LOC);
  }

  RegisterGoogleTestCaseDynamically(
      suite_name, "TwoNodes",
      [=] {
        auto kvstore = options.get_store();
        auto cache_x = KvsBackedTestCache::Make(kvstore, {}, "x");
        auto cache_y = KvsBackedTestCache::Make(kvstore, {}, "y");
        auto get_key = options.get_key_getter();
        auto a_key = get_key("a");
        auto transaction = Transaction(tensorstore::atomic_isolated);
        {
          TENSORSTORE_ASSERT_OK_AND_ASSIGN(
              auto open_transaction,
              tensorstore::internal::AcquireOpenTransactionPtrOrError(
                  transaction));
          TENSORSTORE_ASSERT_OK(GetCacheEntry(cache_x, a_key)
                                    ->Modify(open_transaction, false, "abc"));
          TENSORSTORE_ASSERT_OK(GetCacheEntry(cache_y, a_key)
                                    ->Modify(open_transaction, false, "def"));
        }
        TENSORSTORE_ASSERT_OK(transaction.CommitAsync());
        EXPECT_THAT(kvstore->Read(a_key).result(),
                    MatchesKvsReadResult(absl::Cord("abcdef")));
      },
      TENSORSTORE_LOC);

  RegisterGoogleTestCaseDynamically(
      suite_name, "ThreeNodes",
      [=] {
        auto kvstore = options.get_store();
        auto cache_x = KvsBackedTestCache::Make(kvstore, {}, "x");
        auto cache_y = KvsBackedTestCache::Make(kvstore, {}, "y");
        auto cache_z = KvsBackedTestCache::Make(kvstore);
        auto get_key = options.get_key_getter();
        auto a_key = get_key("a");
        auto transaction = Transaction(tensorstore::atomic_isolated);
        {
          TENSORSTORE_ASSERT_OK_AND_ASSIGN(
              auto open_transaction,
              tensorstore::internal::AcquireOpenTransactionPtrOrError(
                  transaction));
          TENSORSTORE_ASSERT_OK(GetCacheEntry(cache_x, a_key)
                                    ->Modify(open_transaction, false, "abc"));
          TENSORSTORE_ASSERT_OK(GetCacheEntry(cache_y, a_key)
                                    ->Modify(open_transaction, true, "def"));
          TENSORSTORE_ASSERT_OK(GetCacheEntry(cache_z, a_key)
                                    ->Modify(open_transaction, false, "ghi"));
        }
        TENSORSTORE_ASSERT_OK(transaction.CommitAsync());
        EXPECT_THAT(kvstore->Read(a_key).result(),
                    MatchesKvsReadResult(absl::Cord("defghi")));
      },
      TENSORSTORE_LOC);

  RegisterGoogleTestCaseDynamically(
      suite_name, "WriteThenClearThenRevokeThenRead",
      [=] {
        auto kvstore = options.get_store();
        auto cache_x = KvsBackedTestCache::Make(kvstore, {}, "x");
        auto cache_y = KvsBackedTestCache::Make(kvstore, {}, "y");
        auto get_key = options.get_key_getter();
        auto a_key = get_key("a");
        auto transaction = Transaction(tensorstore::isolated);
        {
          TENSORSTORE_ASSERT_OK_AND_ASSIGN(
              auto open_transaction,
              tensorstore::internal::AcquireOpenTransactionPtrOrError(
                  transaction));
          TENSORSTORE_ASSERT_OK(GetCacheEntry(cache_x, a_key)
                                    ->Modify(open_transaction, false, "abc"));
          TENSORSTORE_ASSERT_OK(GetCacheEntry(cache_y, a_key)
                                    ->Modify(open_transaction, true, ""));
          EXPECT_THAT(GetCacheEntry(cache_x, a_key)
                          ->ReadValue(open_transaction)
                          .result(),
                      ::testing::Optional(absl::Cord()));
        }
      },
      TENSORSTORE_LOC);

  if (options.delete_range_supported) {
    RegisterGoogleTestCaseDynamically(
        suite_name, "WriteThenDeleteRangeThenRead",
        [=] {
          auto kvstore = options.get_store();
          auto cache_x = KvsBackedTestCache::Make(kvstore, {}, "x");
          auto cache_y = KvsBackedTestCache::Make(kvstore, {}, "y");
          auto get_key = options.get_key_getter();
          auto a_key = get_key("a");
          auto b_key = get_key("b");
          auto c_key = get_key("c");
          auto transaction = Transaction(tensorstore::isolated);
          {
            TENSORSTORE_ASSERT_OK_AND_ASSIGN(
                auto open_transaction,
                tensorstore::internal::AcquireOpenTransactionPtrOrError(
                    transaction));
            TENSORSTORE_ASSERT_OK(GetCacheEntry(cache_x, b_key)
                                      ->Modify(open_transaction, false, "abc"));
            EXPECT_THAT(GetCacheEntry(cache_x, b_key)
                            ->ReadValue(open_transaction)
                            .result(),
                        ::testing::Optional(absl::Cord("abc")));
            EXPECT_THAT(GetCacheEntry(cache_y, b_key)
                            ->ReadValue(open_transaction)
                            .result(),
                        ::testing::Optional(absl::Cord("abc")));
            TENSORSTORE_ASSERT_OK(kvstore->TransactionalDeleteRange(
                open_transaction, KeyRange{a_key, c_key}));
            EXPECT_THAT(GetCacheEntry(cache_x, b_key)
                            ->ReadValue(open_transaction)
                            .result(),
                        ::testing::Optional(absl::Cord()));
          }
        },
        TENSORSTORE_LOC);
  }

  RegisterGoogleTestCaseDynamically(
      suite_name, "RandomOperationTest/SinglePhase",
      [=] {
        RandomOperationTester tester(options.get_store(),
                                     options.get_key_getter());
        if (!options.delete_range_supported) {
          tester.write_probability = 1;
        }
        tester.barrier_probability = 0;
        tester.PerformRandomActions();
      },
      TENSORSTORE_LOC);

  RegisterGoogleTestCaseDynamically(
      suite_name, "RandomOperationTest/MultiPhase",
      [=] {
        RandomOperationTester tester(options.get_store(),
                                     options.get_key_getter());
        if (!options.delete_range_supported) {
          tester.write_probability = 1;
        }
        tester.PerformRandomActions();
      },
      TENSORSTORE_LOC);
}

}  // namespace internal
}  // namespace tensorstore
