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

#include "tensorstore/internal/cache/kvs_backed_cache_testutil.h"

#include <stddef.h>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/optimization.h"
#include "absl/log/absl_log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/transaction.h"
#include "tensorstore/transaction_impl.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

void KvsBackedTestCache::Entry::DoDecode(std::optional<absl::Cord> value,
                                         DecodeReceiver receiver) {
  if (value && absl::StrContains(value->Flatten(), 'Z')) {
    return execution::set_error(
        receiver, absl::FailedPreconditionError("existing value contains Z"));
  }
  execution::set_value(
      receiver, std::make_shared<absl::Cord>(value.value_or(absl::Cord())));
}

void KvsBackedTestCache::Entry::DoEncode(std::shared_ptr<const absl::Cord> data,
                                         EncodeReceiver receiver) {
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

    void set_cancel() { ABSL_UNREACHABLE(); }  // COV_NF_LINE
    void set_error(absl::Status status) { promise_.SetResult(status); }
    void set_value(AsyncCache::ReadState update) {
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
        receiver, ReadState{{}, TimestampedStorageGeneration::Unconditional()});
    return;
  }
  auto continuation = [this, receiver = std::move(receiver)](
                          ReadyFuture<const void> future) mutable {
    auto& r = future.result();
    if (!r) {
      return execution::set_error(receiver, r.status());
    }
    if (absl::StrContains(value, 'Z')) {
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
    return execution::set_value(receiver, std::move(read_state));
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
    kvstore::DriverPtr kvstore, CachePool::StrongPtr pool,
    std::string_view cache_identifier) {
  if (!pool) {
    pool = CachePool::Make(CachePool::Limits{});
  }
  return pool->GetCache<KvsBackedTestCache>(cache_identifier, [&] {
    return std::make_unique<KvsBackedTestCache>(kvstore);
  });
}

KvsRandomOperationTester::KvsRandomOperationTester(
    absl::BitGenRef gen, kvstore::DriverPtr kvstore,
    std::function<std::string(std::string)> get_key)
    : gen(gen), kvstore(kvstore) {
  for (const auto& key : {"x", "y"}) {
    caches.push_back(KvsBackedTestCache::Make(kvstore, {}, key));
  }
  size_t num_keys = absl::Uniform(gen, 5u, 15u);
  for (size_t i = 0; i < num_keys; ++i) {
    keys.push_back(get_key(std::string{static_cast<char>('a' + i)}));
  }
}

void KvsRandomOperationTester::SimulateDeleteRange(const KeyRange& range) {
  if (range.empty()) return;
  map.erase(map.lower_bound(range.inclusive_min),
            range.exclusive_max.empty() ? map.end()
                                        : map.lower_bound(range.exclusive_max));
}

void KvsRandomOperationTester::SimulateWrite(const std::string& key, bool clear,
                                             const std::string& append) {
  auto& value = map[key];
  if (clear) value.clear();
  value += append;
}

std::string KvsRandomOperationTester::SampleKey() {
  return keys[absl::Uniform(gen, 0u, keys.size())];
}

std::string KvsRandomOperationTester::SampleKeyOrEmpty() {
  size_t key_index =
      absl::Uniform(absl::IntervalClosedClosed, gen, 0u, keys.size());
  if (key_index == 0) return "";
  return keys[key_index - 1];
}

void KvsRandomOperationTester::PerformRandomAction(
    OpenTransactionPtr transaction) {
  if (barrier_probability > 0 && absl::Bernoulli(gen, barrier_probability)) {
    transaction->Barrier();
    if (log) {
      ABSL_LOG(INFO) << "Barrier";
    }
  }
  if (absl::Bernoulli(gen, write_probability)) {
    const auto& key = SampleKey();
    const auto& cache = caches[absl::Uniform(gen, 0u, caches.size())];
    bool clear = absl::Bernoulli(gen, clear_probability);
    std::string append = tensorstore::StrCat(", ", ++write_number);
    SimulateWrite(key, clear, append);
    if (log) {
      ABSL_LOG(INFO) << "Write: key=" << QuoteString(key)
                     << ", cache_key=" << cache->cache_identifier()
                     << ", clear=" << clear << ", append=\"" << append << "\"";
    }
    TENSORSTORE_EXPECT_OK(
        GetCacheEntry(cache, key)->Modify(transaction, clear, append));
  } else {
    KeyRange range{SampleKeyOrEmpty(), SampleKeyOrEmpty()};
    if (log) {
      ABSL_LOG(INFO) << "DeleteRange: " << range;
    }
    SimulateDeleteRange(range);
    TENSORSTORE_EXPECT_OK(
        kvstore->TransactionalDeleteRange(transaction, range));
  }
}

void KvsRandomOperationTester::PerformRandomActions() {
  const size_t num_actions = absl::Uniform(gen, 1u, 100u);
  if (log) {
    ABSL_LOG(INFO) << "--PerformRandomActions-- " << num_actions;
  }

  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    for (size_t i = 0; i < num_actions; ++i) {
      PerformRandomAction(open_transaction);
    }
  }
  TENSORSTORE_ASSERT_OK(transaction.CommitAsync());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto kvstore_cord_map,
                                   tensorstore::internal::GetMap(kvstore));

  EXPECT_THAT(Map(kvstore_cord_map.begin(), kvstore_cord_map.end()),
              ::testing::ElementsAreArray(map));
}

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
          EXPECT_THAT(
              entry->Read(absl::Now()).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            tensorstore::StrCat(
                                "Error reading ", kvstore->DescribeKey(a_key),
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
                          tensorstore::StrCat("Error reading ",
                                              kvstore->DescribeKey(a_key),
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
      suite_name, "BarrierThenUnconditionalWriteback",
      [=] {
        auto kvstore = options.get_store();
        auto cache = KvsBackedTestCache::Make(kvstore);
        auto get_key = options.get_key_getter();
        auto a_key = get_key("a");
        TENSORSTORE_EXPECT_OK(kvstore->Write(a_key, absl::Cord("ghi")));
        auto entry = GetCacheEntry(cache, a_key);
        TENSORSTORE_EXPECT_OK(entry->Read(absl::InfinitePast()).result());

        {
          auto transaction = Transaction(tensorstore::isolated);
          transaction.Barrier();
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
          EXPECT_THAT(
              transaction.CommitAsync().result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            tensorstore::StrCat("Error writing ",
                                                kvstore->DescribeKey(a_key),
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
                      tensorstore::StrCat(
                          "Cannot read/write ", kvstore->DescribeKey(a_key),
                          " and read/write ", kvstore->DescribeKey(b_key),
                          " as single atomic transaction")));
            }
            EXPECT_THAT(
                transaction.future().result(),
                MatchesStatus(
                    absl::StatusCode::kInvalidArgument,
                    tensorstore::StrCat(
                        "Cannot read/write ", kvstore->DescribeKey(a_key),
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
        std::minstd_rand gen{internal::GetRandomSeedForTest(
            "TENSORSTORE_INTERNAL_KVS_TESTUTIL_SINGLEPHASE")};

        KvsRandomOperationTester tester(gen, options.get_store(),
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
        std::minstd_rand gen{internal::GetRandomSeedForTest(
            "TENSORSTORE_INTERNAL_KVS_TESTUTIL_MULTIPHASE")};

        KvsRandomOperationTester tester(gen, options.get_store(),
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
