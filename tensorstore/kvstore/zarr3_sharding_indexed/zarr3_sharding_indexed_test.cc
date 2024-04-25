// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/kvstore/zarr3_sharding_indexed/zarr3_sharding_indexed.h"

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "re2/re2.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/write.h"
#include "riegeli/digests/crc32c_digester.h"
#include "tensorstore/batch.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/kvs_backed_cache_testutil.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/riegeli/digest_suffixed_writer.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/internal/thread/thread_pool.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/kvstore/zarr3_sharding_indexed/key.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;

using ::tensorstore::Batch;
using ::tensorstore::Executor;
using ::tensorstore::Future;
using ::tensorstore::Index;
using ::tensorstore::KvStore;
using ::tensorstore::MatchesStatus;
using ::tensorstore::OptionalByteRangeRequest;
using ::tensorstore::Result;
using ::tensorstore::span;
using ::tensorstore::StorageGeneration;
using ::tensorstore::TimestampedStorageGeneration;
using ::tensorstore::Transaction;
using ::tensorstore::internal::CachePool;
using ::tensorstore::internal::GetCache;
using ::tensorstore::internal::KvsBackedTestCache;
using ::tensorstore::internal::MatchesKvsReadResult;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore::internal::MatchesTimestampedStorageGeneration;
using ::tensorstore::internal::MockKeyValueStore;
using ::tensorstore::internal::UniqueNow;
using ::tensorstore::internal_zarr3::ZarrCodecChainSpec;
using ::tensorstore::kvstore::ReadResult;
using ::tensorstore::zarr3_sharding_indexed::EntryId;
using ::tensorstore::zarr3_sharding_indexed::EntryIdToKey;
using ::tensorstore::zarr3_sharding_indexed::GetShardedKeyValueStore;
using ::tensorstore::zarr3_sharding_indexed::ShardedKeyValueStoreParameters;
using ::tensorstore::zarr3_sharding_indexed::ShardIndexLocation;

constexpr CachePool::Limits kSmallCacheLimits{10000000};

absl::Cord Bytes(std::initializer_list<unsigned char> x) {
  return absl::Cord(std::string(x.begin(), x.end()));
}

absl::Cord WithCrc32c(absl::Cord input) {
  absl::Cord output;
  riegeli::CordWriter writer{&output};
  TENSORSTORE_CHECK_OK(riegeli::Write(
      input, tensorstore::internal::DigestSuffixedWriter<
                 riegeli::Crc32cDigester,
                 tensorstore::internal::LittleEndianDigestWriter>{&writer}));
  ABSL_CHECK(writer.Close());
  return output;
}

class GetKey {
 public:
  GetKey(bool sequential, std::vector<Index> grid_shape)
      : sequential_(sequential),
        grid_shape_(std::move(grid_shape)),
        num_entries_(
            tensorstore::ProductOfExtents(span<const Index>(grid_shape_))) {}

  // NOTE: absl::FunctionRef currently takes objects by const&, so we have to
  // mark operator() const and the members mutable.
  std::string operator()(std::string key) const {
    auto it = key_to_entry_id_.find(key);
    if (it == key_to_entry_id_.end()) {
      ABSL_CHECK_LT(entry_id_to_key_.size(), num_entries_);
      while (true) {
        auto x = sequential_ ? next_entry_id_++ : absl::Uniform<EntryId>(gen_);
        x = x % num_entries_;
        if (entry_id_to_key_.emplace(x, key).second) {
          it = key_to_entry_id_.emplace(key, x).first;
          break;
        }
      }
    }
    return EntryIdToKey(it->second, grid_shape_);
  }

 private:
  bool sequential_;
  std::vector<Index> grid_shape_;
  EntryId num_entries_;
  mutable EntryId next_entry_id_ = 0;
  mutable absl::BitGen gen_;
  mutable absl::flat_hash_map<std::string, EntryId> key_to_entry_id_;
  mutable absl::flat_hash_map<EntryId, std::string> entry_id_to_key_;
};

kvstore::DriverPtr GetDefaultStore(kvstore::DriverPtr base_kvstore,
                                   std::string base_kvstore_path,
                                   Executor executor,
                                   CachePool::StrongPtr cache_pool,
                                   const std::vector<Index>& grid_shape) {
  ShardedKeyValueStoreParameters params;
  params.base_kvstore = base_kvstore;
  params.base_kvstore_path = base_kvstore_path;
  params.executor = executor;
  params.cache_pool = CachePool::WeakPtr(cache_pool);
  TENSORSTORE_CHECK_OK_AND_ASSIGN(
      auto index_codecs,
      ZarrCodecChainSpec::FromJson(
          {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}},
           {{"name", "crc32c"}}}));
  params.index_params.index_location = ShardIndexLocation::kEnd;
  TENSORSTORE_CHECK_OK(
      params.index_params.Initialize(index_codecs, grid_shape));
  return GetShardedKeyValueStore(std::move(params));
}

TEST(ShardedKeyValueStoreTest, BasicFunctionality) {
  std::vector<std::pair<std::string, tensorstore::Executor>> executors{
      {"inline", tensorstore::InlineExecutor{}},
      {"thread_pool", tensorstore::internal::DetachedThreadPool(2)}};
  for (const auto& [executor_name, executor] : executors) {
    for (const auto sequential_ids : {true, false}) {
      auto cache_pool = CachePool::Make(kSmallCacheLimits);
      auto base_kv_store = tensorstore::GetMemoryKeyValueStore();
      const int64_t num_entries = 100;
      SCOPED_TRACE(executor_name);
      auto store = GetDefaultStore(base_kv_store, "shard_path", executor,
                                   cache_pool, {num_entries});
      GetKey get_key_fn(sequential_ids, {num_entries});
      tensorstore::internal::TestKeyValueReadWriteOps(store, get_key_fn);
    }
  }
}

TEST(Uint64ShardedKeyValueStoreTest, DescribeKey) {
  CachePool::StrongPtr cache_pool = CachePool::Make(kSmallCacheLimits);
  kvstore::DriverPtr base_kv_store = tensorstore::GetMemoryKeyValueStore();
  int64_t num_entries = 100;
  std::vector<Index> grid_shape{num_entries};
  kvstore::DriverPtr store =
      GetDefaultStore(base_kv_store, "shard_path",
                      tensorstore::InlineExecutor{}, cache_pool, grid_shape);
  for (const auto& [key, description] :
       std::vector<std::pair<uint32_t, std::string>>{
           {0, "shard entry {0}/{100} in \"shard_path\""},
           {1, "shard entry {1}/{100} in \"shard_path\""},
       }) {
    EXPECT_EQ(description, store->DescribeKey(EntryIdToKey(key, grid_shape)));
  }
}

class RawEncodingTest : public ::testing::Test {
 protected:
  CachePool::StrongPtr cache_pool = CachePool::Make(kSmallCacheLimits);
  kvstore::DriverPtr base_kv_store = tensorstore::GetMemoryKeyValueStore();
  kvstore::DriverPtr GetStore(const std::vector<Index>& grid_shape) {
    return GetDefaultStore(base_kv_store, "shard_path",
                           tensorstore::InlineExecutor{}, cache_pool,
                           grid_shape);
  }
};

TEST_F(RawEncodingTest, MultipleUnconditionalWrites) {
  std::vector<Index> grid_shape{100};
  kvstore::DriverPtr store = GetStore(grid_shape);
  std::vector<absl::Cord> values{absl::Cord("abc"), absl::Cord("aaaaa"),
                                 absl::Cord("efgh")};
  std::vector<Future<TimestampedStorageGeneration>> futures;
  auto key = EntryIdToKey(10, grid_shape);
  tensorstore::Transaction txn(tensorstore::isolated);
  for (auto value : values) {
    futures.push_back(kvstore::WriteCommitted(KvStore{store, txn}, key, value));
  }
  txn.CommitAsync().IgnoreFuture();
  std::vector<Result<TimestampedStorageGeneration>> results;
  for (const auto& future : futures) {
    results.push_back(future.result());
  }
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto shard_read,
                                   base_kv_store->Read("shard_path").result());

  // All writes succeed, but all but one write is assigned a generation of
  // `StorageGeneration::Invalid()` since it is overwritten immediately before
  // it is ever actually committed to the `base_kv_store`.
  EXPECT_THAT(
      results,
      ::testing::UnorderedElementsAre(
          MatchesTimestampedStorageGeneration(StorageGeneration::Invalid()),
          MatchesTimestampedStorageGeneration(StorageGeneration::Invalid()),
          MatchesTimestampedStorageGeneration(shard_read.stamp.generation)));
  for (size_t i = 0; i < results.size(); ++i) {
    if (results[i] && results[i]->generation == shard_read.stamp.generation) {
      EXPECT_THAT(store->Read(key).result(),
                  MatchesKvsReadResult(values[i], results[i]->generation));
    }
  }
}

TEST_F(RawEncodingTest, List) {
  std::vector<Index> grid_shape{100};
  kvstore::DriverPtr store = GetStore(grid_shape);
  std::map<std::string, absl::Cord> values{
      {EntryIdToKey(1, grid_shape), absl::Cord("a")},
      {EntryIdToKey(2, grid_shape), absl::Cord("bc")},
      {EntryIdToKey(3, grid_shape), absl::Cord("def")},
      {EntryIdToKey(10, grid_shape), absl::Cord("xyz")}};
  for (auto [key, value] : values) {
    TENSORSTORE_EXPECT_OK(store->Write(key, value));
  }
  EXPECT_THAT(tensorstore::internal::GetMap(store),
              ::testing::Optional(::testing::ElementsAreArray(values)));
}

TEST_F(RawEncodingTest, WritesAndDeletes) {
  std::vector<Index> grid_shape{100};
  kvstore::DriverPtr store = GetStore(grid_shape);
  StorageGeneration gen1, gen2, gen3;
  {
    tensorstore::Transaction txn(tensorstore::isolated);
    auto init_future1 = kvstore::WriteCommitted(
        KvStore{store, txn}, EntryIdToKey(1, grid_shape), absl::Cord("a"));
    auto init_future2 = kvstore::WriteCommitted(
        KvStore{store, txn}, EntryIdToKey(2, grid_shape), absl::Cord("bc"));
    auto init_future3 = kvstore::WriteCommitted(
        KvStore{store, txn}, EntryIdToKey(3, grid_shape), absl::Cord("def"));
    txn.CommitAsync().IgnoreFuture();

    gen1 = init_future1.value().generation;
    gen2 = init_future2.value().generation;
    gen3 = init_future3.value().generation;
  }

  tensorstore::Transaction txn(tensorstore::isolated);

  // Conditional delete with mismatched generation.
  auto future1 =
      kvstore::DeleteCommitted(KvStore{store, txn}, EntryIdToKey(1, grid_shape),
                               {StorageGeneration::NoValue()});

  // Conditional write with matching generation.
  auto future2 =
      kvstore::WriteCommitted(KvStore{store, txn}, EntryIdToKey(2, grid_shape),
                              absl::Cord("ww"), {gen2});
  auto future3 =
      kvstore::WriteCommitted(KvStore{store, txn}, EntryIdToKey(2, grid_shape),
                              absl::Cord("xx"), {gen2});

  // Conditional write with matching generation
  auto future4 =
      kvstore::WriteCommitted(KvStore{store, txn}, EntryIdToKey(4, grid_shape),
                              absl::Cord("zz"), {StorageGeneration::NoValue()});

  // Conditional delete with matching generation.
  auto future5 = kvstore::DeleteCommitted(KvStore{store, txn},
                                          EntryIdToKey(3, grid_shape), {gen3});

  txn.CommitAsync().IgnoreFuture();

  EXPECT_THAT(future1.result(), MatchesTimestampedStorageGeneration(
                                    StorageGeneration::Unknown()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto shard_read,
                                   base_kv_store->Read("shard_path").result());

  // Exactly one of `future2` and `future3` succeeds, and the other is aborted
  // due to generation mismatch.
  EXPECT_THAT(
      std::vector({future2.result(), future3.result()}),
      ::testing::UnorderedElementsAre(
          MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()),
          MatchesTimestampedStorageGeneration(shard_read.stamp.generation)));

  EXPECT_THAT(store->Read(EntryIdToKey(1, grid_shape)).result(),
              MatchesKvsReadResult(absl::Cord("a")));
  EXPECT_THAT(store->Read(EntryIdToKey(2, grid_shape)).result(),
              MatchesKvsReadResult(
                  !StorageGeneration::IsUnknown(future2.result()->generation)
                      ? absl::Cord("ww")
                      : absl::Cord("xx")));
  EXPECT_THAT(store->Read(EntryIdToKey(3, grid_shape)).result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(store->Read(EntryIdToKey(4, grid_shape)).result(),
              MatchesKvsReadResult(absl::Cord("zz")));
}

// The order in which multiple requests for the same `ChunkId` are attempted
// depends on the order produced by `std::sort`, which is unspecified.  To
// ensure we test both possibilities, we run the test with both orders.  This
// assumes that `std::sort` is deterministic.
std::vector<std::vector<Result<TimestampedStorageGeneration>>>
TestOrderDependentWrites(
    std::function<void()> init,
    std::function<Future<TimestampedStorageGeneration>()> op0,
    std::function<Future<TimestampedStorageGeneration>()> op1,
    std::function<void()> finalize) {
  std::vector<std::vector<Result<TimestampedStorageGeneration>>> all_results;
  for (int i = 0; i < 2; ++i) {
    std::vector<Future<TimestampedStorageGeneration>> futures(2);
    init();
    if (i == 0) {
      futures[0] = op0();
      futures[1] = op1();
    } else {
      futures[1] = op1();
      futures[0] = op0();
    }
    finalize();
    all_results.push_back({futures[0].result(), futures[1].result()});
  }
  return all_results;
}

TEST_F(RawEncodingTest, WriteThenDelete) {
  std::vector<Index> grid_shape{100};
  kvstore::DriverPtr store = GetStore(grid_shape);
  TENSORSTORE_ASSERT_OK(
      store->Write(EntryIdToKey(1, grid_shape), absl::Cord("a")).result());
  EXPECT_THAT(store->Read(EntryIdToKey(1, grid_shape)).result(),
              MatchesKvsReadResult(absl::Cord("a")));
  TENSORSTORE_ASSERT_OK(store->Delete(EntryIdToKey(1, grid_shape)).result());
  EXPECT_THAT(store->Read(EntryIdToKey(1, grid_shape)).result(),
              MatchesKvsReadResultNotFound());
}

TEST_F(RawEncodingTest, MultipleDeleteExisting) {
  std::vector<Index> grid_shape{100};
  kvstore::DriverPtr store = GetStore(grid_shape);
  StorageGeneration gen;
  tensorstore::Transaction txn{tensorstore::no_transaction};
  EXPECT_THAT(
      TestOrderDependentWrites(
          /*init=*/
          [&] {
            gen = store->Write(EntryIdToKey(1, grid_shape), absl::Cord("a"))
                      .value()
                      .generation;
            txn = tensorstore::Transaction(tensorstore::isolated);
          },
          /*op0=*/
          [&] {
            // Delete conditioned on `gen` is guaranteed to succeed.
            return kvstore::DeleteCommitted(KvStore{store, txn},
                                            EntryIdToKey(1, grid_shape),
                                            {/*.if_equal=*/gen});
          },
          /*op1=*/
          [&] {
            // Delete conditioned on `StorageGeneration::NoValue()` succeeds if
            // it is attempted after the other delete, otherwise it fails.
            return kvstore::DeleteCommitted(
                KvStore{store, txn}, EntryIdToKey(1, grid_shape),
                {/*.if_equal=*/StorageGeneration::NoValue()});
          },
          /*finalize=*/[&] { txn.CommitAsync().IgnoreFuture(); }),
      // Test we covered each of the two cases (corresponding to different sort
      // orders) exactly once.
      ::testing::UnorderedElementsAre(
          ::testing::ElementsAre(
              MatchesTimestampedStorageGeneration(StorageGeneration::Invalid()),
              MatchesTimestampedStorageGeneration(
                  StorageGeneration::NoValue())),
          ::testing::ElementsAre(
              MatchesTimestampedStorageGeneration(StorageGeneration::NoValue()),
              MatchesTimestampedStorageGeneration(
                  StorageGeneration::Unknown()))));
}

// Tests that a conditional `Write` performed in the same commit after another
// `Write` fails.
TEST_F(RawEncodingTest, WriteWithUnmatchedConditionAfterDelete) {
  std::vector<Index> grid_shape{100};
  kvstore::DriverPtr store = GetStore(grid_shape);
  tensorstore::Transaction txn{tensorstore::no_transaction};
  EXPECT_THAT(
      TestOrderDependentWrites(
          /*init=*/
          [&] {
            store->Delete(EntryIdToKey(0, grid_shape)).value();
            txn = tensorstore::Transaction(tensorstore::isolated);
          },
          /*op0=*/
          [&] {
            // Write should succeed.
            return kvstore::WriteCommitted(KvStore{store, txn},
                                           EntryIdToKey(0, grid_shape),
                                           absl::Cord("a"));
          },
          /*op1=*/
          [&] {
            // Write should fail due to prior write.
            return kvstore::WriteCommitted(
                KvStore{store, txn}, EntryIdToKey(0, grid_shape),
                absl::Cord("b"),
                {/*.if_equal=*/StorageGeneration::FromString("g")});
          },
          /*finalize=*/[&] { txn.CommitAsync().IgnoreFuture(); }),
      // Regardless of order of operations, the result is the same.
      ::testing::Each(::testing::ElementsAre(
          MatchesTimestampedStorageGeneration(
              ::testing::AllOf(::testing::Not(StorageGeneration::NoValue()),
                               ::testing::Not(StorageGeneration::Invalid()))),
          MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()))));
}

TEST_F(RawEncodingTest, MultipleDeleteNonExisting) {
  std::vector<Index> grid_shape{100};
  kvstore::DriverPtr store = GetStore(grid_shape);
  tensorstore::Transaction txn(tensorstore::isolated);
  std::vector futures{
      kvstore::DeleteCommitted(KvStore{store, txn}, EntryIdToKey(1, grid_shape),
                               {StorageGeneration::NoValue()}),
      kvstore::DeleteCommitted(KvStore{store, txn}, EntryIdToKey(1, grid_shape),
                               {StorageGeneration::NoValue()})};
  txn.CommitAsync().IgnoreFuture();
  std::vector results{futures[0].result(), futures[1].result()};
  EXPECT_THAT(
      results,
      ::testing::UnorderedElementsAre(
          MatchesTimestampedStorageGeneration(StorageGeneration::Invalid()),
          MatchesTimestampedStorageGeneration(StorageGeneration::NoValue())));
}

TEST_F(RawEncodingTest, ShardIndexTooShort) {
  std::vector<Index> grid_shape{100};
  kvstore::DriverPtr store = GetStore(grid_shape);
  base_kv_store->Write("shard_path", Bytes({1, 2, 3})).value();
  EXPECT_THAT(store->Read(EntryIdToKey(1, grid_shape)).result(),
              MatchesStatus(
                  absl::StatusCode::kFailedPrecondition,
                  RE2::QuoteMeta("Error reading shard index in \"shard_path\": "
                                 "Requested byte range [-1604, ?) is not valid "
                                 "for value of size 3")));
  EXPECT_THAT(
      store->Write(EntryIdToKey(10, grid_shape), absl::Cord("abc")).result(),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    "Error reading \"shard_path\": "
                    "Existing shard has size of 3 bytes, but expected at least "
                    "1604 bytes"));
}

TEST_F(RawEncodingTest, ShardIndexByteRangeOverflow) {
  std::vector<Index> grid_shape{2};
  kvstore::DriverPtr store = GetStore(grid_shape);
  auto content = WithCrc32c(Bytes({
      // entries[0].offset
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
      // entries[0].length
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
      // entries[1].offset
      0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
      // entries[1].length
      0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
  }));

  TENSORSTORE_ASSERT_OK(base_kv_store->Write("shard_path", content));
  EXPECT_THAT(
      store->Read(EntryIdToKey(1, grid_shape)).result(),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    "Error reading shard index in \"shard_path\": "
                    "Invalid shard index entry 1 with offset=.*, length=.*"));
}

TEST_F(RawEncodingTest, ShardIndexEntryByteRangeOutOfRange) {
  std::vector<Index> grid_shape{2};
  kvstore::DriverPtr store = GetStore(grid_shape);
  auto content = WithCrc32c(Bytes({
      // entries[0].offset
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
      // entries[0].length
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
      // entries[1].offset
      0, 0, 0, 0, 0, 0, 0, 0,  //
      // entries[1].length
      37, 0, 0, 0, 0, 0, 0, 0,  //
  }));

  TENSORSTORE_ASSERT_OK(base_kv_store->Write("shard_path", content));
  EXPECT_THAT(
      store->Write(EntryIdToKey(1, grid_shape), absl::Cord("x")).result(),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    "Error reading \"shard_path\": "
                    "Shard index entry 1 with byte range .* is invalid "
                    "for shard of size .*"));
}

TEST_F(RawEncodingTest, ShardIndexInvalidChecksum) {
  std::vector<Index> grid_shape{2};
  kvstore::DriverPtr store = GetStore(grid_shape);
  auto content = Bytes({
      // entries[0].offset
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
      // entries[0].length
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
      // entries[1].offset
      0, 0, 0, 0, 0, 0, 0, 0,  //
      // entries[1].length
      5, 0, 0, 0, 0, 0, 0, 0,  //
  });
  content.Append("abcd");
  TENSORSTORE_ASSERT_OK(base_kv_store->Write("shard_path", content));
  EXPECT_THAT(store->Read(EntryIdToKey(1, grid_shape)).result(),
              MatchesStatus(absl::StatusCode::kDataLoss,
                            "Error reading shard index in \"shard_path\": "
                            "Digest mismatch.*"));
}

// Tests of operations issued to underlying KeyValueStore.
class UnderlyingKeyValueStoreTest : public ::testing::Test {
 protected:
  CachePool::StrongPtr cache_pool = CachePool::Make(kSmallCacheLimits);
  MockKeyValueStore::MockPtr mock_store = MockKeyValueStore::Make();
  kvstore::DriverPtr GetStore(std::vector<Index> grid_shape) {
    return GetDefaultStore(mock_store, "shard_path",
                           tensorstore::InlineExecutor{}, cache_pool,
                           grid_shape);
  }
  std::vector<Index> grid_shape{5};
  kvstore::DriverPtr store = GetStore(grid_shape);
};

// Tests that a Read operation results in the expected sequence of calls to the
// underlying KeyValueStore.
TEST_F(UnderlyingKeyValueStoreTest, Read) {
  // Perform initial successful read.
  absl::Time init_time = UniqueNow();
  absl::Time shard_index_time;
  {
    auto future = store->Read(EntryIdToKey(2, grid_shape), {});
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("shard_path", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(),
                req.options.generation_conditions.if_not_equal);
      EXPECT_EQ(StorageGeneration::Unknown(),
                req.options.generation_conditions.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest::SuffixLength(5 * 16 + 4),
                req.options.byte_range);
      EXPECT_THAT(req.options.staleness_bound, ::testing::Gt(init_time));
      shard_index_time = absl::Now();
      req.promise.SetResult(
          ReadResult{ReadResult::kValue,
                     WithCrc32c(Bytes({
                         // entries[0].offset
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[0].length
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[1].offset
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[1].length
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[2].offset
                         10, 0, 0, 0, 0, 0, 0, 0,  //
                         // entries[2].length
                         5, 0, 0, 0, 0, 0, 0, 0,  //
                         // entries[3].offset
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[3].length
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[4].offset
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[4].length
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                     })),
                     {StorageGeneration::FromString("g0"), shard_index_time}});
    }
    ASSERT_FALSE(future.ready()) << future.status();
    absl::Time read_time;
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("shard_path", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(),
                req.options.generation_conditions.if_not_equal);
      EXPECT_EQ(StorageGeneration::FromString("g0"),
                req.options.generation_conditions.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(10, 15), req.options.byte_range);
      read_time = absl::Now();
      req.promise.SetResult(
          ReadResult{ReadResult::kValue,
                     Bytes({5, 6, 7, 8, 9}),
                     {StorageGeneration::FromString("g0"), read_time}});
    }
    ASSERT_EQ(0, mock_store->read_requests.size());
    ASSERT_TRUE(future.ready());
    EXPECT_THAT(
        future.result(),
        MatchesKvsReadResult(Bytes({5, 6, 7, 8, 9}),
                             StorageGeneration::FromString("g0"), read_time));
  }

  // Issue another read for a not-present chunk that hits cached minishard
  // index.
  {
    kvstore::ReadOptions options;
    options.staleness_bound = init_time;
    auto future = store->Read(EntryIdToKey(3, grid_shape), options);
    ASSERT_TRUE(future.ready());
    EXPECT_THAT(future.result(),
                MatchesKvsReadResultNotFound(shard_index_time));
  }

  // Same as above, but ensures cached shard index is up to date.
  {
    auto req_time = UniqueNow();
    auto future = store->Read(EntryIdToKey(3, grid_shape), {});
    // Request for shard index if modified.
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("shard_path", req.key);
      EXPECT_EQ(StorageGeneration::FromString("g0"),
                req.options.generation_conditions.if_not_equal);
      EXPECT_EQ(StorageGeneration::Unknown(),
                req.options.generation_conditions.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest::SuffixLength(5 * 16 + 4),
                req.options.byte_range);
      EXPECT_THAT(req.options.staleness_bound, ::testing::Gt(req_time));
      shard_index_time = absl::Now();
      req.promise.SetResult(ReadResult::Unspecified(
          {StorageGeneration::FromString("g0"), shard_index_time}));
    }
    ASSERT_TRUE(future.ready());
    EXPECT_THAT(future.result(),
                MatchesKvsReadResultNotFound(shard_index_time));
  }

  // Issue a read for present chunk that hits cached minishard index.
  {
    kvstore::ReadOptions options;
    options.staleness_bound = init_time;
    auto future = store->Read(EntryIdToKey(2, grid_shape), options);
    absl::Time read_time;
    // Request for data based on cached minishard.
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("shard_path", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(),
                req.options.generation_conditions.if_not_equal);
      EXPECT_EQ(StorageGeneration::FromString("g0"),
                req.options.generation_conditions.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(10, 15), req.options.byte_range);
      EXPECT_EQ(init_time, req.options.staleness_bound);
      read_time = absl::Now();
      req.promise.SetResult(
          ReadResult{ReadResult::kValue,
                     Bytes({5, 6, 7, 8, 9}),
                     {StorageGeneration::FromString("g0"), read_time}});
    }
    ASSERT_EQ(0, mock_store->read_requests.size());
    ASSERT_TRUE(future.ready());
    EXPECT_THAT(
        future.result(),
        MatchesKvsReadResult(Bytes({5, 6, 7, 8, 9}),
                             StorageGeneration::FromString("g0"), read_time));
  }

  // Issue a read for present chunk while there is a concurrent modification.
  {
    kvstore::ReadOptions options;
    options.staleness_bound = init_time;
    auto future = store->Read(EntryIdToKey(2, grid_shape), options);
    absl::Time abort_time;
    // Initial request for data based on cached shard index.
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("shard_path", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(),
                req.options.generation_conditions.if_not_equal);
      EXPECT_EQ(init_time, req.options.staleness_bound);
      EXPECT_EQ(StorageGeneration::FromString("g0"),
                req.options.generation_conditions.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(10, 15), req.options.byte_range);
      abort_time = absl::Now();
      req.promise.SetResult(ReadResult::Unspecified(
          {StorageGeneration::FromString("g0"), abort_time}));
    }
    // Request for updated shard index.
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("shard_path", req.key);
      EXPECT_EQ(StorageGeneration::FromString("g0"),
                req.options.generation_conditions.if_not_equal);
      EXPECT_EQ(StorageGeneration::Unknown(),
                req.options.generation_conditions.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest::SuffixLength(5 * 16 + 4),
                req.options.byte_range);
      EXPECT_THAT(req.options.staleness_bound, ::testing::Ge(abort_time));
      shard_index_time = absl::Now();
      req.promise.SetResult(
          ReadResult{ReadResult::kValue,
                     WithCrc32c(Bytes({
                         // entries[0].offset
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[0].length
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[1].offset
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[1].length
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[2].offset
                         10, 0, 0, 0, 0, 0, 0, 0,  //
                         // entries[2].length
                         6, 0, 0, 0, 0, 0, 0, 0,  //
                         // entries[3].offset
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[3].length
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[4].offset
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                         // entries[4].length
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                     })),
                     {StorageGeneration::FromString("g1"), shard_index_time}});
    }
    // Request for value.
    absl::Time read_time;
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("shard_path", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(),
                req.options.generation_conditions.if_not_equal);
      EXPECT_EQ(StorageGeneration::FromString("g1"),
                req.options.generation_conditions.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(10, 16), req.options.byte_range);
      read_time = absl::Now();
      req.promise.SetResult(
          ReadResult{ReadResult::kValue,
                     Bytes({4, 5, 6, 7, 8, 9}),
                     {StorageGeneration::FromString("g1"), read_time}});
    }
    ASSERT_EQ(0, mock_store->read_requests.size());
    ASSERT_TRUE(future.ready());
    EXPECT_THAT(
        future.result(),
        MatchesKvsReadResult(Bytes({4, 5, 6, 7, 8, 9}),
                             StorageGeneration::FromString("g1"), read_time));
  }
}

// Verify that a read-only transaction does not do any I/O on commit.
TEST_F(UnderlyingKeyValueStoreTest, TransactionReadThenCommit) {
  tensorstore::Transaction txn(tensorstore::isolated);
  auto memory_store = tensorstore::GetMemoryKeyValueStore();
  {
    auto future =
        kvstore::Read(KvStore{store, txn}, EntryIdToKey(2, grid_shape), {});
    {
      auto req = mock_store->read_requests.pop();
      req(memory_store);
      ASSERT_EQ(0, mock_store->read_requests.size());
    }
    EXPECT_THAT(future.result(),
                ::testing::Optional(MatchesKvsReadResultNotFound()));
  }

  auto commit_future = txn.CommitAsync();
  TENSORSTORE_ASSERT_OK(commit_future.result());
  EXPECT_EQ(0, mock_store->read_requests.size());
}

// Tests issuing read for chunk in uncached shard index while the shard is
// concurrently deleted (after the shard index can be read).
TEST_F(UnderlyingKeyValueStoreTest,
       ReadConcurrentDeleteAfterReadingMinishardIndex) {
  auto req_time = UniqueNow();
  auto future = store->Read(EntryIdToKey(2, grid_shape), {});
  // Request for shard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("shard_path", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(),
              req.options.generation_conditions.if_not_equal);
    EXPECT_EQ(StorageGeneration::Unknown(),
              req.options.generation_conditions.if_equal);
    EXPECT_THAT(req.options.staleness_bound, ::testing::Gt(req_time));
    EXPECT_EQ(OptionalByteRangeRequest::SuffixLength(5 * 16 + 4),
              req.options.byte_range);
    req.promise.SetResult(
        ReadResult{ReadResult::kValue,
                   WithCrc32c(Bytes({
                       // entries[0].offset
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[0].length
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[1].offset
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[1].length
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[2].offset
                       10, 0, 0, 0, 0, 0, 0, 0,  //
                       // entries[2].length
                       5, 0, 0, 0, 0, 0, 0, 0,  //
                       // entries[3].offset
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[3].length
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[4].offset
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[4].length
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                   })),
                   {StorageGeneration::FromString("g0"), absl::Now()}});
  }
  // Request for value.
  absl::Time read_time;
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("shard_path", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(),
              req.options.generation_conditions.if_not_equal);
    EXPECT_EQ(StorageGeneration::FromString("g0"),
              req.options.generation_conditions.if_equal);
    EXPECT_EQ(OptionalByteRangeRequest(10, 15), req.options.byte_range);
    read_time = absl::Now();
    req.promise.SetResult(ReadResult{
        ReadResult::kMissing, {}, {StorageGeneration::NoValue(), read_time}});
  }
  ASSERT_EQ(0, mock_store->read_requests.size());
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(), MatchesKvsReadResultNotFound(read_time));
}

TEST_F(UnderlyingKeyValueStoreTest, ReadErrorReadingShardIndex) {
  auto future = store->Read(EntryIdToKey(2, grid_shape), {});
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("shard_path", req.key);
    EXPECT_EQ(OptionalByteRangeRequest::SuffixLength(5 * 16 + 4),
              req.options.byte_range);
    req.promise.SetResult(absl::UnknownError("Read error"));
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown,
                            "Error reading shard index in \"shard_path\": "
                            "Read error"));
}

TEST_F(UnderlyingKeyValueStoreTest, ReadErrorReadingData) {
  auto future = store->Read(EntryIdToKey(0x2, grid_shape), {});
  // Request for shard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("shard_path", req.key);
    EXPECT_EQ(OptionalByteRangeRequest::SuffixLength(5 * 16 + 4),
              req.options.byte_range);
    req.promise.SetResult(
        ReadResult{ReadResult::kValue,
                   WithCrc32c(Bytes({
                       // entries[0].offset
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[0].length
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[1].offset
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[1].length
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[2].offset
                       10, 0, 0, 0, 0, 0, 0, 0,  //
                       // entries[2].length
                       5, 0, 0, 0, 0, 0, 0, 0,  //
                       // entries[3].offset
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[3].length
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[4].offset
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                       // entries[4].length
                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
                   })),
                   {StorageGeneration::FromString("g0"), absl::Now()}});
  }
  ASSERT_FALSE(future.ready()) << future.status();
  // Request for value.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("shard_path", req.key);
    EXPECT_EQ(OptionalByteRangeRequest(10, 15), req.options.byte_range);
    req.promise.SetResult(absl::UnknownError("Read error"));
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "Read error"));
}

TEST_F(UnderlyingKeyValueStoreTest, ReadInvalidKey) {
  auto future = store->Read("abc", {});
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST_F(UnderlyingKeyValueStoreTest, WriteInvalidKey) {
  auto future = store->Write("abc", absl::Cord("x"));
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST_F(UnderlyingKeyValueStoreTest, DeleteInvalidKey) {
  auto future = store->Delete("abc");
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST_F(UnderlyingKeyValueStoreTest, WriteWithNoExistingShard) {
  grid_shape = {2};
  store = GetStore(grid_shape);
  auto future = store->Write(EntryIdToKey(1, grid_shape), Bytes({1, 2, 3}));
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("shard_path", req.key);
    req.promise.SetResult(ReadResult{
        ReadResult::kMissing, {}, {StorageGeneration::NoValue(), absl::Now()}});
  }
  absl::Time write_time;
  {
    auto req = mock_store->write_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->write_requests.size());
    EXPECT_EQ("shard_path", req.key);
    EXPECT_EQ(StorageGeneration::NoValue(),
              req.options.generation_conditions.if_equal);
    auto expected = Bytes({
        1, 2, 3,  //
    });
    expected.Append(WithCrc32c(Bytes({
        // entries[0].offset
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
        // entries[0].length
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
        // entries[1].offset
        0, 0, 0, 0, 0, 0, 0, 0,  //
        // entries[1].length
        3, 0, 0, 0, 0, 0, 0, 0,  //
    })));
    EXPECT_THAT(req.value, ::testing::Optional(expected));
    write_time = absl::Now();
    req.promise.SetResult(std::in_place, StorageGeneration::FromString("g0"),
                          write_time);
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesTimestampedStorageGeneration(
                  StorageGeneration::FromString("g0"), write_time));
}

TEST_F(UnderlyingKeyValueStoreTest, UnconditionalWrite) {
  grid_shape = {2};
  store = GetStore(grid_shape);
  auto txn = Transaction(tensorstore::isolated);
  auto future1 = kvstore::WriteCommitted(
      KvStore{store, txn}, EntryIdToKey(0, grid_shape), Bytes({1, 2, 3}));
  auto future2 = kvstore::WriteCommitted(
      KvStore{store, txn}, EntryIdToKey(1, grid_shape), Bytes({4, 5, 6}));
  ASSERT_EQ(0, mock_store->read_requests.size());
  ASSERT_EQ(0, mock_store->write_requests.size());
  txn.CommitAsync().IgnoreFuture();
  ASSERT_EQ(0, mock_store->read_requests.size());
  absl::Time write_time;
  {
    auto req = mock_store->write_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->write_requests.size());
    EXPECT_EQ("shard_path", req.key);
    // Since we wrote the maximum number of chunks to shard 0, the write is
    // unconditional.
    EXPECT_EQ(StorageGeneration::Unknown(),
              req.options.generation_conditions.if_equal);
    auto expected = Bytes({
        1, 2, 3,  //
        4, 5, 6,  //
    });
    expected.Append(WithCrc32c(Bytes({
        // entries[0].offset
        0, 0, 0, 0, 0, 0, 0, 0,  //
        // entries[0].length
        3, 0, 0, 0, 0, 0, 0, 0,  //
        // entries[1].offset
        3, 0, 0, 0, 0, 0, 0, 0,  //
        // entries[1].length
        3, 0, 0, 0, 0, 0, 0, 0,  //
    })));
    EXPECT_THAT(req.value, ::testing::Optional(expected));
    write_time = absl::Now();
    req.promise.SetResult(std::in_place, StorageGeneration::FromString("g0"),
                          write_time);
  }
  ASSERT_TRUE(future1.ready());
  ASSERT_TRUE(future2.ready());
  EXPECT_THAT(future1.result(),
              MatchesTimestampedStorageGeneration(
                  StorageGeneration::FromString("g0"), write_time));
  EXPECT_THAT(future2.result(),
              MatchesTimestampedStorageGeneration(
                  StorageGeneration::FromString("g0"), write_time));
}

TEST_F(UnderlyingKeyValueStoreTest, ConditionalWriteDespiteMaxChunks) {
  grid_shape = {2};
  store = GetStore(grid_shape);
  auto future = store->Write(EntryIdToKey(0, grid_shape), Bytes({1, 2, 3}),
                             {/*.if_equal=*/StorageGeneration::NoValue()});
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("shard_path", req.key);
    req.promise.SetResult(ReadResult{
        ReadResult::kMissing, {}, {StorageGeneration::NoValue(), absl::Now()}});
  }
  {
    auto req = mock_store->write_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->write_requests.size());
    EXPECT_EQ("shard_path", req.key);
    // Write is conditional because original write was conditional, despite
    // reaching the maximum number of chunks per shard.
    EXPECT_EQ(StorageGeneration::NoValue(),
              req.options.generation_conditions.if_equal);
  }
}

TEST_F(UnderlyingKeyValueStoreTest, WriteWithNoExistingShardError) {
  auto future = store->Write(EntryIdToKey(1, grid_shape), Bytes({1, 2, 3}));
  future.Force();
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("shard_path", req.key);
    req.promise.SetResult(ReadResult{
        ReadResult::kMissing, {}, {StorageGeneration::NoValue(), absl::Now()}});
  }
  {
    auto req = mock_store->write_requests.pop_nonblock().value();
    req.promise.SetResult(absl::UnknownError("Write error"));
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(), MatchesStatus(absl::StatusCode::kUnknown,
                                             "Error writing \"shard_path\": "
                                             "Write error"));
}

TEST_F(UnderlyingKeyValueStoreTest, WriteWithExistingShard) {
  grid_shape = {2};
  store = GetStore(grid_shape);
  auto future = store->Write(EntryIdToKey(0, grid_shape), Bytes({1, 2, 3}));
  ASSERT_FALSE(future.ready()) << future.status();
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("shard_path", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(),
              req.options.generation_conditions.if_equal);
    EXPECT_EQ(StorageGeneration::Unknown(),
              req.options.generation_conditions.if_not_equal);

    auto content = Bytes({
        4, 5, 6,  //
    });
    content.Append(WithCrc32c(Bytes({
        // entries[0].offset
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
        // entries[0].length
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
        // entries[1].offset
        0, 0, 0, 0, 0, 0, 0, 0,  //
        // entries[1].length
        3, 0, 0, 0, 0, 0, 0, 0,  //
    })));
    req.promise.SetResult(
        ReadResult{ReadResult::kValue,
                   content,
                   {StorageGeneration::FromString("g0"), absl::Now()}});
  }
  ASSERT_FALSE(future.ready()) << future.status();
  absl::Time write_time;
  {
    auto req = mock_store->write_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->write_requests.size());
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("shard_path", req.key);
    EXPECT_EQ(StorageGeneration::FromString("g0"),
              req.options.generation_conditions.if_equal);
    auto content = Bytes({
        1, 2, 3,  //
        4, 5, 6,  //
    });
    content.Append(WithCrc32c(Bytes({
        // entries[0].offset
        0, 0, 0, 0, 0, 0, 0, 0,  //
        // entries[0].length
        3, 0, 0, 0, 0, 0, 0, 0,  //
        // entries[1].offset
        3, 0, 0, 0, 0, 0, 0, 0,  //
        // entries[1].length
        3, 0, 0, 0, 0, 0, 0, 0,  //
    })));
    EXPECT_THAT(req.value, content);
    write_time = absl::Now();
    req.promise.SetResult(std::in_place, StorageGeneration::FromString("g1"),
                          write_time);
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesTimestampedStorageGeneration(
                  StorageGeneration::FromString("g1"), write_time));
}

TEST_F(UnderlyingKeyValueStoreTest, WriteWithExistingShardReadError) {
  auto future = store->Write(EntryIdToKey(1, grid_shape), Bytes({1, 2, 3}));
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("shard_path", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(),
              req.options.generation_conditions.if_equal);
    EXPECT_EQ(StorageGeneration::Unknown(),
              req.options.generation_conditions.if_not_equal);
    req.promise.SetResult(absl::UnknownError("Read error"));
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(), MatchesStatus(absl::StatusCode::kUnknown,
                                             "Error reading \"shard_path\": "
                                             "Read error"));
}

TEST_F(UnderlyingKeyValueStoreTest, DeleteRangeWhenEmpty) {
  grid_shape = {2};
  store = GetStore(grid_shape);
  auto future = store->DeleteRange({});
  future.Force();
  {
    auto req = mock_store->write_requests.pop();
    ASSERT_EQ(0, mock_store->write_requests.size());
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("shard_path", req.key);
    EXPECT_TRUE(StorageGeneration::IsUnknown(
        req.options.generation_conditions.if_equal));
    EXPECT_EQ(std::nullopt, req.value);
    req.promise.SetResult(std::in_place, StorageGeneration::FromString("g1"),
                          absl::Now());
  }
  ASSERT_TRUE(future.ready());
  TENSORSTORE_ASSERT_OK(future);
}

TEST_F(UnderlyingKeyValueStoreTest, BatchRead) {
  cache_pool = CachePool::Make({});
  auto memory_store = tensorstore::GetMemoryKeyValueStore();
  mock_store->forward_to = memory_store;
  mock_store->log_requests = true;
  mock_store->handle_batch_requests = true;
  grid_shape = {3};
  store = GetStore(grid_shape);
  TENSORSTORE_ASSERT_OK(
      store->Write(EntryIdToKey(0, grid_shape), absl::Cord("abc")).result());
  TENSORSTORE_ASSERT_OK(
      store->Write(EntryIdToKey(1, grid_shape), absl::Cord("def")).result());
  mock_store->request_log.pop_all();

  // Read 2/3 entries in a single batch.
  {
    std::vector<Future<kvstore::ReadResult>> futures;
    {
      kvstore::ReadOptions options;
      options.batch = Batch::New();
      futures = {
          store->Read(EntryIdToKey(0, grid_shape), options),
          store->Read(EntryIdToKey(1, grid_shape), options),
      };
    }
    EXPECT_THAT(futures[0].result(), MatchesKvsReadResult(absl::Cord("abc")));
    EXPECT_THAT(futures[1].result(), MatchesKvsReadResult(absl::Cord("def")));
    // Expected to result in a single request for the shard index, followed by a
    // batch request for the two entries.
    EXPECT_THAT(mock_store->request_log.pop_all(), ::testing::SizeIs(2));
  }

  // Read 3/3 entries in a single batch.
  {
    std::vector<Future<kvstore::ReadResult>> futures;
    {
      kvstore::ReadOptions options;
      options.batch = Batch::New();
      futures = {
          store->Read(EntryIdToKey(0, grid_shape), options),
          store->Read(EntryIdToKey(1, grid_shape), options),
          store->Read(EntryIdToKey(2, grid_shape), options),
      };
    }
    EXPECT_THAT(futures[0].result(), MatchesKvsReadResult(absl::Cord("abc")));
    EXPECT_THAT(futures[1].result(), MatchesKvsReadResult(absl::Cord("def")));
    EXPECT_THAT(futures[2].result(), MatchesKvsReadResultNotFound());
    // Expected to result in a single request for the entire shard.
    EXPECT_THAT(mock_store->request_log.pop_all(), ::testing::SizeIs(1));
  }

  // Read 3/3 entries in a single batch with inconsistent generation
  // constraints.
  {
    std::vector<Future<kvstore::ReadResult>> futures;
    {
      kvstore::ReadOptions options1;
      options1.batch = Batch::New();
      kvstore::ReadOptions options2;
      options2.batch = options1.batch;
      options2.generation_conditions.if_not_equal =
          StorageGeneration::Invalid();
      futures = {
          store->Read(EntryIdToKey(0, grid_shape), options1),
          store->Read(EntryIdToKey(1, grid_shape), options1),
          store->Read(EntryIdToKey(2, grid_shape), options2),
      };
    }
    EXPECT_THAT(futures[0].result(), MatchesKvsReadResult(absl::Cord("abc")));
    EXPECT_THAT(futures[1].result(), MatchesKvsReadResult(absl::Cord("def")));
    EXPECT_THAT(futures[2].result(), MatchesKvsReadResultNotFound());
    // Expected to result in a single request for the shard index, followed by a
    // batch request for the two present entries.
    EXPECT_THAT(mock_store->request_log.pop_all(), ::testing::SizeIs(2));
  }
}

// Tests of ReadModifyWrite operations, using `KvsBackedTestCache` ->
// `Uint64ShardedKeyValueStore` -> `MockKeyValueStore`.
class ReadModifyWriteTest : public ::testing::Test {
 protected:
  MockKeyValueStore::MockPtr mock_store = MockKeyValueStore::Make();
  tensorstore::kvstore::DriverPtr memory_store =
      tensorstore::GetMemoryKeyValueStore();

  /// Returns a new (unique) `Uint64ShardedKeyValueStore` backed by
  /// `mock_store`.
  kvstore::DriverPtr GetStore(int64_t num_entries = 100) {
    return GetDefaultStore(mock_store, "shard_path",
                           tensorstore::InlineExecutor{},
                           CachePool::Make(CachePool::Limits{}), {num_entries});
  }

  /// Returns a new (unique) `KvsBackedTestCache` backed by the specified
  /// `KeyValueStore`; if none is specified, calls `GetStore()`.
  auto GetKvsBackedCache(kvstore::DriverPtr store = {}) {
    if (!store) store = GetStore();
    return GetCache<KvsBackedTestCache>(
        CachePool::Make(CachePool::Limits{}).get(), "",
        [&] { return std::make_unique<KvsBackedTestCache>(store); });
  }
};

TEST_F(ReadModifyWriteTest, MultipleCaches) {
  std::vector<Index> grid_shape{100};
  auto cache1 = GetKvsBackedCache();
  auto cache2 = GetKvsBackedCache();
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(GetCacheEntry(cache1, EntryIdToKey(0x0, grid_shape))
                              ->Modify(open_transaction, false, "abc"));
    TENSORSTORE_ASSERT_OK(GetCacheEntry(cache2, EntryIdToKey(0x0, grid_shape))
                              ->Modify(open_transaction, false, "def"));
    auto read_future = GetCacheEntry(cache1, EntryIdToKey(0x0, grid_shape))
                           ->ReadValue(open_transaction);
    // Currently, reading a modified shard is not optimized, such that we end up
    // performing one read of the entire shard, and also one read of the single
    // modified chunk.
    mock_store->read_requests.pop()(memory_store);
    mock_store->read_requests.pop()(memory_store);
    EXPECT_THAT(read_future.result(),
                ::testing::Optional(absl::Cord("abcdef")));
  }
  transaction.CommitAsync().IgnoreFuture();
  auto write_req = mock_store->write_requests.pop();
  write_req(memory_store);
  TENSORSTORE_EXPECT_OK(transaction.future());
}

TEST_F(ReadModifyWriteTest, MultiplePhasesMultipleCaches) {
  std::vector<Index> grid_shape{100};
  auto cache1 = GetKvsBackedCache();
  auto cache2 = GetKvsBackedCache();
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(GetCacheEntry(cache1, EntryIdToKey(0x0, grid_shape))
                              ->Modify(open_transaction, false, "abc"));
    TENSORSTORE_ASSERT_OK(GetCacheEntry(cache2, EntryIdToKey(0x0, grid_shape))
                              ->Modify(open_transaction, false, "def"));
    open_transaction->Barrier();
    TENSORSTORE_ASSERT_OK(GetCacheEntry(cache1, EntryIdToKey(0x0, grid_shape))
                              ->Modify(open_transaction, false, "ghi"));
    TENSORSTORE_ASSERT_OK(GetCacheEntry(cache2, EntryIdToKey(0x0, grid_shape))
                              ->Modify(open_transaction, false, "jkl"));
    auto read_future = GetCacheEntry(cache1, EntryIdToKey(0x0, grid_shape))
                           ->ReadValue(open_transaction);
    // Currently, reading a modified shard is not optimized, such that we end up
    // performing one read of the entire shard, and also one read of the single
    // modified chunk.
    mock_store->read_requests.pop()(memory_store);
    mock_store->read_requests.pop()(memory_store);
    EXPECT_THAT(read_future.result(),
                ::testing::Optional(absl::Cord("abcdefghijkl")));
  }
  transaction.CommitAsync().IgnoreFuture();
  // Handle write request for first phase.
  mock_store->write_requests.pop()(memory_store);
  // Currently, invalidation after the commit of the first phase is not
  // optimized.  We therefore have to re-read the contents of chunk 0, which
  // involves 3 read requests to the underlying key value store (shard index,
  // minishard index, chunk).
  mock_store->read_requests.pop()(memory_store);
  mock_store->read_requests.pop()(memory_store);
  mock_store->read_requests.pop()(memory_store);
  // Handle write request for second phase.
  mock_store->write_requests.pop()(memory_store);
  TENSORSTORE_EXPECT_OK(transaction.future());
}

TENSORSTORE_GLOBAL_INITIALIZER {
  using ::tensorstore::internal::KvsBackedCacheBasicTransactionalTestOptions;
  using ::tensorstore::internal::RegisterKvsBackedCacheBasicTransactionalTest;

  for (bool underlying_atomic : {false, true}) {
    KvsBackedCacheBasicTransactionalTestOptions options;
    const int64_t num_entries = 100;
    options.test_name = tensorstore::StrCat("Uint64Sharded/underlying_atomic=",
                                            underlying_atomic);
    options.get_store = [=] {
      return GetDefaultStore(
          tensorstore::GetMemoryKeyValueStore(/*atomic=*/underlying_atomic),
          "shard_path", tensorstore::InlineExecutor{},
          CachePool::Make(CachePool::Limits{}), {num_entries});
    };
    options.delete_range_supported = true;
    options.multi_key_atomic_supported = true;
    options.get_key_getter = [=] {
      return [getter = std::make_shared<GetKey>(
                  /*sequential_ids=*/true, std::vector<Index>{num_entries})](
                 auto key) { return (*getter)(key); };
    };
    RegisterKvsBackedCacheBasicTransactionalTest(options);
  }
}

TEST(ShardedKeyValueStoreTest, SpecRoundtrip) {
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.roundtrip_key = std::string(8, '\0');
  options.full_base_spec = {{"driver", "memory"}, {"path", "shard_path"}};
  options.full_spec = {
      {"driver", "zarr3_sharding_indexed"},
      {"base", options.full_base_spec},
      {"grid_shape", {100, 200}},
      {"index_location", "end"},
      {"index_codecs",
       {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}}},
  };
  options.check_data_after_serialization = false;
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(ShardedKeyValueStoreTest, SpecRoundtripFile) {
  tensorstore::internal_testing::ScopedTemporaryDirectory tempdir;
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.roundtrip_key = std::string(8, '\0');
  options.full_base_spec = {{"driver", "file"},
                            {"path", tempdir.path() + "/shard_path"}};
  options.full_spec = {
      {"driver", "zarr3_sharding_indexed"},
      {"base", options.full_base_spec},
      {"grid_shape", {100, 200}},
      {"index_location", "end"},
      {"index_codecs",
       {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}}},
  };
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(ShardedKeyValueStoreTest, Base) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      kvstore::Spec::FromJson(
          {{"driver", "zarr3_sharding_indexed"},
           {"base", "memory://abc/"},
           {"grid_shape", {100, 200}},
           {"index_location", "end"},
           {"index_codecs",
            {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}}},
           {"path", "1"}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_spec,
                                   kvstore::Spec::FromJson("memory://abc/"));
  EXPECT_THAT(spec.base(), ::testing::Optional(base_spec));

  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   kvstore::Open(spec, context).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_store,
                                   kvstore::Open(base_spec, context).result());
  EXPECT_THAT(store.base(), ::testing::Optional(base_store));

  // Check that the transaction is propagated to the base.
  auto transaction = tensorstore::Transaction(tensorstore::atomic_isolated);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_with_txn, store | transaction);
  EXPECT_THAT(store_with_txn.base(), base_store | transaction);
}

}  // namespace
