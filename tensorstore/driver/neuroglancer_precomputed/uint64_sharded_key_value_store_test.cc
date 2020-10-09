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

#include "tensorstore/driver/neuroglancer_precomputed/uint64_sharded_key_value_store.h"

#include <functional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/compression/zlib.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/kvs_backed_cache_testutil.h"
#include "tensorstore/internal/thread_pool.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/kvstore/key_value_store_testutil.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

namespace zlib = tensorstore::zlib;
using tensorstore::Future;
using tensorstore::KeyValueStore;
using tensorstore::MatchesStatus;
using tensorstore::OptionalByteRangeRequest;
using tensorstore::Result;
using tensorstore::Status;
using tensorstore::StorageGeneration;
using tensorstore::TimestampedStorageGeneration;
using tensorstore::Transaction;
using tensorstore::internal::CachePool;
using tensorstore::internal::KvsBackedTestCache;
using tensorstore::internal::MatchesKvsReadResult;
using tensorstore::internal::MatchesKvsReadResultNotFound;
using tensorstore::internal::MatchesTimestampedStorageGeneration;
using tensorstore::internal::MockKeyValueStore;
using tensorstore::internal::UniqueNow;
using tensorstore::neuroglancer_uint64_sharded::ChunkIdToKey;
using tensorstore::neuroglancer_uint64_sharded::GetShardedKeyValueStore;
using tensorstore::neuroglancer_uint64_sharded::ShardingSpec;
using ReadResult = KeyValueStore::ReadResult;

constexpr CachePool::Limits kSmallCacheLimits{10000000, 5000000};

absl::Cord Bytes(std::initializer_list<unsigned char> x) {
  return absl::Cord(std::string(x.begin(), x.end()));
}

std::string GetChunkKey(std::uint64_t chunk_id) {
  return ChunkIdToKey({chunk_id});
}

class GetUint64Key {
 public:
  GetUint64Key(bool sequential) : sequential_(sequential) {}

  std::string operator()(std::string key) {
    auto it = key_to_uint64_.find(key);
    if (it == key_to_uint64_.end()) {
      while (true) {
        auto x =
            sequential_ ? next_chunk_id_++ : absl::Uniform<std::uint64_t>(gen_);
        if (uint64_to_key_.emplace(x, key).second) {
          it = key_to_uint64_.emplace(key, x).first;
          break;
        }
      }
    }
    return GetChunkKey(it->second);
  }

 private:
  bool sequential_;
  std::uint64_t next_chunk_id_ = 0;
  absl::BitGen gen_;
  absl::flat_hash_map<std::string, std::uint64_t> key_to_uint64_;
  absl::flat_hash_map<std::uint64_t, std::string> uint64_to_key_;
};

TEST(Uint64ShardedKeyValueStoreTest, BasicFunctionality) {
  std::vector<std::pair<std::string, tensorstore::Executor>> executors{
      {"inline", tensorstore::InlineExecutor{}},
      {"thread_pool", tensorstore::internal::DetachedThreadPool(2)}};
  for (const auto& [executor_name, executor] : executors) {
    for (const auto sequential_ids : {true, false}) {
      for (const auto& hash : {"identity", "murmurhash3_x86_128"}) {
        for (const auto& data_encoding : {"raw", "gzip"}) {
          for (const auto& minishard_index_encoding : {"raw", "gzip"}) {
            for (const auto& sharding_spec_json : ::nlohmann::json::array_t{
                     {{"@type", "neuroglancer_uint64_sharded_v1"},
                      {"hash", hash},
                      {"preshift_bits", 0},
                      {"minishard_bits", 0},
                      {"shard_bits", 0},
                      {"data_encoding", data_encoding},
                      {"minishard_index_encoding", minishard_index_encoding}},
                     {{"@type", "neuroglancer_uint64_sharded_v1"},
                      {"hash", "identity"},
                      {"preshift_bits", 1},
                      {"minishard_bits", 2},
                      {"shard_bits", 3},
                      {"data_encoding", data_encoding},
                      {"minishard_index_encoding", minishard_index_encoding}},
                 }) {
              auto cache_pool = CachePool::Make(kSmallCacheLimits);
              auto base_kv_store = tensorstore::GetMemoryKeyValueStore();
              auto sharding_spec =
                  ShardingSpec::FromJson(sharding_spec_json).value();
              SCOPED_TRACE(executor_name);
              SCOPED_TRACE(sharding_spec_json.dump());

              auto store = GetShardedKeyValueStore(
                  base_kv_store, executor, "prefix", sharding_spec,
                  CachePool::WeakPtr(cache_pool));
              tensorstore::internal::TestKeyValueStoreBasicFunctionality(
                  store, GetUint64Key(sequential_ids));
            }
          }
        }
      }
    }
  }
}

TEST(Uint64ShardedKeyValueStoreTest, DescribeKey) {
  ::nlohmann::json sharding_spec_json{
      {"@type", "neuroglancer_uint64_sharded_v1"},
      {"hash", "identity"},
      {"preshift_bits", 0},
      {"minishard_bits", 1},
      {"shard_bits", 1},
      {"data_encoding", "raw"},
      {"minishard_index_encoding", "raw"}};
  ShardingSpec sharding_spec =
      ShardingSpec::FromJson(sharding_spec_json).value();
  CachePool::StrongPtr cache_pool = CachePool::Make(kSmallCacheLimits);
  KeyValueStore::Ptr base_kv_store = tensorstore::GetMemoryKeyValueStore();
  KeyValueStore::Ptr store = GetShardedKeyValueStore(
      base_kv_store, tensorstore::InlineExecutor{}, "prefix", sharding_spec,
      CachePool::WeakPtr(cache_pool));
  for (const auto& [key, description] :
       std::vector<std::pair<std::uint64_t, std::string>>{
           {0, "chunk 0 in minishard 0 in \"prefix/0.shard\""},
           {1, "chunk 1 in minishard 1 in \"prefix/0.shard\""},
           {2, "chunk 2 in minishard 0 in \"prefix/1.shard\""},
           {3, "chunk 3 in minishard 1 in \"prefix/1.shard\""},
       }) {
    EXPECT_EQ(description, store->DescribeKey(GetChunkKey(key)));
  }
}

class RawEncodingTest : public ::testing::Test {
 protected:
  ::nlohmann::json sharding_spec_json{
      {"@type", "neuroglancer_uint64_sharded_v1"},
      {"hash", "identity"},
      {"preshift_bits", 0},
      {"minishard_bits", 0},
      {"shard_bits", 0},
      {"data_encoding", "raw"},
      {"minishard_index_encoding", "raw"}};
  ShardingSpec sharding_spec =
      ShardingSpec::FromJson(sharding_spec_json).value();
  CachePool::StrongPtr cache_pool = CachePool::Make(kSmallCacheLimits);
  KeyValueStore::Ptr base_kv_store = tensorstore::GetMemoryKeyValueStore();
  KeyValueStore::Ptr store = GetShardedKeyValueStore(
      base_kv_store, tensorstore::InlineExecutor{}, "prefix", sharding_spec,
      CachePool::WeakPtr(cache_pool));
};

TEST_F(RawEncodingTest, MultipleUnconditionalWrites) {
  std::vector<absl::Cord> values{absl::Cord("abc"), absl::Cord("aaaaa"),
                                 absl::Cord("efgh")};
  std::vector<Future<TimestampedStorageGeneration>> futures;
  auto key = GetChunkKey(10);
  for (auto value : values) {
    futures.push_back(store->Write(key, value));
  }
  // Nothing is written until `Force`/`result` is called.
  EXPECT_THAT(GetMap(base_kv_store),
              ::testing::Optional(::testing::ElementsAre()));
  std::vector<Result<TimestampedStorageGeneration>> results;
  for (const auto& future : futures) {
    results.push_back(future.result());
  }
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto shard_read, base_kv_store->Read("prefix/0.shard").result());

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
  std::map<std::string, absl::Cord> values{
      {GetChunkKey(1), absl::Cord("a")},
      {GetChunkKey(2), absl::Cord("bc")},
      {GetChunkKey(3), absl::Cord("def")},
      {GetChunkKey(10), absl::Cord("xyz")}};
  for (auto [key, value] : values) {
    TENSORSTORE_EXPECT_OK(store->Write(key, value));
  }
  EXPECT_THAT(tensorstore::internal::GetMap(store),
              ::testing::Optional(::testing::ElementsAreArray(values)));
}

TEST_F(RawEncodingTest, WritesAndDeletes) {
  auto init_future1 = store->Write(GetChunkKey(1), absl::Cord("a"));
  auto init_future2 = store->Write(GetChunkKey(2), absl::Cord("bc"));
  auto init_future3 = store->Write(GetChunkKey(3), absl::Cord("def"));

  auto gen1 = init_future1.value().generation;
  auto gen2 = init_future2.value().generation;
  auto gen3 = init_future3.value().generation;

  // Conditional delete with mismatched generation.
  auto future1 = store->Delete(GetChunkKey(1), {StorageGeneration::NoValue()});

  // Conditional write with matching generation.
  auto future2 = store->Write(GetChunkKey(2), absl::Cord("ww"), {gen2});
  auto future3 = store->Write(GetChunkKey(2), absl::Cord("xx"), {gen2});

  // Conditional write with matching generation
  auto future4 = store->Write(GetChunkKey(4), absl::Cord("zz"),
                              {StorageGeneration::NoValue()});

  // Conditional delete with matching generation.
  auto future5 = store->Delete(GetChunkKey(3), {gen3});

  EXPECT_THAT(future1.result(), MatchesTimestampedStorageGeneration(
                                    StorageGeneration::Unknown()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto shard_read, base_kv_store->Read("prefix/0.shard").result());

  // Exactly one of `future2` and `future3` succeeds, and the other is aborted
  // due to generation mismatch.
  EXPECT_THAT(
      std::vector({future2.result(), future3.result()}),
      ::testing::UnorderedElementsAre(
          MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()),
          MatchesTimestampedStorageGeneration(shard_read.stamp.generation)));

  EXPECT_THAT(store->Read(GetChunkKey(1)).result(),
              MatchesKvsReadResult(absl::Cord("a")));
  EXPECT_THAT(store->Read(GetChunkKey(2)).result(),
              MatchesKvsReadResult(
                  !StorageGeneration::IsUnknown(future2.result()->generation)
                      ? absl::Cord("ww")
                      : absl::Cord("xx")));
  EXPECT_THAT(store->Read(GetChunkKey(3)).result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(store->Read(GetChunkKey(4)).result(),
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
    std::function<Future<TimestampedStorageGeneration>()> op1) {
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
    all_results.push_back({futures[0].result(), futures[1].result()});
  }
  return all_results;
}

TEST_F(RawEncodingTest, WriteThenDelete) {
  TENSORSTORE_ASSERT_OK(store->Write(GetChunkKey(1), absl::Cord("a")).result());
  EXPECT_THAT(store->Read(GetChunkKey(1)).result(),
              MatchesKvsReadResult(absl::Cord("a")));
  TENSORSTORE_ASSERT_OK(store->Delete(GetChunkKey(1)).result());
  EXPECT_THAT(store->Read(GetChunkKey(1)).result(),
              MatchesKvsReadResultNotFound());
}

TEST_F(RawEncodingTest, MultipleDeleteExisting) {
  StorageGeneration gen;
  EXPECT_THAT(
      TestOrderDependentWrites(
          /*init=*/
          [&] {
            gen = store->Write(GetChunkKey(1), absl::Cord("a"))
                      .value()
                      .generation;
          },
          /*op0=*/
          [&] {
            // Delete conditioned on `gen` is guaranteed to succeed.
            return store->Delete(GetChunkKey(1), {/*.if_equal=*/gen});
          },
          /*op1=*/
          [&] {
            // Delete conditioned on `StorageGeneration::NoValue()` succeeds if
            // it is attempted after the other delete, otherwise it fails.
            return store->Delete(GetChunkKey(1),
                                 {/*.if_equal=*/StorageGeneration::NoValue()});
          }),
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
  EXPECT_THAT(
      TestOrderDependentWrites(
          /*init=*/
          [&] { store->Delete(GetChunkKey(0)).value(); },
          /*op0=*/
          [&] {
            // Write should succeed.
            return store->Write(GetChunkKey(0), absl::Cord("a"));
          },
          /*op1=*/
          [&] {
            // Write should fail due to prior write.
            return store->Write(
                GetChunkKey(0), absl::Cord("b"),
                {/*.if_equal=*/StorageGeneration::FromString("g")});
          }),
      // Regardless of order of operations, the result is the same.
      ::testing::Each(::testing::ElementsAre(
          MatchesTimestampedStorageGeneration(
              ::testing::AllOf(::testing::Not(StorageGeneration::NoValue()),
                               ::testing::Not(StorageGeneration::Invalid()))),
          MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()))));
}

TEST_F(RawEncodingTest, MultipleDeleteNonExisting) {
  std::vector futures{
      store->Delete(GetChunkKey(1), {StorageGeneration::NoValue()}),
      store->Delete(GetChunkKey(1), {StorageGeneration::NoValue()})};
  std::vector results{futures[0].result(), futures[1].result()};
  EXPECT_THAT(
      results,
      ::testing::UnorderedElementsAre(
          MatchesTimestampedStorageGeneration(StorageGeneration::Invalid()),
          MatchesTimestampedStorageGeneration(StorageGeneration::NoValue())));
}

TEST_F(RawEncodingTest, ShardIndexTooShort) {
  base_kv_store->Write("prefix/0.shard", Bytes({1, 2, 3})).value();
  EXPECT_THAT(
      store->Read(GetChunkKey(10)).result(),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Error reading minishard 0 in \"prefix/0\\.shard\": "
          "Error retrieving shard index entry: "
          "Requested byte range \\[0, 16\\) is not valid for value of size 3"));
  EXPECT_THAT(
      store->Write(GetChunkKey(10), absl::Cord("abc")).result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error reading \"prefix/0\\.shard\": "
                    "Existing shard has size 3, but expected at least: 16"));
}

TEST_F(RawEncodingTest, ShardIndexInvalidByteRange) {
  base_kv_store
      ->Write("prefix/0.shard",
              Bytes({10, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0}))
      .value();
  EXPECT_THAT(
      store->Read(GetChunkKey(10)).result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error reading minishard 0 in \"prefix/0\\.shard\": "
                    "Error retrieving shard index entry: "
                    "Shard index specified invalid byte range: \\[10, 2\\)"));
  EXPECT_THAT(store->Write(GetChunkKey(10), absl::Cord("abc")).result(),
              MatchesStatus(
                  absl::StatusCode::kFailedPrecondition,
                  "Error reading \"prefix/0\\.shard\": "
                  "Error decoding existing shard index entry for minishard 0: "
                  "Shard index specified invalid byte range: \\[10, 2\\)"));
}

TEST_F(RawEncodingTest, ShardIndexByteRangeOverflow) {
  base_kv_store
      ->Write("prefix/0.shard",
              Bytes({
                  10, 0, 0, 0, 0, 0, 0, 0,                         //
                  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
              }))
      .value();
  EXPECT_THAT(
      store->Read(GetChunkKey(10)).result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error reading minishard 0 in \"prefix/0\\.shard\": "
                    "Error retrieving shard index entry: "
                    "Byte range .* relative to the end of "
                    "the shard index \\(16\\) is not valid"));
  EXPECT_THAT(store->Write(GetChunkKey(10), absl::Cord("abc")).result(),
              MatchesStatus(
                  absl::StatusCode::kFailedPrecondition,
                  "Error reading \"prefix/0\\.shard\": "
                  "Error decoding existing shard index entry for minishard 0: "
                  "Byte range .* relative to the end of "
                  "the shard index \\(16\\) is not valid"));
}

TEST_F(RawEncodingTest, MinishardIndexOutOfRange) {
  base_kv_store
      ->Write("prefix/0.shard",
              Bytes({0, 0, 0, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0}))
      .value();
  EXPECT_THAT(
      store->Read(GetChunkKey(10)).result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error reading minishard 0 in \"prefix/0\\.shard\": "
                    "Requested byte range \\[16, 64\\) is "
                    "not valid for value of size 16"));
  EXPECT_THAT(store->Write(GetChunkKey(10), absl::Cord("abc")).result(),
              MatchesStatus(
                  absl::StatusCode::kFailedPrecondition,
                  "Error reading \"prefix/0\\.shard\": "
                  "Error decoding existing shard index entry for minishard 0: "
                  "Requested byte range .* is not valid for value of size 16"));
}

TEST_F(RawEncodingTest, MinishardIndexInvalidSize) {
  base_kv_store
      ->Write("prefix/0.shard",
              Bytes({0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}))
      .value();
  EXPECT_THAT(
      store->Read(GetChunkKey(10)).result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error reading minishard 0 in \"prefix/0\\.shard\": "
                    "Invalid minishard index length: 1"));
  EXPECT_THAT(
      store->Write(GetChunkKey(10), absl::Cord("abc")).result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error reading \"prefix/0\\.shard\": "
                    "Error decoding existing minishard index for minishard 0: "
                    "Invalid minishard index length: 1"));
}

TEST_F(RawEncodingTest, MinishardIndexByteRangeOverflow) {
  base_kv_store
      ->Write("prefix/0.shard",
              Bytes({
                  0,    0,    0,    0,    0,    0,    0,    0,     //
                  24,   0,    0,    0,    0,    0,    0,    0,     //
                  10,   0,    0,    0,    0,    0,    0,    0,     //
                  0,    0,    0,    0,    0,    0,    0,    0,     //
                  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
              }))
      .value();
  EXPECT_THAT(
      store->Read(GetChunkKey(10)).result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error reading minishard 0 in \"prefix/0\\.shard\": "
                    "Error decoding minishard index entry "
                    "for chunk 10: Byte range .* relative to the end "
                    "of the shard index \\(16\\) is not valid"));
}

TEST_F(RawEncodingTest, MinishardIndexEntryByteRangeOutOfRange) {
  base_kv_store
      ->Write("prefix/0.shard", Bytes({
                                    0,   0, 0, 0, 0, 0, 0, 0,  //
                                    24,  0, 0, 0, 0, 0, 0, 0,  //
                                    10,  0, 0, 0, 0, 0, 0, 0,  //
                                    0,   0, 0, 0, 0, 0, 0, 0,  //
                                    200, 0, 0, 0, 0, 0, 0, 0,  //
                                }))
      .value();
  EXPECT_THAT(store->Write(GetChunkKey(1), absl::Cord("x")).result(),
              MatchesStatus(
                  absl::StatusCode::kFailedPrecondition,
                  "Error reading \"prefix/0\\.shard\": "
                  "Invalid existing byte range for chunk 10: "
                  "Requested byte range .* is not valid for value of size .*"));
}

TEST_F(RawEncodingTest, MinishardIndexWithDuplicateChunkId) {
  base_kv_store
      ->Write("prefix/0.shard", Bytes({
                                    0,  0, 0, 0, 0, 0, 0, 0,  //
                                    48, 0, 0, 0, 0, 0, 0, 0,  //
                                    10, 0, 0, 0, 0, 0, 0, 0,  //
                                    0,  0, 0, 0, 0, 0, 0, 0,  //
                                    0,  0, 0, 0, 0, 0, 0, 0,  //
                                    0,  0, 0, 0, 0, 0, 0, 0,  //
                                    0,  0, 0, 0, 0, 0, 0, 0,  //
                                    0,  0, 0, 0, 0, 0, 0, 0,  //
                                }))
      .value();
  EXPECT_THAT(store->Write(GetChunkKey(10), absl::Cord("abc")).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error reading \"prefix/0\\.shard\": "
                            "Chunk 10 occurs more than once in the minishard "
                            "index for minishard 0"));
}

class GzipEncodingTest : public ::testing::Test {
 protected:
  ::nlohmann::json sharding_spec_json{
      {"@type", "neuroglancer_uint64_sharded_v1"},
      {"hash", "identity"},
      {"preshift_bits", 0},
      {"minishard_bits", 0},
      {"shard_bits", 0},
      {"data_encoding", "gzip"},
      {"minishard_index_encoding", "gzip"}};
  ShardingSpec sharding_spec =
      ShardingSpec::FromJson(sharding_spec_json).value();
  CachePool::StrongPtr cache_pool = CachePool::Make(kSmallCacheLimits);
  KeyValueStore::Ptr base_kv_store = tensorstore::GetMemoryKeyValueStore();
  KeyValueStore::Ptr store = GetShardedKeyValueStore(
      base_kv_store, tensorstore::InlineExecutor{}, "prefix", sharding_spec,
      CachePool::WeakPtr(cache_pool));
};

TEST_F(GzipEncodingTest, CorruptMinishardGzipEncoding) {
  base_kv_store
      ->Write("prefix/0.shard", Bytes({
                                    0, 0, 0, 0, 0, 0, 0, 0,  //
                                    3, 0, 0, 0, 0, 0, 0, 0,  //
                                    1, 2, 3,                 //
                                }))
      .value();
  EXPECT_THAT(
      store->Read(GetChunkKey(10)).result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error reading minishard 0 in \"prefix/0\\.shard\": "
                    "Error decoding zlib-compressed data"));
  EXPECT_THAT(
      store->Write(GetChunkKey(10), absl::Cord("abc")).result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error reading \"prefix/0\\.shard\": "
                    "Error decoding existing minishard index for minishard 0: "
                    "Error decoding zlib-compressed data"));
}

TEST_F(GzipEncodingTest, CorruptDataGzipEncoding) {
  absl::Cord shard_data("abc");
  zlib::Options zlib_options;
  zlib_options.use_gzip_header = true;
  zlib::Encode(Bytes({
                   10, 0, 0, 0, 0, 0, 0, 0,  //
                   0,  0, 0, 0, 0, 0, 0, 0,  //
                   3,  0, 0, 0, 0, 0, 0, 0,  //
               }),
               &shard_data, zlib_options);
  const unsigned char n = static_cast<unsigned char>(shard_data.size());
  absl::Cord temp = Bytes({
      3, 0, 0, 0, 0, 0, 0, 0,  //
      n, 0, 0, 0, 0, 0, 0, 0,  //
  });
  temp.Append(shard_data);
  TENSORSTORE_ASSERT_OK(base_kv_store->Write("prefix/0.shard", temp));
  EXPECT_THAT(store->Read(GetChunkKey(10)).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error decoding zlib-compressed data"));
}

// Tests of operations issued to underlying KeyValueStore.
class UnderlyingKeyValueStoreTest : public ::testing::Test {
 protected:
  ::nlohmann::json sharding_spec_json{
      {"@type", "neuroglancer_uint64_sharded_v1"},
      {"hash", "identity"},
      {"preshift_bits", 0},
      {"minishard_bits", 1},
      {"shard_bits", 1},
      {"data_encoding", "raw"},
      {"minishard_index_encoding", "raw"}};
  ShardingSpec sharding_spec =
      ShardingSpec::FromJson(sharding_spec_json).value();
  CachePool::StrongPtr cache_pool = CachePool::Make(kSmallCacheLimits);
  KeyValueStore::PtrT<MockKeyValueStore> mock_store{new MockKeyValueStore};
  KeyValueStore::Ptr GetStore(
      tensorstore::neuroglancer_uint64_sharded::GetMaxChunksPerShardFunction
          get_max_chunks_per_shard = {}) {
    return GetShardedKeyValueStore(
        mock_store, tensorstore::InlineExecutor{}, "prefix", sharding_spec,
        CachePool::WeakPtr(cache_pool), std::move(get_max_chunks_per_shard));
  }
  KeyValueStore::Ptr store = GetStore();
};

// Tests that a Read operation results in the expected sequence of calls to the
// underlying KeyValueStore.
TEST_F(UnderlyingKeyValueStoreTest, Read) {
  // Perform initial successful read.
  absl::Time init_time = UniqueNow();
  absl::Time minishard_index_time;
  {
    auto future = store->Read(GetChunkKey(0x50), {});
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(0, 16), req.options.byte_range);
      EXPECT_THAT(req.options.staleness_bound, ::testing::Gt(init_time));
      req.promise.SetResult(KeyValueStore::ReadResult{
          ReadResult::kValue,
          Bytes({
              5, 0, 0, 0, 0, 0, 0, 0,   //
              31, 0, 0, 0, 0, 0, 0, 0,  //
          }),
          {StorageGeneration::FromString("g0"), absl::Now()}});
    }
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
      EXPECT_EQ(StorageGeneration::FromString("g0"), req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(37, 63), req.options.byte_range);
      minishard_index_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          ReadResult::kValue,
          Bytes({
              0x50, 0, 0, 0, 0, 0, 0, 0,  //
              0,    0, 0, 0, 0, 0, 0, 0,  //
              5,    0, 0, 0, 0, 0, 0, 0,  //
          }),
          {StorageGeneration::FromString("g0"), minishard_index_time}});
    }
    absl::Time read_time;
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
      EXPECT_EQ(StorageGeneration::FromString("g0"), req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(32, 37), req.options.byte_range);
      read_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          ReadResult::kValue,
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
    KeyValueStore::ReadOptions options;
    options.staleness_bound = init_time;
    auto future = store->Read(GetChunkKey(0x60), options);
    ASSERT_TRUE(future.ready());
    EXPECT_THAT(future.result(),
                MatchesKvsReadResultNotFound(minishard_index_time));
  }

  // Same as above, but ensures cached minishard index is up to date.
  {
    auto req_time = UniqueNow();
    auto future = store->Read(GetChunkKey(0x60), {});
    // Request for shard index entry if modified.
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::FromString("g0"), req.options.if_not_equal);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(0, 16), req.options.byte_range);
      EXPECT_THAT(req.options.staleness_bound, ::testing::Gt(req_time));
      minishard_index_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          {StorageGeneration::FromString("g0"), minishard_index_time}});
    }
    ASSERT_TRUE(future.ready());
    EXPECT_THAT(future.result(),
                MatchesKvsReadResultNotFound(minishard_index_time));
  }

  // Issue a read for present chunk that hits cached minishard index.
  {
    KeyValueStore::ReadOptions options;
    options.staleness_bound = init_time;
    auto future = store->Read(GetChunkKey(0x50), options);
    absl::Time read_time;
    // Request for data based on cached minishard.
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
      EXPECT_EQ(StorageGeneration::FromString("g0"), req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(32, 37), req.options.byte_range);
      EXPECT_EQ(init_time, req.options.staleness_bound);
      read_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          ReadResult::kValue,
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
    KeyValueStore::ReadOptions options;
    options.staleness_bound = init_time;
    auto future = store->Read(GetChunkKey(0x50), options);
    absl::Time abort_time;
    // Initial request for data based on cached minishard.
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
      EXPECT_EQ(init_time, req.options.staleness_bound);
      EXPECT_EQ(StorageGeneration::FromString("g0"), req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(32, 37), req.options.byte_range);
      abort_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          {StorageGeneration::FromString("g0"), abort_time}});
    }
    // Request for updated shard index.
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::FromString("g0"), req.options.if_not_equal);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(0, 16), req.options.byte_range);
      EXPECT_THAT(req.options.staleness_bound, ::testing::Ge(abort_time));
      req.promise.SetResult(KeyValueStore::ReadResult{
          ReadResult::kValue,
          Bytes({
              6, 0, 0, 0, 0, 0, 0, 0,   //
              32, 0, 0, 0, 0, 0, 0, 0,  //
          }),
          {StorageGeneration::FromString("g1"), absl::Now()}});
    }
    // Request for updated minishard index.
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
      EXPECT_EQ(StorageGeneration::FromString("g1"), req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(38, 64), req.options.byte_range);
      minishard_index_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          ReadResult::kValue,
          Bytes({
              0x50, 0, 0, 0, 0, 0, 0, 0,  //
              0,    0, 0, 0, 0, 0, 0, 0,  //
              6,    0, 0, 0, 0, 0, 0, 0,  //
          }),
          {StorageGeneration::FromString("g1"), minishard_index_time}});
    }
    // Request for value.
    absl::Time read_time;
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
      EXPECT_EQ(StorageGeneration::FromString("g1"), req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(32, 38), req.options.byte_range);
      read_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          ReadResult::kValue,
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

// Tests issuing read for chunk in uncached minishard index while there is a
// concurrent modification.
TEST_F(UnderlyingKeyValueStoreTest,
       ReadConcurrentModificationAfterReadingShardIndex) {
  absl::Time init_time = absl::Now();
  KeyValueStore::ReadOptions options;
  options.staleness_bound = init_time;
  auto future = store->Read(GetChunkKey(0x1), options);
  absl::Time abort_time;
  // Request for shard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
    EXPECT_EQ(init_time, req.options.staleness_bound);
    EXPECT_EQ(OptionalByteRangeRequest(16, 32), req.options.byte_range);
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kValue,
        Bytes({
            6, 0, 0, 0, 0, 0, 0, 0,   //
            32, 0, 0, 0, 0, 0, 0, 0,  //
        }),
        {StorageGeneration::FromString("g2"), absl::Now()}});
  }
  // Request for minishard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ(StorageGeneration::FromString("g2"), req.options.if_equal);
    EXPECT_EQ(OptionalByteRangeRequest(38, 64), req.options.byte_range);
    abort_time = absl::Now();
    req.promise.SetResult(KeyValueStore::ReadResult{
        {StorageGeneration::FromString("g2"), abort_time}});
  }
  // Request for updated shard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
    EXPECT_THAT(req.options.staleness_bound, ::testing::Ge(abort_time));
    EXPECT_EQ(OptionalByteRangeRequest(16, 32), req.options.byte_range);
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kValue,
        Bytes({
            7, 0, 0, 0, 0, 0, 0, 0,   //
            33, 0, 0, 0, 0, 0, 0, 0,  //
        }),
        {StorageGeneration::FromString("g3"), absl::Now()}});
  }
  // Request for updated minishard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ(StorageGeneration::FromString("g3"), req.options.if_equal);
    EXPECT_EQ(OptionalByteRangeRequest(39, 65), req.options.byte_range);
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kValue,
        Bytes({
            0x1, 0, 0, 0, 0, 0, 0, 0,  //
            0,   0, 0, 0, 0, 0, 0, 0,  //
            4,   0, 0, 0, 0, 0, 0, 0,  //
        }),
        {StorageGeneration::FromString("g3"), absl::Now()}});
  }
  // Request for value.
  absl::Time read_time;
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ(StorageGeneration::FromString("g3"), req.options.if_equal);
    EXPECT_EQ(OptionalByteRangeRequest(32, 36), req.options.byte_range);
    read_time = absl::Now();
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kValue,
        Bytes({4, 5, 6, 7}),
        {StorageGeneration::FromString("g3"), read_time}});
  }
  ASSERT_EQ(0, mock_store->read_requests.size());
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(
      future.result(),
      MatchesKvsReadResult(Bytes({4, 5, 6, 7}),
                           StorageGeneration::FromString("g3"), read_time));
}

// Tests issuing read for chunk in uncached minishard index while the shard is
// concurrently deleted (before the minishard index can be read).
TEST_F(UnderlyingKeyValueStoreTest,
       ReadConcurrentDeleteAfterReadingShardIndex) {
  auto req_time = UniqueNow();
  auto future = store->Read(GetChunkKey(0x1), {});
  // Request for shard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
    EXPECT_THAT(req.options.staleness_bound, ::testing::Gt(req_time));
    EXPECT_EQ(OptionalByteRangeRequest(16, 32), req.options.byte_range);
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kValue,
        Bytes({
            6, 0, 0, 0, 0, 0, 0, 0,   //
            32, 0, 0, 0, 0, 0, 0, 0,  //
        }),
        {StorageGeneration::FromString("g4"), absl::Now()}});
  }
  // Request for minishard index.
  absl::Time read_time;
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ(StorageGeneration::FromString("g4"), req.options.if_equal);
    EXPECT_EQ(OptionalByteRangeRequest(38, 64), req.options.byte_range);
    read_time = absl::Now();
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kMissing, {}, {StorageGeneration::NoValue(), read_time}});
  }
  ASSERT_EQ(0, mock_store->read_requests.size());
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(), MatchesKvsReadResultNotFound(read_time));
}

// Tests issuing read for chunk in uncached minishard index while the shard is
// concurrently deleted (after the minishard index can be read).
TEST_F(UnderlyingKeyValueStoreTest,
       ReadConcurrentDeleteAfterReadingMinishardIndex) {
  auto req_time = UniqueNow();
  auto future = store->Read(GetChunkKey(0x1), {});
  // Request for shard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
    EXPECT_THAT(req.options.staleness_bound, ::testing::Gt(req_time));
    EXPECT_EQ(OptionalByteRangeRequest(16, 32), req.options.byte_range);
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kValue,
        Bytes({
            6, 0, 0, 0, 0, 0, 0, 0,   //
            32, 0, 0, 0, 0, 0, 0, 0,  //
        }),
        {StorageGeneration::FromString("g0"), absl::Now()}});
  }
  // Request for minishard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ(StorageGeneration::FromString("g0"), req.options.if_equal);
    EXPECT_EQ(OptionalByteRangeRequest(38, 64), req.options.byte_range);
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kValue,
        Bytes({
            0x1, 0, 0, 0, 0, 0, 0, 0,  //
            0,   0, 0, 0, 0, 0, 0, 0,  //
            4,   0, 0, 0, 0, 0, 0, 0,  //
        }),
        {StorageGeneration::FromString("g0"), absl::Now()}});
  }
  // Request for value.
  absl::Time read_time;
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ(StorageGeneration::FromString("g0"), req.options.if_equal);
    EXPECT_EQ(OptionalByteRangeRequest(32, 36), req.options.byte_range);
    read_time = absl::Now();
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kMissing, {}, {StorageGeneration::NoValue(), read_time}});
  }
  ASSERT_EQ(0, mock_store->read_requests.size());
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(), MatchesKvsReadResultNotFound(read_time));
}

TEST_F(UnderlyingKeyValueStoreTest, ReadErrorReadingShardIndex) {
  auto future = store->Read(GetChunkKey(0x50), {});
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(OptionalByteRangeRequest(0, 16), req.options.byte_range);
    req.promise.SetResult(absl::UnknownError("Read error"));
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(
      future.result(),
      MatchesStatus(absl::StatusCode::kUnknown,
                    "Error reading minishard 0 in \"prefix/0\\.shard\": "
                    "Error retrieving shard index entry: "
                    "Read error"));
}

TEST_F(UnderlyingKeyValueStoreTest, ReadErrorReadingMinishardShardIndex) {
  auto future = store->Read(GetChunkKey(0x1), {});
  // Request for shard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(OptionalByteRangeRequest(16, 32), req.options.byte_range);
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kValue,
        Bytes({
            6, 0, 0, 0, 0, 0, 0, 0,   //
            32, 0, 0, 0, 0, 0, 0, 0,  //
        }),
        {StorageGeneration::FromString("g0"), absl::Now()}});
  }
  // Request for minishard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(OptionalByteRangeRequest(38, 64), req.options.byte_range);
    req.promise.SetResult(absl::UnknownError("Read error"));
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(
      future.result(),
      MatchesStatus(absl::StatusCode::kUnknown,
                    "Error reading minishard 1 in \"prefix/0\\.shard\": "
                    "Read error"));
}

TEST_F(UnderlyingKeyValueStoreTest, ReadErrorReadingData) {
  auto future = store->Read(GetChunkKey(0x1), {});
  // Request for shard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(OptionalByteRangeRequest(16, 32), req.options.byte_range);
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kValue,
        Bytes({
            6, 0, 0, 0, 0, 0, 0, 0,   //
            32, 0, 0, 0, 0, 0, 0, 0,  //
        }),
        {StorageGeneration::FromString("g0"), absl::Now()}});
  }
  // Request for minishard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(OptionalByteRangeRequest(38, 64), req.options.byte_range);
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kValue,
        Bytes({
            0x1, 0, 0, 0, 0, 0, 0, 0,  //
            0,   0, 0, 0, 0, 0, 0, 0,  //
            4,   0, 0, 0, 0, 0, 0, 0,  //
        }),
        {StorageGeneration::FromString("g0"), absl::Now()}});
  }
  // Request for value.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(OptionalByteRangeRequest(32, 36), req.options.byte_range);
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
  for (const bool with_max_chunks : {false, true}) {
    SCOPED_TRACE(tensorstore::StrCat("with_max_chunks=", with_max_chunks));
    // Specifying the `get_max_chunks_per_shard` function has no effect because
    // we only write 1 chunk, which is not equal to the maximum of 2.
    if (with_max_chunks) {
      store = GetStore(
          /*get_max_chunks_per_shard=*/[](std::uint64_t shard)
                                           -> std::uint64_t { return 2; });
    } else {
      store = GetStore();
    }
    auto future = store->Write(GetChunkKey(0x50), Bytes({1, 2, 3}));
    ASSERT_EQ(0, mock_store->read_requests.size());
    ASSERT_EQ(0, mock_store->write_requests.size());
    future.Force();
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      req.promise.SetResult(KeyValueStore::ReadResult{
          ReadResult::kMissing,
          {},
          {StorageGeneration::NoValue(), absl::Now()}});
    }
    absl::Time write_time;
    {
      auto req = mock_store->write_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->write_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::NoValue(), req.options.if_equal);
      EXPECT_THAT(req.value, ::testing::Optional(Bytes({
                                 3,    0, 0, 0, 0, 0, 0, 0,  //
                                 27,   0, 0, 0, 0, 0, 0, 0,  //
                                 0,    0, 0, 0, 0, 0, 0, 0,  //
                                 0,    0, 0, 0, 0, 0, 0, 0,  //
                                 1,    2, 3,                 //
                                 0x50, 0, 0, 0, 0, 0, 0, 0,  //
                                 0,    0, 0, 0, 0, 0, 0, 0,  //
                                 3,    0, 0, 0, 0, 0, 0, 0,  //
                             })));
      write_time = absl::Now();
      req.promise.SetResult(std::in_place, StorageGeneration::FromString("g0"),
                            write_time);
    }
    ASSERT_TRUE(future.ready());
    EXPECT_THAT(future.result(),
                MatchesTimestampedStorageGeneration(
                    StorageGeneration::FromString("g0"), write_time));
  }
}

TEST_F(UnderlyingKeyValueStoreTest, UnconditionalWrite) {
  store = GetStore(
      /*get_max_chunks_per_shard=*/[](std::uint64_t shard) -> std::uint64_t {
        return 2;
      });
  auto future1 = store->Write(GetChunkKey(0x50), Bytes({1, 2, 3}));
  auto future2 = store->Write(GetChunkKey(0x54), Bytes({4, 5, 6}));
  ASSERT_EQ(0, mock_store->read_requests.size());
  ASSERT_EQ(0, mock_store->write_requests.size());
  future1.Force();
  future2.Force();
  ASSERT_EQ(0, mock_store->read_requests.size());
  absl::Time write_time;
  {
    auto req = mock_store->write_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->write_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    // Since we wrote the maximum number of chunks to shard 0, the write is
    // unconditional.
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
    EXPECT_THAT(req.value, ::testing::Optional(Bytes({
                               6,    0, 0, 0, 0, 0, 0, 0,  //
                               54,   0, 0, 0, 0, 0, 0, 0,  //
                               0,    0, 0, 0, 0, 0, 0, 0,  //
                               0,    0, 0, 0, 0, 0, 0, 0,  //
                               1,    2, 3,                 //
                               4,    5, 6,                 //
                               0x50, 0, 0, 0, 0, 0, 0, 0,  //
                               0x04, 0, 0, 0, 0, 0, 0, 0,  //
                               0,    0, 0, 0, 0, 0, 0, 0,  //
                               0,    0, 0, 0, 0, 0, 0, 0,  //
                               3,    0, 0, 0, 0, 0, 0, 0,  //
                               3,    0, 0, 0, 0, 0, 0, 0,  //
                           })));
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
  store = GetStore(
      /*get_max_chunks_per_shard=*/[](std::uint64_t shard) -> std::uint64_t {
        return 1;
      });
  auto future = store->Write(GetChunkKey(0x50), Bytes({1, 2, 3}),
                             {/*.if_equal=*/StorageGeneration::NoValue()});
  ASSERT_EQ(0, mock_store->read_requests.size());
  ASSERT_EQ(0, mock_store->write_requests.size());
  future.Force();
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kMissing, {}, {StorageGeneration::NoValue(), absl::Now()}});
  }
  {
    auto req = mock_store->write_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->write_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    // Write is conditional because original write was conditional, despite
    // reaching the maximum number of chunks per shard.
    EXPECT_EQ(StorageGeneration::NoValue(), req.options.if_equal);
  }
}

TEST_F(UnderlyingKeyValueStoreTest, WriteWithNoExistingShardError) {
  auto future = store->Write(GetChunkKey(0x50), Bytes({1, 2, 3}));
  future.Force();
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kMissing, {}, {StorageGeneration::NoValue(), absl::Now()}});
  }
  {
    auto req = mock_store->write_requests.pop_nonblock().value();
    req.promise.SetResult(absl::UnknownError("Write error"));
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown,
                            "Error writing \"prefix/0\\.shard\": "
                            "Write error"));
}

TEST_F(UnderlyingKeyValueStoreTest, WriteWithExistingShard) {
  auto future = store->Write(GetChunkKey(0x50), Bytes({1, 2, 3}));

  ASSERT_EQ(0, mock_store->read_requests.size());
  ASSERT_EQ(0, mock_store->write_requests.size());
  future.Force();
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    req.promise.SetResult(KeyValueStore::ReadResult{
        ReadResult::kValue,
        Bytes({
            3,    0, 0, 0, 0, 0, 0, 0,  //
            27,   0, 0, 0, 0, 0, 0, 0,  //
            0,    0, 0, 0, 0, 0, 0, 0,  //
            0,    0, 0, 0, 0, 0, 0, 0,  //
            4,    5, 6,                 //
            0x70, 0, 0, 0, 0, 0, 0, 0,  //
            0,    0, 0, 0, 0, 0, 0, 0,  //
            3,    0, 0, 0, 0, 0, 0, 0,  //
        }),
        {StorageGeneration::FromString("g0"), absl::Now()}});
  }
  absl::Time write_time;
  {
    auto req = mock_store->write_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->write_requests.size());
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::FromString("g0"), req.options.if_equal);
    EXPECT_THAT(req.value, ::testing::Optional(Bytes({
                               6,    0, 0, 0, 0, 0, 0, 0,  //
                               54,   0, 0, 0, 0, 0, 0, 0,  //
                               0,    0, 0, 0, 0, 0, 0, 0,  //
                               0,    0, 0, 0, 0, 0, 0, 0,  //
                               1,    2, 3,                 //
                               4,    5, 6,                 //
                               0x50, 0, 0, 0, 0, 0, 0, 0,  //
                               0x20, 0, 0, 0, 0, 0, 0, 0,  //
                               0,    0, 0, 0, 0, 0, 0, 0,  //
                               0,    0, 0, 0, 0, 0, 0, 0,  //
                               3,    0, 0, 0, 0, 0, 0, 0,  //
                               3,    0, 0, 0, 0, 0, 0, 0,  //
                           })));
    write_time = absl::Now();
    req.promise.SetResult(std::in_place, StorageGeneration::FromString("g1"),
                          write_time);
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesTimestampedStorageGeneration(
                  StorageGeneration::FromString("g1"), write_time));
}

TEST_F(UnderlyingKeyValueStoreTest, WriteMaxChunksWithExistingShard) {
  for (const bool specify_max_chunks : {false, true}) {
    if (specify_max_chunks) {
      store = GetStore(
          /*get_max_chunks_per_shard=*/[](std::uint64_t shard)
                                           -> std::uint64_t { return 1; });
    }
    auto future = store->Write(GetChunkKey(0x50), Bytes({1, 2, 3}));

    ASSERT_EQ(0, mock_store->read_requests.size());
    ASSERT_EQ(0, mock_store->write_requests.size());
    future.Force();
    if (!specify_max_chunks) {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      req.promise.SetResult(KeyValueStore::ReadResult{
          ReadResult::kMissing,
          {},
          {StorageGeneration::NoValue(), absl::Now()}});
    }
    absl::Time write_time;
    {
      auto req = mock_store->write_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->write_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ((specify_max_chunks ? StorageGeneration::Unknown()
                                    : StorageGeneration::NoValue()),
                req.options.if_equal);
      EXPECT_THAT(req.value, ::testing::Optional(Bytes({
                                 3,    0, 0, 0, 0, 0, 0, 0,  //
                                 27,   0, 0, 0, 0, 0, 0, 0,  //
                                 0,    0, 0, 0, 0, 0, 0, 0,  //
                                 0,    0, 0, 0, 0, 0, 0, 0,  //
                                 1,    2, 3,                 //
                                 0x50, 0, 0, 0, 0, 0, 0, 0,  //
                                 0,    0, 0, 0, 0, 0, 0, 0,  //
                                 3,    0, 0, 0, 0, 0, 0, 0,  //
                             })));
      write_time = absl::Now();
      req.promise.SetResult(std::in_place, StorageGeneration::FromString("g0"),
                            write_time);
    }
    ASSERT_TRUE(future.ready());
    EXPECT_THAT(future.result(),
                MatchesTimestampedStorageGeneration(
                    StorageGeneration::FromString("g0"), write_time));
  }
}

TEST_F(UnderlyingKeyValueStoreTest, WriteWithExistingShardReadError) {
  auto future = store->Write(GetChunkKey(0x50), Bytes({1, 2, 3}));

  ASSERT_EQ(0, mock_store->read_requests.size());
  ASSERT_EQ(0, mock_store->write_requests.size());
  future.Force();
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    req.promise.SetResult(absl::UnknownError("Read error"));
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown,
                            "Error reading \"prefix/0\\.shard\": "
                            "Read error"));
}

TEST_F(UnderlyingKeyValueStoreTest, DeleteRangeUnimplemented) {
  EXPECT_THAT(store->DeleteRange(tensorstore::KeyRange::Prefix("abc")).result(),
              MatchesStatus(absl::StatusCode::kUnimplemented));
}

TEST_F(UnderlyingKeyValueStoreTest, TransactionalDeleteRangeUnimplemented) {
  EXPECT_THAT(
      store->TransactionalDeleteRange({}, tensorstore::KeyRange::Prefix("abc")),
      MatchesStatus(absl::StatusCode::kUnimplemented));
}

// Tests of ReadModifyWrite operations, using `KvsBackedTestCache` ->
// `Uint64ShardedKeyValueStore` -> `MockKeyValueStore`.
class ReadModifyWriteTest : public ::testing::Test {
 protected:
  ::nlohmann::json sharding_spec_json{
      {"@type", "neuroglancer_uint64_sharded_v1"},
      {"hash", "identity"},
      {"preshift_bits", 0},
      {"minishard_bits", 1},
      {"shard_bits", 1},
      {"data_encoding", "raw"},
      {"minishard_index_encoding", "raw"}};
  ShardingSpec sharding_spec =
      ShardingSpec::FromJson(sharding_spec_json).value();
  KeyValueStore::PtrT<MockKeyValueStore> mock_store{new MockKeyValueStore};
  tensorstore::KeyValueStore::Ptr memory_store =
      tensorstore::GetMemoryKeyValueStore();

  /// Returns a new (unique) `Uint64ShardedKeyValueStore` backed by
  /// `mock_store`.
  KeyValueStore::Ptr GetStore(
      tensorstore::neuroglancer_uint64_sharded::GetMaxChunksPerShardFunction
          get_max_chunks_per_shard = {}) {
    return GetShardedKeyValueStore(
        mock_store, tensorstore::InlineExecutor{}, "prefix", sharding_spec,
        CachePool::WeakPtr(CachePool::Make(CachePool::Limits{})),
        std::move(get_max_chunks_per_shard));
  }

  /// Returns a new (unique) `KvsBackedTestCache` backed by the specified
  /// `KeyValueStore`; if none is specified, calls `GetStore()`.
  auto GetKvsBackedCache(KeyValueStore::Ptr store = {}) {
    if (!store) store = GetStore();
    return CachePool::Make(CachePool::Limits{})
        ->GetCache<KvsBackedTestCache>(
            "", [&] { return std::make_unique<KvsBackedTestCache>(store); });
  }
};

TEST_F(ReadModifyWriteTest, MultipleCaches) {
  auto cache1 = GetKvsBackedCache();
  auto cache2 = GetKvsBackedCache();
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(GetCacheEntry(cache1, GetChunkKey(0x0))
                              ->Modify(open_transaction, false, "abc"));
    TENSORSTORE_ASSERT_OK(GetCacheEntry(cache2, GetChunkKey(0x0))
                              ->Modify(open_transaction, false, "def"));
    auto read_future =
        GetCacheEntry(cache1, GetChunkKey(0x0))->ReadValue(open_transaction);
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
  auto cache1 = GetKvsBackedCache();
  auto cache2 = GetKvsBackedCache();
  auto transaction = Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_transaction,
        tensorstore::internal::AcquireOpenTransactionPtrOrError(transaction));
    TENSORSTORE_ASSERT_OK(GetCacheEntry(cache1, GetChunkKey(0x0))
                              ->Modify(open_transaction, false, "abc"));
    TENSORSTORE_ASSERT_OK(GetCacheEntry(cache2, GetChunkKey(0x0))
                              ->Modify(open_transaction, false, "def"));
    open_transaction->Barrier();
    TENSORSTORE_ASSERT_OK(GetCacheEntry(cache1, GetChunkKey(0x0))
                              ->Modify(open_transaction, false, "ghi"));
    TENSORSTORE_ASSERT_OK(GetCacheEntry(cache2, GetChunkKey(0x0))
                              ->Modify(open_transaction, false, "jkl"));
    auto read_future =
        GetCacheEntry(cache1, GetChunkKey(0x0))->ReadValue(open_transaction);
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
  using tensorstore::internal::KvsBackedCacheBasicTransactionalTestOptions;
  using tensorstore::internal::RegisterKvsBackedCacheBasicTransactionalTest;

  ::nlohmann::json sharding_spec_json{
      {"@type", "neuroglancer_uint64_sharded_v1"},
      {"hash", "identity"},
      {"preshift_bits", 0},
      {"minishard_bits", 1},
      {"shard_bits", 1},
      {"data_encoding", "raw"},
      {"minishard_index_encoding", "raw"}};
  ShardingSpec sharding_spec =
      ShardingSpec::FromJson(sharding_spec_json).value();

  for (bool underlying_atomic : {false, true}) {
    KvsBackedCacheBasicTransactionalTestOptions options;
    options.test_name = tensorstore::StrCat("Uint64Sharded/underlying_atomic=",
                                            underlying_atomic);
    options.get_store = [=] {
      return GetShardedKeyValueStore(
          tensorstore::GetMemoryKeyValueStore(/*atomic=*/underlying_atomic),
          tensorstore::InlineExecutor{}, "prefix", sharding_spec,
          CachePool::WeakPtr(CachePool::Make(CachePool::Limits{})), {});
    };
    options.delete_range_supported = false;
    options.multi_key_atomic_supported = true;
    options.get_key_getter = [] {
      return [getter = std::make_shared<GetUint64Key>(/*sequential_ids=*/true)](
                 auto key) { return (*getter)(key); };
    };
    RegisterKvsBackedCacheBasicTransactionalTest(options);
  }
}

}  // namespace
