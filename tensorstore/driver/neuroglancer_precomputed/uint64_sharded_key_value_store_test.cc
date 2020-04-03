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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/compression/zlib.h"
#include "tensorstore/internal/thread_pool.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/kvstore/key_value_store_testutil.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

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
using tensorstore::internal::CachePool;
using tensorstore::internal::MatchesKvsReadResult;
using tensorstore::internal::MatchesKvsReadResultNotFound;
using tensorstore::internal::MatchesTimestampedStorageGeneration;
using tensorstore::internal::MockKeyValueStore;
using tensorstore::internal::UniqueNow;
using tensorstore::neuroglancer_uint64_sharded::GetShardedKeyValueStore;
using tensorstore::neuroglancer_uint64_sharded::ShardingSpec;

constexpr CachePool::Limits kSmallCacheLimits{10000000, 5000000};

std::string Bytes(std::initializer_list<unsigned char> x) {
  return std::string(x.begin(), x.end());
}

std::string GetChunkKey(std::uint64_t chunk_id) {
  return std::string(reinterpret_cast<const char*>(&chunk_id),
                     sizeof(std::uint64_t));
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
  for (const auto [executor_name, executor] : executors) {
    for (const auto sequential_ids : {true, false}) {
      for (const auto hash : {"identity", "murmurhash3_x86_128"}) {
        for (const auto data_encoding : {"raw", "gzip"}) {
          for (const auto minishard_index_encoding : {"raw", "gzip"}) {
            for (const auto sharding_spec_json : ::nlohmann::json::array_t{
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
  std::vector<std::string> values{"abc", "aaaaa", "efgh"};
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
  auto shard_generation = base_kv_store->Read("prefix/0.shard").result();
  ASSERT_EQ(Status(), GetStatus(shard_generation));

  // All writes succeed, but all but one write is assigned a generation of
  // `StorageGeneration::Invalid()` since it is overwritten immediately before
  // it is ever actually committed to the `base_kv_store`.
  EXPECT_THAT(
      results,
      ::testing::UnorderedElementsAre(
          MatchesTimestampedStorageGeneration(StorageGeneration::Invalid()),
          MatchesTimestampedStorageGeneration(StorageGeneration::Invalid()),
          MatchesTimestampedStorageGeneration(
              shard_generation->generation.generation)));
  for (size_t i = 0; i < results.size(); ++i) {
    if (results[i] &&
        results[i]->generation == shard_generation->generation.generation) {
      EXPECT_THAT(store->Read(key).result(),
                  MatchesKvsReadResult(values[i], results[i]->generation));
    }
  }
}

TEST_F(RawEncodingTest, WritesAndDeletes) {
  auto init_future1 = store->Write(GetChunkKey(1), "a");
  auto init_future2 = store->Write(GetChunkKey(2), "bc");
  auto init_future3 = store->Write(GetChunkKey(3), "def");

  auto gen1 = init_future1.value().generation;
  auto gen2 = init_future2.value().generation;
  auto gen3 = init_future3.value().generation;

  // Conditional delete with mismatched generation.
  auto future1 = store->Delete(GetChunkKey(1), {StorageGeneration::NoValue()});

  // Conditional write with matching generation.
  auto future2 = store->Write(GetChunkKey(2), "ww", {gen2});
  auto future3 = store->Write(GetChunkKey(2), "xx", {gen2});

  // Conditional write with matching generation
  auto future4 =
      store->Write(GetChunkKey(4), "zz", {StorageGeneration::NoValue()});

  // Conditional delete with matching generation.
  auto future5 = store->Delete(GetChunkKey(3), {gen3});

  EXPECT_THAT(future1.result(), MatchesTimestampedStorageGeneration(
                                    StorageGeneration::Unknown()));

  auto shard_generation = base_kv_store->Read("prefix/0.shard").result();
  ASSERT_EQ(Status(), GetStatus(shard_generation));

  // Exactly one of `future2` and `future3` succeeds, and the other is aborted
  // due to generation mismatch.
  EXPECT_THAT(
      std::vector({future2.result(), future3.result()}),
      ::testing::UnorderedElementsAre(
          MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()),
          MatchesTimestampedStorageGeneration(
              shard_generation->generation.generation)));

  EXPECT_THAT(store->Read(GetChunkKey(1)).result(), MatchesKvsReadResult("a"));
  EXPECT_THAT(
      store->Read(GetChunkKey(2)).result(),
      MatchesKvsReadResult(std::string(
          !StorageGeneration::IsUnknown(future2.result()->generation) ? "ww"
                                                                      : "xx")));
  EXPECT_THAT(store->Read(GetChunkKey(3)).result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(store->Read(GetChunkKey(4)).result(), MatchesKvsReadResult("zz"));
}

TEST_F(RawEncodingTest, MultipleDeleteExisting) {
  std::vector<Result<TimestampedStorageGeneration>> no_value_results;
  // The order in which multiple requests for the same `ChunkId` are attempted
  // depends on the order produced by `std::sort`, which is unspecified.  To
  // ensure we test both possibilities, we run the test with both orders.  This
  // assumes that `std::sort` is deterministic.
  for (int i = 0; i < 2; ++i) {
    auto gen = store->Write(GetChunkKey(1), "a").value().generation;
    StorageGeneration if_equal[] = {
        // Delete conditioned on `gen` is guaranteed to succeed.
        gen,
        // Delete conditioned on `StorageGeneration::NoValue()` succeeds if it
        // is attempted before the other delete, otherwise it fails.
        StorageGeneration::NoValue()};
    std::vector futures{store->Delete(GetChunkKey(1), {if_equal[i]}),
                        store->Delete(GetChunkKey(1), {if_equal[1 - i]})};
    std::vector results{futures[i].result(), futures[1 - i].result()};
    no_value_results.push_back(results[1]);
    EXPECT_THAT(results,
                ::testing::AnyOf(
                    ::testing::ElementsAre(MatchesTimestampedStorageGeneration(
                                               StorageGeneration::Invalid()),
                                           MatchesTimestampedStorageGeneration(
                                               StorageGeneration::NoValue())),
                    ::testing::ElementsAre(MatchesTimestampedStorageGeneration(
                                               StorageGeneration::NoValue()),
                                           MatchesTimestampedStorageGeneration(
                                               StorageGeneration::Unknown()))));
  }
  // Ensure that we covered both cases.  This could fail if `std::sort` is not
  // deterministic.
  EXPECT_THAT(
      no_value_results,
      ::testing::UnorderedElementsAre(
          MatchesTimestampedStorageGeneration(StorageGeneration::NoValue()),
          MatchesTimestampedStorageGeneration(StorageGeneration::Unknown())));
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
  base_kv_store->Write("prefix/0.shard", {1, 2, 3}).value();
  EXPECT_THAT(store->Read(GetChunkKey(10)).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error retrieving shard index entry for shard 0 "
                            "minishard 0: Requested byte range \\[0, 16\\) "
                            "is not valid for value of size 3"));
  EXPECT_THAT(
      store->Write(GetChunkKey(10), "abc").result(),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Error decoding existing shard 0: "
          "Existing shard index has size 3, but expected at least: 16"));
}

TEST_F(RawEncodingTest, ShardIndexInvalidByteRange) {
  base_kv_store
      ->Write("prefix/0.shard",
              Bytes({10, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0}))
      .value();
  EXPECT_THAT(store->Read(GetChunkKey(10)).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error retrieving shard index entry for shard 0 "
                            "minishard 0: Shard index specified invalid byte "
                            "range: \\[10, 2\\)"));
  EXPECT_THAT(store->Write(GetChunkKey(10), "abc").result(),
              MatchesStatus(
                  absl::StatusCode::kFailedPrecondition,
                  "Error decoding existing shard 0: "
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
  EXPECT_THAT(store->Read(GetChunkKey(10)).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error retrieving shard index entry for shard 0 "
                            "minishard 0: Byte range .* relative to the end of "
                            "the shard index \\(16\\) is not valid"));
  EXPECT_THAT(store->Write(GetChunkKey(10), "abc").result(),
              MatchesStatus(
                  absl::StatusCode::kFailedPrecondition,
                  "Error decoding existing shard 0: "
                  "Error decoding existing shard index entry for minishard 0: "
                  "Byte range .* relative to the end of "
                  "the shard index \\(16\\) is not valid"));
}

TEST_F(RawEncodingTest, MinishardIndexOutOfRange) {
  base_kv_store
      ->Write("prefix/0.shard",
              Bytes({0, 0, 0, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0}))
      .value();
  EXPECT_THAT(store->Read(GetChunkKey(10)).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error retrieving minishard index for shard 0 "
                            "minishard 0: Requested byte range \\[16, 64\\) is "
                            "not valid for value of size 16"));
  EXPECT_THAT(store->Write(GetChunkKey(10), "abc").result(),
              MatchesStatus(
                  absl::StatusCode::kFailedPrecondition,
                  "Error decoding existing shard 0: "
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
                    "Error retrieving minishard index for shard 0 minishard 0: "
                    "Invalid minishard index length: 1"));
  EXPECT_THAT(
      store->Write(GetChunkKey(10), "abc").result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error decoding existing shard 0: "
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
  EXPECT_THAT(store->Read(GetChunkKey(10)).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error retrieving minishard index for shard 0 "
                            "minishard 0: Error decoding minishard index entry "
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
  EXPECT_THAT(store->Write(GetChunkKey(1), "x").result(),
              MatchesStatus(
                  absl::StatusCode::kFailedPrecondition,
                  "Error decoding existing shard 0: "
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
  EXPECT_THAT(store->Write(GetChunkKey(10), "abc").result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error decoding existing shard 0: "
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
      ->Write("prefix/0.shard",
              {
                  0, 0, 0, 0, 0, 0, 0, 0,  //
                  3, 0, 0, 0, 0, 0, 0, 0,  //
                  1, 2, 3,                 //
              })
      .value();
  EXPECT_THAT(
      store->Read(GetChunkKey(10)).result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error retrieving minishard index for shard 0 minishard 0: "
                    "Error decoding zlib-compressed data"));
  EXPECT_THAT(
      store->Write(GetChunkKey(10), "abc").result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Error decoding existing shard 0: "
                    "Error decoding existing minishard index for minishard 0: "
                    "Error decoding zlib-compressed data"));
}

TEST_F(GzipEncodingTest, CorruptDataGzipEncoding) {
  std::string shard_data = "abc";
  zlib::Options zlib_options;
  zlib_options.use_gzip_header = true;
  zlib::Encode(Bytes({
                   10, 0, 0, 0, 0, 0, 0, 0,  //
                   0,  0, 0, 0, 0, 0, 0, 0,  //
                   3,  0, 0, 0, 0, 0, 0, 0,  //
               }),
               &shard_data, zlib_options);
  const unsigned char n = static_cast<unsigned char>(shard_data.size());
  base_kv_store
      ->Write("prefix/0.shard", Bytes({
                                    3, 0, 0, 0, 0, 0, 0, 0,  //
                                    n, 0, 0, 0, 0, 0, 0, 0,  //
                                }) + shard_data)
      .value();
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
  KeyValueStore::Ptr store = GetShardedKeyValueStore(
      mock_store, tensorstore::InlineExecutor{}, "prefix", sharding_spec,
      CachePool::WeakPtr(cache_pool));
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
      req.promise.SetResult(
          KeyValueStore::ReadResult{Bytes({
                                        5, 0, 0, 0, 0, 0, 0, 0,   //
                                        31, 0, 0, 0, 0, 0, 0, 0,  //
                                    }),
                                    StorageGeneration{"g0"}, absl::Now()});
    }
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
      EXPECT_EQ("g0", req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(37, 63), req.options.byte_range);
      minishard_index_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          Bytes({
              0x50, 0, 0, 0, 0, 0, 0, 0,  //
              0,    0, 0, 0, 0, 0, 0, 0,  //
              5,    0, 0, 0, 0, 0, 0, 0,  //
          }),
          StorageGeneration{"g0"}, minishard_index_time});
    }
    absl::Time read_time;
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
      EXPECT_EQ("g0", req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(32, 37), req.options.byte_range);
      read_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          Bytes({5, 6, 7, 8, 9}), StorageGeneration{"g0"}, read_time});
    }
    ASSERT_EQ(0, mock_store->read_requests.size());
    ASSERT_TRUE(future.ready());
    EXPECT_THAT(future.result(),
                MatchesKvsReadResult(Bytes({5, 6, 7, 8, 9}),
                                     StorageGeneration{"g0"}, read_time));
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
      EXPECT_EQ("g0", req.options.if_not_equal);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(0, 16), req.options.byte_range);
      EXPECT_THAT(req.options.staleness_bound, ::testing::Gt(req_time));
      minishard_index_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          std::nullopt, StorageGeneration::Unknown(), minishard_index_time});
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
      EXPECT_EQ("g0", req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(32, 37), req.options.byte_range);
      EXPECT_EQ(init_time, req.options.staleness_bound);
      read_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          Bytes({5, 6, 7, 8, 9}), StorageGeneration{"g0"}, read_time});
    }
    ASSERT_EQ(0, mock_store->read_requests.size());
    ASSERT_TRUE(future.ready());
    EXPECT_THAT(future.result(),
                MatchesKvsReadResult(Bytes({5, 6, 7, 8, 9}),
                                     StorageGeneration{"g0"}, read_time));
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
      EXPECT_EQ("g0", req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(32, 37), req.options.byte_range);
      abort_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          std::nullopt, StorageGeneration::Unknown(), abort_time});
    }
    // Request for updated shard index.
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ("g0", req.options.if_not_equal);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(0, 16), req.options.byte_range);
      EXPECT_THAT(req.options.staleness_bound, ::testing::Ge(abort_time));
      req.promise.SetResult(
          KeyValueStore::ReadResult{Bytes({
                                        6, 0, 0, 0, 0, 0, 0, 0,   //
                                        32, 0, 0, 0, 0, 0, 0, 0,  //
                                    }),
                                    StorageGeneration{"g1"}, absl::Now()});
    }
    // Request for updated minishard index.
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
      EXPECT_EQ("g1", req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(38, 64), req.options.byte_range);
      minishard_index_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          Bytes({
              0x50, 0, 0, 0, 0, 0, 0, 0,  //
              0,    0, 0, 0, 0, 0, 0, 0,  //
              6,    0, 0, 0, 0, 0, 0, 0,  //
          }),
          StorageGeneration{"g1"}, minishard_index_time});
    }
    // Request for value.
    absl::Time read_time;
    {
      auto req = mock_store->read_requests.pop_nonblock().value();
      ASSERT_EQ(0, mock_store->read_requests.size());
      EXPECT_EQ("prefix/0.shard", req.key);
      EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
      EXPECT_EQ("g1", req.options.if_equal);
      EXPECT_EQ(OptionalByteRangeRequest(32, 38), req.options.byte_range);
      read_time = absl::Now();
      req.promise.SetResult(KeyValueStore::ReadResult{
          Bytes({4, 5, 6, 7, 8, 9}), StorageGeneration{"g1"}, read_time});
    }
    ASSERT_EQ(0, mock_store->read_requests.size());
    ASSERT_TRUE(future.ready());
    EXPECT_THAT(future.result(),
                MatchesKvsReadResult(Bytes({4, 5, 6, 7, 8, 9}),
                                     StorageGeneration{"g1"}, read_time));
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
    req.promise.SetResult(
        KeyValueStore::ReadResult{Bytes({
                                      6, 0, 0, 0, 0, 0, 0, 0,   //
                                      32, 0, 0, 0, 0, 0, 0, 0,  //
                                  }),
                                  StorageGeneration{"g2"}, absl::Now()});
  }
  // Request for minishard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ("g2", req.options.if_equal);
    EXPECT_EQ(OptionalByteRangeRequest(38, 64), req.options.byte_range);
    abort_time = absl::Now();
    req.promise.SetResult(KeyValueStore::ReadResult{
        std::nullopt, StorageGeneration::Unknown(), abort_time});
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
    req.promise.SetResult(
        KeyValueStore::ReadResult{Bytes({
                                      7, 0, 0, 0, 0, 0, 0, 0,   //
                                      33, 0, 0, 0, 0, 0, 0, 0,  //
                                  }),
                                  StorageGeneration{"g3"}, absl::Now()});
  }
  // Request for updated minishard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ("g3", req.options.if_equal);
    EXPECT_EQ(OptionalByteRangeRequest(39, 65), req.options.byte_range);
    req.promise.SetResult(
        KeyValueStore::ReadResult{Bytes({
                                      0x1, 0, 0, 0, 0, 0, 0, 0,  //
                                      0,   0, 0, 0, 0, 0, 0, 0,  //
                                      4,   0, 0, 0, 0, 0, 0, 0,  //
                                  }),
                                  StorageGeneration{"g3"}, absl::Now()});
  }
  // Request for value.
  absl::Time read_time;
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ("g3", req.options.if_equal);
    EXPECT_EQ(OptionalByteRangeRequest(32, 36), req.options.byte_range);
    read_time = absl::Now();
    req.promise.SetResult(KeyValueStore::ReadResult{
        Bytes({4, 5, 6, 7}), StorageGeneration{"g3"}, read_time});
  }
  ASSERT_EQ(0, mock_store->read_requests.size());
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesKvsReadResult(Bytes({4, 5, 6, 7}), StorageGeneration{"g3"},
                                   read_time));
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
    req.promise.SetResult(
        KeyValueStore::ReadResult{Bytes({
                                      6, 0, 0, 0, 0, 0, 0, 0,   //
                                      32, 0, 0, 0, 0, 0, 0, 0,  //
                                  }),
                                  StorageGeneration{"g4"}, absl::Now()});
  }
  // Request for minishard index.
  absl::Time read_time;
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ("g4", req.options.if_equal);
    EXPECT_EQ(OptionalByteRangeRequest(38, 64), req.options.byte_range);
    read_time = absl::Now();
    req.promise.SetResult(KeyValueStore::ReadResult{
        std::nullopt, StorageGeneration::NoValue(), read_time});
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
    req.promise.SetResult(
        KeyValueStore::ReadResult{Bytes({
                                      6, 0, 0, 0, 0, 0, 0, 0,   //
                                      32, 0, 0, 0, 0, 0, 0, 0,  //
                                  }),
                                  StorageGeneration{"g0"}, absl::Now()});
  }
  // Request for minishard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ("g0", req.options.if_equal);
    EXPECT_EQ(OptionalByteRangeRequest(38, 64), req.options.byte_range);
    req.promise.SetResult(
        KeyValueStore::ReadResult{Bytes({
                                      0x1, 0, 0, 0, 0, 0, 0, 0,  //
                                      0,   0, 0, 0, 0, 0, 0, 0,  //
                                      4,   0, 0, 0, 0, 0, 0, 0,  //
                                  }),
                                  StorageGeneration{"g0"}, absl::Now()});
  }
  // Request for value.
  absl::Time read_time;
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ("g0", req.options.if_equal);
    EXPECT_EQ(OptionalByteRangeRequest(32, 36), req.options.byte_range);
    read_time = absl::Now();
    req.promise.SetResult(KeyValueStore::ReadResult{
        std::nullopt, StorageGeneration::NoValue(), read_time});
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
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown,
                            "Error retrieving shard index entry for shard 0 "
                            "minishard 0: Read error"));
}

TEST_F(UnderlyingKeyValueStoreTest, ReadErrorReadingMinishardShardIndex) {
  auto future = store->Read(GetChunkKey(0x1), {});
  // Request for shard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(OptionalByteRangeRequest(16, 32), req.options.byte_range);
    req.promise.SetResult(
        KeyValueStore::ReadResult{Bytes({
                                      6, 0, 0, 0, 0, 0, 0, 0,   //
                                      32, 0, 0, 0, 0, 0, 0, 0,  //
                                  }),
                                  StorageGeneration{"g0"}, absl::Now()});
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
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown,
                            "Error retrieving minishard index for shard 0 "
                            "minishard 1: Read error"));
}

TEST_F(UnderlyingKeyValueStoreTest, ReadErrorReadingData) {
  auto future = store->Read(GetChunkKey(0x1), {});
  // Request for shard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(OptionalByteRangeRequest(16, 32), req.options.byte_range);
    req.promise.SetResult(
        KeyValueStore::ReadResult{Bytes({
                                      6, 0, 0, 0, 0, 0, 0, 0,   //
                                      32, 0, 0, 0, 0, 0, 0, 0,  //
                                  }),
                                  StorageGeneration{"g0"}, absl::Now()});
  }
  // Request for minishard index.
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(OptionalByteRangeRequest(38, 64), req.options.byte_range);
    req.promise.SetResult(
        KeyValueStore::ReadResult{Bytes({
                                      0x1, 0, 0, 0, 0, 0, 0, 0,  //
                                      0,   0, 0, 0, 0, 0, 0, 0,  //
                                      4,   0, 0, 0, 0, 0, 0, 0,  //
                                  }),
                                  StorageGeneration{"g0"}, absl::Now()});
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
  auto future = store->Write("abc", "x");
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
  auto future = store->Write(GetChunkKey(0x50), Bytes({1, 2, 3}));
  ASSERT_EQ(0, mock_store->read_requests.size());
  ASSERT_EQ(0, mock_store->write_requests.size());
  future.Force();
  ASSERT_EQ(0, mock_store->read_requests.size());
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
    req.promise.SetResult(std::in_place, StorageGeneration{"g0"}, write_time);
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesTimestampedStorageGeneration("g0", write_time));
}

TEST_F(UnderlyingKeyValueStoreTest, WriteWithNoExistingShardError) {
  auto future = store->Write(GetChunkKey(0x50), Bytes({1, 2, 3}));
  future.Force();
  {
    auto req = mock_store->write_requests.pop_nonblock().value();
    req.promise.SetResult(absl::UnknownError("Write error"));
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "Write error"));
}

TEST_F(UnderlyingKeyValueStoreTest, WriteWithExistingShard) {
  auto future = store->Write(GetChunkKey(0x50), Bytes({1, 2, 3}));

  ASSERT_EQ(0, mock_store->read_requests.size());
  ASSERT_EQ(0, mock_store->write_requests.size());
  future.Force();
  ASSERT_EQ(0, mock_store->read_requests.size());
  absl::Time abort_time;
  {
    auto req = mock_store->write_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->write_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::NoValue(), req.options.if_equal);
    abort_time = absl::Now();
    req.promise.SetResult(std::in_place, StorageGeneration::Unknown(),
                          abort_time);
  }
  {
    auto req = mock_store->read_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
    EXPECT_EQ(StorageGeneration::Unknown(), req.options.if_not_equal);
    EXPECT_EQ(abort_time, req.options.staleness_bound);
    req.promise.SetResult(
        KeyValueStore::ReadResult{Bytes({
                                      3,    0, 0, 0, 0, 0, 0, 0,  //
                                      27,   0, 0, 0, 0, 0, 0, 0,  //
                                      0,    0, 0, 0, 0, 0, 0, 0,  //
                                      0,    0, 0, 0, 0, 0, 0, 0,  //
                                      4,    5, 6,                 //
                                      0x70, 0, 0, 0, 0, 0, 0, 0,  //
                                      0,    0, 0, 0, 0, 0, 0, 0,  //
                                      3,    0, 0, 0, 0, 0, 0, 0,  //
                                  }),
                                  StorageGeneration{"g0"}, absl::Now()});
  }
  absl::Time write_time;
  {
    auto req = mock_store->write_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->write_requests.size());
    ASSERT_EQ(0, mock_store->read_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ("g0", req.options.if_equal);
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
    req.promise.SetResult(std::in_place, StorageGeneration{"g1"}, write_time);
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesTimestampedStorageGeneration("g1", write_time));
}

TEST_F(UnderlyingKeyValueStoreTest, WriteWithExistingShardReadError) {
  auto future = store->Write(GetChunkKey(0x50), Bytes({1, 2, 3}));

  ASSERT_EQ(0, mock_store->read_requests.size());
  ASSERT_EQ(0, mock_store->write_requests.size());
  future.Force();
  ASSERT_EQ(0, mock_store->read_requests.size());
  {
    auto req = mock_store->write_requests.pop_nonblock().value();
    ASSERT_EQ(0, mock_store->write_requests.size());
    EXPECT_EQ("prefix/0.shard", req.key);
    EXPECT_EQ(StorageGeneration::NoValue(), req.options.if_equal);
    req.promise.SetResult(std::in_place, StorageGeneration::Unknown(),
                          absl::Now());
  }
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
              MatchesStatus(absl::StatusCode::kUnknown, "Read error"));
}

TEST_F(UnderlyingKeyValueStoreTest, List) {
  auto future = ListFuture(store.get());
  future.Force();
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(), MatchesStatus(absl::StatusCode::kUnimplemented));
}

TEST_F(UnderlyingKeyValueStoreTest, DeletePrefixInvalid) {
  auto future = store->DeletePrefix("abc");
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(), MatchesStatus(absl::StatusCode::kInvalidArgument,
                                             "Only empty prefix is supported"));
}

TEST_F(UnderlyingKeyValueStoreTest, DeletePrefix) {
  auto future = store->DeletePrefix("");
  {
    auto req = mock_store->delete_prefix_requests.pop_nonblock().value();
    EXPECT_EQ(0, mock_store->delete_prefix_requests.size());
    EXPECT_EQ("prefix/", req.prefix);
    req.promise.SetResult(5);
  }
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(), ::testing::Optional(5));
}

}  // namespace
