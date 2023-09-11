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

/// Tests of the zarr3 driver `GetStorageStatistics` support.

#include <stdint.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/schema.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::ArrayStorageStatistics;
using ::tensorstore::ChunkLayout;
using ::tensorstore::Context;
using ::tensorstore::dtype_v;
using ::tensorstore::JsonSubValuesMatch;
using ::tensorstore::Schema;

class StorageStatisticsTest : public ::testing::Test {
 protected:
  Context context = Context::Default();
  tensorstore::internal::MockKeyValueStore::MockPtr mock_kvstore =
      *context.GetResource<tensorstore::internal::MockKeyValueStoreResource>()
           .value();
  tensorstore::kvstore::DriverPtr memory_store =
      tensorstore::GetMemoryKeyValueStore();

 public:
  StorageStatisticsTest() {
    mock_kvstore->forward_to = memory_store;
    mock_kvstore->log_requests = true;
  }
};

TEST_F(StorageStatisticsTest, FullyLexicographicOrder) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(
                      {
                          {"driver", "zarr3"},
                          {"kvstore", {{"driver", "mock_key_value_store"}}},
                      },
                      Schema::Shape({100, 200, 300}), dtype_v<uint8_t>,
                      ChunkLayout::ChunkShape({10, 20, 30}), context,
                      tensorstore::OpenMode::create)
                      .result());
  mock_kvstore->request_log.pop_all();
  {
    auto transformed =
        store | tensorstore::AllDims().HalfOpenInterval({1, 1, 1}, {20, 5, 5});
    EXPECT_THAT(tensorstore::GetStorageStatistics(
                    transformed, ArrayStorageStatistics::query_not_stored)
                    .result(),
                ::testing::Optional(ArrayStorageStatistics{
                    /*.mask=*/ArrayStorageStatistics::query_not_stored,
                    /*.not_stored=*/true}));
    EXPECT_THAT(mock_kvstore->request_log.pop_all(),
                ::testing::UnorderedElementsAreArray({
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "c/0/0/0"},
                                        {"/byte_range_exclusive_max", 0}}),
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "c/1/0/0"},
                                        {"/byte_range_exclusive_max", 0}}),
                }));
    TENSORSTORE_ASSERT_OK(
        tensorstore::Write(tensorstore::MakeScalarArray<uint8_t>(42),
                           transformed)
            .result());
    mock_kvstore->request_log.pop_all();
    EXPECT_THAT(tensorstore::GetStorageStatistics(
                    transformed, ArrayStorageStatistics::query_not_stored)
                    .result(),
                ::testing::Optional(ArrayStorageStatistics{
                    /*.mask=*/ArrayStorageStatistics::query_not_stored,
                    /*.not_stored=*/false}));
    EXPECT_THAT(mock_kvstore->request_log.pop_all(),
                ::testing::UnorderedElementsAreArray({
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "c/0/0/0"},
                                        {"/byte_range_exclusive_max", 0}}),
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "c/1/0/0"},
                                        {"/byte_range_exclusive_max", 0}}),
                }));
    EXPECT_THAT(tensorstore::GetStorageStatistics(
                    transformed, ArrayStorageStatistics::query_not_stored,
                    ArrayStorageStatistics::query_fully_stored)
                    .result(),
                ::testing::Optional(ArrayStorageStatistics{
                    /*.mask=*/ArrayStorageStatistics::query_not_stored |
                        ArrayStorageStatistics::query_fully_stored,
                    /*.not_stored=*/false, /*.fully_stored=*/true}));
    EXPECT_THAT(mock_kvstore->request_log.pop_all(),
                ::testing::UnorderedElementsAreArray({
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "c/0/0/0"},
                                        {"/byte_range_exclusive_max", 0}}),
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "c/1/0/0"},
                                        {"/byte_range_exclusive_max", 0}}),
                }));
  }

  // Test listing entire array
  {
    EXPECT_THAT(tensorstore::GetStorageStatistics(
                    store, ArrayStorageStatistics::query_not_stored,
                    ArrayStorageStatistics::query_fully_stored)
                    .result(),
                ::testing::Optional(ArrayStorageStatistics{
                    /*.mask=*/ArrayStorageStatistics::query_not_stored |
                        ArrayStorageStatistics::query_fully_stored,
                    /*.not_stored=*/false, /*.fully_stored=*/false}));
    EXPECT_THAT(
        mock_kvstore->request_log.pop_all(),
        ::testing::UnorderedElementsAreArray({
            JsonSubValuesMatch({{"/type", "list"}, {"/range", {"c/", "c0"}}}),
        }));
  }

  // Test listing with single-dimension prefix
  {
    EXPECT_THAT(tensorstore::GetStorageStatistics(
                    store | tensorstore::Dims(0).HalfOpenInterval(12, 15),
                    ArrayStorageStatistics::query_not_stored,
                    ArrayStorageStatistics::query_fully_stored)
                    .result(),
                ::testing::Optional(ArrayStorageStatistics{
                    /*.mask=*/ArrayStorageStatistics::query_not_stored |
                        ArrayStorageStatistics::query_fully_stored,
                    /*.not_stored=*/false, /*.fully_stored=*/false}));
    EXPECT_THAT(mock_kvstore->request_log.pop_all(),
                ::testing::ElementsAre(JsonSubValuesMatch(
                    {{"/type", "list"}, {"/range", {"c/1/", "c/10"}}})));
  }

  // Test listing with a single (not present) chunk.
  {
    EXPECT_THAT(tensorstore::GetStorageStatistics(
                    store | tensorstore::AllDims().IndexSlice({10, 25, 35}),
                    ArrayStorageStatistics::query_not_stored,
                    ArrayStorageStatistics::query_fully_stored)
                    .result(),
                ::testing::Optional(ArrayStorageStatistics{
                    /*.mask=*/ArrayStorageStatistics::query_not_stored |
                        ArrayStorageStatistics::query_fully_stored,
                    /*.not_stored=*/true, /*.fully_stored=*/false}));
    EXPECT_THAT(mock_kvstore->request_log.pop_all(),
                ::testing::ElementsAre(
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "c/1/1/1"},
                                        {"/byte_range_exclusive_max", 0}})));
  }

  // Test listing with a single (present) chunk.
  {
    EXPECT_THAT(tensorstore::GetStorageStatistics(
                    store | tensorstore::AllDims().IndexSlice({2, 2, 2}),
                    ArrayStorageStatistics::query_not_stored,
                    ArrayStorageStatistics::query_fully_stored)
                    .result(),
                ::testing::Optional(ArrayStorageStatistics{
                    /*.mask=*/ArrayStorageStatistics::query_not_stored |
                        ArrayStorageStatistics::query_fully_stored,
                    /*.not_stored=*/false, /*.fully_stored=*/true}));
    EXPECT_THAT(mock_kvstore->request_log.pop_all(),
                ::testing::ElementsAre(
                    JsonSubValuesMatch({{"/type", "read"},
                                        {"/key", "c/0/0/0"},
                                        {"/byte_range_exclusive_max", 0}})));
  }
}

TEST_F(StorageStatisticsTest, SemiLexicographicOrder) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(
                      {
                          {"driver", "zarr3"},
                          {"kvstore", {{"driver", "mock_key_value_store"}}},
                      },
                      Schema::Shape({100, 100, 100}), dtype_v<uint8_t>,
                      ChunkLayout::ChunkShape({1, 1, 1}), context,
                      tensorstore::OpenMode::create)
                      .result());
  mock_kvstore->request_log.pop_all();

  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store | tensorstore::Dims(0).HalfOpenInterval(8, 15),
                  ArrayStorageStatistics::query_not_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored,
                  /*.not_stored=*/true}));
  EXPECT_THAT(
      mock_kvstore->request_log.pop_all(),
      ::testing::UnorderedElementsAreArray({
          JsonSubValuesMatch({{"/type", "list"}, {"/range", {"c/8/", "c/80"}}}),
          JsonSubValuesMatch({{"/type", "list"}, {"/range", {"c/9/", "c/90"}}}),
          JsonSubValuesMatch(
              {{"/type", "list"}, {"/range", {"c/10/", "c/140"}}}),
      }));

  EXPECT_THAT(
      tensorstore::GetStorageStatistics(
          store | tensorstore::Dims(0, 1).HalfOpenInterval({3, 8}, {4, 15}),
          ArrayStorageStatistics::query_not_stored)
          .result(),
      ::testing::Optional(ArrayStorageStatistics{
          /*.mask=*/ArrayStorageStatistics::query_not_stored,
          /*.not_stored=*/true}));
  EXPECT_THAT(mock_kvstore->request_log.pop_all(),
              ::testing::UnorderedElementsAreArray({
                  JsonSubValuesMatch(
                      {{"/type", "list"}, {"/range", {"c/3/8/", "c/3/80"}}}),
                  JsonSubValuesMatch(
                      {{"/type", "list"}, {"/range", {"c/3/9/", "c/3/90"}}}),
                  JsonSubValuesMatch(
                      {{"/type", "list"}, {"/range", {"c/3/10/", "c/3/140"}}}),
              }));
}

TEST_F(StorageStatisticsTest, Sharded) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(
                      {
                          {"driver", "zarr3"},
                          {"kvstore", {{"driver", "mock_key_value_store"}}},
                      },
                      Schema::Shape({100, 100, 100}), dtype_v<uint8_t>,
                      ChunkLayout::ReadChunkShape({1, 1, 1}),
                      ChunkLayout::WriteChunkShape({8, 8, 8}), context,
                      tensorstore::OpenMode::create)
                      .result());
  mock_kvstore->request_log.pop_all();

  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store | tensorstore::Dims(0).HalfOpenInterval(8, 15),
                  ArrayStorageStatistics::query_not_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored,
                  /*.not_stored=*/true}));
  // No data present.
  EXPECT_THAT(
      mock_kvstore->request_log.pop_all(),
      ::testing::UnorderedElementsAreArray({
          JsonSubValuesMatch({{"/type", "list"}, {"/range", {"c/1/", "c/10"}}}),
      }));

  // Write to one chunk.
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(
          tensorstore::MakeScalarArray<uint8_t>(42),
          store | tensorstore::AllDims().IndexSlice({30, 50, 70}))
          .result());

  mock_kvstore->request_log.pop_all();

  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store | tensorstore::Dims(0).HalfOpenInterval(24, 32),
                  ArrayStorageStatistics::query_not_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored,
                  /*.not_stored=*/false}));

  // Shard index is not retrieved since query covers entire shard.
  EXPECT_THAT(
      mock_kvstore->request_log.pop_all(),
      ::testing::UnorderedElementsAreArray({
          JsonSubValuesMatch({{"/type", "list"}, {"/range", {"c/3/", "c/30"}}}),
      }));

  mock_kvstore->request_log.pop_all();

  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store | tensorstore::Dims(0).HalfOpenInterval(24, 31),
                  ArrayStorageStatistics::query_not_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored,
                  /*.not_stored=*/false}));

  // Shard index is retrieved since query does not cover entire shard.
  EXPECT_THAT(
      mock_kvstore->request_log.pop_all(),
      ::testing::UnorderedElementsAreArray({
          JsonSubValuesMatch({{"/type", "list"}, {"/range", {"c/3/", "c/30"}}}),
          JsonSubValuesMatch({{"/type", "read"},
                              {"/byte_range_inclusive_min", -8196},
                              {"/key", "c/3/6/8"}}),
      }));
}

TEST_F(StorageStatisticsTest, ShardedTranspose) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(
                      {
                          {"driver", "zarr3"},
                          {"kvstore", {{"driver", "mock_key_value_store"}}},
                          {"cache_pool", {{"total_bytes_limit", 100000000}}},
                          {"metadata",
                           {{"codecs",
                             {{{"name", "transpose"},
                               {"configuration", {{"order", {2, 1, 0}}}}},
                              {{"name", "sharding_indexed"}}}}}},
                      },
                      Schema::Shape({100, 100, 100}), dtype_v<uint8_t>,
                      ChunkLayout::ReadChunkShape({1, 1, 1}),
                      ChunkLayout::WriteChunkShape({8, 8, 8}), context,
                      tensorstore::OpenMode::create)
                      .result());
  mock_kvstore->request_log.pop_all();

  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store | tensorstore::Dims(0).HalfOpenInterval(8, 15),
                  ArrayStorageStatistics::query_not_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored,
                  /*.not_stored=*/true}));
  // No data present.
  EXPECT_THAT(
      mock_kvstore->request_log.pop_all(),
      ::testing::UnorderedElementsAreArray({
          JsonSubValuesMatch({{"/type", "list"}, {"/range", {"c/1/", "c/10"}}}),
      }));

  // Write to one chunk.
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(
          tensorstore::MakeScalarArray<uint8_t>(42),
          store | tensorstore::AllDims().IndexSlice({30, 50, 70}))
          .result());

  mock_kvstore->request_log.pop_all();

  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store | tensorstore::Dims(0).HalfOpenInterval(24, 32),
                  ArrayStorageStatistics::query_not_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored,
                  /*.not_stored=*/false}));

  // Shard index is not retrieved since query covers entire shard.
  EXPECT_THAT(
      mock_kvstore->request_log.pop_all(),
      ::testing::UnorderedElementsAreArray({
          JsonSubValuesMatch({{"/type", "list"}, {"/range", {"c/3/", "c/30"}}}),
      }));

  mock_kvstore->request_log.pop_all();

  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store | tensorstore::Dims(0).HalfOpenInterval(24, 31),
                  ArrayStorageStatistics::query_not_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored,
                  /*.not_stored=*/false}));

  // Shard index is retrieved since query does not cover entire shard.
  EXPECT_THAT(
      mock_kvstore->request_log.pop_all(),
      ::testing::UnorderedElementsAreArray({
          JsonSubValuesMatch({{"/type", "list"}, {"/range", {"c/3/", "c/30"}}}),
          JsonSubValuesMatch({{"/type", "read"},
                              {"/byte_range_inclusive_min", -8196},
                              {"/key", "c/3/6/8"}}),
      }));
}

}  // namespace
