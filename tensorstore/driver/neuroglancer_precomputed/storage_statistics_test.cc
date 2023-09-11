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

/// Tests of the neuroglancer_precomputed driver `GetStorageStatistics` support.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/open.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::ArrayStorageStatistics;
using ::tensorstore::ChunkLayout;
using ::tensorstore::Context;
using ::tensorstore::dtype_v;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
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
                          {"driver", "neuroglancer_precomputed"},
                          {"kvstore", {{"driver", "mock_key_value_store"}}},
                      },
                      Schema::Shape({100, 200, 300, 1}), dtype_v<uint8_t>,
                      ChunkLayout::ReadChunkShape({10, 20, 30, 1}), context,
                      tensorstore::OpenMode::create)
                      .result());
  mock_kvstore->request_log.pop_all();
  {
    auto transformed = store | tensorstore::AllDims().HalfOpenInterval(
                                   {1, 1, 1, 0}, {20, 5, 5, 1});
    EXPECT_THAT(tensorstore::GetStorageStatistics(
                    transformed, ArrayStorageStatistics::query_not_stored)
                    .result(),
                ::testing::Optional(ArrayStorageStatistics{
                    /*.mask=*/ArrayStorageStatistics::query_not_stored,
                    /*.not_stored=*/true}));
    EXPECT_THAT(mock_kvstore->request_log.pop_all(),
                ::testing::UnorderedElementsAreArray({
                    MatchesJson({{"type", "read"},
                                 {"key", "1_1_1/0-10_0-20_0-30"},
                                 {"byte_range_exclusive_max", 0}}),
                    MatchesJson({{"type", "read"},
                                 {"key", "1_1_1/10-20_0-20_0-30"},
                                 {"byte_range_exclusive_max", 0}}),
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
                    MatchesJson({{"type", "read"},
                                 {"key", "1_1_1/0-10_0-20_0-30"},
                                 {"byte_range_exclusive_max", 0}}),
                    MatchesJson({{"type", "read"},
                                 {"key", "1_1_1/10-20_0-20_0-30"},
                                 {"byte_range_exclusive_max", 0}}),
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
                    MatchesJson({{"type", "read"},
                                 {"key", "1_1_1/0-10_0-20_0-30"},
                                 {"byte_range_exclusive_max", 0}}),
                    MatchesJson({{"type", "read"},
                                 {"key", "1_1_1/10-20_0-20_0-30"},
                                 {"byte_range_exclusive_max", 0}}),
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
    EXPECT_THAT(mock_kvstore->request_log.pop_all(),
                ::testing::UnorderedElementsAreArray({
                    MatchesJson({{"type", "list"},
                                 {"range", {"1_1_1/", "1_1_10"}},
                                 {"strip_prefix_length", 6}}),
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
                ::testing::ElementsAre(
                    MatchesJson({{"type", "list"},
                                 {"range", {"1_1_1/10-20_", "1_1_1/10-20`"}},
                                 {"strip_prefix_length", 6}})));
  }

  // Test listing with a single (not present) chunk.
  {
    EXPECT_THAT(tensorstore::GetStorageStatistics(
                    store | tensorstore::AllDims().IndexSlice({10, 25, 35, 0}),
                    ArrayStorageStatistics::query_not_stored,
                    ArrayStorageStatistics::query_fully_stored)
                    .result(),
                ::testing::Optional(ArrayStorageStatistics{
                    /*.mask=*/ArrayStorageStatistics::query_not_stored |
                        ArrayStorageStatistics::query_fully_stored,
                    /*.not_stored=*/true, /*.fully_stored=*/false}));
    EXPECT_THAT(
        mock_kvstore->request_log.pop_all(),
        ::testing::ElementsAre(MatchesJson({{"type", "read"},
                                            {"key", "1_1_1/10-20_20-40_30-60"},
                                            {"byte_range_exclusive_max", 0}})));
  }

  // Test listing with a single (present) chunk.
  {
    EXPECT_THAT(tensorstore::GetStorageStatistics(
                    store | tensorstore::AllDims().IndexSlice({2, 2, 2, 0}),
                    ArrayStorageStatistics::query_not_stored,
                    ArrayStorageStatistics::query_fully_stored)
                    .result(),
                ::testing::Optional(ArrayStorageStatistics{
                    /*.mask=*/ArrayStorageStatistics::query_not_stored |
                        ArrayStorageStatistics::query_fully_stored,
                    /*.not_stored=*/false, /*.fully_stored=*/true}));
    EXPECT_THAT(
        mock_kvstore->request_log.pop_all(),
        ::testing::ElementsAre(MatchesJson({{"type", "read"},
                                            {"key", "1_1_1/0-10_0-20_0-30"},
                                            {"byte_range_exclusive_max", 0}})));
  }
}

TEST_F(StorageStatisticsTest, SemiLexicographicOrder) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(
                      {
                          {"driver", "neuroglancer_precomputed"},
                          {"kvstore", {{"driver", "mock_key_value_store"}}},
                      },
                      Schema::Shape({100, 100, 100, 1}), dtype_v<uint8_t>,
                      ChunkLayout::ReadChunkShape({1, 1, 1, 1}), context,
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
  EXPECT_THAT(mock_kvstore->request_log.pop_all(),
              ::testing::UnorderedElementsAreArray({
                  MatchesJson({{"type", "list"},
                               {"range", {"1_1_1/8-9_", "1_1_1/8-9`"}},
                               {"strip_prefix_length", 6}}),
                  MatchesJson({{"type", "list"},
                               {"range", {"1_1_1/9-10_", "1_1_1/9-10`"}},
                               {"strip_prefix_length", 6}}),
                  MatchesJson({{"type", "list"},
                               {"range", {"1_1_1/10-11_", "1_1_1/14-15`"}},
                               {"strip_prefix_length", 6}}),
              }));

  EXPECT_THAT(
      tensorstore::GetStorageStatistics(
          store | tensorstore::Dims(0, 1).HalfOpenInterval({3, 8}, {4, 15}),
          ArrayStorageStatistics::query_not_stored)
          .result(),
      ::testing::Optional(ArrayStorageStatistics{
          /*.mask=*/ArrayStorageStatistics::query_not_stored,
          /*.not_stored=*/true}));
  EXPECT_THAT(
      mock_kvstore->request_log.pop_all(),
      ::testing::UnorderedElementsAreArray({
          MatchesJson({{"type", "list"},
                       {"range", {"1_1_1/3-4_8-9_", "1_1_1/3-4_8-9`"}},
                       {"strip_prefix_length", 6}}),
          MatchesJson({{"type", "list"},
                       {"range", {"1_1_1/3-4_9-10_", "1_1_1/3-4_9-10`"}},
                       {"strip_prefix_length", 6}}),
          MatchesJson({{"type", "list"},
                       {"range", {"1_1_1/3-4_10-11_", "1_1_1/3-4_14-15`"}},
                       {"strip_prefix_length", 6}}),
      }));
}

TEST_F(StorageStatisticsTest, Sharded) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(
                      {
                          {"driver", "neuroglancer_precomputed"},
                          {"kvstore", {{"driver", "mock_key_value_store"}}},
                      },
                      Schema::Shape({100, 100, 100, 1}), dtype_v<uint8_t>,
                      ChunkLayout::ReadChunkShape({1, 1, 1, 1}),
                      ChunkLayout::WriteChunkShape({8, 8, 8, 1}), context,
                      tensorstore::OpenMode::create)
                      .result());
  mock_kvstore->request_log.pop_all();

  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store | tensorstore::Dims(0).HalfOpenInterval(8, 15),
                  ArrayStorageStatistics::query_not_stored)
                  .result(),
              MatchesStatus(absl::StatusCode::kUnimplemented));
}

}  // namespace
