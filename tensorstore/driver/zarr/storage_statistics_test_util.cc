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

#include "tensorstore/driver/zarr/storage_statistics_test_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/open.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_zarr {

using tensorstore::StrCat;

TEST_P(ZarrLikeStorageStatisticsTest, FullyLexicographicOrder) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(json_spec, Schema::Shape({100, 200, 300}),
                                    dtype_v<uint8_t>,
                                    ChunkLayout::ReadChunkShape({10, 20, 30}),
                                    tensorstore::OpenMode::create, context)
                      .result());
  // Ignore kvstore operations from opening the array.
  mock_kvstore->request_log.pop_all();
  {
    auto transformed =
        store | tensorstore::AllDims().HalfOpenInterval({1, 1, 1}, {20, 5, 5});

    // Query region before it has been written: not stored.
    EXPECT_THAT(tensorstore::GetStorageStatistics(
                    transformed, ArrayStorageStatistics::query_not_stored)
                    .result(),
                ::testing::Optional(ArrayStorageStatistics{
                    /*.mask=*/ArrayStorageStatistics::query_not_stored,
                    /*.not_stored=*/true}));
    EXPECT_THAT(mock_kvstore->request_log.pop_all(),
                ::testing::UnorderedElementsAreArray({
                    MatchesJson({{"type", "read"},
                                 {"key", StrCat("0", sep, "0", sep, "0")},
                                 {"byte_range_exclusive_max", 0}}),
                    MatchesJson({{"type", "read"},
                                 {"key", StrCat("1", sep, "0", sep, "0")},
                                 {"byte_range_exclusive_max", 0}}),
                }));

    // Write to the region.
    TENSORSTORE_ASSERT_OK(
        tensorstore::Write(tensorstore::MakeScalarArray<uint8_t>(42),
                           transformed)
            .result());
    // Ignore kvstore operations from the write.
    mock_kvstore->request_log.pop_all();

    // Query not_stored after writing: not_stored=false because it is now fully
    // stored.
    EXPECT_THAT(tensorstore::GetStorageStatistics(
                    transformed, ArrayStorageStatistics::query_not_stored)
                    .result(),
                ::testing::Optional(ArrayStorageStatistics{
                    /*.mask=*/ArrayStorageStatistics::query_not_stored,
                    /*.not_stored=*/false}));
    EXPECT_THAT(mock_kvstore->request_log.pop_all(),
                ::testing::UnorderedElementsAreArray({
                    MatchesJson({{"type", "read"},
                                 {"key", StrCat("0", sep, "0", sep, "0")},
                                 {"byte_range_exclusive_max", 0}}),
                    MatchesJson({{"type", "read"},
                                 {"key", StrCat("1", sep, "0", sep, "0")},
                                 {"byte_range_exclusive_max", 0}}),
                }));

    // Query not_stored and fully_stored.
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
                                 {"key", StrCat("0", sep, "0", sep, "0")},
                                 {"byte_range_exclusive_max", 0}}),
                    MatchesJson({{"type", "read"},
                                 {"key", StrCat("1", sep, "0", sep, "0")},
                                 {"byte_range_exclusive_max", 0}}),
                }));
  }

  // Test listing entire array: partially stored.
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
                    MatchesJson({{"type", "list"}, {"range", {"", ""}}}),
                }));
  }

  // Test listing with single-dimension prefix: partially stored.
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
                ::testing::ElementsAre(MatchesJson(
                    {{"type", "list"},
                     {"range", {StrCat("1", sep), StrCat("1", sep_next)}}})));
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
                    MatchesJson({{"type", "read"},
                                 {"key", StrCat("1", sep, "1", sep, "1")},
                                 {"byte_range_exclusive_max", 0}})));
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
                    MatchesJson({{"type", "read"},
                                 {"key", StrCat("0", sep, "0", sep, "0")},
                                 {"byte_range_exclusive_max", 0}})));
  }
}

TEST_P(ZarrLikeStorageStatisticsTest, SemiLexicographicOrder) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, Schema::Shape({100, 100}), dtype_v<uint8_t>,
                        ChunkLayout::ReadChunkShape({1, 1}),
                        tensorstore::OpenMode::create, context)
          .result());
  mock_kvstore->request_log.pop_all();

  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store | tensorstore::Dims(0).HalfOpenInterval(8, 15),
                  ArrayStorageStatistics::query_not_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored,
                  /*.not_stored=*/true, /*.fully_stored=*/false}));
  EXPECT_THAT(
      mock_kvstore->request_log.pop_all(),
      ::testing::UnorderedElementsAreArray({
          MatchesJson({{"type", "list"},
                       {"range", {StrCat("8", sep), StrCat("8", sep_next)}}}),
          MatchesJson({{"type", "list"},
                       {"range", {StrCat("9", sep), StrCat("9", sep_next)}}}),
          MatchesJson({{"type", "list"},
                       {"range", {StrCat("10", sep), StrCat("14", sep_next)}}}),
      }));

  EXPECT_THAT(
      tensorstore::GetStorageStatistics(
          store | tensorstore::Dims(0, 1).HalfOpenInterval({3, 8}, {4, 15}),
          ArrayStorageStatistics::query_not_stored)
          .result(),
      ::testing::Optional(ArrayStorageStatistics{
          /*.mask=*/ArrayStorageStatistics::query_not_stored,
          /*.not_stored=*/true, /*.fully_stored=*/false}));
  EXPECT_THAT(
      mock_kvstore->request_log.pop_all(),
      ::testing::UnorderedElementsAreArray({
          MatchesJson({{"type", "read"},
                       {"key", StrCat("3", sep, "8")},
                       {"byte_range_exclusive_max", 0}}),
          MatchesJson({{"type", "read"},
                       {"key", StrCat("3", sep, "9")},
                       {"byte_range_exclusive_max", 0}}),
          MatchesJson(
              {{"type", "list"},
               {"range", {StrCat("3", sep, "10"), StrCat("3", sep, "15")}}}),
      }));
}

TEST_P(ZarrLikeStorageStatisticsTest, Rank0) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, Schema::Shape({}), dtype_v<uint8_t>,
                        tensorstore::OpenMode::create, context)
          .result());
  mock_kvstore->request_log.pop_all();

  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store, ArrayStorageStatistics::query_not_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored,
                  /*.not_stored=*/true, /*.fully_stored=*/false}));
  EXPECT_THAT(mock_kvstore->request_log.pop_all(),
              ::testing::UnorderedElementsAreArray({
                  MatchesJson({{"type", "read"},
                               {"key", "0"},
                               {"byte_range_exclusive_max", 0}}),
              }));

  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(tensorstore::MakeScalarArray<uint8_t>(42), store)
          .result());
  mock_kvstore->request_log.pop_all();
  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store, ArrayStorageStatistics::query_not_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored,
                  /*.not_stored=*/false, /*.fully_stored=*/false}));
  EXPECT_THAT(mock_kvstore->request_log.pop_all(),
              ::testing::UnorderedElementsAreArray({
                  MatchesJson({{"type", "read"},
                               {"key", "0"},
                               {"byte_range_exclusive_max", 0}}),
              }));

  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store, ArrayStorageStatistics::query_not_stored,
                  ArrayStorageStatistics::query_fully_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored |
                      ArrayStorageStatistics::query_fully_stored,
                  /*.not_stored=*/false,
                  /*.fully_stored=*/true}));
  EXPECT_THAT(mock_kvstore->request_log.pop_all(),
              ::testing::UnorderedElementsAreArray({
                  MatchesJson({{"type", "read"},
                               {"key", "0"},
                               {"byte_range_exclusive_max", 0}}),
              }));
}

TEST_P(ZarrLikeStorageStatisticsTest, Example) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, Schema::Shape({100, 200}), dtype_v<uint8_t>,
                        tensorstore::OpenMode::create, context)
          .result());
  mock_kvstore->request_log.pop_all();

  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store, ArrayStorageStatistics::query_not_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored,
                  /*.not_stored=*/true}));
  EXPECT_THAT(mock_kvstore->request_log.pop_all(),
              ::testing::UnorderedElementsAreArray({
                  MatchesJson({{"type", "read"},
                               {"key", StrCat("0", sep, "0")},
                               {"byte_range_exclusive_max", 0}}),
              }));
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(
          tensorstore::MakeScalarArray<uint8_t>(42),
          store | tensorstore::Dims(0, 1).HalfOpenInterval({10, 30}, {20, 40}))
          .result());
  mock_kvstore->request_log.pop_all();
  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store, ArrayStorageStatistics::query_not_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored,
                  /*.not_stored=*/false}));
  EXPECT_THAT(mock_kvstore->request_log.pop_all(),
              ::testing::UnorderedElementsAreArray({
                  MatchesJson({{"type", "read"},
                               {"key", StrCat("0", sep, "0")},
                               {"byte_range_exclusive_max", 0}}),
              }));

  EXPECT_THAT(tensorstore::GetStorageStatistics(
                  store, ArrayStorageStatistics::query_not_stored,
                  ArrayStorageStatistics::query_fully_stored)
                  .result(),
              ::testing::Optional(ArrayStorageStatistics{
                  /*.mask=*/ArrayStorageStatistics::query_not_stored |
                      ArrayStorageStatistics::query_fully_stored,
                  /*.not_stored=*/false,
                  /*.fully_stored=*/true}));
  EXPECT_THAT(mock_kvstore->request_log.pop_all(),
              ::testing::UnorderedElementsAreArray({
                  MatchesJson({{"type", "read"},
                               {"key", StrCat("0", sep, "0")},
                               {"byte_range_exclusive_max", 0}}),
              }));
}

}  // namespace internal_zarr
}  // namespace tensorstore
