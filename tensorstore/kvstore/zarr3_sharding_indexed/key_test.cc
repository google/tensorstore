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

#include "tensorstore/kvstore/zarr3_sharding_indexed/key.h"

#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index.h"
#include "tensorstore/kvstore/key_range.h"

namespace {

using ::tensorstore::Index;
using ::tensorstore::KeyRange;
using ::tensorstore::zarr3_sharding_indexed::EntryId;
using ::tensorstore::zarr3_sharding_indexed::EntryIdToInternalKey;
using ::tensorstore::zarr3_sharding_indexed::EntryIdToKey;
using ::tensorstore::zarr3_sharding_indexed::IndicesToKey;
using ::tensorstore::zarr3_sharding_indexed::InternalKeyLowerBoundToEntryId;
using ::tensorstore::zarr3_sharding_indexed::InternalKeyRangeToEntryRange;
using ::tensorstore::zarr3_sharding_indexed::InternalKeyToEntryId;
using ::tensorstore::zarr3_sharding_indexed::KeyRangeToEntryRange;
using ::tensorstore::zarr3_sharding_indexed::KeyRangeToInternalKeyRange;
using ::tensorstore::zarr3_sharding_indexed::KeyToEntryId;
using ::tensorstore::zarr3_sharding_indexed::KeyToIndices;
using ::tensorstore::zarr3_sharding_indexed::LowerBoundToEntryId;

TEST(KeyToEntryIdTest, Basic) {
  EntryId entry_id = 1 * 5 * 6 + 2 * 6 + 3;
  std::string key{0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3};
  Index grid_shape[] = {4, 5, 6};
  EXPECT_THAT(KeyToEntryId(key, grid_shape), ::testing::Optional(entry_id));
  EXPECT_THAT(EntryIdToKey(entry_id, grid_shape), ::testing::Eq(key));
}

TEST(KeyToEntryIdTest, OutOfRange) {
  EXPECT_THAT(KeyToEntryId(std::string{0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 3},
                           {{4, 5, 6}}),
              ::testing::Eq(std::nullopt));
}

TEST(KeyToEntryIdTest, Invalid) {
  EXPECT_THAT(
      KeyToEntryId(std::string{0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0}, {{4, 5, 6}}),
      ::testing::Eq(std::nullopt));
}

TEST(IndicesToKeyTest, Basic) {
  const Index indices[] = {1, 2, 3};
  std::string key{0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3};
  EXPECT_THAT(IndicesToKey(indices), ::testing::Eq(key));
  Index decoded_indices[3];
  EXPECT_TRUE(KeyToIndices(key, decoded_indices));
  EXPECT_THAT(decoded_indices, ::testing::ElementsAreArray(indices));
  EXPECT_FALSE(KeyToIndices(key.substr(1), decoded_indices));
}

TEST(LowerBoundToEntryId, Exact) {
  Index grid_shape[] = {4, 5, 6};
  EXPECT_THAT(LowerBoundToEntryId(
                  std::string{0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3}, grid_shape),
              ::testing::Eq(1 * 5 * 6 + 2 * 6 + 3));
}

TEST(LowerBoundToEntryId, Longer) {
  Index grid_shape[] = {4, 5, 6};
  EXPECT_THAT(
      LowerBoundToEntryId(std::string{0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0},
                          grid_shape),
      ::testing::Eq(1 * 5 * 6 + 2 * 6 + 4));
}

TEST(KeyRangeToEntryRange, Full) {
  Index grid_shape[] = {4, 5, 6};
  EXPECT_THAT(KeyRangeToEntryRange("", "", grid_shape),
              ::testing::Pair(0, 4 * 5 * 6));
}

TEST(KeyRangeToEntryRange, Partial) {
  Index grid_shape[] = {4, 5, 6};
  EXPECT_THAT(
      KeyRangeToEntryRange(
          std::string{
              0, 0, 0, 2,  //
              0, 0, 0, 3,  //
              0, 0, 0, 4,  //
          },
          std::string{
              0, 0, 0, 2,  //
              0, 0, 0, 4,  //
              0, 0, 0, 5,  //
          },
          grid_shape),
      ::testing::Pair(2 * (5 * 6) + 3 * 6 + 4, 2 * (5 * 6) + 4 * 6 + 5));
  EXPECT_THAT(KeyRangeToInternalKeyRange(KeyRange{std::string{
                                                      0, 0, 0, 2,  //
                                                      0, 0, 0, 3,  //
                                                      0, 0, 0, 4,  //
                                                  },
                                                  std::string{
                                                      0, 0, 0, 2,  //
                                                      0, 0, 0, 4,  //
                                                      0, 0, 0, 5,  //
                                                  }},
                                         grid_shape),
              KeyRange(EntryIdToInternalKey(2 * (5 * 6) + 3 * 6 + 4),
                       EntryIdToInternalKey(2 * (5 * 6) + 4 * 6 + 5)));
}

TEST(EntryIdToInternalKeyTest, Basic) {
  EntryId entry_id = 0x01020304;
  std::string internal_key{0x01, 0x02, 0x03, 0x04};
  EXPECT_THAT(EntryIdToInternalKey(entry_id), ::testing::Eq(internal_key));
  EXPECT_THAT(InternalKeyToEntryId(internal_key), ::testing::Eq(entry_id));
}

TEST(InternalKeyLowerBoundToEntryIdTest, Basic) {
  EXPECT_THAT(InternalKeyLowerBoundToEntryId(
                  std::string{0x01, 0x02, 0x03, 0x04}, 0x88888888),
              ::testing::Eq(0x01020304));
  EXPECT_THAT(InternalKeyLowerBoundToEntryId(
                  std::string{0x01, 0x02, 0x03, 0x04, 0x0}, 0x88888888),
              ::testing::Eq(0x01020304 + 1));
  EXPECT_THAT(
      InternalKeyLowerBoundToEntryId(std::string{0x01, 0x02, 0x03}, 0x88888888),
      ::testing::Eq(0x01020300));
  EXPECT_THAT(InternalKeyLowerBoundToEntryId(
                  std::string{0x01, 0x02, 0x03, 0x04}, 0x01020302),
              ::testing::Eq(0x01020302));
}

TEST(InternalKeyRangeToEntryRange, Basic) {
  EXPECT_THAT(InternalKeyRangeToEntryRange(std::string{0x01, 0x02, 0x03, 0x04},
                                           std::string{0x01, 0x02, 0x03, 0x07},
                                           0x88888888),
              ::testing::Pair(0x01020304, 0x01020307));
  EXPECT_THAT(InternalKeyRangeToEntryRange(std::string{0x01, 0x02, 0x03, 0x04},
                                           {}, 0x88888888),
              ::testing::Pair(0x01020304, 0x88888888));
}

}  // namespace
