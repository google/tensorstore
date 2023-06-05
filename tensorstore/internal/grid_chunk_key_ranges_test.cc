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

#include "tensorstore/internal/grid_chunk_key_ranges.h"

#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/box.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/grid_chunk_key_ranges_base10.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Box;
using ::tensorstore::BoxView;
using ::tensorstore::CeilOfRatio;
using ::tensorstore::DimensionIndex;
using ::tensorstore::dynamic_rank;
using ::tensorstore::Index;
using ::tensorstore::IndexInterval;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::IndexTransformView;
using ::tensorstore::KeyRange;
using ::tensorstore::kMaxRank;
using ::tensorstore::Result;
using ::tensorstore::span;
using ::tensorstore::internal::Base10LexicographicalGridIndexKeyParser;
using ::tensorstore::internal::MinValueWithMaxBase10Digits;
using ::testing::ElementsAre;
using ::testing::Optional;

using R = std::tuple<KeyRange, size_t, std::vector<IndexInterval>>;

absl::Status GetChunkKeyRangesForRegularGridWithBase10Keys(
    IndexTransformView<> transform,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> chunk_shape, span<const Index> shape,
    char dimension_separator,
    absl::FunctionRef<absl::Status(std::string key)> handle_key,
    absl::FunctionRef<absl::Status(KeyRange key_range, size_t prefix,
                                   BoxView<> grid_bounds)>
        handle_key_range) {
  const DimensionIndex rank = grid_output_dimensions.size();
  assert(rank == chunk_shape.size());
  assert(rank == shape.size());
  Box<dynamic_rank(kMaxRank)> grid_bounds(rank);
  for (DimensionIndex i = 0; i < shape.size(); ++i) {
    const Index grid_size = CeilOfRatio(shape[i], chunk_shape[i]);
    grid_bounds[i] = IndexInterval::UncheckedSized(0, grid_size);
  }
  return GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys(
      transform, grid_output_dimensions, chunk_shape, grid_bounds,
      dimension_separator, Base10LexicographicalGridIndexKeyParser{},
      handle_key, handle_key_range);
}

Result<std::vector<R>> GetRanges(
    IndexTransformView<> transform,
    span<const DimensionIndex> grid_output_dimensions,
    span<const Index> chunk_shape, span<const Index> shape,
    char dimension_separator) {
  std::vector<R> ranges;
  const auto handle_key = [&](std::string key) -> absl::Status {
    ranges.emplace_back(KeyRange::Singleton(key), key.size(),
                        std::vector<IndexInterval>());
    return absl::OkStatus();
  };
  const auto handle_key_range = [&](KeyRange key_range, size_t prefix,
                                    BoxView<> grid_bounds) -> absl::Status {
    std::vector<IndexInterval> grid_bounds_copy(grid_bounds.rank());
    for (DimensionIndex i = 0; i < grid_bounds.rank(); ++i) {
      grid_bounds_copy[i] = grid_bounds[i];
    }
    ranges.emplace_back(std::move(key_range), prefix,
                        std::move(grid_bounds_copy));
    return absl::OkStatus();
  };
  TENSORSTORE_RETURN_IF_ERROR(GetChunkKeyRangesForRegularGridWithBase10Keys(
      transform, grid_output_dimensions, chunk_shape, shape,
      dimension_separator, handle_key, handle_key_range));
  return ranges;
}

TEST(ChunkKeyRangesTest, Rank0) {
  EXPECT_THAT(GetRanges(IndexTransformBuilder(0, 0).Finalize().value(),
                        /*grid_output_dimensions=*/{}, /*chunk_shape=*/{},
                        /*shape=*/{}, /*dimension_separator=*/'/'),
              Optional(ElementsAre(R{KeyRange::Singleton("0"), 1, {}})));
}

TEST(ChunkKeyRangesTest, Rank1Unconstrained) {
  EXPECT_THAT(GetRanges(IndexTransformBuilder(1, 1)
                            .input_shape({50})
                            .output_identity_transform()
                            .Finalize()
                            .value(),
                        /*grid_output_dimensions=*/{{0}}, /*chunk_shape=*/{{5}},
                        /*shape=*/{{50}}, /*dimension_separator=*/'/'),
              Optional(ElementsAre(
                  R{KeyRange(), 0, {IndexInterval::UncheckedSized(0, 10)}})));
}

TEST(ChunkKeyRangesTest, Rank1Constrained) {
  // Grid dimension 0:
  //   Output range: [7, 36]
  //   Grid range: [1, 7]
  EXPECT_THAT(
      GetRanges(IndexTransformBuilder(1, 1)
                    .input_origin({7})
                    .input_shape({30})
                    .output_identity_transform()
                    .Finalize()
                    .value(),
                /*grid_output_dimensions=*/{{0}}, /*chunk_shape=*/{{5}},
                /*shape=*/{{50}}, /*dimension_separator=*/'/'),
      Optional(ElementsAre(
          R{KeyRange("1", "8"), 0, {IndexInterval::UncheckedClosed(1, 7)}})));
}

TEST(ChunkKeyRangesTest, Rank1ConstrainedSplit) {
  // Grid dimension 0:
  //   Output range: [8, 12]
  //   Grid range: [8, 12]
  EXPECT_THAT(
      GetRanges(IndexTransformBuilder(1, 1)
                    .input_origin({8})
                    .input_exclusive_max({13})
                    .output_identity_transform()
                    .Finalize()
                    .value(),
                /*grid_output_dimensions=*/{{0}}, /*chunk_shape=*/{{1}},
                /*shape=*/{{20}}, /*dimension_separator=*/'/'),
      Optional(ElementsAre(R{KeyRange::Singleton("8"), 1, {}},
                           R{KeyRange::Singleton("9"), 1, {}},
                           R{KeyRange("10", "13"),
                             0,
                             {IndexInterval::UncheckedClosed(10, 12)}})));
}

TEST(ChunkKeyRangesTest, Rank2ConstrainedBothDims) {
  // Grid dimension 0:
  //   Output range: [6, 13]
  //   Grid range: [1, 2]
  // Grid dimension 1:
  //   Output range: [7, 37)
  //   Grid range: [0, 3]
  EXPECT_THAT(
      GetRanges(IndexTransformBuilder(2, 2)
                    .input_origin({6, 7})
                    .input_shape({8, 30})
                    .output_identity_transform()
                    .Finalize()
                    .value(),
                /*grid_output_dimensions=*/{{0, 1}}, /*chunk_shape=*/{{5, 10}},
                /*shape=*/{{25, 100}}, /*dimension_separator=*/'/'),
      Optional(ElementsAre(
          R{KeyRange("1/0", "1/4"), 2, {IndexInterval::UncheckedClosed(0, 3)}},
          R{KeyRange("2/0", "2/4"),
            2,
            {IndexInterval::UncheckedClosed(0, 3)}})));
}

TEST(ChunkKeyRangesTest, Rank2ConstrainedFirstDimOnly) {
  // Grid dimension 0:
  //   Output range: [6, 13]
  //   Grid range: [1, 2]
  // Grid dimension 1:
  //   Output range: [0, 49]
  //   Grid range: [0, 9] (unconstrained)
  EXPECT_THAT(
      GetRanges(IndexTransformBuilder(2, 2)
                    .input_origin({6, 0})
                    .input_shape({8, 50})
                    .output_identity_transform()
                    .Finalize()
                    .value(),
                /*grid_output_dimensions=*/{{0, 1}}, /*chunk_shape=*/{{5, 5}},
                /*shape=*/{{25, 50}}, /*dimension_separator=*/'/'),
      Optional(ElementsAre(R{KeyRange("1", "3"),
                             0,
                             {IndexInterval::UncheckedClosed(1, 2),
                              IndexInterval::UncheckedSized(0, 10)}})));
}

TEST(ChunkKeyRangesTest, Rank2ConstrainedFirstDimOnlySplit) {
  // Grid dimension 0:
  //   Output range: [8, 12]
  //   Grid range: [8, 12]
  // Grid dimension 1:
  //   Output range: [0, 49]
  //   Grid range: [0, 9] (unconstrained)
  EXPECT_THAT(
      GetRanges(IndexTransformBuilder(2, 2)
                    .input_origin({8, 0})
                    .input_shape({5, 50})
                    .output_identity_transform()
                    .Finalize()
                    .value(),
                /*grid_output_dimensions=*/{{0, 1}}, /*chunk_shape=*/{{1, 5}},
                /*shape=*/{{25, 50}}, /*dimension_separator=*/'/'),
      Optional(ElementsAre(
          R{KeyRange::Prefix("8/"), 2, {IndexInterval::UncheckedSized(0, 10)}},
          R{KeyRange::Prefix("9/"), 2, {IndexInterval::UncheckedSized(0, 10)}},
          R{KeyRange("10", "13"),
            0,
            {IndexInterval::UncheckedClosed(10, 12),
             IndexInterval::UncheckedSized(0, 10)}})));
}

TEST(ChunkKeyRangesTest, Rank2ConstrainedSecondDimOnly) {
  // Grid dimension 0:
  //   Output range: [0, 24]
  //   Grid range: [0, 4] (unconstrained)
  // Grid dimension 1:
  //   Output range: [7, 36]
  //   Grid range: [1, 7]
  EXPECT_THAT(
      GetRanges(IndexTransformBuilder(2, 2)
                    .input_origin({0, 7})
                    .input_shape({25, 30})
                    .output_identity_transform()
                    .Finalize()
                    .value(),
                /*grid_output_dimensions=*/{{0, 1}}, /*chunk_shape=*/{{5, 5}},
                /*shape=*/{{25, 50}}, /*dimension_separator=*/'/'),
      Optional(ElementsAre(
          R{KeyRange("0/1", "0/8"), 2, {IndexInterval::UncheckedClosed(1, 7)}},
          R{KeyRange("1/1", "1/8"), 2, {IndexInterval::UncheckedClosed(1, 7)}},
          R{KeyRange("2/1", "2/8"), 2, {IndexInterval::UncheckedClosed(1, 7)}},
          R{KeyRange("3/1", "3/8"), 2, {IndexInterval::UncheckedClosed(1, 7)}},
          R{KeyRange("4/1", "4/8"),
            2,
            {IndexInterval::UncheckedClosed(1, 7)}})));
}

TEST(MinValueWithMaxBase10DigitsTest, Basic) {
  EXPECT_EQ(0, MinValueWithMaxBase10Digits(0));
  EXPECT_EQ(0, MinValueWithMaxBase10Digits(1));
  EXPECT_EQ(0, MinValueWithMaxBase10Digits(9));
  EXPECT_EQ(0, MinValueWithMaxBase10Digits(10));
  EXPECT_EQ(10, MinValueWithMaxBase10Digits(11));
  EXPECT_EQ(10, MinValueWithMaxBase10Digits(100));
  EXPECT_EQ(100, MinValueWithMaxBase10Digits(101));
  EXPECT_EQ(100, MinValueWithMaxBase10Digits(999));
  EXPECT_EQ(100, MinValueWithMaxBase10Digits(1000));
  EXPECT_EQ(1000, MinValueWithMaxBase10Digits(1001));
}

}  // namespace
