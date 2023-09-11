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

#include "tensorstore/internal/grid_chunk_key_ranges_base10.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"

namespace {

using ::tensorstore::Index;
using ::tensorstore::IndexInterval;
using ::tensorstore::internal::Base10LexicographicalGridIndexKeyParser;
using ::tensorstore::internal::MinValueWithMaxBase10Digits;

TEST(Base10LexicographicalGridIndexKeyParserTest, FormatKeyRank0) {
  Base10LexicographicalGridIndexKeyParser parser(/*rank=*/0,
                                                 /*dimension_separator=*/'/');
  EXPECT_THAT(parser.FormatKey({}), "0");
}

TEST(Base10LexicographicalGridIndexKeyParserTest, FormatKeyRank1) {
  Base10LexicographicalGridIndexKeyParser parser(/*rank=*/1,
                                                 /*dimension_separator=*/'/');
  EXPECT_THAT(parser.FormatKey({{2}}), "2");
  EXPECT_THAT(parser.FormatKey({}), "");
}

TEST(Base10LexicographicalGridIndexKeyParserTest, FormatKeyRank2) {
  Base10LexicographicalGridIndexKeyParser parser(/*rank=*/2,
                                                 /*dimension_separator=*/'/');
  EXPECT_THAT(parser.FormatKey({{2, 3}}), "2/3");
  EXPECT_THAT(parser.FormatKey({{2}}), "2/");
  EXPECT_THAT(parser.FormatKey({}), "");
}

TEST(Base10LexicographicalGridIndexKeyParserTest, ParseKeyRank1) {
  Base10LexicographicalGridIndexKeyParser parser(/*rank=*/1,
                                                 /*dimension_separator=*/'/');
  Index indices[1];
  EXPECT_TRUE(parser.ParseKey("2", indices));
  EXPECT_THAT(indices, ::testing::ElementsAre(2));
  EXPECT_FALSE(parser.ParseKey("", indices));
  EXPECT_FALSE(parser.ParseKey("-1", indices));
  EXPECT_FALSE(parser.ParseKey("a", indices));
  EXPECT_FALSE(parser.ParseKey("2/3", indices));
  EXPECT_FALSE(parser.ParseKey("2/", indices));
}

TEST(Base10LexicographicalGridIndexKeyParserTest, ParseKeyRank2) {
  Base10LexicographicalGridIndexKeyParser parser(/*rank=*/2,
                                                 /*dimension_separator=*/'/');
  Index indices[2];
  EXPECT_TRUE(parser.ParseKey("2/3", indices));
  EXPECT_THAT(indices, ::testing::ElementsAre(2, 3));
  EXPECT_TRUE(parser.ParseKey("212/335", indices));
  EXPECT_THAT(indices, ::testing::ElementsAre(212, 335));
  EXPECT_FALSE(parser.ParseKey("1", indices));
  EXPECT_FALSE(parser.ParseKey("", indices));
  EXPECT_FALSE(parser.ParseKey("1/2/3", indices));
  EXPECT_FALSE(parser.ParseKey("1/2/", indices));
}

TEST(Base10LexicographicalGridIndexKeyParserTest,
     MinGridIndexForLexicographicalOrder) {
  Base10LexicographicalGridIndexKeyParser parser(/*rank=*/2,
                                                 /*dimension_separator=*/'/');
  EXPECT_THAT(parser.MinGridIndexForLexicographicalOrder(
                  0, IndexInterval::UncheckedHalfOpen(0, 1)),
              0);
  EXPECT_THAT(parser.MinGridIndexForLexicographicalOrder(
                  0, IndexInterval::UncheckedHalfOpen(0, 9)),
              0);
  EXPECT_THAT(parser.MinGridIndexForLexicographicalOrder(
                  0, IndexInterval::UncheckedHalfOpen(0, 10)),
              0);
  EXPECT_THAT(parser.MinGridIndexForLexicographicalOrder(
                  0, IndexInterval::UncheckedHalfOpen(0, 11)),
              10);
  EXPECT_THAT(parser.MinGridIndexForLexicographicalOrder(
                  0, IndexInterval::UncheckedHalfOpen(0, 100)),
              10);
  EXPECT_THAT(parser.MinGridIndexForLexicographicalOrder(
                  0, IndexInterval::UncheckedHalfOpen(0, 101)),
              100);
  EXPECT_THAT(parser.MinGridIndexForLexicographicalOrder(
                  0, IndexInterval::UncheckedHalfOpen(0, 999)),
              100);
  EXPECT_THAT(parser.MinGridIndexForLexicographicalOrder(
                  0, IndexInterval::UncheckedHalfOpen(0, 1000)),
              100);
  EXPECT_THAT(parser.MinGridIndexForLexicographicalOrder(
                  0, IndexInterval::UncheckedHalfOpen(0, 1001)),
              1000);
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
