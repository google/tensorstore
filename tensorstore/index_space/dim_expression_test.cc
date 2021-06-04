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

#include "tensorstore/index_space/dim_expression.h"

/// \file
///
/// This file serves as an example of how to apply a DimExpression to
/// a tensorstore array. The tests are minimal because they are redundant
/// with the individual *_op_test.cc files.

#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::AllDims;
using ::tensorstore::BoxView;
using ::tensorstore::DimRange;
using ::tensorstore::Dims;
using ::tensorstore::Index;
using ::tensorstore::MakeArray;
using ::tensorstore::MakeOffsetArrayView;
using ::tensorstore::Materialize;  // TransformedArray to OffsetArray
                                   // conversion.

static const Index default_origin[3] = {0, 0, 0};

auto TestArray(tensorstore::span<const Index, 3> origin = default_origin) {
  // Test array values are constructed so that the value is a 1-indexed
  // representation of the coordinate position. Thus when comparing values
  // we can easily map the input to a numeric literal.
  //
  // Also, in some operations the origin matters, while in others it
  // does not; in those cases where the comparison does not differ,
  // the default origin of {0,0,0} is used.
  static const int test_array[4][4][8] = {
      {
          {111, 112, 113, 114, 115, 116, 117, 118},
          {121, 122, 123, 124, 125, 126, 127, 128},
          {131, 132, 133, 134, 135, 136, 137, 138},
          {141, 142, 143, 144, 145, 146, 147, 148},
      },
      {
          {211, 212, 213, 214, 215, 216, 217, 218},
          {221, 222, 223, 224, 225, 226, 227, 228},
          {231, 232, 233, 234, 235, 236, 237, 238},
          {241, 242, 243, 244, 245, 246, 247, 248},
      },
      {
          {311, 312, 313, 314, 315, 316, 317, 318},
          {321, 322, 323, 324, 325, 326, 327, 328},
          {331, 332, 333, 334, 335, 336, 337, 338},
          {341, 342, 343, 344, 345, 346, 347, 348},
      },
      {
          {411, 412, 413, 414, 415, 416, 417, 418},
          {421, 422, 423, 424, 425, 426, 427, 428},
          {431, 432, 433, 434, 435, 436, 437, 438},
          {441, 442, 443, 444, 445, 446, 447, 448},
      }};

  return MakeOffsetArrayView(origin, test_array);
}

TEST(DimExpressionTest, TranslateBy) {
  /// For example: `Dims(0, 2).TranslateBy({10, 20})` has the following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {0, 2}
  /// Input domain        | [1, 3], [2, 5], [3, 4] | [11, 13], [2, 5], [23, 24]
  /// Labels              | {"x", "y", "z"}        | {"x", "y", "z"}
  /// Equiv. input indices| {2, 3, 3}              | {12, 3, 23}
  /// Equiv. input indices| {x, y, z}              | {x + 10, y, z + 20}
  auto view = TestArray() | Dims(0, 2).TranslateBy({10, 20}) | Materialize();
  TENSORSTORE_EXPECT_OK(view);

  EXPECT_EQ(344, ((*view)({12, 3, 23})));
}

TEST(DimExpressionTest, TranslateBySingle) {
  // All use the same.
  auto view = TestArray() | Dims(0, 2).TranslateBy(10);
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, TranslateTo) {
  const Index origin[3] = {1, 2, 3};
  /// For example: `Dims(0, 2).TranslateTo({10, 20})` has the following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {0, 2}
  /// Input domain        | [1, 3], [2, 5], [3, 4] | [10, 12], [2, 5], [20, 21]
  /// Labels              | {"x", "y", "z"}        | {"x", "y", "z"}
  /// Equiv. input indices| {2, 3, 3}              | {11, 3, 20}
  /// Equiv. input indices| {x, y, z}              | {x + 9, y, z + 17}
  auto view =
      TestArray(origin) | Dims(0, 2).TranslateTo({10, 20}) | Materialize();
  TENSORSTORE_EXPECT_OK(view);

  EXPECT_EQ(344 - 123, ((*view)({11, 3, 20})));
}

TEST(DimExpressionTest, TranslateToSingle) {
  auto view = TestArray() | AllDims().TranslateTo(0);
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, IndexSlice) {
  /// For example: `Dims(0, 2).IndexSlice({2, 4})` has the following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {}
  /// Input domain        | [1, 3], [2, 5], [3, 4] | [2, 5]
  /// Labels              | {"x", "y", "z"}        | {"y"}
  /// Equiv. input indices| {2, 3, 4}              | {3}
  /// Equiv. input indices| {2, y, 4}              | {y}
  auto view = TestArray() | Dims(0, 2).IndexSlice({2, 4}) | Materialize();
  TENSORSTORE_EXPECT_OK(view);

  EXPECT_EQ(345, ((*view)({3})));
}

TEST(DimExpressionTest, IndexSliceSingle) {
  auto view = TestArray() | Dims(0, 2).IndexSlice(1);
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, BoxSlice) {
  /// For example: `Dims(0, 2).BoxSlice(BoxView({1, 4}, {3, 4}))` has the
  /// following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {0, 2}
  /// Input domain        | [0, 6], [2, 5], [0, 9] | [1, 3], [2, 5], [4, 7]
  /// Labels              | {"x", "y", "z"}        | {"x", "y", "z"}
  /// Equiv. input indices| {1, 3, 4}              | {1, 3, 4}
  /// Equiv. input indices| {x, y, z}              | {x, y, z}
  auto view = TestArray() | Dims(0, 2).BoxSlice(BoxView({1, 4}, {3, 4})) |
              Materialize();
  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(245, ((*view)({1, 3, 4})));
}

TEST(DimExpressionTest, TranslateBoxSlice) {
  const Index origin[3] = {0, 2, 0};

  /// For example: `Dims(0, 2).TranslateBoxSlice(BoxView({1, 4}, {3, 4}))` has
  /// the following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {0, 2}
  /// Input domain        | [0, 6], [2, 5], [0, 9] | [0, 2], [2, 5], [0, 3]
  /// Labels              | {"x", "y", "z"}        | {"x", "y", "z"}
  /// Equiv. input indices| {1, 3, 4}              | {0, 3, 0}
  /// Equiv. input indices| {x + 1, y, z + 4}      | {x, y, z}
  auto view = TestArray(origin) |
              Dims(0, 2).TranslateBoxSlice(BoxView({1, 4}, {3, 4})) |
              Materialize();
  TENSORSTORE_EXPECT_OK(view);

  EXPECT_EQ(245 - 20, ((*view)({0, 3, 0})));
}

TEST(DimExpressionTest, ClosedInterval) {
  /// For example: `Dims(0, 2).ClosedInterval({1, 8}, {4, 3}, {1, -2})` has the
  /// following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {0, 2}
  /// Input domain        | [0, 6], [2, 5], [0, 9] | [1, 4], [2, 5], [-4, -2]
  /// Labels              | {"x", "y", "z"}        | {"x", "y", "z"}
  /// Equiv. input indices| {2, 3, 6}              | {2, 3, -3}
  /// Equiv. input indices| {x, y, z * -2}         | {x, y, z}
  auto view = TestArray() | Dims(0, 2).ClosedInterval({1, 6}, {3, 0}, {1, -2}) |
              Materialize();
  TENSORSTORE_EXPECT_OK(view);

  EXPECT_EQ(347, ((*view)({2, 3, -3})));
}

TEST(DimExpressionTest, ClosedInterval1) {
  auto view = TestArray() | Dims(0, 2).ClosedInterval(1, 1);
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, HalfOpenInterval) {
  /// For example: `Dims(0, 2).HalfOpenInterval({1, 8}, {4, 3}, {1, -2})` has
  /// the following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {0, 2}
  /// Input domain        | [0, 6], [2, 5], [0, 9] | [1, 3], [2, 5], [-4, -2]
  /// Labels              | {"x", "y", "z"}        | {"x", "y", "z"}
  /// Equiv. input indices| {2, 3, 6}              | {2, 3, -3}
  /// Equiv. input indices| {x, y, z * -2}         | {x, y, z}
  auto view = TestArray() |
              Dims(0, 2).HalfOpenInterval({1, 6}, {3, 0}, {1, -2}) |
              Materialize();
  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(347, ((*view)({2, 3, -3})));
}

TEST(DimExpressionTest, HalfOpenInterval1) {
  auto view = TestArray() | Dims(0, 2).HalfOpenInterval(1, 2);
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, SizedInterval) {
  /// For example: `Dims(0, 2).SizedInterval({1, 8}, {3, 2}, {1, -2})` has the
  /// following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {0, 2}
  /// Input domain        | [0, 6], [2, 5], [0, 9] | [1, 3], [2, 5], [-4, -3]
  /// Labels              | {"x", "y", "z"}        | {"x", "y", "z"}
  /// Equiv. input indices| {2, 3, 6}              | {2, 3, -3}
  /// Equiv. input indices| {x, y, z * -2}         | {x, y, z}
  auto view = TestArray() | Dims(0, 2).SizedInterval({1, 6}, {3, 2}, {1, -2}) |
              Materialize();
  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(347, ((*view)({2, 3, -3})));
}

TEST(DimExpressionTest, SizedInterval1) {
  auto view = TestArray() | Dims(0, 2).SizedInterval(1, 2);
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, TranslateClosedInterval) {
  auto view = TestArray() | Dims(0, 2).TranslateClosedInterval({0, 1}, {1, 1});
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, TranslateClosedInterval1) {
  auto view = TestArray() | Dims(0, 2).TranslateClosedInterval(1, 1);
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, TranslateHalfOpenInterval) {
  auto view =
      TestArray() | Dims(0, 2).TranslateHalfOpenInterval({0, 1}, {1, 1});
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, TranslateHalfOpenInterval1) {
  auto view = TestArray() | Dims(0, 2).TranslateHalfOpenInterval(1, 2);
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, TranslateSizedInterval) {
  auto view = TestArray() | Dims(0, 2).TranslateSizedInterval({0, 1}, {1, 1});
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, TranslateSizedInterval1) {
  auto view = TestArray() | Dims(0, 2).TranslateSizedInterval(1, 2);
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, IndexArraySlice) {
  /// For example:
  ///
  ///     Dims(0, 2).IndexArraySlice(MakeArray<Index>({{1, 2, 3}, {4, 5, 6}}),
  ///                                MakeArray<Index>({{7, 8, 9}, {0, 1, 2}}))
  ///
  /// has the following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {0, 1}
  /// Input domain        | [0, 6], [2, 5], [0, 9] | [0, 1], [0, 2], [2, 5]
  /// Labels              | {"x", "y", "z"}        | {"", "", "y"}
  /// Equiv. input indices| {1, y, 7}              | {0, 0, y}
  /// Equiv. input indices| {2, y, 8}              | {0, 1, y}
  /// Equiv. input indices| {3, y, 9}              | {0, 2, y}
  /// Equiv. input indices| {6, y, 2}              | {1, 2, y}
  /// Equiv. input indices| {xi(a, b), y, zi(a, b)}| {a, b, y}
  auto view = TestArray() |
              Dims(0, 2).IndexArraySlice(
                  /*xi=*/MakeArray<Index>({{1, 2, 3}, {3, 2, 1}}),
                  /*yi=*/MakeArray<Index>({{7, 6, 5}, {1, 2, 4}})) |
              Materialize();
  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(248, ((*view)({0, 0, 3})));
}

TEST(DimExpressionTest, IndexVectorArraySlice) {
  /// For example:
  ///
  ///     Dims(0, 2).IndexVectorArraySlice(
  ///         MakeArray<Index>({{{1, 7}, {2, 8}, {3, 9}},
  ///                           {{4, 0}, {5, 1}, {6, 2}}}),
  ///         -1)
  ///
  /// is equivalent to
  ///
  ///     Dims(0, 2).IndexArraySlice(MakeArray<Index>({{1, 2, 3}, {4, 5, 6}}),
  ///                                MakeArray<Index>({{7, 8, 9}, {0, 1, 2}}))
  ///
  /// and has the following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {0, 1}
  /// Input domain        | [0, 6], [2, 5], [0, 9] | [0, 1], [0, 2], [2, 5]
  /// Labels              | {"x", "y", "z"}        | {"", "", "y"}
  /// Equiv. input indices| {1, y, 7}              | {0, 0, y}
  /// Equiv. input indices| {2, y, 8}              | {0, 1, y}
  /// Equiv. input indices| {3, y, 9}              | {0, 2, y}
  /// Equiv. input indices| {6, y, 2}              | {1, 2, y}
  /// Equiv. input indices| {v(a,b,0), y, v(a,b,1)}| {a, b, y}
  ///
  auto view = TestArray() |
              Dims(0, 2).IndexVectorArraySlice(
                  /*v[a][b][c]*/ MakeArray<Index>(
                      {{{1, 7}, {2, 6}, {3, 5}}, {{3, 1}, {2, 2}, {1, 4}}}),
                  -1) |
              Materialize();

  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(248, ((*view)({0, 0, 3})));
}

TEST(DimExpressionTest, OuterIndexArraySlice) {
  /// For example:
  ///
  ///     Dims(2, 0).OuterIndexArraySlice(MakeArray<Index>({{2, 3}, {4, 5}}),
  ///                                     MakeArray<Index>({6, 7}))
  ///
  /// has the following effects:
  ///
  /// *              | Prior                  | New
  /// ---            | ---                    | ---
  /// Dim. selection | {2, 0}                 | {2, 3, 0}
  /// Input domain   | [4, 8], [2, 5], [0, 9] | [0, 1], [2, 5], [0, 1], [0, 1]
  /// Labels         | {"x", "y", "z"}        | {"", "y", "", ""}
  /// Equiv. inputs  | {6, 3, 3}              | {0, 3, 0, 1}
  /// Equiv. inputs  | {7, 3, 4}              | {1, 3, 1, 0}
  /// Equiv. inputs  | {xi(a), y, zi(b,c)}    | {a, y, b, c}
  auto view = TestArray() |
              Dims(2, 0).OuterIndexArraySlice(
                  /*zi[b][c]*/ MakeArray<Index>({{4, 5}, {6, 7}}),
                  /*xi[a]*/ MakeArray<Index>({3, 2})) |
              Materialize();

  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(438, ((*view)({0, 2, 1, 1})));
}

TEST(DimExpressionTest, Label) {
  /// For example: `Dims(0, 2).Label({"a", "b"})` has the following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {0, 2}
  /// Input domain        | [1, 3], [2, 5], [3, 4] | [1, 3], [2, 5], [3, 4]
  /// Labels              | {"x", "y", "z"}        | {"a", "y", "b"}
  ///
  auto view = TestArray() | Dims(0, 2).Label({"a", "b"});
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, LabelB) {
  auto view = TestArray() | Dims(0, 2).Label("a", "b");
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, MoveTo) {
  /// For example, `Dims(2, 0).MoveTo(1)` has the following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {1, 2}
  /// Input domain        | [1, 3], [2, 5], [3, 4] | [2, 5], [3, 4], [1, 3]
  /// Labels              | {"x", "y", "z"}        | {"y", "z", "x"}
  /// Equiv. input indices| {2, 3, 4}              | {3, 4, 2}
  /// Equiv. input indices| {x, y, z}              | {y, z, x}
  auto view = TestArray() | Dims(2, 0).MoveTo(1) | Materialize();
  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(345, ((*view)({3, 4, 2})));
}

TEST(DimExpressionTest, MoveToFront) {
  auto view = TestArray() | Dims(0, 2).MoveToFront();
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, MoveToBack) {
  auto view = TestArray() | Dims(0, 2).MoveToFront();
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, Diagonal) {
  /// For example: `Dims(0, 2).Diagonal()` has the following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {0}
  /// Input domain        | [1, 5], [2, 5], [3, 7] | [3, 5], [2, 5]
  /// Labels              | {"x", "y", "z"}        | {"", "y"}
  /// Equiv. input indices| {4, 3, 4}              | {4, 3}
  /// Equiv. input indices| {d, y, d}              | {d, y}
  auto view = TestArray() | Dims(0, 2).Diagonal() | Materialize();
  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(343, ((*view)({2, 3})));
}

TEST(DimExpressionTest, AddNew) {
  /// For example, `Dims(0, -1).AddNew()` (equivalent to `Dims(0, 2).AddNew()`)
  /// has the following effects:
  ///
  /// *                   | Prior  | New
  /// ------------------- | ---    | ---
  /// Dimension selection | {0, 2} | {0, 2}
  /// Input domain        | [1, 5] | [-inf*, +inf*], [1, 5], [-inf*, +inf*]
  /// Labels              | {"x"}  | {"", "x", ""}
  /// Equiv. input indices| {2}    | {1, 2, 8}
  /// Equiv. input indices| {x}    | {a, x, b}
  auto view = TestArray() | Dims(0, -1).AddNew() | Materialize();
  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(333, ((*view)({0, 2, 2, 2, 0})));
}

TEST(DimExpressionTest, Transpose) {
  /// For example, `Dims(2, 0, 1).Transpose()` has the following effects:
  ///
  /// *                   | Prior                    | New
  /// ------------------- | ---                      | ---
  /// Dimension selection | {2, 0, 1}                | {0, 1, 2}
  /// Input domain        | [1*, 3], [2, 5*], [3, 4] | [3, 4], [1*, 3], [2, 5*]
  /// Labels              | {"x", "y", "z"}          | {"z", "x", "y"}
  /// Equiv. input indices| {2, 3, 4}                | {4, 2, 3}
  /// Equiv. input indices| {x, y, z}                | {z, x, y}
  auto view = TestArray() | Dims(2, 0, 1).Transpose() | Materialize();
  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(234, ((*view)({3, 1, 2})));
}

TEST(DimExpressionTest, TransposeB) {
  /// For example, `Dims(2, 0).Transpose({1,2})` has the following effects:
  ///
  /// *                   | Prior                    | New
  /// ------------------- | ---                      | ---
  /// Dimension selection | {2, 0}                   | {1, 2}
  /// Input domain        | [1*, 3], [2, 5*], [3, 4] | [2, 5*], [3, 4], [1*, 3]
  /// Labels              | {"x", "y", "z"}          | {"y", "z", "x"}
  /// Equiv. input indices| {2, 3, 4}                | {3, 4, 2}
  /// Equiv. input indices| {x, y, z}                | {y, z, x}
  auto view = TestArray() | Dims(2, 0).Transpose({1, 2}) | Materialize();
  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(345, ((*view)({3, 4, 2})));
}

TEST(DimExpressionTest, MarkBoundsExplicit) {
  /// For example: `Dims(0, 2).MarkBoundsExplicit()` has the following effects:
  ///
  /// *                   | Prior                     | New
  /// ------------------- | ---                       | ---
  /// Dimension selection | {0, 2}                    | {0, 2}
  /// Input domain        | [1, 3*], [2*, 5], [3*, 4] | [1, 3], [2*, 5], [3, 4]
  /// Labels              | {"x", "y", "z"}           | {"x", "y", "z"}
  auto view = TestArray() | Dims(2, 0).MarkBoundsExplicit();
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, UnsafeMarkBoundsImplicit) {
  /// For example: `Dims(0, 2).UnsafeMarkBoundsImplicit()` has the following
  /// effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {0, 2}
  /// Input domain        | [1, 3], [2, 5], [3, 4] | [1*, 3*], [2, 5], [3*, 4*]
  /// Labels              | {"x", "y", "z"}        | {"x", "y", "z"}
  auto view = TestArray() | Dims(2, 0).UnsafeMarkBoundsImplicit();
  TENSORSTORE_EXPECT_OK(view);
}

TEST(DimExpressionTest, Stride) {
  /// For example: `Dims(0, 2).Stride({-2, 3})` has the following effects:
  ///
  /// *                   | Prior                  | New
  /// ------------------- | ---                    | ---
  /// Dimension selection | {0, 2}                 | {0, 2}
  /// Input domain        | [0, 6], [2, 5], [1, 8] | [-3, 0], [2, 5], [1, 2]
  /// Labels              | {"x", "y", "z"}        | {"x", "y", "z"}
  /// Equiv. input indices| {4, 3, 3}              | {-2, 3, 1}
  /// Equiv. input indices| {-2 * x, y, 3 * z}     | {x, y, z}
  auto view = TestArray() | Dims(0, 2).Stride({-2, 3}) | Materialize();
  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(344, ((*view)({-1, 3, 1})));
}

TEST(DimExpressionTest, AllDims) {
  /// AllDims.IndexSlice(x) resolves to a 0-rank array with the first value.
  auto view = TestArray() | AllDims().IndexSlice(1) | Materialize();
  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(222, ((*view)()));
}

TEST(DimExpressionTest, DimRange) {
  /// DimRange(1).IndexSlice(x) resolves to a 1-rank array.
  auto view =
      TestArray() | tensorstore::DimRange(1).IndexSlice(1) | Materialize();
  TENSORSTORE_EXPECT_OK(view);
  EXPECT_EQ(322, ((*view)(2)));
}

}  // namespace
