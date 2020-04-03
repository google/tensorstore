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

#include "tensorstore/index_space/output_index_map.h"

#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/strided_layout.h"

namespace {

using tensorstore::dynamic_rank;
using tensorstore::Index;
using tensorstore::IndexInterval;
using tensorstore::IndexTransformBuilder;
using tensorstore::MakeOffsetArray;
using tensorstore::offset_origin;
using tensorstore::OutputIndexMapIterator;
using tensorstore::OutputIndexMapRange;
using tensorstore::OutputIndexMapRef;
using tensorstore::OutputIndexMethod;
using tensorstore::span;
using tensorstore::StaticRank;
using tensorstore::StridedLayout;

TEST(OutputIndexMapTest, StaticRanks) {
  auto index_array = MakeOffsetArray<Index>({1, 2, 3}, {{{5}, {6}, {7}, {8}}});
  // Rank of index_array is equal to the input rank of transform to be created.
  auto t = IndexTransformBuilder<3, 4>()
               .input_origin({1, 2, 3})
               .input_shape({4, 4, 3})
               .output_constant(0, 10)
               .output_single_input_dimension(1, 20, 2, 2)
               .output_index_array(2, 30, 3, index_array,
                                   IndexInterval::Closed(3, 10))
               .Finalize()
               .value();
  auto range = t.output_index_maps();
  static_assert(std::is_same<decltype(range), OutputIndexMapRange<3, 4>>::value,
                "");
  static_assert(std::is_same<StaticRank<4>, decltype(range.size())>::value, "");
  static_assert(
      std::is_same<StaticRank<3>, decltype(range.input_rank())>::value, "");
  // range.size() is equal to t.output_rank().
  EXPECT_EQ(4, range.size());
  EXPECT_EQ(3, range.input_rank());
  EXPECT_EQ(false, range.empty());

  auto it = range.begin();
  static_assert(std::is_same<OutputIndexMapIterator<3>, decltype(it)>::value,
                "");
  EXPECT_EQ(range.begin(), it);
  EXPECT_NE(range.end(), it);
  EXPECT_EQ(range.end(), range.end());
  {
    auto output0 = *it;
    static_assert(std::is_same<OutputIndexMapRef<3>, decltype(output0)>::value,
                  "");
    EXPECT_EQ(OutputIndexMethod::constant, output0.method());
    EXPECT_EQ(10, output0.offset());
  }

  {
    auto it0 = it;
    // Check that preincrement returns a reference to itself.
    EXPECT_EQ(&++it0, &it0);
    EXPECT_EQ(20, it0->offset());

    // Check that predecrement returns a reference to itself.
    EXPECT_EQ(&--it0, &it0);
    EXPECT_EQ(10, it0->offset());
  }

  {
    // Test adding a distance.
    auto it0 = it + 1;
    EXPECT_EQ(20, it0->offset());

    it0 = 2 + it;
    EXPECT_EQ(30, it0->offset());

    // Test subtracting a distance.
    it0 = it0 - 2;
    EXPECT_EQ(10, it0->offset());
  }

  {
    auto it0 = it + 1;

    // Check distance operator.
    EXPECT_EQ(1, it0 - it);
    EXPECT_EQ(-1, it - it0);

    // Check comparison operators.
    EXPECT_TRUE(it < it0);
    EXPECT_TRUE(it <= it0);
    EXPECT_TRUE(it != it0);
    EXPECT_FALSE(it == it0);
    EXPECT_FALSE(it >= it0);
    EXPECT_FALSE(it > it0);

    EXPECT_FALSE(it0 < it);
    EXPECT_FALSE(it0 <= it);
    EXPECT_TRUE(it0 != it);
    EXPECT_FALSE(it0 == it);
    EXPECT_TRUE(it0 >= it);
    EXPECT_TRUE(it0 > it);

    EXPECT_FALSE(it < it);
    EXPECT_TRUE(it <= it);
    EXPECT_FALSE(it != it);
    EXPECT_TRUE(it == it);
    EXPECT_TRUE(it >= it);
    EXPECT_FALSE(it > it);
  }

  {
    auto it0 = it;
    // Test post increment.
    auto it1 = it0++;
    EXPECT_EQ(it1, it);
    EXPECT_EQ(it0, it + 1);
    EXPECT_EQ(10, it1->offset());
    EXPECT_EQ(20, it0->offset());

    // Test post decrement.
    auto it2 = it0--;
    EXPECT_EQ(it2, it + 1);
    EXPECT_EQ(it0, it);
  }

  ++it;
  {
    auto output1 = *it;
    EXPECT_EQ(OutputIndexMethod::single_input_dimension, output1.method());
    EXPECT_EQ(2, output1.input_dimension());
    EXPECT_EQ(20, output1.offset());
    EXPECT_EQ(2, output1.stride());
  }

  {
    auto output1a = range.begin()[1];
    static_assert(std::is_same<OutputIndexMapRef<3>, decltype(output1a)>::value,
                  "");
    EXPECT_EQ(OutputIndexMethod::single_input_dimension, output1a.method());
    EXPECT_EQ(2, output1a.input_dimension());
    EXPECT_EQ(20, output1a.offset());
    EXPECT_EQ(2, output1a.stride());
  }

  {
    auto output1b = range[1];
    static_assert(std::is_same<OutputIndexMapRef<3>, decltype(output1b)>::value,
                  "");
    EXPECT_EQ(OutputIndexMethod::single_input_dimension, output1b.method());
    EXPECT_EQ(2, output1b.input_dimension());
    EXPECT_EQ(20, output1b.offset());
    EXPECT_EQ(2, output1b.stride());
  }

  {
    auto output1c = t.output_index_map(1);
    static_assert(std::is_same<OutputIndexMapRef<3>, decltype(output1c)>::value,
                  "");
    EXPECT_EQ(OutputIndexMethod::single_input_dimension, output1c.method());
    EXPECT_EQ(2, output1c.input_dimension());
    EXPECT_EQ(20, output1c.offset());
    EXPECT_EQ(2, output1c.stride());
  }

  ++it;
  {
    auto output2 = *it;
    EXPECT_EQ(OutputIndexMethod::array, output2.method());
    EXPECT_EQ(30, output2.offset());
    EXPECT_EQ(3, output2.stride());
    auto index_array_ref = output2.index_array();
    EXPECT_EQ(&index_array(1, 2, 3), &index_array_ref.array_ref()(1, 2, 3));
    EXPECT_EQ(IndexInterval::UncheckedClosed(3, 10),
              index_array_ref.index_range());
    static_assert(
        std::is_same<StaticRank<3>, decltype(index_array_ref.rank())>::value,
        "");
    const StridedLayout<3, offset_origin> expected_layout(
        {1, 2, 3}, {4, 4, 3}, {0, sizeof(Index), 0});
    EXPECT_EQ(expected_layout, index_array_ref.layout());
    EXPECT_EQ(&index_array(1, 2, 3),
              &index_array_ref.shared_array_ref()(1, 2, 3));
    EXPECT_EQ(expected_layout, index_array_ref.shared_array_ref().layout());
    EXPECT_EQ(expected_layout, index_array_ref.array_ref().layout());
    EXPECT_THAT(index_array_ref.byte_strides(),
                testing::ElementsAreArray(expected_layout.byte_strides()));
    EXPECT_EQ(0, index_array_ref.byte_strides()[0]);
    EXPECT_EQ(sizeof(Index), index_array_ref.byte_strides()[1]);
    EXPECT_EQ(0, index_array_ref.byte_strides()[2]);
  }

  ++it;
  {
    auto output3 = *it;
    EXPECT_EQ(OutputIndexMethod::constant, output3.method());
    EXPECT_EQ(0, output3.offset());
  }

  ++it;
  EXPECT_EQ(range.end(), it);
}

TEST(OutputIndexMapTest, ZeroRank) {
  auto t = IndexTransformBuilder<3, 0>()
               .input_origin({1, 2, 3})
               .input_shape({4, 4, 3})
               .Finalize()
               .value();
  auto range = t.output_index_maps();
  EXPECT_EQ(0, range.size());
  EXPECT_EQ(3, range.input_rank());
  EXPECT_TRUE(range.empty());
}

TEST(OutputIndexMapTest, DynamicRanks) {
  auto index_array = MakeOffsetArray<Index>({1, 2, 3}, {{{5}, {6}, {7}, {8}}});
  auto t = IndexTransformBuilder<>(3, 4)
               .input_origin({1, 2, 3})
               .input_shape({4, 4, 3})
               .output_constant(0, 10)
               .output_single_input_dimension(1, 20, 2, 2)
               .output_index_array(2, 30, 3, index_array,
                                   IndexInterval::Closed(3, 10))
               .Finalize()
               .value();
  auto range = t.output_index_maps();
  static_assert(std::is_same<decltype(range), OutputIndexMapRange<>>::value,
                "");
  EXPECT_EQ(4, range.size());
  EXPECT_EQ(3, range.input_rank());
  EXPECT_EQ(false, range.empty());

  auto it = range.begin();
  static_assert(std::is_same<OutputIndexMapIterator<>, decltype(it)>::value,
                "");
  {
    auto output0 = *it;
    static_assert(std::is_same<OutputIndexMapRef<>, decltype(output0)>::value,
                  "");
    EXPECT_EQ(OutputIndexMethod::constant, output0.method());
    EXPECT_EQ(10, output0.offset());
  }

  {
    auto output2 = range[2];
    static_assert(std::is_same<OutputIndexMapRef<>, decltype(output2)>::value,
                  "");
    EXPECT_EQ(OutputIndexMethod::array, output2.method());
    EXPECT_EQ(30, output2.offset());
    EXPECT_EQ(3, output2.stride());
    auto index_array_ref = output2.index_array();
    EXPECT_EQ(&index_array(1, 2, 3), &index_array_ref.array_ref()(1, 2, 3));
    EXPECT_EQ(IndexInterval::UncheckedClosed(3, 10),
              index_array_ref.index_range());
    EXPECT_EQ(3, index_array.rank());
    const StridedLayout<dynamic_rank, offset_origin> expected_layout(
        {1, 2, 3}, {4, 4, 3}, {0, sizeof(Index), 0});
    EXPECT_EQ(expected_layout, index_array_ref.layout());
    EXPECT_EQ(&index_array(1, 2, 3),
              &index_array_ref.shared_array_ref()(1, 2, 3));
    EXPECT_EQ(expected_layout, index_array_ref.shared_array_ref().layout());
  }
}

}  // namespace
