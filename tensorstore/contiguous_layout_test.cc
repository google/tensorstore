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

#include "tensorstore/contiguous_layout.h"

#include <array>
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index.h"
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::ComputeStrides;
using ::tensorstore::ContiguousLayoutOrder;
using ::tensorstore::GetContiguousIndices;
using ::tensorstore::GetContiguousOffset;
using ::tensorstore::Index;
using ::tensorstore::span;

TEST(ContiguousLayoutOrderTest, PrintToOstream) {
  {
    std::ostringstream ostr;
    ostr << ContiguousLayoutOrder::c;
    EXPECT_EQ("C", ostr.str());
  }
  {
    std::ostringstream ostr;
    ostr << ContiguousLayoutOrder::fortran;
    EXPECT_EQ("F", ostr.str());
  }
}

TEST(ComputeStridesTest, COrder) {
  {
    std::array<Index, 3> strides;
    ComputeStrides(ContiguousLayoutOrder::c, /*element_stride=*/1,
                   span<const Index>({3l, 4l, 5l}), strides);
    EXPECT_THAT(strides, ::testing::ElementsAre(20, 5, 1));
  }
  {
    std::array<Index, 3> strides;
    ComputeStrides(ContiguousLayoutOrder::c, /*element_stride=*/2,
                   span<const Index>({3l, 4l, 5l}), strides);
    EXPECT_THAT(strides, ::testing::ElementsAre(40, 10, 2));
  }
}

TEST(ComputeStridesTest, FOrder) {
  std::array<Index, 3> strides;
  ComputeStrides(ContiguousLayoutOrder::fortran, /*element_stride=*/1,
                 span<const Index>({3l, 4l, 5l}), strides);
  EXPECT_THAT(strides, ::testing::ElementsAre(1, 3, 12));
}

TEST(GetContiguousOffsetTest, Basic) {
  Index indices[2];
  EXPECT_EQ(3 * 11 + 4,
            GetContiguousOffset<ContiguousLayoutOrder::c>({{7, 11}}, {{3, 4}}));
  GetContiguousIndices<ContiguousLayoutOrder::c, Index>(3 * 11 + 4, {{7, 11}},
                                                        indices);
  EXPECT_THAT(indices, ::testing::ElementsAre(3, 4));
  EXPECT_EQ(3 + 4 * 7, GetContiguousOffset<ContiguousLayoutOrder::fortran>(
                           {{7, 11}}, {{3, 4}}));
  GetContiguousIndices<ContiguousLayoutOrder::fortran, Index>(
      3 + 4 * 7, {{7, 11}}, indices);
  EXPECT_THAT(indices, ::testing::ElementsAre(3, 4));
  EXPECT_EQ(
      2 * (7 * 11) + 3 * 11 + 4,
      GetContiguousOffset<ContiguousLayoutOrder::c>({{5, 7, 11}}, {{2, 3, 4}}));
  EXPECT_EQ(2 + 5 * 3 + (5 * 7) * 4,
            GetContiguousOffset<ContiguousLayoutOrder::fortran>({{5, 7, 11}},
                                                                {{2, 3, 4}}));
  EXPECT_EQ(0, GetContiguousOffset<ContiguousLayoutOrder::c>({}, {}));
  EXPECT_EQ(0, GetContiguousOffset<ContiguousLayoutOrder::fortran>({}, {}));
}

}  // namespace
