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

#include "tensorstore/util/iterate_over_index_range.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::ContiguousLayoutOrder;
using ::tensorstore::Index;
using ::tensorstore::IterateOverIndexRange;
using ::tensorstore::span;

TEST(IterateOverIndexRange, COrder) {
  using R = std::vector<int>;

  std::vector<R> result;
  const std::vector<R> expected_result{{0, 0}, {0, 1}, {0, 2},
                                       {1, 0}, {1, 1}, {1, 2}};
  IterateOverIndexRange<ContiguousLayoutOrder::c>(
      span({2, 3}),
      [&](span<const int, 2> x) { result.emplace_back(x.begin(), x.end()); });
  EXPECT_EQ(expected_result, result);
}

TEST(IterateOverIndexRange, FortranOrder) {
  using R = std::vector<int>;

  std::vector<R> result;
  // The expected result is simply a permutation of the C-order expected result.
  // Fortran order does not reverse the index vectors themselves (it is not
  // equivalent to C-order iteration with the shape vector reversed).
  const std::vector<R> expected_result{{0, 0}, {1, 0}, {0, 1},
                                       {1, 1}, {0, 2}, {1, 2}};
  IterateOverIndexRange<ContiguousLayoutOrder::fortran>(
      span({2, 3}),
      [&](span<const int, 2> x) { result.emplace_back(x.begin(), x.end()); });
  EXPECT_EQ(expected_result, result);
}

TEST(IterateOverIndexRange, COrderWithOrigin) {
  using R = std::vector<int>;

  std::vector<R> result;
  const std::vector<R> expected_result{{0, 1}, {0, 2}, {1, 1}, {1, 2}};
  IterateOverIndexRange<ContiguousLayoutOrder::c>(
      span({0, 1}), span({2, 2}),
      [&](span<const int, 2> x) { result.emplace_back(x.begin(), x.end()); });
  EXPECT_EQ(expected_result, result);
}

TEST(IterateOverIndexRange, FortranOrderWithOrigin) {
  using R = std::vector<int>;

  std::vector<R> result;
  // The expected result is simply a permutation of the C-order expected result.
  // Fortran order does not reverse the index vectors themselves (it is not
  // equivalent to C-order iteration with the shape vector reversed).
  const std::vector<R> expected_result{{0, 1}, {1, 1}, {0, 2}, {1, 2}};
  IterateOverIndexRange<ContiguousLayoutOrder::fortran>(
      span({0, 1}), span({2, 2}),
      [&](span<const int, 2> x) { result.emplace_back(x.begin(), x.end()); });
  EXPECT_EQ(expected_result, result);
}

TEST(IterateOverIndexRange, COrderWithBox) {
  using R = std::vector<Index>;

  std::vector<R> result;
  const std::vector<R> expected_result{{0, 1}, {0, 2}, {1, 1}, {1, 2}};
  IterateOverIndexRange(
      tensorstore::BoxView({0, 1}, {2, 2}),
      [&](span<const Index, 2> x) { result.emplace_back(x.begin(), x.end()); },
      ContiguousLayoutOrder::c);
  EXPECT_EQ(expected_result, result);
}

TEST(IterateOverIndexRange, RankZero) {
  using R = std::vector<int>;

  std::vector<R> result;
  const std::vector<R> expected_result{R{}};
  IterateOverIndexRange<ContiguousLayoutOrder::fortran>(
      span<const int, 0>(),
      [&](span<const int, 0> x) { result.emplace_back(x.begin(), x.end()); });
  EXPECT_EQ(expected_result, result);
}

TEST(IterateOverIndexRange, Stop) {
  using R = std::vector<int>;

  std::vector<R> result;
  const std::vector<R> expected_result{{0, 0}, {0, 1}};
  EXPECT_EQ(false, IterateOverIndexRange<ContiguousLayoutOrder::c>(
                       span({2, 3}), [&](span<const int, 2> x) {
                         result.emplace_back(x.begin(), x.end());
                         return x[1] != 1;
                       }));
  EXPECT_EQ(expected_result, result);
}

TEST(IterateOverIndexRange, ZeroElementsBoolReturn) {
  // Check that iterating over a zero-element shape with a return value of
  // `bool` returns `true` due to the `DefaultIterationResult<bool>`
  // specialization.
  EXPECT_EQ(true, IterateOverIndexRange<ContiguousLayoutOrder::c>(
                      span({0}), [&](span<const int, 1> x) { return false; }));
}

TEST(IterateOverIndexRange, StaticRankZero) {
  using R = std::vector<int>;
  std::vector<R> result;
  const std::vector<R> expected_result{R{}};
  IterateOverIndexRange(span<const int, 0>{}, [&](span<const int, 0> x) {
    result.emplace_back(x.begin(), x.end());
  });
  EXPECT_EQ(expected_result, result);
}

TEST(IterateOverIndexRange, DynamicRankZero) {
  using R = std::vector<int>;
  std::vector<R> result;
  const std::vector<R> expected_result{R{}};
  IterateOverIndexRange(span<const int>(nullptr, 0), [&](span<const int> x) {
    result.emplace_back(x.begin(), x.end());
  });
  EXPECT_EQ(expected_result, result);
}

}  // namespace
