// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/internal/insertion_sort.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal::InsertionSort;
using ::tensorstore::internal::ShellSort;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::Pointee;

constexpr std::array<int, 5> SortConstexpr() {
  int vals[5] = {4, 1, 5, 2, 3};
  InsertionSort(vals, vals + 5);
  return {vals[0], vals[1], vals[2], vals[3], vals[4]};
}
static_assert(SortConstexpr() == std::array<int, 5>{1, 2, 3, 4, 5});

TEST(InsertionSortTest, EmptyAndSingleton) {
  std::vector<int> empty;
  InsertionSort(empty.begin(), empty.end());
  EXPECT_THAT(empty, IsEmpty());

  std::vector<int> single = {42};
  InsertionSort(single.begin(), single.end());
  EXPECT_THAT(single, ElementsAre(42));
}

TEST(InsertionSortTest, SortedReverseAndDuplicates) {
  std::vector<int> sorted = {1, 2, 3, 4, 5};
  InsertionSort(sorted.begin(), sorted.end());
  EXPECT_THAT(sorted, ElementsAre(1, 2, 3, 4, 5));

  std::vector<int> reversed = {5, 4, 3, 2, 1};
  InsertionSort(reversed.begin(), reversed.end());
  EXPECT_THAT(reversed, ElementsAre(1, 2, 3, 4, 5));

  std::vector<int> duplicates = {3, 1, 2, 3, 1, 0, 2};
  InsertionSort(duplicates.begin(), duplicates.end());
  EXPECT_THAT(duplicates, ElementsAre(0, 1, 1, 2, 2, 3, 3));
}

TEST(InsertionSortTest, CustomComparatorAndStability) {
  std::vector<int> desc = {2, 9, 1, 5, 4};
  InsertionSort(desc.begin(), desc.end(), std::greater<>{});
  EXPECT_THAT(desc, ElementsAre(9, 5, 4, 2, 1));

  // Equal keys preserve their initial relative order.
  std::vector<std::pair<int, int>> items = {
      {2, 0}, {1, 0}, {2, 1}, {1, 1}, {2, 2},
  };
  InsertionSort(items.begin(), items.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
  EXPECT_THAT(items, ElementsAre(Pair(1, 0), Pair(1, 1), Pair(2, 0), Pair(2, 1),
                                 Pair(2, 2)));
}

TEST(InsertionSortTest, MoveOnlyElements) {
  std::vector<std::unique_ptr<int>> ptrs;
  ptrs.push_back(std::make_unique<int>(30));
  ptrs.push_back(std::make_unique<int>(10));
  ptrs.push_back(std::make_unique<int>(20));
  InsertionSort(ptrs.begin(), ptrs.end(),
                [](const auto& a, const auto& b) { return *a < *b; });
  EXPECT_THAT(ptrs, ElementsAre(Pointee(10), Pointee(20), Pointee(30)));
}

constexpr std::array<int, 12> ShellSortConstexpr() {
  int vals[12] = {11, 4, 9, 1, 12, 5, 2, 10, 3, 8, 6, 7};
  ShellSort(vals, vals + 12);
  std::array<int, 12> out{};
  for (size_t i = 0; i < 12; ++i) out[i] = vals[i];
  return out;
}
static_assert(ShellSortConstexpr() ==
              std::array<int, 12>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

TEST(ShellSortTest, EmptyAndSingleton) {
  std::vector<int> empty;
  ShellSort(empty.begin(), empty.end());
  EXPECT_THAT(empty, IsEmpty());

  std::vector<int> single = {42};
  ShellSort(single.begin(), single.end());
  EXPECT_THAT(single, ElementsAre(42));
}

TEST(ShellSortTest, SortedReverseAndDuplicates) {
  std::vector<int> sorted = {1, 2, 3, 4, 5, 6, 7, 8};
  ShellSort(sorted.begin(), sorted.end());
  EXPECT_THAT(sorted, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8));

  std::vector<int> reversed = {8, 7, 6, 5, 4, 3, 2, 1};
  ShellSort(reversed.begin(), reversed.end());
  EXPECT_THAT(reversed, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8));

  std::vector<int> duplicates = {3, 1, 2, 3, 1, 0, 2, 1, 3, 0, 2};
  ShellSort(duplicates.begin(), duplicates.end());
  EXPECT_THAT(duplicates, ElementsAre(0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3));
}

std::vector<int32_t> RunShellSort(std::vector<int32_t> v) {
  ShellSort(v.begin(), v.end());
  return v;
}

std::vector<int32_t> MakePattern(size_t n, uint32_t mult) {
  std::vector<int32_t> v(n);
  uint32_t x = 0x9e3779b9u;
  for (size_t i = 0; i < n; ++i) {
    x = x * mult + 0x7f4a7c15u;
    v[i] = static_cast<int32_t>(x % 199u) - 99;
  }
  return v;
}

std::vector<int32_t> ExpectedSorted(std::vector<int32_t> v) {
  std::sort(v.begin(), v.end());
  return v;
}

TEST(ShellSortTest, AcrossCiuraGapBoundaries) {
  EXPECT_EQ(RunShellSort(MakePattern(4, 1664525u)),
            ExpectedSorted(MakePattern(4, 1664525u)));
  EXPECT_EQ(RunShellSort(MakePattern(5, 1664525u)),
            ExpectedSorted(MakePattern(5, 1664525u)));
  EXPECT_EQ(RunShellSort(MakePattern(10, 1664525u)),
            ExpectedSorted(MakePattern(10, 1664525u)));
  EXPECT_EQ(RunShellSort(MakePattern(23, 1664525u)),
            ExpectedSorted(MakePattern(23, 1664525u)));
  EXPECT_EQ(RunShellSort(MakePattern(24, 1664525u)),
            ExpectedSorted(MakePattern(24, 1664525u)));
  EXPECT_EQ(RunShellSort(MakePattern(58, 1664525u)),
            ExpectedSorted(MakePattern(58, 1664525u)));
  EXPECT_EQ(RunShellSort(MakePattern(302, 1664525u)),
            ExpectedSorted(MakePattern(302, 1664525u)));
  EXPECT_EQ(RunShellSort(MakePattern(5000, 1664525u)),
            ExpectedSorted(MakePattern(5000, 1664525u)));
}

TEST(ShellSortTest, CustomComparatorAndMoveOnly) {
  std::vector<int> desc = {2, 9, 1, 5, 4, 8, 3, 7, 6, 0, 11, 10};
  ShellSort(desc.begin(), desc.end(), std::greater<>{});
  EXPECT_THAT(desc, ElementsAre(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

  std::vector<std::unique_ptr<int>> ptrs;
  for (int x : {50, 10, 40, 20, 60, 30}) {
    ptrs.push_back(std::make_unique<int>(x));
  }
  ShellSort(ptrs.begin(), ptrs.end(),
            [](const auto& a, const auto& b) { return *a < *b; });
  EXPECT_THAT(ptrs, ElementsAre(Pointee(10), Pointee(20), Pointee(30),
                                Pointee(40), Pointee(50), Pointee(60)));

  std::vector<std::string> words = {"pear", "apple", "fig", "banana", "apple"};
  ShellSort(words.begin(), words.end());
  EXPECT_THAT(words, ElementsAre("apple", "apple", "banana", "fig", "pear"));
}

}  // namespace
