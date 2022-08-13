// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"

#include <optional>
#include <tuple>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "tensorstore/internal/estimate_heap_usage/std_optional.h"
#include "tensorstore/internal/estimate_heap_usage/std_variant.h"
#include "tensorstore/internal/estimate_heap_usage/std_vector.h"
#include "tensorstore/util/apply_members/std_tuple.h"

namespace {

using ::tensorstore::internal::EstimateHeapUsage;

TEST(EstimateHeapUsageTest, Trivial) {
  EXPECT_EQ(0, EstimateHeapUsage(5));

  struct Trivial {};

  EXPECT_EQ(0, EstimateHeapUsage(Trivial{}));
}

TEST(EstimateHeapUsageTest, String) {
  std::string s(1000, 'x');
  EXPECT_EQ(s.capacity(), EstimateHeapUsage(s));
}

TEST(EstimateHeapUsageTest, Cord) {
  auto cord = absl::Cord(std::string(1000, 'x'));
  EXPECT_EQ(cord.size(), EstimateHeapUsage(cord));
}

TEST(EstimateHeapUsageTest, Optional) {
  EXPECT_EQ(0, EstimateHeapUsage(std::optional<int>()));
  EXPECT_EQ(0, EstimateHeapUsage(std::optional<int>(42)));
  EXPECT_EQ(0, EstimateHeapUsage(std::optional<std::string>()));
  auto o = std::optional<std::string>(std::in_place, 1000, 'x');
  EXPECT_EQ(o->capacity(), EstimateHeapUsage(o));
}

TEST(EstimateHeapUsageTest, Vector) {
  std::vector<std::string> v;
  v.push_back(std::string(1000, 'x'));
  v.push_back(std::string(5000, 'x'));
  size_t expected =
      v[0].capacity() + v[1].capacity() + v.capacity() * sizeof(std::string);
  EXPECT_EQ(expected, EstimateHeapUsage(v));
  EXPECT_EQ(v.capacity() * sizeof(std::string), EstimateHeapUsage(v, 0));
}

TEST(EstimateHeapUsageTest, Composite) {
  std::variant<std::vector<std::string>, std::vector<int>> v;
  v = std::vector<std::string>({"a", "b"});
  {
    auto& string_vec = std::get<std::vector<std::string>>(v);
    EXPECT_EQ(string_vec.capacity() * sizeof(std::string) +
                  string_vec[0].capacity() + string_vec[1].capacity(),
              EstimateHeapUsage(v));
    EXPECT_EQ(string_vec.capacity() * sizeof(std::string),
              EstimateHeapUsage(v, /*max_depth=*/0));
  }

  v = std::vector<int>({1, 2, 3});
  {
    auto& int_vec = std::get<std::vector<int>>(v);
    EXPECT_EQ(int_vec.capacity() * sizeof(int), EstimateHeapUsage(v));
  }
}

TEST(EstimateHeapUsageTest, Tuple) {
  auto t = std::tuple{std::string(1000, 'x'), std::string(5000, 'x')};
  auto& [s0, s1] = t;
  EXPECT_EQ(s0.capacity() + s1.capacity(), EstimateHeapUsage(t));
}

TEST(EstimateHeapUsageTest, Variant) {
  using Variant = std::variant<int, std::string>;

  EXPECT_EQ(0, EstimateHeapUsage(Variant(5)));
  std::string s(1000, 'x');
  size_t capacity = s.capacity();
  EXPECT_EQ(capacity, EstimateHeapUsage(Variant(std::move(s))));
}

}  // namespace
