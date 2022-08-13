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

#include "tensorstore/serialization/serialization.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/serialization/std_map.h"
#include "tensorstore/serialization/std_optional.h"
#include "tensorstore/serialization/std_set.h"
#include "tensorstore/serialization/std_tuple.h"
#include "tensorstore/serialization/std_variant.h"
#include "tensorstore/serialization/std_vector.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::serialization::IsNonSerializableLike;
using ::tensorstore::serialization::NonSerializable;
using ::tensorstore::serialization::SerializationRoundTrip;
using ::tensorstore::serialization::TestSerializationRoundTrip;

TEST(SerializationTest, Bool) {
  TestSerializationRoundTrip(true);
  TestSerializationRoundTrip(false);
}

TEST(SerializationTest, Float) {
  TestSerializationRoundTrip(3.14f);
  TestSerializationRoundTrip(0.0f);
}

TEST(SerializationTest, String) {
  TestSerializationRoundTrip(std::string("abcdefg"));
  TestSerializationRoundTrip(std::string(""));
}

TEST(CordTest, SerializationRoundTrip) {
  TestSerializationRoundTrip(absl::Cord(""));
  TestSerializationRoundTrip(absl::Cord("abc"));
}

TEST(SerializationTest, Int32) {
  TestSerializationRoundTrip(static_cast<int32_t>(0));
  TestSerializationRoundTrip(static_cast<int32_t>(3));
  TestSerializationRoundTrip(static_cast<int32_t>(2147483647));
  TestSerializationRoundTrip(static_cast<int32_t>(-2147483648));
}

TEST(SerializationTest, VectorInt) {
  TestSerializationRoundTrip(std::vector<int>{});
  TestSerializationRoundTrip(std::vector<int>{1, 2, 3});
}

TEST(SerializationTest, VectorString) {
  TestSerializationRoundTrip(std::vector<std::string>{});
  TestSerializationRoundTrip(std::vector<std::string>{"a", "b", "def"});
}

TEST(SerializationTest, VectorVectorString) {
  TestSerializationRoundTrip(
      std::vector<std::vector<std::string>>{{"a", "b", "def"}, {"e", "f"}});
}

TEST(SerializationTest, Map) {
  TestSerializationRoundTrip(std::map<int, std::string>{{1, "a"}, {2, "b"}});
}

TEST(SerializationTest, Set) {
  //
  TestSerializationRoundTrip(std::set<int>{1, 2, 3});
}

TEST(SerializationTest, Tuple) {
  TestSerializationRoundTrip(
      std::tuple(std::string("abc"), 3, std::string("def")));
}

TEST(SerializationTest, UniquePtrNull) {
  std::unique_ptr<int> ptr;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto ptr2, SerializationRoundTrip(ptr));
  EXPECT_FALSE(ptr2);
}

TEST(SerializationTest, UniquePtrNonNull) {
  auto ptr = std::make_unique<int>(5);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto ptr2, SerializationRoundTrip(ptr));
  EXPECT_THAT(ptr2, ::testing::Pointee(5));
}

TEST(SerializationTest, SharedPtrNull) {
  std::shared_ptr<int> ptr;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto ptr2, SerializationRoundTrip(ptr));
  EXPECT_FALSE(ptr2);
}

TEST(SerializationTest, SharedPtrNonNull) {
  auto ptr = std::make_shared<int>(5);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto ptr2, SerializationRoundTrip(ptr));
  EXPECT_THAT(ptr2, ::testing::Pointee(5));
}

TEST(SerializationTest, SharedPtrDuplicate) {
  auto ptr = std::make_shared<int>(5);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tuple2, SerializationRoundTrip(std::make_tuple(ptr, ptr)));
  EXPECT_THAT(std::get<0>(tuple2), ::testing::Pointee(5));
  EXPECT_EQ(std::get<0>(tuple2), std::get<1>(tuple2));
}

struct Foo {
  std::string a;
  std::string b;
  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.a, x.b);
  };
  bool operator==(const Foo& other) const {
    return a == other.a && b == other.b;
  }
};

TEST(SerializationTest, ApplyMembers) {
  TestSerializationRoundTrip(Foo{"xyz", "abcd"});
  TestSerializationRoundTrip(Foo{"", "abcd"});
}

TEST(SerialiationTest, Optional) {
  TestSerializationRoundTrip(std::optional<int>());
  TestSerializationRoundTrip(std::optional<int>(42));
}

TEST(SerialiationTest, Variant) {
  TestSerializationRoundTrip(std::variant<int, std::string>(42));
  TestSerializationRoundTrip(std::variant<int, std::string>("abc"));
  TestSerializationRoundTrip(std::variant<int, int>(std::in_place_index<1>, 1));
  TestSerializationRoundTrip(std::variant<int, int>(std::in_place_index<0>, 0));
}

static_assert(!IsNonSerializableLike<Foo>);
static_assert(!IsNonSerializableLike<std::pair<Foo, Foo>>);
static_assert(IsNonSerializableLike<NonSerializable<Foo>>);
static_assert(IsNonSerializableLike<std::pair<Foo, NonSerializable<Foo>>>);

}  // namespace
