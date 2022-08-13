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

#include "tensorstore/serialization/function.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::serialization::BindFront;
using ::tensorstore::serialization::NonSerializable;
using ::tensorstore::serialization::SerializableFunction;
using ::tensorstore::serialization::SerializationRoundTrip;

TEST(SerializationTest, Function) {
  SerializableFunction<int()> func([] { return 3; });
  EXPECT_EQ(3, func());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto func_decoded,
                                   SerializationRoundTrip(func));
  EXPECT_EQ(3, func_decoded());
}

TEST(SerializationTest, BindFront) {
  SerializableFunction<int()> func =
      BindFront([](int a, int b) { return a + b; }, 2, 5);
  EXPECT_EQ(7, func());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto func_decoded,
                                   SerializationRoundTrip(func));
  EXPECT_EQ(7, func_decoded());
}

TEST(SerializationTest, NonSerializable) {
  // Lambdas with captures are non-serializable.
  SerializableFunction<int()> func = NonSerializable{[y = 5] { return y; }};

  EXPECT_EQ(5, func());
  EXPECT_THAT(SerializationRoundTrip(func),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Serialization not supported.*"));
}

struct FunctionWithId1 {
  constexpr static const char id[] = "my_test_function1";
  int operator()() const { return 1; }
};

struct FunctionWithId2 {
  constexpr static const char id[] = "my_test_function2";
  int operator()() const { return 2; }
};

TEST(SerializationTest, Id) {
  SerializableFunction<int()> func1 = FunctionWithId1{};
  SerializableFunction<int()> func2 = FunctionWithId2{};
  EXPECT_EQ(1, func1());
  EXPECT_EQ(2, func2());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto func1_copy,
                                   SerializationRoundTrip(func1));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto func2_copy,
                                   SerializationRoundTrip(func2));
  EXPECT_EQ(1, func1_copy());
  EXPECT_EQ(2, func2_copy());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto func1_encoded, tensorstore::serialization::EncodeBatch(func1));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_func1_encoded,
                                   tensorstore::serialization::EncodeBatch(
                                       std::string_view(FunctionWithId1::id)));
  EXPECT_EQ(expected_func1_encoded, func1_encoded);
}

}  // namespace
