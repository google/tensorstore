// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/serialization/registry.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"

namespace {

using ::tensorstore::serialization::Register;
using ::tensorstore::serialization::RegistrySerializer;
using ::tensorstore::serialization::SerializationRoundTrip;

struct Base {
  virtual ~Base() = default;
};

using BasePtr = std::shared_ptr<const Base>;

struct DerivedA : public Base {
  constexpr static const char id[] = "a";
  int x;

  static constexpr auto ApplyMembers = [](auto&& x, auto f) { return f(x.x); };
};

struct DerivedB : public Base {
  constexpr static const char id[] = "b";
  std::string y;
  static constexpr auto ApplyMembers = [](auto&& x, auto f) { return f(x.y); };
};

static const auto init = [] {
  Register<BasePtr, DerivedA>();
  Register<BasePtr, DerivedB>();
  return nullptr;
}();

TEST(RegistryTest, RoundTripA) {
  auto ptr = std::make_shared<DerivedA>();
  ptr->x = 42;

  EXPECT_THAT(
      SerializationRoundTrip(BasePtr(ptr), RegistrySerializer<BasePtr>{}),
      ::testing::Optional(
          ::testing::Pointee(::testing::WhenDynamicCastTo<const DerivedA&>(
              ::testing::Field(&DerivedA::x, 42)))));
}

TEST(RegistryTest, RoundTripB) {
  auto ptr = std::make_shared<DerivedB>();
  ptr->y = "abc";

  EXPECT_THAT(
      SerializationRoundTrip(BasePtr(ptr), RegistrySerializer<BasePtr>{}),
      ::testing::Optional(
          ::testing::Pointee(::testing::WhenDynamicCastTo<const DerivedB&>(
              ::testing::Field(&DerivedB::y, "abc")))));
}

}  // namespace
