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

#include "tensorstore/util/garbage_collection/std_variant.h"  // IWYU pragma: keep

#include <string>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"

namespace {

using ::tensorstore::garbage_collection::GarbageCollectionVisit;
using ::tensorstore::garbage_collection::GarbageCollectionVisitor;
using ::tensorstore::garbage_collection::IsGarbageCollectionRequired;

struct CustomGcType {
  int id = 0;
  ~CustomGcType() {}
};

}  // namespace

template <>
struct tensorstore::garbage_collection::GarbageCollection<CustomGcType> {
  constexpr static bool required() { return true; }
  static void Visit(GarbageCollectionVisitor& visitor,
                    const CustomGcType& value) {
    visitor.Indirect(value);
  }
};

namespace {

class TestVisitor final : public GarbageCollectionVisitor {
 public:
  std::vector<const void*> visited;
  void DoIndirect(const std::type_info& type, ErasedVisitFunction visit,
                  const void* ptr) override {
    visited.push_back(ptr);
  }
};

TEST(StdVariantTest, Required) {
  // Trivially destructible variant does not require GC.
  static_assert(!IsGarbageCollectionRequired<std::variant<int, float>>);

  // Non-trivially destructible variant whose members do not require GC.
  static_assert(
      !IsGarbageCollectionRequired<std::variant<std::string, absl::Cord>>);

  // Variant containing a member that requires GC.
  static_assert(IsGarbageCollectionRequired<std::variant<int, CustomGcType>>);
  static_assert(
      IsGarbageCollectionRequired<std::variant<std::string, CustomGcType>>);
}

TEST(StdVariantTest, Visit) {
  TestVisitor visitor;
  using V = std::variant<int, CustomGcType>;

  // When active alternative does not require GC:
  V v1(10);
  GarbageCollectionVisit(visitor, v1);
  EXPECT_TRUE(visitor.visited.empty());

  // When active alternative requires GC:
  V v2(std::in_place_type<CustomGcType>, CustomGcType{42});
  GarbageCollectionVisit(visitor, v2);
  EXPECT_THAT(visitor.visited,
              ::testing::ElementsAre(&std::get<CustomGcType>(v2)));
}

}  // namespace
