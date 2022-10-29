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

#include "tensorstore/internal/json_registry.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::JsonRegistry;

class MyInterface
    : public tensorstore::internal::AtomicReferenceCount<MyInterface> {
 public:
  virtual int Whatever() const = 0;
  virtual ~MyInterface() = default;
};

class MyInterfacePtr : public IntrusivePtr<MyInterface> {
 public:
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(MyInterfacePtr,
                                          tensorstore::JsonSerializationOptions,
                                          tensorstore::JsonSerializationOptions)
};

using Registry =
    JsonRegistry<MyInterface, tensorstore::JsonSerializationOptions,
                 tensorstore::JsonSerializationOptions>;

Registry& GetRegistry() {
  static tensorstore::internal::NoDestructor<Registry> registry;
  return *registry;
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(MyInterfacePtr, [](auto is_loading,
                                                          const auto& options,
                                                          auto* obj,
                                                          ::nlohmann::json* j) {
  namespace jb = tensorstore::internal_json_binding;
  return jb::Object(GetRegistry().MemberBinder("id"))(is_loading, options, obj,
                                                      j);
})

class FooImpl : public MyInterface {
 public:
  int x;
  int Whatever() const override { return x; }
};

class BarImpl : public MyInterface {
 public:
  float y;
  int Whatever() const override { return static_cast<int>(y); }
};

struct FooRegistration {
  FooRegistration() {
    namespace jb = tensorstore::internal_json_binding;
    GetRegistry().Register<FooImpl>(
        "foo", jb::Object(jb::Member("x", jb::Projection(&FooImpl::x))));
  }
} foo_registration;

struct BarRegistration {
  BarRegistration() {
    namespace jb = tensorstore::internal_json_binding;
    GetRegistry().Register<BarImpl>(
        "bar", jb::Object(jb::Member("y", jb::Projection(&BarImpl::y))));
  }
} bar_registration;

TEST(RegistryTest, Foo) {
  const ::nlohmann::json j{{"id", "foo"}, {"x", 10}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto obj, MyInterfacePtr::FromJson(j));
  EXPECT_EQ(10, obj->Whatever());
  EXPECT_EQ(j, obj.ToJson());
}

TEST(RegistryTest, Bar) {
  const ::nlohmann::json j{{"id", "bar"}, {"y", 42.5}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto obj, MyInterfacePtr::FromJson(j));
  EXPECT_EQ(42, obj->Whatever());
  EXPECT_EQ(j, obj.ToJson());
}

TEST(RegistryTest, Unknown) {
  EXPECT_THAT(MyInterfacePtr::FromJson({{"id", "baz"}, {"y", 42.5}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"id\": "
                            "\"baz\" is not registered"));
}

}  // namespace
