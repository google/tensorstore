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

#include "tensorstore/context.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/context_impl.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/concurrent_testutil.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/std_tuple.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

// Workaround for GoogleMock `::testing::Pointee` requiring a nested
// `element_type` alias, which `Context::Resource<Provider>` can't provide
// without requiring `Provider` to be a complete type, which would prevent
// recursive resource types.
namespace tensorstore {
template <typename Provider>
auto* GetRawPointer(const Context::Resource<Provider>& resource) {
  return resource.get();
}
}  // namespace tensorstore

namespace {

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::Context;
using ::tensorstore::IncludeDefaults;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::internal::ContextResourceCreationContext;
using ::tensorstore::internal::ContextResourceRegistration;
using ::tensorstore::internal::ContextResourceTraits;
using ::tensorstore::internal::ContextSpecBuilder;
using ::tensorstore::internal::TestConcurrent;
using ::tensorstore::serialization::SerializationRoundTrip;

struct IntResource : public ContextResourceTraits<IntResource> {
  struct Spec {
    std::int64_t value;
  };
  using Resource = std::int64_t;
  static constexpr char id[] = "int_resource";
  static Spec Default() { return {42}; }
  static constexpr auto JsonBinder() {
    return jb::Object(jb::Member(
        "value", jb::Projection(&Spec::value,
                                jb::DefaultValue([](auto* v) { *v = 42; }))));
  }
  static Result<Resource> Create(Spec v,
                                 ContextResourceCreationContext context) {
    return v.value;
  }
  static Spec GetSpec(Resource v, const ContextSpecBuilder& builder) {
    return {v};
  }
};

struct StrongRefResource : public ContextResourceTraits<StrongRefResource> {
  struct Spec {
    std::int64_t value;
  };
  struct Resource {
    size_t num_strong_references = 0;
  };
  static constexpr char id[] = "strongref";
  static Spec Default() { return Spec{42}; }
  static constexpr auto JsonBinder() {
    namespace jb = tensorstore::internal_json_binding;
    return jb::Object(jb::Member(
        "value", jb::Projection(&Spec::value, jb::DefaultValue([](auto* obj) {
          *obj = 7;
        }))));
  }
  static Result<Resource> Create(Spec v,
                                 ContextResourceCreationContext context) {
    return Resource{};
  }
  static Spec GetSpec(const Resource& v, const ContextSpecBuilder& builder) {
    return {42};
  }
  static void AcquireContextReference(Resource& v) {
    ++v.num_strong_references;
  }
  static void ReleaseContextReference(Resource& v) {
    --v.num_strong_references;
  }
};

struct OptionalResource : public ContextResourceTraits<OptionalResource> {
  using Spec = std::optional<size_t>;
  using Resource = Spec;
  static constexpr char id[] = "optional_resource";
  static Spec Default() { return {}; }
  static constexpr auto JsonBinder() {
    namespace jb = tensorstore::internal_json_binding;
    return jb::Object(jb::Member(
        "limit", jb::DefaultInitializedValue(jb::Optional(
                     jb::Integer<size_t>(1), [] { return "shared"; }))));
  }
  static Result<Resource> Create(Spec v,
                                 ContextResourceCreationContext context) {
    return v;
  }
  static Spec GetSpec(Resource v, const ContextSpecBuilder& builder) {
    return v;
  }
};

const ContextResourceRegistration<IntResource> int_resource_registration;
const ContextResourceRegistration<StrongRefResource>
    strong_ref_resource_registration;
const ContextResourceRegistration<OptionalResource>
    optional_resource_registration;

TEST(IntResourceTest, InvalidDirectSpec) {
  EXPECT_THAT(Context::Resource<IntResource>::FromJson(nullptr),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected string or object, but received: null"));

  EXPECT_THAT(Context::Resource<IntResource>::FromJson(3),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected string or object, but received: 3"));

  EXPECT_THAT(
      Context::Resource<IntResource>::FromJson("foo"),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Invalid reference to \"int_resource\" resource: \"foo\""));
}

TEST(IntResourceTest, Default) {
  auto context = Context::Default();
  EXPECT_EQ(context, context);
  EXPECT_FALSE(context.parent());
  auto context2 = Context::Default();
  EXPECT_NE(context, context2);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec,
      Context::Resource<IntResource>::FromJson("int_resource"));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource,
                                   context.GetResource(resource_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource2,
                                   context.GetResource(resource_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource3,
                                   context2.GetResource(resource_spec));
  EXPECT_EQ(resource, resource);
  EXPECT_EQ(resource, resource2);
  EXPECT_NE(resource, resource3);
  EXPECT_THAT(context.GetResource<IntResource>(),
              ::testing::Optional(resource));
  EXPECT_THAT(context.GetResource<IntResource>("int_resource"),
              ::testing::Optional(resource));
  EXPECT_THAT(resource, ::testing::Pointee(42));
  EXPECT_THAT(context.GetResource<IntResource>({{"value", 50}}),
              ::testing::Optional(::testing::Pointee(50)));
}

TEST(IntResourceTest, ValidDirectSpec) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec,
      Context::Resource<IntResource>::FromJson({{"value", 7}}));
  EXPECT_THAT(context.GetResource(resource_spec),
              ::testing::Optional(::testing::Pointee(7)));
}

TEST(IntResourceTest, ValidIndirectSpecDefaultId) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, Context::Spec::FromJson({{"int_resource", {{"value", 7}}}}));
  auto context = Context(spec);
  auto resource_spec = Context::Resource<IntResource>::DefaultSpec();
  EXPECT_THAT(context.GetResource(resource_spec),
              ::testing::Optional(::testing::Pointee(7)));
}

TEST(IntResourceTest, ContextFromJson) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context, Context::FromJson({{"int_resource", {{"value", 7}}}}));
  EXPECT_THAT(context.GetResource<IntResource>(),
              ::testing::Optional(::testing::Pointee(7)));
}

TEST(IntResourceTest, ValidIndirectSpecDefault) {
  auto context = Context::Default();
  auto resource_spec = Context::Resource<IntResource>::DefaultSpec();
  EXPECT_THAT(context.GetResource(resource_spec),
              ::testing::Optional(::testing::Pointee(42)));
}

TEST(IntResourceTest, ValidIndirectSpecIdentifier) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, Context::Spec::FromJson({{"int_resource#x", {{"value", 7}}}}));
  auto context = Context(spec);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec,
      Context::Resource<IntResource>::FromJson("int_resource#x"));
  EXPECT_THAT(context.GetResource(resource_spec),
              ::testing::Optional(::testing::Pointee(7)));
}

TEST(IntResourceTest, UndefinedIndirectReference) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec,
      Context::Resource<IntResource>::FromJson("int_resource#x"));
  EXPECT_THAT(context.GetResource(resource_spec),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Resource not defined: \"int_resource#x\""));
}

TEST(IntResourceTest, SimpleReference) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   Context::Spec::FromJson({
                                       {"int_resource#x", {{"value", 7}}},
                                       {"int_resource#y", "int_resource#x"},
                                   }));
  auto context = Context(spec);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec,
      Context::Resource<IntResource>::FromJson("int_resource#y"));
  EXPECT_THAT(context.GetResource(resource_spec),
              ::testing::Optional(::testing::Pointee(7)));
}

TEST(IntResourceTest, ReferenceCycle1) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, Context::Spec::FromJson({{"int_resource", "int_resource"}}));
  auto context = Context(spec);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec,
      Context::Resource<IntResource>::FromJson("int_resource"));
  EXPECT_THAT(context.GetResource(resource_spec),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Context resource reference cycle: "
                            "\"int_resource\":\"int_resource\""));
}

TEST(IntResourceTest, ReferenceCycle2) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   Context::Spec::FromJson({
                                       {"int_resource#a", "int_resource#b"},
                                       {"int_resource#b", "int_resource#a"},
                                   }));
  auto context = Context(spec);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec,
      Context::Resource<IntResource>::FromJson("int_resource#a"));
  EXPECT_THAT(context.GetResource(resource_spec),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Context resource reference cycle: "
                            "\"int_resource#b\":\"int_resource#a\" -> "
                            "\"int_resource#a\":\"int_resource#b\""));
}

TEST(IntResourceTest, Inherit) {
  const ::nlohmann::json json_spec1{
      {"int_resource", {{"value", 7}}},
      {"int_resource#a", {{"value", 9}}},
      {"int_resource#d", {{"value", 42}}},
      {"int_resource#c", nullptr},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec1,
                                   Context::Spec::FromJson(json_spec1));
  EXPECT_THAT(spec1.ToJson(IncludeDefaults{false}),
              ::testing::Optional(MatchesJson({
                  {"int_resource", {{"value", 7}}},
                  {"int_resource#a", {{"value", 9}}},
                  {"int_resource#d", ::nlohmann::json::object_t{}},
                  {"int_resource#c", nullptr},
              })));
  EXPECT_THAT(spec1.ToJson(IncludeDefaults{true}),
              ::testing::Optional(MatchesJson({
                  {"int_resource", {{"value", 7}}},
                  {"int_resource#a", {{"value", 9}}},
                  {"int_resource#d", {{"value", 42}}},
                  {"int_resource#c", nullptr},
              })));
  ::nlohmann::json json_spec2{
      {"int_resource", {{"value", 8}}},
      {"int_resource#b", nullptr},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec2,
                                   Context::Spec::FromJson(json_spec2));
  auto context1 = Context(spec1);
  auto context2 = Context(spec2, context1);
  EXPECT_EQ(context1, context2.parent());
  EXPECT_THAT(context1.spec().ToJson(IncludeDefaults{true}),
              ::testing::Optional(MatchesJson(json_spec1)));
  EXPECT_THAT(context2.spec().ToJson(),
              ::testing::Optional(MatchesJson(json_spec2)));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource1,
      context2.GetResource(
          Context::Resource<IntResource>::FromJson("int_resource").value()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource2,
      context2.GetResource(
          Context::Resource<IntResource>::FromJson("int_resource#a").value()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource3,
      context2.GetResource(
          Context::Resource<IntResource>::FromJson("int_resource#b").value()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource4,
      context2.GetResource(
          Context::Resource<IntResource>::FromJson("int_resource#c").value()));
  EXPECT_EQ(8, *resource1);
  EXPECT_EQ(9, *resource2);
  EXPECT_EQ(7, *resource3);
  EXPECT_EQ(42, *resource4);
}

TEST(IntResourceTest, Unknown) {
  EXPECT_THAT(Context::Spec::FromJson({
                  {"foo", {{"value", 7}}},
              }),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid context resource identifier: \"foo\""));
}

TEST(StrongRefResourceTest, DirectSpec) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec, Context::Resource<StrongRefResource>::FromJson(
                              ::nlohmann::json::object_t{}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource,
                                   context.GetResource(resource_spec));
  // The `StrongRefResource` is held only by a `Context::Resource` handle
  // (`resource`), but not by the `Context` object `context`, since it was
  // specified by a JSON object directly, rather than a string reference.
  // Therefore, there are no strong references.
  EXPECT_EQ(0, resource->num_strong_references);
}

TEST(StrongRefResourceTest, IndirectSpec) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      Context::Spec::FromJson({{"strongref", ::nlohmann::json::object_t{}}}));
  auto context = Context(spec);
  auto resource_spec = Context::Resource<StrongRefResource>::DefaultSpec();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource,
                                   context.GetResource(resource_spec));
  // The `context` object maintains a strong reference to the resource.
  EXPECT_EQ(1, resource->num_strong_references);
  // The `resource` handle remains valid, but the `StrongRefResource` is no
  // longer held in a `Context` object; therefore, there are no strong
  // references.
  context = Context();
  EXPECT_EQ(0, resource->num_strong_references);
}

TEST(ContextSpecBuilderTest, Simple) {
  auto spec =
      Context::Spec::FromJson({{"int_resource", {{"value", 5}}}}).value();
  auto context = Context(spec);
  auto resource_spec = Context::Resource<IntResource>::DefaultSpec();
  auto resource = context.GetResource(resource_spec).value();

  Context::Spec new_spec;
  Context::Resource<IntResource> new_resource_spec;
  {
    auto builder = ContextSpecBuilder::Make();
    new_spec = builder.spec();
    new_resource_spec = builder.AddResource(resource);
  }
  EXPECT_THAT(
      new_spec.ToJson(),
      ::testing::Optional(MatchesJson({{"int_resource", {{"value", 5}}}})));
  EXPECT_THAT(new_resource_spec.ToJson(IncludeDefaults{true}),
              ::testing::Optional(MatchesJson("int_resource")));

  // Test that we can convert back to resources.
  auto new_context = Context(new_spec);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_resource,
                                   new_context.GetResource(new_resource_spec));
  EXPECT_EQ(5, *new_resource);
}

TEST(ContextSpecBuilderTest, Default) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource, Context::Default().GetResource<IntResource>());

  Context::Spec new_spec;
  Context::Resource<IntResource> new_resource_spec;
  {
    auto builder = ContextSpecBuilder::Make();
    new_spec = builder.spec();
    new_resource_spec = builder.AddResource(resource);
  }
  EXPECT_THAT(
      jb::ToJson(new_spec,
                 tensorstore::internal::ContextSpecDefaultableJsonBinder,
                 IncludeDefaults{false}),
      ::testing::Optional(
          MatchesJson({{"int_resource", ::nlohmann::json::object_t()}})));
  EXPECT_THAT(
      new_spec.ToJson(IncludeDefaults{true}),
      ::testing::Optional(MatchesJson({{"int_resource", {{"value", 42}}}})));
  EXPECT_THAT(new_resource_spec.ToJson(IncludeDefaults{true}),
              ::testing::Optional(MatchesJson("int_resource")));

  // Test that we can convert back to resources.
  auto new_context = Context(new_spec);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_resource,
                                   new_context.GetResource(new_resource_spec));
  EXPECT_THAT(new_resource, ::testing::Pointee(42));
}

TEST(ContextSpecBuilderTest, MultipleContexts) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec1, Context::Spec::FromJson({{"int_resource", {{"value", 5}}}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec2, Context::Spec::FromJson({{"int_resource", {{"value", 6}}}}));
  auto context1 = Context(spec1);
  auto context2 = Context(spec2);
  auto resource_spec = Context::Resource<IntResource>::DefaultSpec();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource1,
                                   context1.GetResource(resource_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource2,
                                   context2.GetResource(resource_spec));

  Context::Spec new_spec;
  Context::Resource<IntResource> new_resource_spec1;
  Context::Resource<IntResource> new_resource_spec2;
  {
    auto builder = ContextSpecBuilder::Make();
    new_spec = builder.spec();
    new_resource_spec1 = builder.AddResource(resource1);
    new_resource_spec2 = builder.AddResource(resource2);
  }
  EXPECT_THAT(new_spec.ToJson(), ::testing::Optional(MatchesJson({
                                     {"int_resource#0", {{"value", 5}}},
                                     {"int_resource#1", {{"value", 6}}},
                                 })));
  EXPECT_EQ("int_resource#0", new_resource_spec1.ToJson());
  EXPECT_EQ("int_resource#1", new_resource_spec2.ToJson());
}

TEST(ContextSpecBuilderTest, Inline) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec,
      Context::Resource<IntResource>::FromJson({{"value", 5}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource,
                                   context.GetResource(resource_spec));

  Context::Spec new_spec;
  Context::Resource<IntResource> new_resource_spec;
  {
    auto builder = ContextSpecBuilder::Make();
    new_spec = builder.spec();
    new_resource_spec = builder.AddResource(resource);
  }
  EXPECT_THAT(new_spec.ToJson(),
              ::testing::Optional(MatchesJson(::nlohmann::json::object_t())));
  EXPECT_THAT(new_resource_spec.ToJson(),
              ::testing::Optional(MatchesJson({{"value", 5}})));
}

TEST(ContextSpecBuilderTest, InlineEqualToDefault) {
  auto context = Context::Default();
  auto resource_spec =
      Context::Resource<IntResource>::FromJson({{"value", 42}}).value();
  auto resource = context.GetResource(resource_spec).value();

  Context::Spec new_spec;
  Context::Resource<IntResource> new_resource_spec;
  {
    auto builder = ContextSpecBuilder::Make();
    new_spec = builder.spec();
    new_resource_spec = builder.AddResource(resource);
  }
  EXPECT_EQ(::nlohmann::json({}), new_spec.ToJson());
  EXPECT_EQ(::nlohmann::json::object_t{},
            new_resource_spec.ToJson(IncludeDefaults{false}));
}

TEST(ContextSpecBuilderTest, InlineShared) {
  auto context = Context::Default();
  auto resource_spec =
      Context::Resource<IntResource>::FromJson({{"value", 5}}).value();
  auto resource = context.GetResource(resource_spec).value();

  Context::Spec new_spec;
  Context::Resource<IntResource> new_resource_spec1;
  Context::Resource<IntResource> new_resource_spec2;
  {
    auto builder = ContextSpecBuilder::Make();
    new_spec = builder.spec();
    new_resource_spec1 = builder.AddResource(resource);
    new_resource_spec2 = builder.AddResource(resource);
  }
  EXPECT_EQ(::nlohmann::json({{"int_resource#0", {{"value", 5}}}}),
            new_spec.ToJson());
  EXPECT_EQ("int_resource#0", new_resource_spec1.ToJson());
  EXPECT_EQ("int_resource#0", new_resource_spec2.ToJson());
}

TEST(ContextSpecBuilderTest, ExcludeDefaultsJson) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context, Context::FromJson({
                        {"optional_resource", {{"limit", "shared"}}},
                        {"optional_resource#a", {{"limit", 5}}},
                    }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource1,
                                   context.GetResource<OptionalResource>());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource2,
      context.GetResource<OptionalResource>("optional_resource#a"));
  Context::Spec new_spec;
  Context::Resource<OptionalResource> new_resource_spec1;
  Context::Resource<OptionalResource> new_resource_spec2;
  {
    auto builder = ContextSpecBuilder::Make();
    new_spec = builder.spec();
    new_resource_spec1 = builder.AddResource(resource1);
    new_resource_spec2 = builder.AddResource(resource2);
  }
  EXPECT_THAT(new_spec.ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(MatchesJson({
                  {"optional_resource#a", {{"limit", 5}}},
                  {"optional_resource", ::nlohmann::json::object_t()},
              })));
  EXPECT_THAT(new_spec.ToJson(tensorstore::IncludeDefaults{true}),
              ::testing::Optional(MatchesJson({
                  {"optional_resource#a", {{"limit", 5}}},
                  {"optional_resource", {{"limit", "shared"}}},
              })));
}

TEST(ContextTest, WeakCreator) {
  using ::tensorstore::internal_context::Access;
  using ::tensorstore::internal_context::GetCreator;
  using ::tensorstore::internal_context::ResourceImplBase;

  const ::nlohmann::json json_spec1{
      {"int_resource", {{"value", 7}}},
      {"int_resource#a", {{"value", 9}}},
      {"int_resource#d", {{"value", 42}}},
      {"int_resource#c", nullptr},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec1,
                                   Context::Spec::FromJson(json_spec1));
  ::nlohmann::json json_spec2{
      {"int_resource", {{"value", 8}}},
      {"int_resource#b", nullptr},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec2,
                                   Context::Spec::FromJson(json_spec2));
  auto context1 = Context(spec1);
  auto context2 = Context(spec2, context1);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource1,
                                   context1.GetResource<IntResource>());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource2,
                                   context2.GetResource<IntResource>());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource2_a, context2.GetResource<IntResource>("int_resource#a"));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource2_b, context2.GetResource<IntResource>("int_resource#b"));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource2_c, context2.GetResource<IntResource>("int_resource#c"));
  EXPECT_EQ(
      Access::impl(context1),
      GetCreator(static_cast<ResourceImplBase&>(*Access::impl(resource1))));
  EXPECT_EQ(
      Access::impl(context1),
      GetCreator(static_cast<ResourceImplBase&>(*Access::impl(resource2_a))));
  EXPECT_EQ(
      Access::impl(context1),
      GetCreator(static_cast<ResourceImplBase&>(*Access::impl(resource2_b))));
  EXPECT_EQ(
      Access::impl(context1),
      GetCreator(static_cast<ResourceImplBase&>(*Access::impl(resource2_c))));
  EXPECT_EQ(
      Access::impl(context2),
      GetCreator(static_cast<ResourceImplBase&>(*Access::impl(resource2))));
  context2 = Context();
  EXPECT_EQ(
      Access::impl(context1),
      GetCreator(static_cast<ResourceImplBase&>(*Access::impl(resource1))));
  EXPECT_FALSE(
      GetCreator(static_cast<ResourceImplBase&>(*Access::impl(resource2))));
}

/// Resource provider used for testing context resources that depend on other
/// context resources.
///
/// In this case we are also testing a recursive context resource, which
/// requires that `Context::Resource` can be instantiated with an incomplete
/// `Provider` type.
struct NestedResource : public ContextResourceTraits<NestedResource> {
  struct Spec {
    int value;
    Context::Resource<NestedResource> parent;
    int GetTotal() const {
      int total = value;
      if (parent.has_resource()) total += parent->GetTotal();
      return total;
    }
  };
  using Resource = Spec;
  static constexpr char id[] = "nested_resource";
  static Spec Default() { return {42}; }
  static constexpr auto JsonBinder() {
    return jb::Object(
        jb::Member("value",
                   jb::Projection(&Spec::value,
                                  jb::DefaultValue([](auto* v) { *v = 42; }))),
        jb::Member(
            "parent",
            jb::Projection(
                &Spec::parent,
                jb::DefaultInitializedPredicate<jb::kNeverIncludeDefaults>(
                    [](auto* obj) { return !obj->valid(); }))));
  }

  static Result<Resource> Create(const Spec& spec,
                                 ContextResourceCreationContext context) {
    Resource resource = spec;
    TENSORSTORE_RETURN_IF_ERROR(resource.parent.BindContext(context));
    return resource;
  }

  static Spec GetSpec(const Resource& resource,
                      const ContextSpecBuilder& builder) {
    Spec spec = resource;
    UnbindContext(spec, builder);
    return spec;
  }

  static void UnbindContext(Spec& spec, const ContextSpecBuilder& builder) {
    spec.parent.UnbindContext(builder);
  }
};

const ContextResourceRegistration<NestedResource> nested_resource_registration;

TEST(NestedResourceTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context, Context::FromJson({
                        {"nested_resource#a", {{"value", 1}}},
                        {"nested_resource#b",
                         {{"value", 3}, {"parent", "nested_resource#a"}}},
                        {"nested_resource#c",
                         {{"value", 5}, {"parent", "nested_resource#b"}}},
                        {"nested_resource#d",
                         {{"value", 10}, {"parent", "nested_resource#e"}}},
                        {"nested_resource#e",
                         {{"value", 15}, {"parent", "nested_resource#d"}}},
                    }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto a, context.GetResource<NestedResource>("nested_resource#a"));
  EXPECT_FALSE(a->parent.valid());
  EXPECT_EQ(1, a->GetTotal());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto b, context.GetResource<NestedResource>("nested_resource#b"));
  EXPECT_EQ(a, b->parent);
  EXPECT_EQ(4, b->GetTotal());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto c, context.GetResource<NestedResource>("nested_resource#c"));
  EXPECT_EQ(b, c->parent);
  EXPECT_EQ(9, c->GetTotal());

  EXPECT_THAT(
      context.GetResource<NestedResource>("nested_resource#d"),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Context resource reference cycle: "
                    "\"nested_resource#d\" -> "
                    "\"nested_resource#d\":"
                    "\\{\"parent\":\"nested_resource#e\",\"value\":10\\} -> "
                    "\"nested_resource#e\" -> "
                    "\"nested_resource#e\":"
                    "\\{\"parent\":\"nested_resource#d\",\"value\":15\\}"));
  EXPECT_THAT(context.GetResource<NestedResource>("nested_resource#e"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Context resource reference cycle: .*"));
}

// Tests unbinding a collection of context resources, where some are bound and
// some are not, to ensure that the ids chosen in the resultant `Context::Spec`
// are all unique.
TEST(ContextSpecBuilderTest, PartiallyBound) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context_spec, Context::Spec::FromJson({
                             {"nested_resource#a", {{"value", 2}}},
                             {"nested_resource#b",
                              {{"value", 3}, {"parent", "nested_resource#a"}}},
                         }));
  auto context = Context(context_spec);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec,
      Context::Resource<NestedResource>::FromJson("nested_resource#b"));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource,
                                   context.GetResource(resource_spec));

  Context::Spec new_spec;
  Context::Resource<NestedResource> new_resource_spec1;
  Context::Resource<NestedResource> new_resource_spec2;
  {
    auto builder = ContextSpecBuilder::Make({}, context_spec);
    new_spec = builder.spec();
    new_resource_spec1 = builder.AddResource(resource_spec);
    new_resource_spec2 = builder.AddResource(resource);
  }

  // If unbound independently, `new_resource_spec2` would use the identifier
  // "nested_resource#b".  However, since `new_resource_spec1` was registered
  // with `builder`, a unique identifier is chosen.
  EXPECT_THAT(new_spec.ToJson(),
              ::testing::Optional(MatchesJson({
                  {"nested_resource#a", {{"value", 2}}},
                  {"nested_resource#b",
                   {{"value", 3}, {"parent", "nested_resource#a"}}},
                  {"nested_resource#1", {{"value", 2}}},
                  {"nested_resource#0",
                   {{"value", 3}, {"parent", "nested_resource#1"}}},
              })));
  EXPECT_THAT(new_resource_spec1.ToJson(),
              ::testing::Optional(MatchesJson("nested_resource#b")));
  EXPECT_THAT(new_resource_spec2.ToJson(),
              ::testing::Optional(MatchesJson("nested_resource#0")));
}

TEST(ContextSpecSerializationTest, Empty) {
  // Test empty spec
  Context::Spec spec;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_copy,
                                   SerializationRoundTrip(spec));
  EXPECT_THAT(spec.ToJson(),
              ::testing::Optional(MatchesJson(::nlohmann::json::object_t())));
}

TEST(ContextSpecSerializationTest, NonEmpty) {
  ::nlohmann::json json_spec{
      {"int_resource", ::nlohmann::json::object_t()},
      {"int_resource#a", "int_resource"},
      {"int_resource#b", ::nlohmann::json::object_t()},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   Context::Spec::FromJson(json_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_copy,
                                   SerializationRoundTrip(spec));
  EXPECT_THAT(spec.ToJson(), ::testing::Optional(MatchesJson(json_spec)));
}

TEST(ContextSerializationTest, Null) {
  Context context;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto context_copy,
                                   SerializationRoundTrip(context));
  EXPECT_FALSE(context);
}

TEST(ContextSerializationTest, NonNull) {
  ::nlohmann::json parent_json_spec{
      {"int_resource#c", ::nlohmann::json::object_t()},
  };
  ::nlohmann::json child_json_spec{
      {"int_resource", ::nlohmann::json::object_t()},
      {"int_resource#a", "int_resource"},
      {"int_resource#b", ::nlohmann::json::object_t()},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto parent_spec,
                                   Context::Spec::FromJson(parent_json_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto child_spec,
                                   Context::Spec::FromJson(child_json_spec));
  Context parent_context(parent_spec);
  Context child_context(child_spec, parent_context);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto child_context_copy,
                                   SerializationRoundTrip(child_context));
  EXPECT_THAT(child_context_copy.spec().ToJson(),
              ::testing::Optional(child_json_spec));
  EXPECT_THAT(child_context_copy.parent().spec().ToJson(),
              ::testing::Optional(parent_json_spec));
  EXPECT_FALSE(child_context_copy.parent().parent());
}

TEST(ContextSerializationTest, Shared) {
  ::nlohmann::json parent_json_spec{
      {"int_resource#c", {{"value", 7}}},
  };
  ::nlohmann::json child_json_spec{
      {"int_resource", {{"value", 5}}},
      {"int_resource#a", "int_resource"},
      {"int_resource#b", {{"value", 6}}},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto parent_spec,
                                   Context::Spec::FromJson(parent_json_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto child_spec,
                                   Context::Spec::FromJson(child_json_spec));
  Context parent_context(parent_spec);
  Context child_context(child_spec, parent_context);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto res_parent,
                                   parent_context.GetResource<IntResource>());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto res_child,
                                   child_context.GetResource<IntResource>());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto res_a, child_context.GetResource<IntResource>("int_resource#a"));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto res_b, child_context.GetResource<IntResource>("int_resource#b"));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto res_c_child,
      child_context.GetResource<IntResource>("int_resource#c"));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto res_c_parent,
      parent_context.GetResource<IntResource>("int_resource#c"));
  EXPECT_EQ(res_child, res_a);
  EXPECT_EQ(res_c_child, res_c_parent);
  EXPECT_NE(res_child, res_parent);
  EXPECT_NE(res_a, res_b);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto copy, SerializationRoundTrip(std::make_tuple(
                     parent_context, child_context, res_parent, res_child,
                     res_a, res_b, res_c_child, res_c_parent)));
  auto [copy_parent_context, copy_child_context, copy_res_parent,
        copy_res_child, copy_res_a, copy_res_b, copy_res_c_child,
        copy_res_c_parent] = copy;
  EXPECT_EQ(copy_parent_context, copy_child_context.parent());
  EXPECT_THAT(copy_child_context.GetResource<IntResource>("int_resource#a"),
              ::testing::Optional(copy_res_a));
  EXPECT_THAT(copy_child_context.GetResource<IntResource>("int_resource#b"),
              ::testing::Optional(copy_res_b));
  EXPECT_THAT(copy_child_context.GetResource<IntResource>("int_resource#c"),
              ::testing::Optional(copy_res_c_child));
  EXPECT_THAT(copy_parent_context.GetResource<IntResource>("int_resource#c"),
              ::testing::Optional(copy_res_c_parent));
}

TEST(ContextTest, ConcurrentCreateSingleResource) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, Context::Spec::FromJson({{"int_resource", {{"value", 5}}}}));
  Context context;
  TestConcurrent<3>(
      /*num_iterations=*/100,
      /*initialize=*/[&] { context = Context(spec); },
      /*finalize=*/[&] {},
      [&](auto i) {
        TENSORSTORE_EXPECT_OK(context.GetResource<IntResource>());
      });
}

TEST(ContextTest, ConcurrentCreateMultipleResources) {
  std::vector<std::string> resource_keys{"int_resource#a", "int_resource#b"};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   Context::Spec::FromJson({
                                       {resource_keys[0], {{"value", 5}}},
                                       {resource_keys[1], {{"value", 6}}},
                                   }));
  Context context;
  TestConcurrent<4>(
      /*num_iterations=*/100,
      /*initialize=*/[&] { context = Context(spec); },
      /*finalize=*/[&] {},
      [&](auto i) {
        TENSORSTORE_EXPECT_OK(context.GetResource<IntResource>(
            resource_keys[i % resource_keys.size()]));
      });
}

TEST(ContextTest, ConcurrentCreateInParent) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, Context::Spec::FromJson({{"int_resource", {{"value", 5}}}}));
  Context context;
  TestConcurrent<3>(
      /*num_iterations=*/100,
      /*initialize=*/[&] { context = Context(spec); },
      /*finalize=*/[&] {},
      [&](auto i) {
        Context child({}, context);
        TENSORSTORE_EXPECT_OK(child.GetResource<IntResource>());
      });
}

}  // namespace
