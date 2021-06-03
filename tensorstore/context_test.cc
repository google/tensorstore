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
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {
using tensorstore::Context;
using tensorstore::IncludeDefaults;
using tensorstore::MatchesJson;
using tensorstore::MatchesStatus;
using tensorstore::Result;
using tensorstore::internal::ContextResourceCreationContext;
using tensorstore::internal::ContextResourceRegistration;
using tensorstore::internal::ContextResourceTraits;
using tensorstore::internal::ContextSpecBuilder;
namespace jb = tensorstore::internal_json_binding;

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
  static void AcquireContextReference(Resource* v) {
    ++v->num_strong_references;
  }
  static void ReleaseContextReference(Resource* v) {
    --v->num_strong_references;
  }
};

struct OptionalResource : public ContextResourceTraits<OptionalResource> {
  using Spec = std::optional<size_t>;
  using Resource = Spec;
  static constexpr char id[] = "optional_resource";
  static Spec Default() { return {}; }
  static constexpr auto JsonBinder() {
    namespace jb = tensorstore::internal_json_binding;
    return jb::DefaultInitializedValue(jb::Object(jb::Member(
        "limit", jb::DefaultInitializedValue(jb::Optional(
                     jb::Integer<size_t>(1), [] { return "shared"; })))));
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
  EXPECT_THAT(Context::ResourceSpec<IntResource>::FromJson(nullptr),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected string or object, but received: null"));

  EXPECT_THAT(Context::ResourceSpec<IntResource>::FromJson(3),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected string or object, but received: 3"));

  EXPECT_THAT(
      Context::ResourceSpec<IntResource>::FromJson("foo"),
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
      Context::ResourceSpec<IntResource>::FromJson("int_resource"));
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
      Context::ResourceSpec<IntResource>::FromJson({{"value", 7}}));
  EXPECT_THAT(context.GetResource(resource_spec),
              ::testing::Optional(::testing::Pointee(7)));
}

TEST(IntResourceTest, ValidIndirectSpecDefaultId) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, Context::Spec::FromJson({{"int_resource", {{"value", 7}}}}));
  auto context = Context(spec);
  auto resource_spec = Context::ResourceSpec<IntResource>::Default();
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
  auto resource_spec = Context::ResourceSpec<IntResource>::Default();
  EXPECT_THAT(context.GetResource(resource_spec),
              ::testing::Optional(::testing::Pointee(42)));
}

TEST(IntResourceTest, ValidIndirectSpecIdentifier) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, Context::Spec::FromJson({{"int_resource#x", {{"value", 7}}}}));
  auto context = Context(spec);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec,
      Context::ResourceSpec<IntResource>::FromJson("int_resource#x"));
  EXPECT_THAT(context.GetResource(resource_spec),
              ::testing::Optional(::testing::Pointee(7)));
}

TEST(IntResourceTest, UndefinedIndirectReference) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec,
      Context::ResourceSpec<IntResource>::FromJson("int_resource#x"));
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
      Context::ResourceSpec<IntResource>::FromJson("int_resource#y"));
  EXPECT_THAT(context.GetResource(resource_spec),
              ::testing::Optional(::testing::Pointee(7)));
}

TEST(IntResourceTest, ReferenceCycle1) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, Context::Spec::FromJson({{"int_resource", "int_resource"}}));
  auto context = Context(spec);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec,
      Context::ResourceSpec<IntResource>::FromJson("int_resource"));
  EXPECT_THAT(
      context.GetResource(resource_spec),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Context resource reference cycle: \"int_resource\""));
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
      Context::ResourceSpec<IntResource>::FromJson("int_resource#a"));
  EXPECT_THAT(context.GetResource(resource_spec),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Context resource reference cycle: "
                            "\"int_resource#a\" -> \"int_resource#b\""));
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
          Context::ResourceSpec<IntResource>::FromJson("int_resource")
              .value()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource2,
      context2.GetResource(
          Context::ResourceSpec<IntResource>::FromJson("int_resource#a")
              .value()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource3,
      context2.GetResource(
          Context::ResourceSpec<IntResource>::FromJson("int_resource#b")
              .value()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource4,
      context2.GetResource(
          Context::ResourceSpec<IntResource>::FromJson("int_resource#c")
              .value()));
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
      auto resource_spec, Context::ResourceSpec<StrongRefResource>::FromJson(
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
  auto resource_spec = Context::ResourceSpec<StrongRefResource>::Default();
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
  auto resource_spec = Context::ResourceSpec<IntResource>::Default();
  auto resource = context.GetResource(resource_spec).value();

  Context::Spec new_spec;
  Context::ResourceSpec<IntResource> new_resource_spec;
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
  auto context = Context::Default();
  auto resource_spec = Context::ResourceSpec<IntResource>::Default();
  auto resource = context.GetResource(resource_spec).value();

  Context::Spec new_spec;
  Context::ResourceSpec<IntResource> new_resource_spec;
  {
    auto builder = ContextSpecBuilder::Make();
    new_spec = builder.spec();
    new_resource_spec = builder.AddResource(resource);
  }
  EXPECT_THAT(
      jb::ToJson(new_spec,
                 tensorstore::internal::ContextSpecDefaultableJsonBinder,
                 IncludeDefaults{false}),
      ::testing::Optional(MatchesJson(::nlohmann::json::value_t::discarded)));
  EXPECT_THAT(
      new_spec.ToJson(IncludeDefaults{true}),
      ::testing::Optional(MatchesJson({{"int_resource", {{"value", 42}}}})));
  EXPECT_THAT(new_resource_spec.ToJson(IncludeDefaults{true}),
              ::testing::Optional(MatchesJson("int_resource")));

  // Test that we can convert back to resources.
  auto new_context = Context(new_spec);
  auto new_resource = new_context.GetResource(new_resource_spec).value();
  EXPECT_EQ(42, *new_resource);
}

TEST(ContextSpecBuilderTest, MultipleContexts) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec1, Context::Spec::FromJson({{"int_resource", {{"value", 5}}}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec2, Context::Spec::FromJson({{"int_resource", {{"value", 6}}}}));
  auto context1 = Context(spec1);
  auto context2 = Context(spec2);
  auto resource_spec = Context::ResourceSpec<IntResource>::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource1,
                                   context1.GetResource(resource_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource2,
                                   context2.GetResource(resource_spec));

  Context::Spec new_spec;
  Context::ResourceSpec<IntResource> new_resource_spec1;
  Context::ResourceSpec<IntResource> new_resource_spec2;
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
      Context::ResourceSpec<IntResource>::FromJson({{"value", 5}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource,
                                   context.GetResource(resource_spec));

  Context::Spec new_spec;
  Context::ResourceSpec<IntResource> new_resource_spec;
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
      Context::ResourceSpec<IntResource>::FromJson({{"value", 42}}).value();
  auto resource = context.GetResource(resource_spec).value();

  Context::Spec new_spec;
  Context::ResourceSpec<IntResource> new_resource_spec;
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
      Context::ResourceSpec<IntResource>::FromJson({{"value", 5}}).value();
  auto resource = context.GetResource(resource_spec).value();

  Context::Spec new_spec;
  Context::ResourceSpec<IntResource> new_resource_spec1;
  Context::ResourceSpec<IntResource> new_resource_spec2;
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

// Tests that the JSON never includes `discarded` values.
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
  Context::ResourceSpec<OptionalResource> new_resource_spec1;
  Context::ResourceSpec<OptionalResource> new_resource_spec2;
  {
    auto builder = ContextSpecBuilder::Make();
    new_spec = builder.spec();
    new_resource_spec1 = builder.AddResource(resource1);
    new_resource_spec2 = builder.AddResource(resource2);
  }
  EXPECT_THAT(new_spec.ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(MatchesJson({
                  {"optional_resource#a", {{"limit", 5}}},
              })));
  EXPECT_THAT(new_spec.ToJson(tensorstore::IncludeDefaults{true}),
              ::testing::Optional(MatchesJson({
                  {"optional_resource#a", {{"limit", 5}}},
                  {"optional_resource", {{"limit", "shared"}}},
              })));
}

TEST(ContextTest, WeakCreator) {
  using tensorstore::internal_context::Access;
  using tensorstore::internal_context::GetCreator;

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
  EXPECT_EQ(Access::impl(context1), GetCreator(*Access::impl(resource1)));
  EXPECT_EQ(Access::impl(context1), GetCreator(*Access::impl(resource2_a)));
  EXPECT_EQ(Access::impl(context1), GetCreator(*Access::impl(resource2_b)));
  EXPECT_EQ(Access::impl(context1), GetCreator(*Access::impl(resource2_c)));
  EXPECT_EQ(Access::impl(context2), GetCreator(*Access::impl(resource2)));
  context2 = Context();
  EXPECT_EQ(Access::impl(context1), GetCreator(*Access::impl(resource1)));
  EXPECT_FALSE(GetCreator(*Access::impl(resource2)));
}

}  // namespace
