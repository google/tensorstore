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
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {
using tensorstore::AllowUnregistered;
using tensorstore::Context;
using tensorstore::IncludeDefaults;
using tensorstore::MatchesStatus;
using tensorstore::Result;
using tensorstore::internal::ContextResourceCreationContext;
using tensorstore::internal::ContextResourceRegistration;
using tensorstore::internal::ContextResourceTraits;
using tensorstore::internal::ContextSpecBuilder;

struct IntResource : public ContextResourceTraits<IntResource> {
  struct Spec {
    std::int64_t value;
  };
  using Resource = std::int64_t;
  static constexpr char id[] = "int_resource";
  static Spec Default() { return {42}; }
  static constexpr auto JsonBinder() {
    namespace jb = tensorstore::internal::json_binding;
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
    namespace jb = tensorstore::internal::json_binding;
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

const ContextResourceRegistration<IntResource> int_resource_registration;
const ContextResourceRegistration<StrongRefResource>
    strong_ref_resource_registration;

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
  auto resource_spec =
      Context::ResourceSpec<IntResource>::FromJson("int_resource");
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource_spec));
  auto resource = context.GetResource(*resource_spec);
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource));
  EXPECT_EQ(42, **resource);
}

TEST(IntResourceTest, ValidDirectSpec) {
  auto context = Context::Default();
  auto resource_spec =
      Context::ResourceSpec<IntResource>::FromJson({{"value", 7}});
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource_spec));
  auto resource = context.GetResource(*resource_spec);
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource));
  EXPECT_EQ(7, **resource);
}

TEST(IntResourceTest, ValidIndirectSpecDefaultId) {
  auto spec_result =
      Context::Spec::FromJson({{"int_resource", {{"value", 7}}}});
  ASSERT_EQ(absl::OkStatus(), GetStatus(spec_result));
  auto context = Context(*spec_result);
  auto resource_spec = Context::ResourceSpec<IntResource>::Default();
  auto resource = context.GetResource(resource_spec);
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource));
  EXPECT_EQ(7, **resource);
}

TEST(IntResourceTest, ValidIndirectSpecDefault) {
  auto context = Context::Default();
  auto resource_spec = Context::ResourceSpec<IntResource>::Default();
  auto resource = context.GetResource(resource_spec);
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource));
  EXPECT_EQ(42, **resource);
}

TEST(IntResourceTest, ValidIndirectSpecIdentifier) {
  auto spec_result =
      Context::Spec::FromJson({{"int_resource#x", {{"value", 7}}}});
  ASSERT_EQ(absl::OkStatus(), GetStatus(spec_result));
  auto context = Context(*spec_result);
  auto resource_spec =
      Context::ResourceSpec<IntResource>::FromJson("int_resource#x");
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource_spec));
  auto resource = context.GetResource(*resource_spec);
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource));
  EXPECT_EQ(7, **resource);
}

TEST(IntResourceTest, UndefinedIndirectReference) {
  auto context = Context::Default();
  auto resource_spec =
      Context::ResourceSpec<IntResource>::FromJson("int_resource#x");
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource_spec));
  EXPECT_THAT(context.GetResource(*resource_spec),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Resource not defined: \"int_resource#x\""));
}

TEST(IntResourceTest, SimpleReference) {
  auto spec_result = Context::Spec::FromJson({
      {"int_resource#x", {{"value", 7}}},
      {"int_resource#y", "int_resource#x"},
  });
  ASSERT_EQ(absl::OkStatus(), GetStatus(spec_result));
  auto context = Context(*spec_result);
  auto resource_spec =
      Context::ResourceSpec<IntResource>::FromJson("int_resource#y");
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource_spec));
  auto resource = context.GetResource(*resource_spec);
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource));
  EXPECT_EQ(7, **resource);
}

TEST(IntResourceTest, ReferenceCycle1) {
  auto spec_result =
      Context::Spec::FromJson({{"int_resource", "int_resource"}});
  ASSERT_EQ(absl::OkStatus(), GetStatus(spec_result));
  auto context = Context(*spec_result);
  auto resource_spec =
      Context::ResourceSpec<IntResource>::FromJson("int_resource");
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource_spec));
  EXPECT_THAT(
      context.GetResource(*resource_spec),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Context resource reference cycle: \"int_resource\""));
}

TEST(IntResourceTest, ReferenceCycle2) {
  auto spec_result = Context::Spec::FromJson({
      {"int_resource#a", "int_resource#b"},
      {"int_resource#b", "int_resource#a"},
  });
  ASSERT_EQ(absl::OkStatus(), GetStatus(spec_result));
  auto context = Context(*spec_result);
  auto resource_spec =
      Context::ResourceSpec<IntResource>::FromJson("int_resource#a");
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource_spec));
  EXPECT_THAT(context.GetResource(*resource_spec),
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
  auto spec1 = Context::Spec::FromJson(json_spec1).value();
  EXPECT_EQ(::nlohmann::json({
                {"int_resource", {{"value", 7}}},
                {"int_resource#a", {{"value", 9}}},
                {"int_resource#d", ::nlohmann::json::object_t{}},
                {"int_resource#c", nullptr},
            }),
            ::nlohmann::json(spec1.ToJson(IncludeDefaults{false}).value()));
  EXPECT_EQ(::nlohmann::json({
                {"int_resource", {{"value", 7}}},
                {"int_resource#a", {{"value", 9}}},
                {"int_resource#d", {{"value", 42}}},
                {"int_resource#c", nullptr},
            }),
            ::nlohmann::json(spec1.ToJson(IncludeDefaults{true}).value()));
  auto spec2 = Context::Spec::FromJson({
                                           {"int_resource", {{"value", 8}}},
                                           {"int_resource#b", nullptr},
                                       })
                   .value();
  auto context1 = Context(spec1);
  auto context2 = Context(spec2, context1);
  auto resource1 = context2.GetResource(
      Context::ResourceSpec<IntResource>::FromJson("int_resource").value());
  auto resource2 = context2.GetResource(
      Context::ResourceSpec<IntResource>::FromJson("int_resource#a").value());
  auto resource3 = context2.GetResource(
      Context::ResourceSpec<IntResource>::FromJson("int_resource#b").value());
  auto resource4 = context2.GetResource(
      Context::ResourceSpec<IntResource>::FromJson("int_resource#c").value());
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource1));
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource2));
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource3));
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource4));
  EXPECT_EQ(8, **resource1);
  EXPECT_EQ(9, **resource2);
  EXPECT_EQ(7, **resource3);
  EXPECT_EQ(42, **resource4);
}

TEST(IntResourceTest, UnknownAllowUnregisteredFalse) {
  EXPECT_THAT(Context::Spec::FromJson({
                  {"foo", {{"value", 7}}},
              }),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid context resource identifier: \"foo\""));
}

TEST(IntResourceTest, UnknownAllowUnregisteredTrue) {
  const ::nlohmann::json json_spec = {
      {"foo", {{"value", 7}}},
  };
  auto spec_result =
      Context::Spec::FromJson(json_spec, AllowUnregistered{true});
  ASSERT_EQ(absl::OkStatus(), GetStatus(spec_result));
  EXPECT_EQ(json_spec, spec_result->ToJson());
  EXPECT_EQ(json_spec, spec_result->ToJson(IncludeDefaults{true}));
}

TEST(StrongRefResourceTest, DirectSpec) {
  auto context = Context::Default();
  auto resource_spec = Context::ResourceSpec<StrongRefResource>::FromJson(
      ::nlohmann::json::object_t{});
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource_spec));
  auto resource = context.GetResource(*resource_spec);
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource));
  // The `StrongRefResource` is held only by a `Context::Resource` handle
  // (`resource`), but not by the `Context` object `context`, since it was
  // specified by a JSON object directly, rather than a string reference.
  // Therefore, there are no strong references.
  EXPECT_EQ(0, resource.value()->num_strong_references);
}

TEST(StrongRefResourceTest, IndirectSpec) {
  auto spec_result =
      Context::Spec::FromJson({{"strongref", ::nlohmann::json::object_t{}}});
  ASSERT_EQ(absl::OkStatus(), GetStatus(spec_result));
  auto context = Context(*spec_result);
  auto resource_spec = Context::ResourceSpec<StrongRefResource>::Default();
  auto resource = context.GetResource(resource_spec);
  ASSERT_EQ(absl::OkStatus(), GetStatus(resource));
  // The `context` object maintains a strong reference to the resource.
  EXPECT_EQ(1, resource.value()->num_strong_references);
  // The `resource` handle remains valid, but the `StrongRefResource` is no
  // longer held in a `Context` object; therefore, there are no strong
  // references.
  context = Context();
  EXPECT_EQ(0, resource.value()->num_strong_references);
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
  EXPECT_EQ(::nlohmann::json({{"int_resource", {{"value", 5}}}}),
            new_spec.ToJson());
  EXPECT_EQ("int_resource", new_resource_spec.ToJson());

  // Test that we can convert back to resources.
  auto new_context = Context(new_spec);
  auto new_resource = new_context.GetResource(new_resource_spec).value();
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
  EXPECT_TRUE(new_spec.ToJson(IncludeDefaults{false}).value().is_discarded());
  EXPECT_THAT(new_spec.ToJson(IncludeDefaults{true}),
              ::testing::Optional(
                  ::nlohmann::json({{"int_resource", {{"value", 42}}}})));
  EXPECT_THAT(new_resource_spec.ToJson(),
              ::testing::Optional(::nlohmann::json("int_resource")));

  // Test that we can convert back to resources.
  auto new_context = Context(new_spec);
  auto new_resource = new_context.GetResource(new_resource_spec).value();
  EXPECT_EQ(42, *new_resource);
}

TEST(ContextSpecBuilderTest, MultipleContexts) {
  auto spec1 =
      Context::Spec::FromJson({{"int_resource", {{"value", 5}}}}).value();
  auto spec2 =
      Context::Spec::FromJson({{"int_resource", {{"value", 6}}}}).value();
  auto context1 = Context(spec1);
  auto context2 = Context(spec2);
  auto resource_spec = Context::ResourceSpec<IntResource>::Default();
  auto resource1 = context1.GetResource(resource_spec).value();
  auto resource2 = context2.GetResource(resource_spec).value();

  Context::Spec new_spec;
  Context::ResourceSpec<IntResource> new_resource_spec1;
  Context::ResourceSpec<IntResource> new_resource_spec2;
  {
    auto builder = ContextSpecBuilder::Make();
    new_spec = builder.spec();
    new_resource_spec1 = builder.AddResource(resource1);
    new_resource_spec2 = builder.AddResource(resource2);
  }
  EXPECT_EQ(::nlohmann::json({
                {"int_resource#0", {{"value", 5}}},
                {"int_resource#1", {{"value", 6}}},
            }),
            new_spec.ToJson());
  EXPECT_EQ("int_resource#0", new_resource_spec1.ToJson());
  EXPECT_EQ("int_resource#1", new_resource_spec2.ToJson());
}

TEST(ContextSpecBuilderTest, Inline) {
  auto context = Context::Default();
  auto resource_spec =
      Context::ResourceSpec<IntResource>::FromJson({{"value", 5}}).value();
  auto resource = context.GetResource(resource_spec).value();

  Context::Spec new_spec;
  Context::ResourceSpec<IntResource> new_resource_spec;
  {
    auto builder = ContextSpecBuilder::Make();
    new_spec = builder.spec();
    new_resource_spec = builder.AddResource(resource);
  }
  EXPECT_EQ(::nlohmann::json({}), new_spec.ToJson());
  EXPECT_EQ(::nlohmann::json({{"value", 5}}), new_resource_spec.ToJson());
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

}  // namespace
