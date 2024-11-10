// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/kvstore/s3/aws_credentials_resource.h"

#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::Context;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_kvstore_s3::AwsCredentialsResource;

namespace {

TEST(AwsCredentialsResourceTest, InvalidDirectSpec) {
  EXPECT_THAT(Context::Resource<AwsCredentialsResource>::FromJson(nullptr),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected non-null value, but received: null"));

  EXPECT_THAT(Context::Resource<AwsCredentialsResource>::FromJson(3),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected object, but received: 3"));

  EXPECT_THAT(Context::Resource<AwsCredentialsResource>::FromJson("anonymous"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid spec or reference to \"aws_credentials\" "
                            "resource: \"anonymous\".*"));
}

TEST(AwsCredentialsResourceTest, Default) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec,
      Context::Resource<AwsCredentialsResource>::FromJson("aws_credentials"));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource,
                                   context.GetResource(resource_spec));

  EXPECT_THAT(resource->spec.filename, ::testing::IsEmpty());
  EXPECT_THAT(resource->spec.profile, ::testing::IsEmpty());
  EXPECT_THAT(resource->spec.metadata_endpoint, ::testing::IsEmpty());
  EXPECT_THAT(resource->spec.anonymous, false);
}

TEST(AwsCredentialsResourceTest, ExplicitDefault) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec, Context::Resource<AwsCredentialsResource>::FromJson(
                              ::nlohmann::json::object_t()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource,
                                   context.GetResource(resource_spec));

  EXPECT_THAT(resource->spec.filename, ::testing::IsEmpty());
  EXPECT_THAT(resource->spec.profile, ::testing::IsEmpty());
  EXPECT_THAT(resource->spec.metadata_endpoint, ::testing::IsEmpty());
  EXPECT_THAT(resource->spec.anonymous, false);
}

TEST(AwsCredentialsResourceTest, ValidSpec) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec, Context::Resource<AwsCredentialsResource>::FromJson(
                              {{"profile", "my_profile"}}));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource,
                                   context.GetResource(resource_spec));

  EXPECT_THAT(resource->spec.filename, ::testing::IsEmpty());
  EXPECT_THAT(resource->spec.profile, "my_profile");
  EXPECT_THAT(resource->spec.metadata_endpoint, ::testing::IsEmpty());
  EXPECT_THAT(resource->spec.anonymous, false);
}

TEST(AwsCredentialsResourceTest, ValidAnonymousSpec) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec, Context::Resource<AwsCredentialsResource>::FromJson(
                              {{"anonymous", true}}));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resource,
                                   context.GetResource(resource_spec));

  EXPECT_THAT(resource->spec.filename, ::testing::IsEmpty());
  EXPECT_THAT(resource->spec.profile, ::testing::IsEmpty());
  EXPECT_THAT(resource->spec.metadata_endpoint, ::testing::IsEmpty());
  EXPECT_THAT(resource->spec.anonymous, true);

  EXPECT_THAT(resource->GetCredentials(),
              tensorstore::IsOkAndHolds(::testing::Eq(std::nullopt)));
}

TEST(AwsCredentialsResourceTest, InvalidSpecs) {
  EXPECT_THAT(Context::Resource<AwsCredentialsResource>::FromJson({
                  {"anonymous", true},
                  {"profile", "xyz"},
              }),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
