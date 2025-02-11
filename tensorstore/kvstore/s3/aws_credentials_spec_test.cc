// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/kvstore/s3/aws_credentials_spec.h"

#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/status_testutil.h"

using Spec = ::tensorstore::internal_kvstore_s3::AwsCredentialsSpec;

using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;

namespace jb = ::tensorstore::internal_json_binding;

namespace {

struct Params {
  Spec spec;
  ::nlohmann::json::object_t json;
};

class SpecParamTest : public ::testing::TestWithParam<Params> {
 public:
  constexpr static Spec::PartialBinder binder = {};
};

INSTANTIATE_TEST_SUITE_P(  //
    VariousCredentials, SpecParamTest,
    ::testing::Values(
        []() {
          Params result;
          result.spec = Spec{Spec::Default{}};
          result.json["type"] = "default";
          return result;
        }(),
        []() {
          Params result;
          result.spec = Spec{Spec::Default{"my_user"}};
          result.json["type"] = "default";
          result.json["profile"] = "my_user";
          return result;
        }(),
        []() {
          Params result;
          result.spec = Spec{Spec::Anonymous{}};
          result.json["type"] = "anonymous";
          return result;
        }(),
        []() {
          Params result;
          result.spec = Spec{Spec::Environment{}};
          result.json["type"] = "environment";
          return result;
        }(),
        []() {
          Params result;
          result.spec = Spec{Spec::Imds{}};
          result.json["type"] = "imds";
          return result;
        }(),
        []() {
          Params result;
          result.spec = Spec{Spec::Profile{}};
          result.json["type"] = "profile";
          return result;
        }(),
        []() {
          Params result;
          result.spec =
              Spec{Spec::Profile{"my_user", "config_file", "credentials_file"}};
          result.json["type"] = "profile";
          result.json["profile"] = "my_user";
          result.json["config_file"] = "config_file";
          result.json["credentials_file"] = "credentials_file";
          return result;
        }(),
        []() {
          Params result;
          result.spec = Spec{Spec::EcsRole{}};
          result.json["type"] = "ecs";
          return result;
        }(),
        []() {
          Params result;
          result.spec =
              Spec{Spec::EcsRole{"http://localhost/latest/meta-data/iam/",
                                 "/path/to/auth-token-file"}};
          result.json["type"] = "ecs";
          result.json["endpoint"] = "http://localhost/latest/meta-data/iam/";
          result.json["auth_token_file"] = "/path/to/auth-token-file";
          return result;
        }()
        /**/
        ));

TEST_P(SpecParamTest, Load) {
  Params param = GetParam();
  ::nlohmann::json::object_t json = param.json;
  Spec from_json;
  EXPECT_THAT(binder(std::true_type{}, jb::NoOptions{}, &from_json, &json),
              tensorstore::IsOk());
  EXPECT_THAT(from_json, ::testing::Eq(param.spec));
}

TEST_P(SpecParamTest, Save) {
  const Params param = GetParam();
  ::nlohmann::json::object_t to_json;
  EXPECT_THAT(binder(std::false_type{}, tensorstore::IncludeDefaults{},
                     &param.spec, &to_json),
              tensorstore::IsOk());
  EXPECT_THAT(to_json, MatchesJson(param.json));
}

TEST(SpecTest, Error) {
  ::nlohmann::json::object_t json{{"type", "_missing_"}};

  Spec spec;
  EXPECT_THAT(
      Spec::PartialBinder{}(std::true_type{}, tensorstore::IncludeDefaults(),
                            &spec, &json),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Failed to parse AWS credentials spec.*"));
}

}  // namespace
