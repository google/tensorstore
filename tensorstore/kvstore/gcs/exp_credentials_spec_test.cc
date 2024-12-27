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

#include "tensorstore/kvstore/gcs/exp_credentials_spec.h"

#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "grpcpp/support/channel_arguments.h"  // third_party
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::MatchesJson;
using ::tensorstore::internal_storage_gcs::MakeGrpcAuthenticationStrategy;

using Spec =
    ::tensorstore::internal_storage_gcs::ExperimentalGcsGrpcCredentialsSpec;

namespace jb = ::tensorstore::internal_json_binding;

namespace {

// See grpc/test/cpp/client/credentials_test.cc
const auto kServiceAccountJsonObject = ::nlohmann::json::object_t({
    {"type", "service_account"},
    {"project_id", "foo-project"},
    {"private_key_id", "a1a111aa1111a11a11a11aa111a111a1a1111111"},
    {"private_key",
     "-----BEGIN PRIVATE "
     "KEY-----"
     "\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCltiF2o"
     "P3KJJ+S\ntTc1McylY+"
     "TuAi3AdohX7mmqIjd8a3eBYDHs7FlnUrFC4CRijCr0rUqYfg2pmk4a\n6Ta"
     "KbQRAhWDJ7XD931g7EBvCtd8+"
     "JQBNWVKnP9ByJUaO0hWVniM50KTsWtyX3up/"
     "\nfS0W2R8Cyx4yvasE8QHH8gnNGtr94iiORDC7De2BwHi/"
     "iU8FxMVJAIyDLNfyk0hN\neheYKfIDBgJV2v6VaCOGWaZyEuD0FJ6wFeLyb"
     "FBwibrLIBE5Y/StCrZoVZ5LocFP\nT4o8kT7bU6yonudSCyNMedYmqHj/"
     "iF8B2UN1WrYx8zvoDqZk0nxIglmEYKn/"
     "6U7U\ngyETGcW9AgMBAAECggEAC231vmkpwA7JG9UYbviVmSW79UecsLzsO"
     "AZnbtbn1VLT\nPg7sup7tprD/LXHoyIxK7S/"
     "jqINvPU65iuUhgCg3Rhz8+UiBhd0pCH/"
     "arlIdiPuD\n2xHpX8RIxAq6pGCsoPJ0kwkHSw8UTnxPV8ZCPSRyHV71oQHQ"
     "gSl/WjNhRi6PQroB\nSqc/"
     "pS1m09cTwyKQIopBBVayRzmI2BtBxyhQp9I8t5b7PYkEZDQlbdq0j5Xipoo"
     "v\n9EW0+Zvkh1FGNig8IJ9Wp+SZi3rd7KLpkyKPY7BK/"
     "g0nXBkDxn019cET0SdJOHQG\nDiHiv4yTRsDCHZhtEbAMKZEpku4WxtQ+"
     "JjR31l8ueQKBgQDkO2oC8gi6vQDcx/"
     "CX\nZ23x2ZUyar6i0BQ8eJFAEN+IiUapEeCVazuxJSt4RjYfwSa/"
     "p117jdZGEWD0GxMC\n+iAXlc5LlrrWs4MWUc0AHTgXna28/"
     "vii3ltcsI0AjWMqaybhBTTNbMFa2/"
     "fV2OX2\nUimuFyBWbzVc3Zb9KAG4Y7OmJQKBgQC5324IjXPq5oH8UWZTdJP"
     "uO2cgRsvKmR/"
     "r\n9zl4loRjkS7FiOMfzAgUiXfH9XCnvwXMqJpuMw2PEUjUT+"
     "OyWjJONEK4qGFJkbN5\n3ykc7p5V7iPPc7Zxj4mFvJ1xjkcj+i5LY8Me+"
     "gL5mGIrJ2j8hbuv7f+PWIauyjnp\nNx/"
     "0GVFRuQKBgGNT4D1L7LSokPmFIpYh811wHliE0Fa3TDdNGZnSPhaD9/"
     "aYyy78\nLkxYKuT7WY7UVvLN+gdNoVV5NsLGDa4cAV+CWPfYr5PFKGXMT/"
     "Wewcy1WOmJ5des\nAgMC6zq0TdYmMBN6WpKUpEnQtbmh3eMnuvADLJWxbH3"
     "wCkg+"
     "4xDGg2bpAoGAYRNk\nMGtQQzqoYNNSkfus1xuHPMA8508Z8O9pwKU795R3z"
     "Qs1NAInpjI1sOVrNPD7Ymwc\nW7mmNzZbxycCUL/"
     "yzg1VW4P1a6sBBYGbw1SMtWxun4ZbnuvMc2CTCh+43/1l+FHe\nMmt46kq/"
     "2rH2jwx5feTbOE6P6PINVNRJh/9BDWECgYEAsCWcH9D3cI/"
     "QDeLG1ao7\nrE2NcknP8N783edM07Z/"
     "zxWsIsXhBPY3gjHVz2LDl+QHgPWhGML62M0ja/"
     "6SsJW3\nYvLLIc82V7eqcVJTZtaFkuht68qu/"
     "Jn1ezbzJMJ4YXDYo1+KFi+2CAGR06QILb+I\nlUtj+/"
     "nH3HDQjM4ltYfTPUg=\n-----END PRIVATE KEY-----\n"},
    {"client_email", "foo-email@foo-project.iam.gserviceaccount.com"},
    {"client_id", "100000000000000000001"},
    {"auth_uri", "https://accounts.google.com/o/oauth2/auth"},
    {"token_uri", "https://oauth2.googleapis.com/token"},
    {"auth_provider_x509_cert_url",
     "https://www.googleapis.com/oauth2/v1/certs"},
    {"client_x509_cert_url",
     "https://www.googleapis.com/robot/v1/metadata/x509/"
     "foo-email%40foo-project.iam.gserviceaccount.com"},
});

const auto kExternalAWSAccountJsonObject = ::nlohmann::json::object_t({
    {"type", "external_account"},
    {"audience", "audience"},
    {"subject_token_type", "subject_token_type"},
    {"service_account_impersonation_url", "service_account_impersonation_url"},
    {"token_url", "https://foo.com:5555/token"},
    {"token_info_url", "https://foo.com:5555/token_info"},
    {"credential_source",
     ::nlohmann::json::object_t({
         {"environment_id", "aws1"},
         {"region_url", "https://169.254.169.254:5555/region_url"},
         {"url", "https://169.254.169.254:5555/url"},
         {"regional_cred_verification_url",
          "https://foo.com:5555/regional_cred_verification_url_{region}"},
     })},
    {"quota_project_id", "quota_project_id"},
    {"client_id", "client_id"},
    {"client_secret", "client_secret"},
});

struct Params {
  Spec spec;
  ::nlohmann::json::object_t json;
};

class SpecTest : public ::testing::TestWithParam<Params> {
 public:
  constexpr static Spec::PartialBinder binder = {};
};

INSTANTIATE_TEST_SUITE_P(
    VariousCredentials, SpecTest,
    ::testing::Values(
        Params{{}, {}},
        []() {
          Params result;
          result.spec = Spec{Spec::Insecure{}};
          result.json["type"] = "insecure";
          return result;
        }(),
        []() {
          Params result;
          result.spec = Spec{Spec::GoogleDefault{}};
          result.json["type"] = "google_default";
          return result;
        }(),
        []() {
          Params result;
          result.spec = Spec{Spec::AccessToken{{"token"}}};
          result.json["type"] = "access_token";
          result.json["access_token"] = "token";
          return result;
        }(),
        []() {
          Params result;
          result.spec = Spec{Spec::ServiceAccount{{"path"}}};
          result.json["type"] = "service_account";
          result.json["path"] = "path";
          return result;
        }(),
        []() {
          Params result;
          result.spec =
              Spec{Spec::ServiceAccount{{}, kServiceAccountJsonObject}};
          result.json = kServiceAccountJsonObject;
          return result;
        }(),
        []() {
          Params result;
          result.spec = Spec{Spec::ExternalAccount{{"path"}}};
          result.json["type"] = "external_account";
          result.json["path"] = "path";
          return result;
        }(),
        []() {
          Params result;
          result.spec = Spec{
              Spec::ExternalAccount{{}, {}, kExternalAWSAccountJsonObject}};
          result.json = kExternalAWSAccountJsonObject;
          return result;
        }(),
        []() {
          Params result;
          result.spec = Spec{Spec::ImpersonateServiceAccount{
              {"target"},
              {},
              {},
              ::nlohmann::json::object_t{{"type", "google_default"}}}};
          result.json["type"] = "impersonate_service_account";
          result.json["target_service_account"] = "target";
          result.json["base"]["type"] = "google_default";
          return result;
        }() /**/
        ));

TEST_P(SpecTest, Load) {
  Params param = GetParam();
  Spec from_json;
  EXPECT_THAT(
      binder(std::true_type{}, jb::NoOptions{}, &from_json, &param.json),
      tensorstore::IsOk());
  EXPECT_THAT(from_json, ::testing::Eq(param.spec));
}

TEST_P(SpecTest, Save) {
  const Params param = GetParam();
  ::nlohmann::json::object_t to_json;
  EXPECT_THAT(binder(std::false_type{}, tensorstore::IncludeDefaults{},
                     &param.spec, &to_json),
              tensorstore::IsOk());
  EXPECT_THAT(to_json, MatchesJson(param.json));
}

TEST_P(SpecTest, Create) {
  Params p = GetParam();
  // Skip default-initialized UnifiedAccessCredentialsSpec.
  // Trying to create a GrpcAuthenticationStrategy from it will return nullptr.
  if (p.spec.IsDefault()) return;

  if (auto type = p.spec.GetType();
      type != "insecure" && type != "access_token") {
    // Skip most auth mechanisms for the tests.
    return;
  }

  auto strategy = MakeGrpcAuthenticationStrategy(p.spec, {});
  EXPECT_THAT(strategy, ::tensorstore::IsOk());
  EXPECT_THAT(strategy.value(), ::testing::NotNull());

  grpc::ChannelArguments args;
  auto creds = strategy.value()->GetChannelCredentials("localhost:1", args);
  EXPECT_THAT(creds.get(), testing::NotNull());
}

}  // namespace
