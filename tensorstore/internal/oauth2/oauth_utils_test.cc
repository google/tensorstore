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

#include "tensorstore/internal/oauth2/oauth_utils.h"

#include <gtest/gtest.h>
#include "absl/strings/escaping.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/oauth2/fake_private_key.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

using tensorstore::internal_oauth2::GetFakePrivateKey;
using tensorstore::internal_oauth2::ParseGoogleServiceAccountCredentials;
using tensorstore::internal_oauth2::ParseOAuthResponse;
using tensorstore::internal_oauth2::ParseRefreshToken;

namespace {

std::string GetJsonKeyFileContents() {
  constexpr char kJsonKeyfilePrefix[] = R"""({
      "type": "service_account",
      "project_id": "foo-project",
      "private_key_id": "a1a111aa1111a11a11a11aa111a111a1a1111111",
      "client_email": "foo-email@foo-project.iam.gserviceaccount.com",
      "client_id": "100000000000000000001",
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token",
      "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
      "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/foo-email%40foo-project.iam.gserviceaccount.com",
)""";
  return tensorstore::StrCat(kJsonKeyfilePrefix, "  \"private_key\": \"",
                             absl::CEscape(GetFakePrivateKey()), "\" }");
}

TEST(OAuthUtilTest, GoogleServiceAccountCredentials_Invalid) {
  EXPECT_FALSE(ParseGoogleServiceAccountCredentials("{ }").ok());

  // Empty fields
  EXPECT_FALSE(ParseGoogleServiceAccountCredentials(R"({
    "private_key" : "",
    "private_key_id": "",
    "client_email": "",
    "token_uri": ""
  })")
                   .ok());

  EXPECT_FALSE(ParseGoogleServiceAccountCredentials(R"({
    "private_key" : "",
    "private_key_id": "abc",
    "client_email": "456"
  })")
                   .ok());

  EXPECT_FALSE(ParseGoogleServiceAccountCredentials(R"({
    "private_key" : "123",
    "private_key_id": "",
    "client_email": "456"
  })")
                   .ok());

  EXPECT_FALSE(ParseGoogleServiceAccountCredentials(R"({
    "private_key" : "123",
    "private_key_id": "abc",
    "client_email": ""
  })")
                   .ok());

  EXPECT_FALSE(ParseGoogleServiceAccountCredentials(R"({
    "private_key" : "123",
    "private_key_id": "abc",
    "client_email": "456"
    "token_uri": ""
  })")
                   .ok());

  // Missing fields
  // token_uri can be missing or not.
  EXPECT_FALSE(ParseGoogleServiceAccountCredentials(R"({
    "private_key_id": "abc",
    "client_email": "456",
  })")
                   .ok());

  EXPECT_FALSE(ParseGoogleServiceAccountCredentials(R"({
    "private_key" : "123",
    "client_email": "456",
  })")
                   .ok());

  EXPECT_FALSE(ParseGoogleServiceAccountCredentials(R"({
    "private_key" : "123",
    "private_key_id": "abc",
  })")
                   .ok());
}

TEST(OAuthUtilTest, GoogleServiceAccountCredentials) {
  // Empty, but valid.
  auto result = ParseGoogleServiceAccountCredentials(R"({
    "private_key" : "123",
    "private_key_id": "abc",
    "client_email": "456",
    "token_uri": "wxy"
  })");
  ASSERT_TRUE(result.ok()) << result.status();

  EXPECT_EQ("123", result.value().private_key);
  EXPECT_EQ("abc", result.value().private_key_id);
  EXPECT_EQ("456", result.value().client_email);
  EXPECT_EQ("wxy", result.value().token_uri);

  // Missing token_uri
  result = ParseGoogleServiceAccountCredentials(R"({
    "private_key" : "123",
    "private_key_id": "abc",
    "client_email": "456"
  })");
  ASSERT_TRUE(result.ok()) << result.status();

  EXPECT_EQ("123", result.value().private_key);
  EXPECT_EQ("abc", result.value().private_key_id);
  EXPECT_EQ("456", result.value().client_email);
  EXPECT_EQ("", result.value().token_uri);
}

TEST(OAuthUtilTest, GoogleServiceAccountCredentialsFile) {
  auto result = ParseGoogleServiceAccountCredentials(GetJsonKeyFileContents());

  ASSERT_TRUE(result.ok()) << result.status();

  EXPECT_EQ("foo-email@foo-project.iam.gserviceaccount.com",
            result->client_email);
}

TEST(OAuthUtilTest, ParseRefreshToken_Invalid) {
  EXPECT_FALSE(ParseRefreshToken("{ }").ok());

  // Empty fields.
  EXPECT_FALSE(ParseRefreshToken(R"({
    "client_id" : "",
    "client_secret": "",
    "refresh_token": ""
  })")
                   .ok());

  EXPECT_FALSE(ParseRefreshToken(R"({
    "client_id" : "",
    "client_secret": "abc",
    "refresh_token": "456"
  })")
                   .ok());

  EXPECT_FALSE(ParseRefreshToken(R"({
    "client_id" : "123",
    "client_secret": "",
    "refresh_token": "456"
  })")
                   .ok());

  EXPECT_FALSE(ParseRefreshToken(R"({
    "client_id" : "123",
    "client_secret": "abc",
    "refresh_token": ""
  })")
                   .ok());

  // Wrong type
  EXPECT_FALSE(ParseRefreshToken(R"({
    "client_id" : "123",
    "client_secret": "abc",
    "refresh_token": 456
  })")
                   .ok());

  // Missing fields
  EXPECT_FALSE(ParseRefreshToken(R"({
    "client_secret": "abc",
    "refresh_token": "456"
  })")
                   .ok());

  EXPECT_FALSE(ParseRefreshToken(R"({
    "client_id" : "123",
    "refresh_token": "456"
  })")
                   .ok());

  EXPECT_FALSE(ParseRefreshToken(R"({
    "client_id" : "123",
    "client_secret": "abc",
  })")
                   .ok());
}

TEST(OAuthUtilTest, ParseRefreshToken) {
  // Empty, but valid.
  auto result = ParseRefreshToken(R"({
    "client_id" : "123",
    "client_secret": "abc",
    "refresh_token": "456"
  })");
  ASSERT_TRUE(result.ok()) << result.status();

  EXPECT_EQ("123", result.value().client_id);
  EXPECT_EQ("abc", result.value().client_secret);
  EXPECT_EQ("456", result.value().refresh_token);
}

TEST(OAuthUtilTest, ParseOAuthResponse_Invalid) {
  EXPECT_FALSE(ParseOAuthResponse("{ }").ok());

  // Empty fields.
  EXPECT_FALSE(ParseOAuthResponse(R"({
    "token_type" : "",
    "access_token": "abc",
    "expires_in": 456
  })")
                   .ok());

  EXPECT_FALSE(ParseOAuthResponse(R"({
    "token_type" : "123",
    "access_token": "",
    "expires_in": 456
  })")
                   .ok());

  // Missing field.
  EXPECT_FALSE(ParseOAuthResponse(R"({
    "token_type" : "123",
    "access_token": "abc",
  })")
                   .ok());
}

TEST(OAuthUtilTest, ParseOAuthResponse) {
  // Fields of wrong type can be parsed correctly.
  EXPECT_TRUE(ParseOAuthResponse(R"({
    "token_type" : "123",
    "access_token": "abc",
    "expires_in": "456"
  })")
                  .ok());

  auto result = ParseOAuthResponse(R"({
    "token_type" : "123",
    "access_token": "abc",
    "expires_in": 456
  })");
  ASSERT_TRUE(result.ok()) << result.status();

  EXPECT_EQ("123", result.value().token_type);
  EXPECT_EQ("abc", result.value().access_token);
  EXPECT_EQ(456, result.value().expires_in);

  // Extra fields are ignored.
  result = ParseOAuthResponse(R"({
    "token_type" : "123",
    "access_token": "abc",
    "expires_in": 456,
    "extra_fields": "are ignored"
  })");
  ASSERT_TRUE(result.ok()) << result.status();
}

TEST(OAuthUtilTest, BuildJWTClaimTest) {
  using tensorstore::internal_oauth2::BuildJWTClaimBody;
  using tensorstore::internal_oauth2::BuildJWTHeader;

  EXPECT_EQ("eyJhbGciOiJSUzI1NiIsImtpZCI6ImEiLCJ0eXAiOiJKV1QifQ",
            BuildJWTHeader("a"));

  EXPECT_EQ(
      "eyJhdWQiOiI0IiwiZXhwIjoxNTQ3NjY5NzAzLCJpYXQiOjE1NDc2NjYxMDMsImlzcyI6ImIi"
      "LCJzY29wZSI6ImMifQ",
      BuildJWTClaimBody("b", "c", "4", absl::FromUnixSeconds(1547666103),
                        3600));
}

TEST(OAuthUtilTest, Sign) {
  using tensorstore::internal_oauth2::SignWithRSA256;

  // Empty private key.
  {
    auto result = SignWithRSA256("", "something");
    EXPECT_FALSE(result.ok());
  }

  // Invalid private key.
  {
    constexpr char kBadKey[] =
        "-----BEGIN PRIVATE KEY-----\n"
        "Z23x2ZUyar6i0BQ8eJFAEN+IiUapEeCVazuxJSt4RjYfwSa/"
        "p117jdZGEWD0GxMC\nlUtj+/nH3HDQjM4ltYfTPUg=\n"
        "-----END PRIVATE KEY-----\n";
    auto result = SignWithRSA256(kBadKey, "something");
    EXPECT_FALSE(result.ok());
  }

  auto creds = ParseGoogleServiceAccountCredentials(GetJsonKeyFileContents());
  ASSERT_TRUE(creds.ok());

  // Valid private key.
  {
    auto result = SignWithRSA256(creds->private_key, "something");
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(
        "A-sH4BVqtxu-6LECWJCb0VKGDj46pnpBpZB1KViuhG2CwugRVR6V3-"
        "w8eBvAUbIRewSnXp_lWkxdy_rZBMau9VuILnLOC0t692-"
        "L8WEqHsoFYBWvTZGCT5XkslVXhxt4d8jgM6U_8If4Cf3fGA4XAxpP-pyrbPGz-"
        "VXn6R7jcLGOLsFtcuAXpJ9zkwYE72pGUtI_hiU-"
        "tquIEayOQW9frXJlxt2oR4ld1l3p0FWibkNY8OfYPdTlRS0WcsgpWngTamHEBplJ5xNLD5"
        "Ye5bG1DFqBJn0evxW0btbcfKCYuyirvgvHPsTt-"
        "YMcPGo1xtlhT5c4ycEHOObFUGDpKPjljw",
        *result);
  }
}

TEST(OAuthUtilTest, BuildJWTRequestBody) {
  using tensorstore::internal_oauth2::BuildSignedJWTRequest;

  auto creds = ParseGoogleServiceAccountCredentials(GetJsonKeyFileContents());
  ASSERT_TRUE(creds.ok());

  auto result =
      BuildSignedJWTRequest(creds->private_key, "header", "something");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(
      "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer&"
      "assertion=header.something.LyvY9ZVG6tL34g5Wji--3G5JGQP-"
      "fza47yBQIrRHJqecVUTVGuEXti_deBjSbB36gvpBOE67-U9h1wgD2VR_"
      "MDx8JaQHGct04gVZdKC7m4uqu5lI8u0jqXGG4UbRwfUMZ0UCjxJfyUbg6KUR7iyiqoH5szZv"
      "31rJISnM4RQvH-lQFrE6BuXpvB09Hve4T3q5mtq7E9pd5rXz_"
      "vlqL5ib5tkdBEg2cbydDZHeCx5uA9qcg3hGidrU1fLgreFKu3dSvzu4qFZL3-"
      "0Pnt4XMqwslx2vBbFQB7_K8Dnz10F1TA5njOvwFRWNjKM1I0cRZ5N3O1CnGv1wyAz-"
      "FIcKdk5_7Q",
      *result);
}

}  // namespace
