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

#include "tensorstore/internal/oauth2/google_auth_provider.h"

#include <fstream>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/oauth2/fake_private_key.h"
#include "tensorstore/internal/oauth2/fixed_token_auth_provider.h"
#include "tensorstore/internal/oauth2/gce_auth_provider.h"
#include "tensorstore/internal/oauth2/google_auth_test_utils.h"
#include "tensorstore/internal/oauth2/google_service_account_auth_provider.h"
#include "tensorstore/internal/oauth2/oauth2_auth_provider.h"
#include "tensorstore/internal/oauth2/oauth_utils.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::Future;
using ::tensorstore::StrCat;
using ::tensorstore::internal::JoinPath;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::UnsetEnv;
using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_http::SetDefaultHttpTransport;
using ::tensorstore::internal_oauth2::AuthProvider;
using ::tensorstore::internal_oauth2::GetFakePrivateKey;
using ::tensorstore::internal_oauth2::GetGoogleAuthProvider;
using ::tensorstore::internal_oauth2::GoogleAuthTestScope;

class TestData : public tensorstore::internal::ScopedTemporaryDirectory {
 public:
  std::string WriteApplicationDefaultCredentials() {
    auto p = JoinPath(path(), "application_default_credentials.json");
    std::ofstream ofs(p);
    ofs << R"({
  "client_id": "fake-client-id.apps.googleusercontent.com",
  "client_secret": "fake-client-secret",
  "refresh_token": "fake-refresh-token",
  "type": "authorized_user"
})";
    return p;
  }

  std::string WriteServiceAccountCredentials() {
    auto p = JoinPath(path(), "service_account_credentials.json");
    std::ofstream ofs(p);
    ofs << R"({
  "type": "service_account",
  "project_id": "fake_project_id",
  "private_key_id": "fake_key_id",
  "client_email": "fake-test-project.iam.gserviceaccount.com",
  "client_id": "fake_client_id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://accounts.google.com/o/oauth2/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/fake-test-project.iam.gserviceaccount.com",
)";
    ofs << "  \"private_key\": \"" << absl::CEscape(GetFakePrivateKey())
        << "\" }";
    return p;
  }
};

class GoogleAuthProviderTest : public ::testing::Test {
  GoogleAuthTestScope google_auth_test_scope;
};

TEST_F(GoogleAuthProviderTest, Invalid) {
  // All environment variables are unset by default; this will look for
  // GCE, which will fail, and will return an error status.
  //
  // Set GCE_METADATA_ROOT to dummy value to ensure GCE detection fails even if
  // the test is really being run on GCE.
  SetEnv("GCE_METADATA_ROOT", "invalidmetadata.google.internal");
  auto auth_provider = GetGoogleAuthProvider();
  EXPECT_FALSE(auth_provider.ok());
  UnsetEnv("GCE_METADATA_ROOT");
}

TEST_F(GoogleAuthProviderTest, AuthTokenForTesting) {
  SetEnv("GOOGLE_AUTH_TOKEN_FOR_TESTING", "abc");

  // GOOGLE_AUTH_TOKEN_FOR_TESTING, so a FixedTokenAuthProvider
  // with the provided token will be returned.
  auto auth_provider = GetGoogleAuthProvider();
  ASSERT_TRUE(auth_provider.ok()) << auth_provider.status();

  // Expect an instance of FixedTokenAuthProvider.
  {
    auto instance =
        dynamic_cast<tensorstore::internal_oauth2::FixedTokenAuthProvider*>(
            auth_provider->get());
    EXPECT_FALSE(instance == nullptr);
  }

  // The token value is the same as was set by setenv()
  std::unique_ptr<AuthProvider> auth = std::move(*auth_provider);
  auto token = auth->GetToken();
  ASSERT_TRUE(token.ok());
  EXPECT_EQ("abc", token->token);
}

TEST_F(GoogleAuthProviderTest, GoogleOAuth2AccountCredentialsFromSDKConfig) {
  TestData test_data;
  test_data.WriteServiceAccountCredentials();
  test_data.WriteApplicationDefaultCredentials();
  SetEnv("CLOUDSDK_CONFIG", test_data.path().c_str());

  // CLOUDSDK_CONFIG has been set to the path of the credentials file.
  // We will attempt to parse the "application_default_credentials.json"
  // file in that location, which happens to be an OAuth2 token.
  auto auth_provider = GetGoogleAuthProvider();
  ASSERT_TRUE(auth_provider.ok()) << auth_provider.status();

  // Expect an instance of OAuth2AuthProvider
  {
    auto instance =
        dynamic_cast<tensorstore::internal_oauth2::OAuth2AuthProvider*>(
            auth_provider->get());
    EXPECT_FALSE(instance == nullptr);
  }
}

/// GOOGLE_APPLICATION_CREDENTIALS
TEST_F(GoogleAuthProviderTest, GoogleOAuth2AccountCredentials) {
  TestData test_data;
  SetEnv("GOOGLE_APPLICATION_CREDENTIALS",
         test_data.WriteApplicationDefaultCredentials().c_str());

  // GOOGLE_APPLICATION_CREDENTIALS has been set to the path of the
  // application_default_credentials.json file, which is an OAuth2 token.
  auto auth_provider = GetGoogleAuthProvider();
  ASSERT_TRUE(auth_provider.ok()) << auth_provider.status();

  // Expect an instance of OAuth2AuthProvider
  {
    auto instance =
        dynamic_cast<tensorstore::internal_oauth2::OAuth2AuthProvider*>(
            auth_provider->get());
    EXPECT_FALSE(instance == nullptr);
  }
}

TEST_F(GoogleAuthProviderTest, GoogleServiceAccountCredentials) {
  TestData test_data;
  SetEnv("GOOGLE_APPLICATION_CREDENTIALS",
         test_data.WriteServiceAccountCredentials().c_str());

  // GOOGLE_APPLICATION_CREDENTIALS has been set to the path of the
  // service_account_credentials.json file, which is an Google Service Account
  // credentials token.
  auto auth_provider = GetGoogleAuthProvider();
  ASSERT_TRUE(auth_provider.ok()) << auth_provider.status();

  // Expect an instance of GoogleServiceAccountAuthProvider
  {
    auto instance = dynamic_cast<
        tensorstore::internal_oauth2::GoogleServiceAccountAuthProvider*>(
        auth_provider->get());
    EXPECT_FALSE(instance == nullptr);
  }
}

// Responds to a "metadata.google.internal" request.
class MetadataMockTransport : public HttpTransport {
 public:
  explicit MetadataMockTransport(bool has_service_account)
      : has_service_account_(has_service_account) {}
  Future<HttpResponse> IssueRequest(const HttpRequest& request,
                                    absl::Cord payload,
                                    absl::Duration request_timeout,
                                    absl::Duration connect_timeout) override {
    auto parsed = tensorstore::internal::ParseGenericUri(request.url());

    if (!absl::StartsWith(parsed.authority_and_path,
                          "metadata.google.internal")) {
      return absl::UnimplementedError("Mock cannot satisfy the request.");
    }

    // Respond with the GCE OAuth2 token
    constexpr char kOAuthPath[] =
        "metadata.google.internal/computeMetadata/v1/"
        "instance/service-accounts/user@nowhere.com/token";
    if (absl::StartsWith(parsed.authority_and_path, kOAuthPath)) {
      if (!has_service_account_) {
        return HttpResponse{404, absl::Cord()};
      }

      return HttpResponse{
          200,
          absl::Cord(
              R"({ "token_type" : "refresh", "access_token": "abc", "expires_in": 3600 })")};
    }

    // Respond with the GCE context metadata.
    constexpr char kServiceAccountPath[] =
        "metadata.google.internal/computeMetadata/v1/"
        "instance/service-accounts/default/";
    if (absl::StartsWith(parsed.authority_and_path, kServiceAccountPath)) {
      if (!has_service_account_) {
        return HttpResponse{404, absl::Cord()};
      }

      return HttpResponse{
          200, absl::Cord(
                   R"({ "email": "user@nowhere.com", "scopes": [ "test" ] })")};
    }

    // Pretend to run on GCE.
    return HttpResponse{200, absl::Cord()};
  }

  bool has_service_account_;
};

struct DefaultHttpTransportSetter {
  DefaultHttpTransportSetter(std::shared_ptr<HttpTransport> transport) {
    SetDefaultHttpTransport(transport);
    tensorstore::internal_oauth2::ResetSharedGoogleAuthProvider();
  }
  ~DefaultHttpTransportSetter() {
    tensorstore::internal_oauth2::ResetSharedGoogleAuthProvider();
    SetDefaultHttpTransport(nullptr);
  }
};

TEST_F(GoogleAuthProviderTest, GceWithServiceAccount) {
  auto mock_transport = std::make_shared<MetadataMockTransport>(true);
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto auth_provider, GetGoogleAuthProvider());

  // Expect an instance of GceAuthProvider
  {
    auto instance =
        dynamic_cast<tensorstore::internal_oauth2::GceAuthProvider*>(
            auth_provider.get());
    EXPECT_FALSE(instance == nullptr);
  }

  EXPECT_THAT(auth_provider->GetAuthHeader(),
              ::testing::Optional(std::string("Authorization: Bearer abc")));
}

TEST_F(GoogleAuthProviderTest, GceWithoutServiceAccount) {
  auto mock_transport = std::make_shared<MetadataMockTransport>(false);
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  EXPECT_THAT(GetGoogleAuthProvider(),
              tensorstore::MatchesStatus(absl::StatusCode::kNotFound));
}

// NOTE: ${HOME}/.cloud/config/application_default_credentials.json is not
// tested.

}  // namespace
