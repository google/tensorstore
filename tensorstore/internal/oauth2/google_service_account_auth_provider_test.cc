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

#include "tensorstore/internal/oauth2/google_service_account_auth_provider.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "tensorstore/internal/oauth2/fake_private_key.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

using tensorstore::Result;
using tensorstore::internal_http::HttpResponse;
using tensorstore::internal_oauth2::GetFakePrivateKey;
using tensorstore::internal_oauth2::GoogleServiceAccountAuthProvider;
using tensorstore::internal_oauth2::GoogleServiceAccountCredentials;

namespace {

const char kServiceAccountInfo[] = R"({
    "token_type" : "123",
    "access_token": "abc",
    "expires_in": 456
  })";

const GoogleServiceAccountCredentials kCreds{
    /*private_key_id=*/"a1a111aa1111a11a11a11aa111a111a1a1111111",
    /*private_key=*/GetFakePrivateKey(),
    /*token_uri=*/"https://oauth2.googleapis.com/token",
    /*email=*/"foo-email@foo-project.iam.gserviceaccount.com",
};

constexpr char kBody[] =
    "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer&"
    "assertion="
    "eyJhbGciOiJSUzI1NiIsImtpZCI6ImExYTExMWFhMTExMWExMWExMWExMWFhMTExYTExMWExYT"
    "ExMTExMTEiLCJ0eXAiOiJKV1QifQ."
    "eyJhdWQiOiJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9vYXV0aDIvdjQvdG9rZW4iLCJleH"
    "AiOjE1NDc2Njk3MDMsImlhdCI6MTU0NzY2NjEwMywiaXNzIjoiZm9vLWVtYWlsQGZvby1wcm9q"
    "ZWN0LmlhbS5nc2VydmljZWFjY291bnQuY29tIiwic2NvcGUiOiJodHRwczovL3d3dy5nb29nbG"
    "VhcGlzLmNvbS9hdXRoL2Nsb3VkLXBsYXRmb3JtIn0.gvM1sjnFXwQkBTTqobnTJqE8ZCrAR-"
    "SEevEZB4Quqxd836v7iHjnWBiOkUCZl_o5wQouz5pFuhkQ1BlhhAZNih_Ko2yxBi0W_NuhI-"
    "18We8gSMhi8pwfNu6WqNqXkHlQAJebhJQH23yP_A2dxU3Z50maUJaAl9G0e60CIynsaeW-"
    "o7QneaPxPEWjOi--XMvkOu-z8eD0CXx1dUrlzINDxWzJFoXzCk2_NZ9-"
    "UPzHWai68qKo2FjbtTT3fEPA-L1IN908OWhuN2UHdvPrg_"
    "h13GO7kY3K7TsWotsgsLon2KxWYaDpasaY_ZqCIXCeS4jW89gVtsOB3E6B-xdR1Gq-9g";

class TestAuthProvider : public GoogleServiceAccountAuthProvider {
 public:
  TestAuthProvider(const GoogleServiceAccountCredentials& creds)
      : GoogleServiceAccountAuthProvider(creds, nullptr,
                                         [this] { return this->time; }),
        time(absl::FromUnixSeconds(1547666103)),
        idx(0) {}

  virtual Result<HttpResponse> IssueRequest(absl::string_view uri,
                                            absl::Cord body) {
    request.push_back(std::make_pair(std::string(uri), std::string(body)));
    if (responses.count(idx) != 0) {
      return responses[idx++];
    }
    return HttpResponse{};
  }

  absl::Time time;
  int idx;
  absl::flat_hash_map<int, HttpResponse> responses;
  std::vector<std::pair<std::string, std::string>> request;
};

TEST(GoogleServiceAccountAuthProviderTest, InitialState) {
  TestAuthProvider auth({"a", "b", "c", "d"});
  EXPECT_FALSE(auth.IsValid());
  EXPECT_TRUE(auth.IsExpired());
}

TEST(GoogleServiceAccountAuthProviderTest, BadKeys) {
  // The GoogleServiceAccountCredentials are invalid.
  TestAuthProvider auth({"a", "b", "c", "d"});

  auto result = auth.GetToken();
  EXPECT_FALSE(result.ok()) << result.status();
  EXPECT_EQ(0, auth.request.size());
}

TEST(OAuth2AuthProviderTest, NoResponse) {
  TestAuthProvider auth(kCreds);

  auto result = auth.GetToken();
  EXPECT_FALSE(result.ok()) << result.status();

  ASSERT_EQ(1, auth.request.size());
  EXPECT_EQ("https://www.googleapis.com/oauth2/v4/token",
            auth.request[0].first);
  EXPECT_EQ(kBody, auth.request[0].second);
}

TEST(GoogleServiceAccountAuthProviderTest, Status200) {
  TestAuthProvider auth(kCreds);

  auth.responses = {
      {0,
       {200,
        absl::Cord(kServiceAccountInfo),
        {}}},  // RetrieveServiceAccountInfo
      {1,
       {200,
        absl::Cord(kServiceAccountInfo),
        {}}},  // RetrieveServiceAccountInfo
  };

  {
    auto result = auth.GetToken();

    EXPECT_EQ(1, auth.idx);
    EXPECT_TRUE(result.ok()) << result.status();

    EXPECT_EQ(1, auth.request.size());
    EXPECT_EQ(auth.time + absl::Seconds(456), result->expiration);
    EXPECT_EQ("abc", result->token);
  }

  EXPECT_FALSE(auth.IsExpired());
  EXPECT_TRUE(auth.IsValid());

  // time passes.
  auth.time += absl::Seconds(600);
  {
    auto result = auth.GetToken();

    EXPECT_EQ(2, auth.idx);
    EXPECT_TRUE(result.ok()) << result.status();

    EXPECT_EQ(2, auth.request.size());
    EXPECT_EQ(auth.time + absl::Seconds(456), result->expiration);
    EXPECT_EQ("abc", result->token);
  }
}

}  // namespace
