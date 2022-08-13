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

#include "tensorstore/internal/oauth2/oauth2_auth_provider.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/time/clock.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace {

using ::tensorstore::Result;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_oauth2::OAuth2AuthProvider;

const char kServiceAccountInfo[] = R"({
    "token_type" : "123",
    "access_token": "abc",
    "expires_in": 456
  })";

// The URL to retrieve the auth bearer token via OAuth with a refresh token.
constexpr char kOAuthV3Url[] = "https://www.googleapis.com/oauth2/v3/token";

class TestAuthProvider : public OAuth2AuthProvider {
 public:
  TestAuthProvider(const RefreshToken& creds)
      : OAuth2AuthProvider(creds, kOAuthV3Url, nullptr,
                           [this] { return this->time; }),
        time(absl::Now()),
        idx(0) {}

  virtual Result<HttpResponse> IssueRequest(std::string_view method,
                                            std::string_view uri,
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

TEST(OAuth2AuthProviderTest, InitialState) {
  TestAuthProvider auth({"a", "b", "c"});
  EXPECT_FALSE(auth.IsValid());
  EXPECT_TRUE(auth.IsExpired());
}

TEST(OAuth2AuthProviderTest, NoResponse) {
  TestAuthProvider auth({"a", "b", "c"});

  auto result = auth.GetToken();
  EXPECT_FALSE(result.ok()) << result.status();

  ASSERT_EQ(1, auth.request.size());
  EXPECT_EQ("https://www.googleapis.com/oauth2/v3/token",
            auth.request[0].first);
  EXPECT_EQ(
      "grant_type=refresh_token&client_id=a&client_secret=b&refresh_token=c",
      auth.request[0].second);
}

TEST(OAuth2AuthProviderTest, Status200) {
  TestAuthProvider auth({"a", "b", "c"});
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

    ASSERT_EQ(1, auth.request.size());
    EXPECT_EQ("https://www.googleapis.com/oauth2/v3/token",
              auth.request[0].first);
    EXPECT_EQ(
        "grant_type=refresh_token&client_id=a&client_secret=b&refresh_token=c",
        auth.request[0].second);

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

    ASSERT_EQ(2, auth.request.size());
    EXPECT_EQ("https://www.googleapis.com/oauth2/v3/token",
              auth.request[1].first);
    EXPECT_EQ(
        "grant_type=refresh_token&client_id=a&client_secret=b&refresh_token=c",
        auth.request[1].second);

    EXPECT_EQ(auth.time + absl::Seconds(456), result->expiration);
    EXPECT_EQ("abc", result->token);
  }
}

}  // namespace
