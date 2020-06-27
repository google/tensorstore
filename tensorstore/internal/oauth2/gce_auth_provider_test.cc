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

#include "tensorstore/internal/oauth2/gce_auth_provider.h"

#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/time/clock.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

using tensorstore::Result;
using tensorstore::internal_http::HttpResponse;
using tensorstore::internal_oauth2::GceAuthProvider;

namespace {

const char kServiceAccountInfo[] = R"(
{
  "email": "nobody@nowhere.com",
  "scopes": [ "abc", "xyz" ]
}
)";

const char kOAuthResponse[] = R"(
{
  "token_type" : "refresh",
  "access_token": "abc",
  "expires_in": 456
}
)";

class TestAuthProvider : public GceAuthProvider {
 public:
  TestAuthProvider()
      : GceAuthProvider(nullptr, [this] { return this->time; }),
        time(absl::Now()),
        idx(0) {}

  virtual Result<HttpResponse> IssueRequest(std::string path, bool recursive) {
    request.emplace_back(std::move(path));
    if (responses.count(idx) != 0) {
      return responses[idx++];
    }
    return HttpResponse{};
  }

  absl::Time time;
  int idx;
  absl::flat_hash_map<int, HttpResponse> responses;
  std::vector<std::string> request;
};

TEST(GceAuthProviderTest, InitialState) {
  TestAuthProvider auth;
  EXPECT_FALSE(auth.IsValid());
  EXPECT_TRUE(auth.IsExpired());
}

TEST(GceAuthProviderTest, Status200) {
  TestAuthProvider auth;
  auth.responses = {
      {0,
       {200,
        absl::Cord(kServiceAccountInfo),
        {}}},                                      // RetrieveServiceAccountInfo
      {1, {200, absl::Cord(kOAuthResponse), {}}},  // OAuth request
      {2,
       {200,
        absl::Cord(kServiceAccountInfo),
        {}}},                                      // RetrieveServiceAccountInfo
      {3, {200, absl::Cord(kOAuthResponse), {}}},  // OAuth request
  };

  EXPECT_FALSE(auth.IsValid());

  {
    auto result = auth.GetToken();
    EXPECT_EQ(2, auth.idx);
    EXPECT_TRUE(result.ok()) << result.status();

    EXPECT_EQ(auth.time + absl::Seconds(456), result->expiration);
    EXPECT_EQ("abc", result->token);
  }

  EXPECT_FALSE(auth.IsExpired());
  EXPECT_TRUE(auth.IsValid());

  // time passes.
  auth.time += absl::Seconds(600);
  {
    auto result = auth.GetToken();
    EXPECT_EQ(4, auth.idx);
    EXPECT_TRUE(result.ok()) << result.status();

    EXPECT_EQ(auth.time + absl::Seconds(456), result->expiration);
    EXPECT_EQ("abc", result->token);
  }
}

TEST(GceAuthProviderTest, NoResponse) {
  TestAuthProvider auth;

  auto result = auth.GetToken();
  EXPECT_FALSE(result.ok()) << result.status();

  ASSERT_EQ(1, auth.request.size());
  EXPECT_EQ("/computeMetadata/v1/instance/service-accounts/default/",
            auth.request[0]);
}

TEST(GceAuthProviderTest, Status400) {
  TestAuthProvider auth;
  auth.responses = {
      {0,
       {400,
        absl::Cord(kServiceAccountInfo),
        {}}},  // RetrieveServiceAccountInfo
  };

  auto result = auth.GetToken();
  EXPECT_FALSE(result.ok()) << result.status();
}

TEST(GceAuthProviderTest, Status400OnSecondCall) {
  TestAuthProvider auth;
  auth.responses = {
      {0,
       {200,
        absl::Cord(kServiceAccountInfo),
        {}}},                                      // RetrieveServiceAccountInfo
      {1, {400, absl::Cord(kOAuthResponse), {}}},  // OAuth request
  };

  auto result = auth.GetToken();
  EXPECT_EQ(2, auth.idx);
  EXPECT_FALSE(result.ok()) << result.status();
}

TEST(GceAuthProviderTest, Hostname) {
  // GCE_METADATA_ROOT overrides the default GCE metata hostname.
  EXPECT_EQ("metadata.google.internal",
            tensorstore::internal_oauth2::GceMetadataHostname());

  tensorstore::internal::SetEnv("GCE_METADATA_ROOT", "localhost");
  EXPECT_EQ("localhost", tensorstore::internal_oauth2::GceMetadataHostname());
  tensorstore::internal::UnsetEnv("GCE_METADATA_ROOT");
}

}  // namespace
