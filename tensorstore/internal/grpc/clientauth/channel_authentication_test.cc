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

#include "tensorstore/internal/grpc/clientauth/channel_authentication.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::internal_grpc::CreateInsecureAuthenticationStrategy;
using ::tensorstore::internal_grpc::GrpcChannelCredentialsAuthentication;
using ::testing::IsNull;
using ::testing::NotNull;

namespace {

TEST(GrpcChannelCredentialsAuthenticationTest, Basic) {
  GrpcChannelCredentialsAuthentication auth(grpc::InsecureChannelCredentials());

  grpc::ChannelArguments args;
  auto creds = auth.GetChannelCredentials("localhost:1", args);
  EXPECT_THAT(creds.get(), NotNull());

  for (auto _ : {1, 2, 3}) {
    auto context = std::make_shared<grpc::ClientContext>();
    EXPECT_THAT(context->credentials(), IsNull());
    auto configured = auth.ConfigureContext(context).result();
    EXPECT_THAT(configured.status(), tensorstore::IsOk());
    EXPECT_THAT(configured.value()->credentials(), IsNull());
  }
}

TEST(GrpcChannelCredentialsAuthenticationTest, Insecure) {
  auto auth = CreateInsecureAuthenticationStrategy();

  grpc::ChannelArguments args;
  auto creds = auth->GetChannelCredentials("localhost:1", args);
  EXPECT_THAT(creds.get(), NotNull());
}

}  // namespace
