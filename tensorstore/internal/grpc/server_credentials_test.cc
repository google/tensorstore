// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/grpc/server_credentials.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "grpcpp/security/server_credentials.h"  // third_party
#include "tensorstore/context.h"
#include "tensorstore/util/result.h"

namespace {

using ::tensorstore::GrpcServerCredentials;
using ::testing::Eq;
using ::testing::Ne;

TEST(GrpcServerCredentials, Use) {
  auto use = grpc::experimental::LocalServerCredentials(LOCAL_TCP);
  auto ctx = tensorstore::Context::Default();

  EXPECT_TRUE(GrpcServerCredentials::Use(ctx, use));
  auto a = ctx.GetResource<GrpcServerCredentials>()
               .value()
               ->GetAuthenticationStrategy();
  EXPECT_THAT(a->GetServerCredentials().get(), Eq(use.get()));
}

TEST(GrpcServerCredentials, Default) {
  auto ctx = tensorstore::Context::Default();
  auto a = ctx.GetResource<GrpcServerCredentials>()
               .value()
               ->GetAuthenticationStrategy();
  auto b = ctx.GetResource<GrpcServerCredentials>()
               .value()
               ->GetAuthenticationStrategy();
  EXPECT_THAT(a.get(), Ne(b.get()));
}

}  // namespace
