// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/internal/grpc/client_credentials.h"

#include <memory>

#include <gtest/gtest.h>
#include "grpcpp/security/credentials.h"  // third_party
#include "tensorstore/context.h"
#include "tensorstore/util/result.h"

namespace {

using ::tensorstore::GrpcClientCredentials;

TEST(GrpcClientCredentials, Use) {
  auto use = grpc::experimental::LocalCredentials(LOCAL_TCP);
  auto ctx = tensorstore::Context::Default();

  EXPECT_TRUE(GrpcClientCredentials::Use(ctx, use));
  auto a = ctx.GetResource<GrpcClientCredentials>().value()->GetCredentials();
  EXPECT_EQ(a.get(), use.get());
}

TEST(GrpcClientCredentials, Default) {
  auto ctx = tensorstore::Context::Default();
  auto a = ctx.GetResource<GrpcClientCredentials>().value()->GetCredentials();
  auto b = ctx.GetResource<GrpcClientCredentials>().value()->GetCredentials();
  EXPECT_NE(a.get(), b.get());
}

}  // namespace
