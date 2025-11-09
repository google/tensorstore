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

#include "tensorstore/internal/aws/aws_api.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <aws/common/logging.h>

using ::tensorstore::internal_aws::GetAwsAllocator;
using ::tensorstore::internal_aws::GetAwsClientBootstrap;
using ::tensorstore::internal_aws::GetAwsTlsCtx;

namespace {

TEST(AwsApiTest, Basic) {
  // This does not validate anything; it just initializes the AWS API and
  // verifies that it doesn't crash, and then leaks the library setup.
  EXPECT_THAT(GetAwsAllocator(), ::testing::NotNull());
  EXPECT_THAT(GetAwsClientBootstrap(), ::testing::NotNull());
  EXPECT_THAT(GetAwsTlsCtx(), ::testing::NotNull());

  AWS_LOGF_INFO(AWS_LS_COMMON_GENERAL, "info log call");
  AWS_LOGF_WARN(AWS_LS_COMMON_GENERAL, "warn log call");
}

}  // namespace
