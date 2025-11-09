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

#include "tensorstore/internal/env.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal::GetEnv;
using ::tensorstore::internal::GetEnvironmentMap;
using ::tensorstore::internal::GetEnvValue;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::UnsetEnv;

TEST(GetEnvTest, Basic) {
  // Env is set
  SetEnv("TENSORSTORE_TEST_ENV_VAR", "test env var");
  {
    auto var = GetEnv("TENSORSTORE_TEST_ENV_VAR");
    EXPECT_TRUE(var);
    EXPECT_EQ("test env var", *var);
  }

  // Env is not set
  UnsetEnv("TENSORSTORE_TEST_ENV_VAR");
  {
    auto var = GetEnv("TENSORSTORE_TEST_ENV_VAR");
    EXPECT_FALSE(var);
  }
}

TEST(GetEnvTest, GetEnvironmentMap) {
  // Env is set
  SetEnv("TENSORSTORE_TEST_ENV_VAR", "test env var");

  auto allenv = GetEnvironmentMap();
  EXPECT_FALSE(allenv.empty());
  EXPECT_THAT(allenv.count("TENSORSTORE_TEST_ENV_VAR"), 1);
}

TEST(GetEnvTest, ParseBool) {
  // Env is set
  SetEnv("TENSORSTORE_TEST_ENV_VAR", "trUe");
  {
    EXPECT_THAT(GetEnvValue<bool>("TENSORSTORE_TEST_ENV_VAR"),
                testing::Optional(true));
  }

  // Env is not set
  UnsetEnv("TENSORSTORE_TEST_ENV_VAR");
  {
    auto var = GetEnvValue<bool>("TENSORSTORE_TEST_ENV_VAR");
    EXPECT_FALSE(var);
  }
}

TEST(GetEnvTest, ParseInt) {
  // Env is set
  SetEnv("TENSORSTORE_TEST_ENV_VAR", "123");
  {
    EXPECT_THAT(GetEnvValue<int>("TENSORSTORE_TEST_ENV_VAR"),
                testing::Optional(123));
  }

  // Env is not set
  UnsetEnv("TENSORSTORE_TEST_ENV_VAR");
  {
    auto var = GetEnvValue<int>("TENSORSTORE_TEST_ENV_VAR");
    EXPECT_FALSE(var);
  }
}

}  // namespace
