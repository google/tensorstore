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

#include <optional>

#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal::GetEnv;
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

}  // namespace
