// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/internal/testing/env_saver.h"

#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/env.h"

namespace {

using ::tensorstore::internal::GetEnv;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::UnsetEnv;
using ::tensorstore::internal_testing::EnvSaver;

TEST(EnvSaverTest, SavesAndRestoresEnvironment) {
  EnvSaver outer_saver;
  SetEnv("TENSORSTORE_TEST_ENV_EXISTING", "original_value");
  SetEnv("TENSORSTORE_TEST_ENV_TO_UNSET", "unset_me");
  UnsetEnv("TENSORSTORE_TEST_ENV_ADDED");

  {
    EnvSaver inner_saver;
    SetEnv("TENSORSTORE_TEST_ENV_EXISTING", "modified_value");
    UnsetEnv("TENSORSTORE_TEST_ENV_TO_UNSET");
    SetEnv("TENSORSTORE_TEST_ENV_ADDED", "added_value");

    EXPECT_THAT(GetEnv("TENSORSTORE_TEST_ENV_EXISTING"),
                testing::Optional(std::string("modified_value")));
    EXPECT_EQ(GetEnv("TENSORSTORE_TEST_ENV_TO_UNSET"), std::nullopt);
    EXPECT_THAT(GetEnv("TENSORSTORE_TEST_ENV_ADDED"),
                testing::Optional(std::string("added_value")));
  }

  EXPECT_THAT(GetEnv("TENSORSTORE_TEST_ENV_EXISTING"),
              testing::Optional(std::string("original_value")));
  EXPECT_THAT(GetEnv("TENSORSTORE_TEST_ENV_TO_UNSET"),
              testing::Optional(std::string("unset_me")));
  EXPECT_EQ(GetEnv("TENSORSTORE_TEST_ENV_ADDED"), std::nullopt);
}

}  // namespace
