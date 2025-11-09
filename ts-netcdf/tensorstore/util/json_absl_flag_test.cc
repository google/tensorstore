// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/util/json_absl_flag.h"

#include <stdint.h>

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/json_binding/json_binding.h"  // IWYU pragma: keep
#include "tensorstore/kvstore/spec.h"

namespace {

TEST(JsonAbslFlag, IntFlag) {
  // Validate that the default value can roundtrip.
  tensorstore::JsonAbslFlag<int64_t> flag = {};
  EXPECT_THAT(AbslUnparseFlag(flag), ::testing::StrEq("0"));

  std::string error;
  EXPECT_TRUE(AbslParseFlag("", &flag, &error));
  EXPECT_TRUE(error.empty());

  EXPECT_TRUE(AbslParseFlag("1", &flag, &error));
  EXPECT_TRUE(error.empty()) << error;
}

TEST(JsonAbslFlag, KvStoreSpecFlag) {
  // Validate that the default value can roundtrip.
  tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec> flag = {};
  EXPECT_THAT(AbslUnparseFlag(flag), ::testing::IsEmpty());

  // Try to parse as an empty value JSON string.
  std::string error;
  EXPECT_TRUE(AbslParseFlag("", &flag, &error));
  EXPECT_TRUE(error.empty()) << error;

  EXPECT_TRUE(AbslParseFlag("  ", &flag, &error));
  EXPECT_TRUE(error.empty()) << error;

  // Try to parse a bad json object.
  error.clear();
  EXPECT_FALSE(AbslParseFlag("{ \"driver\": \"memory\" ", &flag, &error));
  EXPECT_THAT(error, testing::HasSubstr("Failed to parse JSON"));

  // Try to parse as a json object.
  error.clear();
  EXPECT_FALSE(AbslParseFlag("{ \"driver\": \"memory\" }", &flag, &error));
  EXPECT_THAT(error, testing::HasSubstr("Failed to parse or bind JSON"));

  // Try to parse as a raw string.
  error.clear();
  EXPECT_FALSE(AbslParseFlag("memory://", &flag, &error));
  EXPECT_THAT(error, testing::HasSubstr("Failed to parse or bind JSON"));
}

}  // namespace
