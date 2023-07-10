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

#include <string>

#include <gtest/gtest.h>
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/kvstore/spec.h"

namespace {

TEST(JsonAbslFlag, IntFlag) {
  // Validate that the default value can roundtrip.
  tensorstore::JsonAbslFlag<int64_t> flag = {};
  std::string default_value = AbslUnparseFlag(flag);

  std::string error;
  EXPECT_TRUE(AbslParseFlag(default_value, &flag, &error));
  EXPECT_TRUE(error.empty());
}

TEST(JsonAbslFlag, KvStoreSpecFlag) {
  // Validate that the default value can roundtrip.
  tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec> flag = {};
  std::string default_value = AbslUnparseFlag(flag);

  std::string error;
  EXPECT_TRUE(AbslParseFlag(default_value, &flag, &error))
      << "value: " << default_value;
  EXPECT_TRUE(error.empty()) << error;
}

}  // namespace
