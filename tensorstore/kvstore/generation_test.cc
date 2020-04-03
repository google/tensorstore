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

#include "tensorstore/kvstore/generation.h"

#include <gtest/gtest.h>

namespace {

TEST(StorageGenerationTest, Basic) {
  EXPECT_TRUE(tensorstore::StorageGeneration::IsUnknown(
      tensorstore::StorageGeneration::Unknown()));
  EXPECT_FALSE(tensorstore::StorageGeneration::IsUnknown(
      tensorstore::StorageGeneration::NoValue()));

  EXPECT_FALSE(tensorstore::StorageGeneration::IsNoValue(
      tensorstore::StorageGeneration::Unknown()));
  EXPECT_TRUE(tensorstore::StorageGeneration::IsNoValue(
      tensorstore::StorageGeneration::NoValue()));
}

}  // namespace
