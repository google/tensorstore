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

#include "tensorstore/kvstore/operations.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "tensorstore/kvstore/generation.h"

namespace {

using ::tensorstore::StorageGeneration;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ReadGenerationConditions;

TEST(ReadGenerationConditionsTest, Stringify) {
  ReadGenerationConditions cond;
  EXPECT_EQ("{}", absl::StrCat(cond));
  cond.if_equal = StorageGeneration::FromString("abc");
  EXPECT_EQ("{if_equal=\"abc\"}", absl::StrCat(cond));
  cond.if_not_equal = StorageGeneration::NoValue();
  EXPECT_EQ("{if_not_equal=NoValue, if_equal=\"abc\"}", absl::StrCat(cond));
}

TEST(ListEntryTest, Stringify) {
  ListEntry entry;
  entry.key = "abc";
  EXPECT_EQ("abc", absl::StrCat(entry));
}

}  // namespace
