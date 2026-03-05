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

#include "tensorstore/array_storage_statistics.h"

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"

namespace {

using ::tensorstore::ArrayStorageStatistics;

TEST(ArrayStorageStatisticsTest, AbslStringify) {
  EXPECT_EQ("{not_stored=<unknown>, fully_stored=<unknown>}",
            absl::StrCat(ArrayStorageStatistics{}));

  EXPECT_EQ("{not_stored=true, fully_stored=<unknown>}",
            absl::StrCat(ArrayStorageStatistics{
                ArrayStorageStatistics::query_not_stored, true}));

  EXPECT_EQ("{not_stored=false, fully_stored=<unknown>}",
            absl::StrCat(ArrayStorageStatistics{
                ArrayStorageStatistics::query_not_stored, false}));

  EXPECT_EQ("{not_stored=<unknown>, fully_stored=true}",
            absl::StrCat(ArrayStorageStatistics{
                ArrayStorageStatistics::query_fully_stored, false, true}));

  EXPECT_EQ("{not_stored=<unknown>, fully_stored=false}",
            absl::StrCat(ArrayStorageStatistics{
                ArrayStorageStatistics::query_fully_stored, false, false}));

  EXPECT_EQ("{not_stored=true, fully_stored=false}",
            absl::StrCat(ArrayStorageStatistics{
                ArrayStorageStatistics::query_not_stored |
                    ArrayStorageStatistics::query_fully_stored,
                true, false}));
}

}  // namespace
