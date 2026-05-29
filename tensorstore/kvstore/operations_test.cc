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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::StatusIs;
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

TEST(KvStoreOperationsTest, ReadInvalid) {
  tensorstore::kvstore::KvStore store;
  EXPECT_THAT(
      tensorstore::kvstore::Read(store, "key").result(),
      StatusIs(absl::StatusCode::kInvalidArgument, "KvStore is not valid"));
}

TEST(KvStoreOperationsTest, WriteInvalid) {
  tensorstore::kvstore::KvStore store;
  EXPECT_THAT(
      tensorstore::kvstore::Write(store, "key", absl::Cord("value")).result(),
      StatusIs(absl::StatusCode::kInvalidArgument, "KvStore is not valid"));
}

TEST(KvStoreOperationsTest, DeleteRangeInvalid) {
  tensorstore::kvstore::KvStore store;
  EXPECT_THAT(
      tensorstore::kvstore::DeleteRange(store, {}).result(),
      StatusIs(absl::StatusCode::kInvalidArgument, "KvStore is not valid"));
}

TEST(KvStoreOperationsTest, ListInvalid) {
  tensorstore::kvstore::KvStore store;
  EXPECT_THAT(
      tensorstore::kvstore::ListFuture(store, {}).result(),
      StatusIs(absl::StatusCode::kInvalidArgument, "KvStore is not valid"));
}

TEST(KvStoreOperationsTest, CopyRangeInvalidSource) {
  tensorstore::kvstore::KvStore store;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto valid_store, tensorstore::kvstore::Open("memory://").result());
  EXPECT_THAT(
      tensorstore::kvstore::ExperimentalCopyRange(store, valid_store).result(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Source KvStore is not valid"));
}

TEST(KvStoreOperationsTest, CopyRangeInvalidTarget) {
  tensorstore::kvstore::KvStore store;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto valid_store, tensorstore::kvstore::Open("memory://").result());
  EXPECT_THAT(
      tensorstore::kvstore::ExperimentalCopyRange(valid_store, store).result(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Target KvStore is not valid"));
}

}  // namespace
