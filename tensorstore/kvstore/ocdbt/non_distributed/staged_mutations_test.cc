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

#include "tensorstore/kvstore/ocdbt/non_distributed/staged_mutations.h"

#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/compare.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/future.h"

namespace {

using ::tensorstore::internal_ocdbt::DeleteRangeEntry;
using ::tensorstore::internal_ocdbt::MutationEntry;
using ::tensorstore::internal_ocdbt::MutationEntryUniquePtr;
using ::tensorstore::internal_ocdbt::PendingRequests;
using ::tensorstore::internal_ocdbt::StagedMutations;
using ::tensorstore::internal_ocdbt::StageMutations;
using ::tensorstore::internal_ocdbt::WriteEntry;

TEST(StagedMutationsTest, WriteThenDeleteRange) {
  StagedMutations staged;

  {
    PendingRequests pending;
    // 1. Stage a Write for "b"
    auto entry = std::make_unique<WriteEntry>();
    entry->kind_ = MutationEntry::kWrite;
    entry->key_ = "b";
    auto [promise1, future1] = tensorstore::PromiseFuturePair<
        tensorstore::TimestampedStorageGeneration>::Make(std::in_place);
    entry->promise_ = std::move(promise1);
    pending.requests.push_back(MutationEntryUniquePtr(entry.release()));

    // 2. Stage a Write for "d"
    entry = std::make_unique<WriteEntry>();
    entry->kind_ = MutationEntry::kWrite;
    entry->key_ = "d";
    auto [promise2, future2] = tensorstore::PromiseFuturePair<
        tensorstore::TimestampedStorageGeneration>::Make(std::in_place);
    entry->promise_ = std::move(promise2);
    pending.requests.push_back(MutationEntryUniquePtr(entry.release()));
    StageMutations(staged, std::move(pending));
  }

  EXPECT_EQ(std::distance(staged.entries.begin(), staged.entries.end()), 2);

  // 3. Stage a DeleteRange for ["a", "c")
  {
    PendingRequests pending;
    auto entry = std::make_unique<DeleteRangeEntry>();
    entry->kind_ = MutationEntry::kDeleteRange;
    entry->key_ = "a";
    entry->exclusive_max_ = "c";
    pending.requests.push_back(MutationEntryUniquePtr(entry.release()));
    StageMutations(staged, std::move(pending));
  }

  // Verify that "b" is superseded and "d" remains.
  EXPECT_EQ(std::distance(staged.entries.begin(), staged.entries.end()), 2);

  auto it = staged.entries.begin();
  ASSERT_NE(it, staged.entries.end());
  EXPECT_EQ(it->kind_, MutationEntry::kDeleteRange);
  auto* dr_entry = static_cast<DeleteRangeEntry*>(&*it);
  EXPECT_EQ(dr_entry->key_, "a");
  EXPECT_EQ(dr_entry->exclusive_max_, "c");

  auto superseded_it = dr_entry->superseded_.begin();
  ASSERT_NE(superseded_it, dr_entry->superseded_.end());
  EXPECT_EQ(superseded_it->key_, "b");

  ++it;
  ASSERT_NE(it, staged.entries.end());
  EXPECT_EQ(it->kind_, MutationEntry::kWrite);
  EXPECT_EQ(it->key_, "d");

  // Clean up staged mutations to avoid memory leaks on test exit
  CommitFailed(staged, absl::CancelledError("Clean up"));
}

TEST(StagedMutationsTest, OverlappingDeleteRanges) {
  StagedMutations staged;

  // 1. Stage a DeleteRange for ["a", "c")
  {
    PendingRequests pending;
    auto entry = std::make_unique<DeleteRangeEntry>();
    entry->kind_ = MutationEntry::kDeleteRange;
    entry->key_ = "a";
    entry->exclusive_max_ = "c";
    pending.requests.push_back(MutationEntryUniquePtr(entry.release()));
    StageMutations(staged, std::move(pending));
  }

  ASSERT_EQ(std::distance(staged.entries.begin(), staged.entries.end()), 1);

  // 2. Stage a DeleteRange for ["b", "d")
  {
    PendingRequests pending;
    auto entry = std::make_unique<DeleteRangeEntry>();
    entry->kind_ = MutationEntry::kDeleteRange;
    entry->key_ = "b";
    entry->exclusive_max_ = "d";
    pending.requests.push_back(MutationEntryUniquePtr(entry.release()));
    StageMutations(staged, std::move(pending));
  }

  // Verify that the merged entry can be found.
  auto* root = staged.entries.root();
  ASSERT_NE(root, nullptr);
  EXPECT_EQ(std::distance(staged.entries.begin(), staged.entries.end()), 1);
  EXPECT_EQ(root->key_, "a");
  EXPECT_EQ(static_cast<DeleteRangeEntry*>(root)->exclusive_max_, "d");

  // Find "a".
  auto found = staged.entries.Find([&](MutationEntry& entry) {
    return entry.key_ < "a"    ? absl::weak_ordering::less
           : entry.key_ == "a" ? absl::weak_ordering::equivalent
                               : absl::weak_ordering::greater;
  });
  EXPECT_TRUE(found.found);

  // Clean up staged mutations to avoid memory leaks on test exit
  CommitFailed(staged, absl::CancelledError("Clean up"));
}

}  // namespace
