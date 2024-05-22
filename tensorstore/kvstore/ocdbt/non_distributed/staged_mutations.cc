// Copyright 2022 The TensorStore Authors
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

#include <array>
#include <cassert>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "absl/types/compare.h"
#include "tensorstore/internal/compare.h"
#include "tensorstore/internal/container/intrusive_red_black_tree.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_ocdbt {

void AbortPendingRequestsWithError(const PendingRequests& pending,
                                   const absl::Status& error) {
  assert(!error.ok());
  if (!pending.delete_range_promise.null()) {
    pending.delete_range_promise.SetResult(error);
  }
  for (const auto& request : pending.requests) {
    auto& mutation = *request;
    if (mutation.kind_ == MutationEntry::kWrite) {
      static_cast<WriteEntry&>(mutation).promise_.SetResult(error);
    }
  }
}

auto CompareToEntry(MutationEntry& e) {
  return [&e](MutationEntry& other) {
    return internal::CompareResultAsWeakOrdering(e.key_.compare(other.key_));
  };
}

void InsertDeleteRangeEntry(StagedMutations& staged,
                            std::unique_ptr<DeleteRangeEntry> new_entry) {
  auto& range_inclusive_min = new_entry->key_;
  auto& range_exclusive_max = new_entry->exclusive_max_;

  // Find the first existing entry that intersects or is after
  // `[range_inclusive_min, range_exclusive_max)`.  We iterate forwards starting
  // from this entry to find all existing entries that intersect `range`.
  auto find_result = staged.entries.FindBound<MutationEntryTree::kLeft>(
      [&](MutationEntry& existing_entry) {
        if (existing_entry.kind_ == MutationEntry::kWrite) {
          return existing_entry.key_ < range_inclusive_min;
        } else {
          return KeyRange::CompareExclusiveMaxAndKey(
                     static_cast<DeleteRangeEntry&>(existing_entry)
                         .exclusive_max_,
                     range_inclusive_min) <= 0;
        }
      });

  staged.entries.Insert(find_result.insert_position(), *new_entry);

  for (MutationEntry *existing_entry = find_result.found_node(), *next;
       existing_entry; existing_entry = next) {
    if (KeyRange::CompareKeyAndExclusiveMax(existing_entry->key_,
                                            range_exclusive_max) >= 0) {
      break;
    }
    next =
        MutationEntryTree::Traverse(*existing_entry, MutationEntryTree::kRight);
    staged.entries.Remove(*existing_entry);
    if (existing_entry->kind_ == MutationEntry::kWrite) {
      // Add existing write entry to tree of superseded entries.
      auto* existing_write_entry = static_cast<WriteEntry*>(existing_entry);
      [[maybe_unused]] bool inserted =
          new_entry->superseded_
              .FindOrInsert(CompareToEntry(*existing_write_entry),
                            [=] { return existing_write_entry; })
              .second;
      assert(inserted);
    } else {
      auto* existing_dr_entry = static_cast<DeleteRangeEntry*>(existing_entry);
      // Merge existing delete range entry into the new entry.
      new_entry->superseded_ = WriteEntryTree::Join(
          new_entry->superseded_, existing_dr_entry->superseded_);
      if (existing_dr_entry->key_ < range_inclusive_min) {
        range_inclusive_min = std::move(existing_dr_entry->key_);
      }
      if (KeyRange::CompareExclusiveMax(existing_dr_entry->exclusive_max_,
                                        range_exclusive_max) > 0) {
        range_exclusive_max = std::move(existing_dr_entry->exclusive_max_);
      }
      delete existing_dr_entry;
    }
  }

  new_entry.release();
}

void InsertWriteEntry(StagedMutations& staged,
                      std::unique_ptr<WriteEntry> new_entry) {
  auto* entry = new_entry.release();
  // We need to insert `entry` into the map.  This may involve marking an
  // existing `WriteEntry` for the same key as superseded, or splitting an
  // existing `DeleteRangeEntry` that contains `entry->key`.

  // Find either an existing `WriteEntry` with the same key, or an existing
  // `DeleteRangeEntry` that contains `key`.
  auto find_result = staged.entries.Find([key = std::string_view(entry->key_)](
                                             MutationEntry& existing_entry) {
    auto c = key.compare(existing_entry.key_);
    if (c <= 0) return internal::CompareResultAsWeakOrdering(c);
    if (existing_entry.kind_ == MutationEntry::kWrite) {
      return absl::weak_ordering::greater;
    }
    return KeyRange::CompareKeyAndExclusiveMax(
               key,
               static_cast<DeleteRangeEntry&>(existing_entry).exclusive_max_) <
                   0
               ? absl::weak_ordering::equivalent
               : absl::weak_ordering::greater;
  });

  if (!find_result.found) {
    // No existing `WriteEntry` or `DeleteRangeEntry` covering `key` was found.
    staged.entries.Insert(find_result.insert_position(), *entry);
    return;
  }

  // Existing `WriteEntry` or `DeleteRangeEntry` covering `key` was found.
  staged.entries.Replace(*find_result.node, *entry);
  if (find_result.node->kind_ == MutationEntry::kWrite) {
    // New WriteEntry supersedes existing WriteEntry.
    entry->supersedes_ = static_cast<WriteEntry*>(find_result.node);
    return;
  }

  // `DeleteRangeEntry` contains `key`.  It needs to be split into a
  // "before" range and an "after" range.
  auto* existing_entry = static_cast<DeleteRangeEntry*>(find_result.node);
  assert(existing_entry->key_ <= entry->key_);
  assert(KeyRange::CompareKeyAndExclusiveMax(
             entry->key_, existing_entry->exclusive_max_) < 0);

  // We must split its `superseded_` tree of `WriteEntry` nodes.
  auto split_result = existing_entry->superseded_.FindSplit(
      [key = std::string_view(entry->key_)](MutationEntry& e) {
        return internal::CompareResultAsWeakOrdering(key.compare(e.key_));
      });
  if (split_result.center) {
    entry->supersedes_ = WriteEntry::Supersedes(split_result.center, 1);
  }
  if (existing_entry->key_ != entry->key_) {
    // "Left" interval is non-empty.
    auto* dr_entry = new DeleteRangeEntry;
    dr_entry->kind_ = MutationEntry::kDeleteRange;
    dr_entry->key_ = std::move(existing_entry->key_);
    dr_entry->exclusive_max_ = entry->key_;
    staged.entries.Insert({entry, MutationEntryTree::kLeft}, *dr_entry);
    dr_entry->superseded_ = std::move(split_result.trees[0]);
  } else {
    assert(split_result.trees[0].empty());
  }
  existing_entry->key_ = KeyRange::Successor(entry->key_);
  if (existing_entry->key_ != existing_entry->exclusive_max_) {
    // "Right" interval is non-empty.  Re-use the existing entry for the
    // right interval.
    staged.entries.Insert({entry, MutationEntryTree::kRight}, *existing_entry);
    existing_entry->superseded_ = std::move(split_result.trees[1]);
  } else {
    assert(split_result.trees[1].empty());
    delete existing_entry;
  }
}

void StageMutations(StagedMutations& staged, PendingRequests&& pending) {
  for (auto& request : pending.requests) {
    if (request->kind_ == MutationEntry::kWrite) {
      InsertWriteEntry(staged,
                       std::unique_ptr<WriteEntry>(
                           static_cast<WriteEntry*>(request.release())));
    } else {
      InsertDeleteRangeEntry(
          staged, std::unique_ptr<DeleteRangeEntry>(
                      static_cast<DeleteRangeEntry*>(request.release())));
    }
  }
  if (!pending.delete_range_promise.null()) {
    if (!staged.delete_range_promise.null()) {
      auto future = staged.delete_range_promise.future();
      if (!future.null()) {
        LinkResult(std::move(pending.delete_range_promise), std::move(future));
      } else {
        staged.delete_range_promise = std::move(pending.delete_range_promise);
      }
    } else {
      staged.delete_range_promise = std::move(pending.delete_range_promise);
    }
  }
}

template <typename Callback>
void ForEachWriteEntryPromise(WriteEntry* e, Callback callback) {
  do {
    callback(e->promise_);
    WriteEntry* next = e->supersedes_;
    delete e;
    e = next;
  } while (e);
}

template <typename Callback>
void ForEachWriteEntryPromise(MutationEntryTree& tree, Callback callback) {
  for (auto it = tree.begin(); it != tree.end();) {
    auto& mutation = *it;
    auto next = std::next(it);
    tree.Remove(*it);
    if (mutation.kind_ == MutationEntry::kWrite) {
      ForEachWriteEntryPromise(static_cast<WriteEntry*>(&mutation), callback);
    } else {
      auto& dr_entry = static_cast<DeleteRangeEntry&>(mutation);
      auto& superseded_writes = dr_entry.superseded_;
      for (auto superseded_it = superseded_writes.begin();
           superseded_it != superseded_writes.end();) {
        auto superseded_next = std::next(superseded_it);
        superseded_writes.Remove(*superseded_it);
        ForEachWriteEntryPromise(superseded_it.to_pointer(), callback);
        superseded_it = superseded_next;
      }
      delete &dr_entry;
    }
    it = next;
  }
}

void CommitSuccessful(StagedMutations& staged, absl::Time time) {
  if (!staged.delete_range_promise.null()) {
    staged.delete_range_promise.SetResult(absl::OkStatus());
  }
  ForEachWriteEntryPromise(staged.entries,
                           [&](Promise<TimestampedStorageGeneration>& p) {
                             auto& stamp = *p.raw_result();
                             stamp.time = time;
                             p.SetReady();
                           });
}

void CommitFailed(StagedMutations& staged, absl::Status error) {
  assert(!error.ok());
  if (!staged.delete_range_promise.null()) {
    staged.delete_range_promise.SetResult(error);
  }
  ForEachWriteEntryPromise(
      staged.entries,
      [&](Promise<TimestampedStorageGeneration>& p) { p.SetResult(error); });
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
