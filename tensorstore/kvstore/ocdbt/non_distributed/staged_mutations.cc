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
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/internal/container/intrusive_red_black_tree.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/storage_generation.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {
ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");
}

void AbortPendingRequestsWithError(const PendingRequests& pending,
                                   const absl::Status& error) {
  assert(!error.ok());
  if (!pending.delete_range_promise.null()) {
    pending.delete_range_promise.SetResult(error);
  }
  for (const auto& request : pending.requests) {
    auto& mutation = *request;
    if (mutation.kind == MutationEntry::kWrite) {
      static_cast<WriteEntry&>(mutation).promise.SetResult(error);
    }
  }
}

auto CompareToEntry(MutationEntry& e) {
  return [&e](MutationEntry& other) { return e.key.compare(other.key); };
}

void InsertDeleteRangeEntry(StagedMutations& staged,
                            std::unique_ptr<DeleteRangeEntry> new_entry) {
  auto& range_inclusive_min = new_entry->key;
  auto& range_exclusive_max = new_entry->exclusive_max;

  // Find the first existing entry that intersects or is after
  // `[range_inclusive_min, range_exclusive_max)`.  We iterate forwards starting
  // from this entry to find all existing entries that intersect `range`.
  auto find_result = staged.entries.FindBound<MutationEntryTree::kLeft>(
      [&](MutationEntry& existing_entry) {
        if (existing_entry.kind == MutationEntry::kWrite) {
          return existing_entry.key < range_inclusive_min;
        } else {
          return KeyRange::CompareExclusiveMaxAndKey(
                     static_cast<DeleteRangeEntry&>(existing_entry)
                         .exclusive_max,
                     range_inclusive_min) <= 0;
        }
      });

  staged.entries.Insert(find_result.insert_position(), *new_entry);

  for (MutationEntry *existing_entry = find_result.found_node(), *next;
       existing_entry; existing_entry = next) {
    if (KeyRange::CompareKeyAndExclusiveMax(existing_entry->key,
                                            range_exclusive_max) >= 0) {
      break;
    }
    next =
        MutationEntryTree::Traverse(*existing_entry, MutationEntryTree::kRight);
    staged.entries.Remove(*existing_entry);
    if (existing_entry->kind == MutationEntry::kWrite) {
      // Add existing write entry to tree of superseded entries.
      auto* existing_write_entry = static_cast<WriteEntry*>(existing_entry);
      [[maybe_unused]] bool inserted =
          new_entry->superseded_writes
              .FindOrInsert(CompareToEntry(*existing_write_entry),
                            [=] { return existing_write_entry; })
              .second;
      assert(inserted);
    } else {
      auto* existing_dr_entry = static_cast<DeleteRangeEntry*>(existing_entry);
      // Merge existing delete range entry into the new entry.
      new_entry->superseded_writes = WriteEntryTree::Join(
          new_entry->superseded_writes, existing_dr_entry->superseded_writes);
      if (existing_dr_entry->key < range_inclusive_min) {
        range_inclusive_min = std::move(existing_dr_entry->key);
      }
      if (KeyRange::CompareExclusiveMax(existing_dr_entry->exclusive_max,
                                        range_exclusive_max) > 0) {
        range_exclusive_max = std::move(existing_dr_entry->exclusive_max);
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
  auto find_result = staged.entries.Find([key = std::string_view(entry->key)](
                                             MutationEntry& existing_entry) {
    auto c = key.compare(existing_entry.key);
    if (c <= 0) return c;
    if (existing_entry.kind == MutationEntry::kWrite) return 1;
    return KeyRange::CompareKeyAndExclusiveMax(
               key,
               static_cast<DeleteRangeEntry&>(existing_entry).exclusive_max) < 0
               ? 0
               : 1;
  });

  if (!find_result.found) {
    // No existing `WriteEntry` or `DeleteRangeEntry` covering `key` was found.
    staged.entries.Insert(find_result.insert_position(), *entry);
    return;
  }

  // Existing `WriteEntry` or `DeleteRangeEntry` covering `key` was found.
  staged.entries.Replace(*find_result.node, *entry);
  if (find_result.node->kind == MutationEntry::kWrite) {
    // New WriteEntry supersedes existing WriteEntry.
    entry->supersedes = static_cast<WriteEntry*>(find_result.node);
    return;
  }

  // `DeleteRangeEntry` contains `key`.  It needs to be split into a
  // "before" range and an "after" range.
  auto* existing_entry = static_cast<DeleteRangeEntry*>(find_result.node);
  assert(existing_entry->key <= entry->key);
  assert(KeyRange::CompareKeyAndExclusiveMax(
             entry->key, existing_entry->exclusive_max) < 0);

  // We must split its `superseded_writes` tree of `WriteEntry` nodes.
  auto split_result = existing_entry->superseded_writes.FindSplit(
      [key = std::string_view(entry->key)](MutationEntry& e) {
        return key.compare(e.key);
      });
  if (split_result.center) {
    entry->supersedes = WriteEntry::Supersedes(split_result.center, 1);
  }
  if (existing_entry->key != entry->key) {
    // "Left" interval is non-empty.
    auto* dr_entry = new DeleteRangeEntry;
    dr_entry->kind = MutationEntry::kDeleteRange;
    dr_entry->key = std::move(existing_entry->key);
    dr_entry->exclusive_max = entry->key;
    staged.entries.Insert({entry, MutationEntryTree::kLeft}, *dr_entry);
    dr_entry->superseded_writes = std::move(split_result.trees[0]);
  } else {
    assert(split_result.trees[0].empty());
  }
  existing_entry->key = KeyRange::Successor(entry->key);
  if (existing_entry->key != existing_entry->exclusive_max) {
    // "Right" interval is non-empty.  Re-use the existing entry for the
    // right interval.
    staged.entries.Insert({entry, MutationEntryTree::kRight}, *existing_entry);
    existing_entry->superseded_writes = std::move(split_result.trees[1]);
  } else {
    assert(split_result.trees[1].empty());
    delete existing_entry;
  }
}

void StageMutations(StagedMutations& staged, PendingRequests&& pending) {
  for (auto& request : pending.requests) {
    if (request->kind == MutationEntry::kWrite) {
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
    callback(e->promise);
    WriteEntry* next = e->supersedes;
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
    if (mutation.kind == MutationEntry::kWrite) {
      ForEachWriteEntryPromise(static_cast<WriteEntry*>(&mutation), callback);
    } else {
      auto& dr_entry = static_cast<DeleteRangeEntry&>(mutation);
      auto& superseded_writes = dr_entry.superseded_writes;
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

WriteEntryTree::Range GetWriteEntryInterval(WriteEntryTree& tree,
                                            std::string_view inclusive_min,
                                            std::string_view exclusive_max) {
  WriteEntry* lower_bound =
      tree.FindBound<WriteEntryTree::kLeft>(
              [&](auto& entry) { return entry.key < inclusive_min; })
          .found_node();
  WriteEntry* upper_bound =
      exclusive_max.empty()
          ? nullptr
          : tree.FindBound<WriteEntryTree::kRight>([&](auto& entry) {
                  return KeyRange::CompareKeyAndExclusiveMax(entry.key,
                                                             exclusive_max) < 0;
                })
                .found_node();
  return WriteEntryTree::Range(lower_bound, upper_bound);
}

WriteEntryTree::Range GetWriteEntryInterval(WriteEntryTree& tree,
                                            const KeyRange& key_range) {
  return GetWriteEntryInterval(tree, key_range.inclusive_min,
                               key_range.exclusive_max);
}

void PartitionInteriorNodeMutations(
    span<const InteriorNodeEntry> existing_entries,
    std::string_view existing_key_prefix, const KeyRange& key_range,
    MutationEntryTree::Range entry_range,
    absl::FunctionRef<void(const InteriorNodeEntry& existing_child,
                           KeyRange child_key_range,
                           MutationEntryTree::Range mutation_sub_range)>
        partition_callback) {
  assert(!existing_entries.empty());

  ComparePrefixedKeyToUnprefixedKey compare_existing_and_new_keys{
      existing_key_prefix};

  // Next mutation entry to assign to a partition.
  auto entry_it = entry_range.begin();

  // Exclusive end point of current partition/child node.
  //
  // In non-leaf nodes, the keys for entries after the first specify partition
  // points.  The key for the first entry does not specify a partition point.
  auto existing_it = existing_entries.begin() + 1;

  // First mutation entry in the partition that ends at `existing_it`.
  MutationEntry* first_mutation_in_partition = entry_it.to_pointer();

  // Move to next partition/child node.  `end_mutation_in_partition` is one past
  // the last mutation assigned to the partition.
  const auto end_of_partition = [&](MutationEntry* end_mutation_in_partition) {
    if (first_mutation_in_partition != end_mutation_in_partition) {
      auto& existing_entry = *(existing_it - 1);
      KeyRange new_key_range;
      if (&existing_entry == &existing_entries.front()) {
        new_key_range.inclusive_min = key_range.inclusive_min;
      } else {
        new_key_range.inclusive_min =
            tensorstore::StrCat(existing_key_prefix, existing_entry.key);
      }
      if (existing_it == existing_entries.end()) {
        new_key_range.exclusive_max = key_range.exclusive_max;
      } else {
        new_key_range.exclusive_max =
            tensorstore::StrCat(existing_key_prefix, existing_it->key);
      }
      partition_callback(existing_entry, std::move(new_key_range),
                         MutationEntryTree::Range(first_mutation_in_partition,
                                                  end_mutation_in_partition));
      first_mutation_in_partition = entry_it.to_pointer();
    } else {
      ABSL_LOG_IF(INFO, ocdbt_logging)
          << "PartitionInteriorNodeMutations: existing child "
          << tensorstore::QuoteString(existing_key_prefix) << "+"
          << tensorstore::QuoteString((existing_it - 1)->key)
          << " has no mutations";
    }
    ++existing_it;
  };

  while (entry_it != entry_range.end()) {
    auto* mutation = &*entry_it;
    int c = existing_it != existing_entries.end()
                ? compare_existing_and_new_keys(existing_it->key, mutation->key)
                : 1;
    if (c <= 0) {
      // Current partition ends before lower bound of mutation.
      end_of_partition(entry_it.to_pointer());
      continue;
    }

    if (mutation->kind == MutationEntry::kDeleteRange) {
      auto& dr_entry = *static_cast<const DeleteRangeEntry*>(mutation);

      // Compare `dr_entry.exclusive_max` to the exclusive max of the current
      // partition.

      // Indicates the 3-way comparison between `dr_entry.exclusive_max` and the
      // exclusive max of the current partition.
      int c_max;

      if (existing_it != existing_entries.end()) {
        // Current partition is not the last partition: compare against the
        // starting key of the next partition.
        c_max = dr_entry.exclusive_max.empty()
                    ? 1
                    : -compare_existing_and_new_keys(existing_it->key,
                                                     dr_entry.exclusive_max);
      } else {
        // Current partition is the last partition.
        c_max = -1;
      }

      if (c_max < 0) {
        // `dr_entry` is contained in current partition and not the next
        // partition, and does not end the current partition.
        ++entry_it;
        continue;
      }

      if (c_max == 0) {
        // `dr_entry` ends exactly at the end of the current partition.
        ++entry_it;
        end_of_partition(entry_it.to_pointer());
        continue;
      }

      assert(c_max > 0);
      // `dr_entry` extends past the current partition.
      end_of_partition(std::next(entry_it).to_pointer());
      continue;
    }

    assert(mutation->kind == MutationEntry::kWrite);
    ++entry_it;
  }

  end_of_partition(entry_it.to_pointer());
}

std::optional<const LeafNodeValueReference*> ApplyWriteEntryChain(
    StorageGeneration existing_generation, const WriteEntry& last_write_entry) {
  std::optional<const LeafNodeValueReference*> new_value;
  absl::InlinedVector<const WriteEntry*, 8> write_entries;
  for (const WriteEntry* e = &last_write_entry;;) {
    write_entries.push_back(e);
    if (!(e = e->supersedes)) break;
  }
  for (auto entry_it = write_entries.rbegin(); entry_it != write_entries.rend();
       ++entry_it) {
    const WriteEntry* e = *entry_it;
    if (e->supersedes.tag()) {
      // Previous entry was deleted by a `DeleteRange` request before being
      // superseded.
      existing_generation = StorageGeneration::NoValue();
      new_value = nullptr;
    }
    if (StorageGeneration::EqualOrUnspecified(existing_generation,
                                              e->if_equal)) {
      // `if_equal` condition was satisfied, write will be marked as having
      // completed successfully.
      if (e->value) {
        existing_generation =
            internal_ocdbt::ComputeStorageGeneration(*e->value);
        e->promise.raw_result()->generation = existing_generation;
        new_value = &*e->value;
      } else {
        e->promise.raw_result()->generation = StorageGeneration::NoValue();
        new_value = nullptr;
        existing_generation = StorageGeneration::NoValue();
      }
    } else {
      // `if_equal` condition was not satisfied, write will be marked as having
      // failed.
      e->promise.raw_result()->generation = StorageGeneration::Unknown();
    }
  }
  return new_value;
}

span<const LeafNodeEntry>::iterator ValidateSupersededWriteEntries(
    WriteEntryTree::Range superseded_writes,
    span<const LeafNodeEntry> existing_entries,
    std::string_view existing_prefix) {
  ComparePrefixedKeyToUnprefixedKey compare_existing_and_new_keys{
      existing_prefix};

  auto existing_it = existing_entries.begin();
  auto superseded_it = superseded_writes.begin();
  while (superseded_it != superseded_writes.end() &&
         existing_it != existing_entries.end()) {
    int c = compare_existing_and_new_keys(existing_it->key, superseded_it->key);
    if (c < 0) {
      // Existing key comes before mutation.
      ++existing_it;
      continue;
    }
    if (c == 0) {
      // Existing key matches mutation key.
      ApplyWriteEntryChain(internal_ocdbt::ComputeStorageGeneration(
                               existing_it->value_reference),
                           *superseded_it);
      ++existing_it;
      ++superseded_it;
      continue;
    }
    // Existing key comes after mutation.
    ApplyWriteEntryChain(StorageGeneration::NoValue(), *superseded_it);
    ++superseded_it;
  }
  for (; superseded_it != superseded_writes.end(); ++superseded_it) {
    ApplyWriteEntryChain(StorageGeneration::NoValue(), *superseded_it);
  }
  return existing_it;
}

bool MustReadNodeToApplyMutations(const KeyRange& key_range,
                                  MutationEntryTree::Range entry_range) {
  assert(!entry_range.empty());
  if (entry_range.end() != std::next(entry_range.begin())) {
    // More than one mutation, which means a single `DeleteRangeEntry` does not
    // cover the entire `key_range`.
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "MustReadNodeToApplyMutations: more than one mutation";
    return true;
  }
  MutationEntry* mutation = entry_range.begin().to_pointer();
  if (mutation->kind != MutationEntry::kDeleteRange) {
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "MustReadNodeToApplyMutations: not delete range mutation";
    return true;
  }

  // `entry_range` consists of a single `DeleteRangeEntry`.
  auto& dr_entry = *static_cast<DeleteRangeEntry*>(mutation);
  if (dr_entry.key > key_range.inclusive_min ||
      KeyRange::CompareExclusiveMax(dr_entry.exclusive_max,
                                    key_range.exclusive_max) < 0) {
    // `DeleteRangeEntry` does not cover the entire key space.
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "MustReadNodeToApplyMutations: does not cover entire key space: "
           "dr_entry.key="
        << tensorstore::QuoteString(dr_entry.key) << ", dr_entry.exclusive_max="
        << tensorstore::QuoteString(dr_entry.exclusive_max)
        << ", key_range.exclusive_max="
        << tensorstore::QuoteString(key_range.exclusive_max);
    return true;
  }

  // `DeleteRangeEntry` covers the entire key space.

  auto writes =
      GetWriteEntryInterval(dr_entry.superseded_writes, key_range.inclusive_min,
                            key_range.exclusive_max);
  if (!writes.empty()) {
    // There are superseded writes that need to be validated, in order to be
    // able to correctly indicate in the response whether the write "succeeded"
    // (before being overwritten) or was aborted due to a generation mismatch.
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "MustReadNodeToApplyMutations: superseded writes";
    return true;
  }

  // No superseded writes need to be validated.
  return false;
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
