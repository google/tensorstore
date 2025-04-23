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

#include "tensorstore/kvstore/transaction.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <iterator>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/btree_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/compare.h"
#include "tensorstore/internal/compare.h"
#include "tensorstore/internal/container/intrusive_red_black_tree.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_modify_write.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/flow_sender_operation_state.h"
#include "tensorstore/util/execution/future_sender.h"  // IWYU pragma: keep
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_kvstore {

namespace {

auto& kvstore_transaction_retries = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/transaction_retries",
    internal_metrics::MetricMetadata("Count of kvstore transaction retries"));

template <typename Controller>
void ReportWritebackError(Controller controller, std::string_view action,
                          const absl::Status& error,
                          SourceLocation loc = SourceLocation::current()) {
  controller.Error(kvstore::Driver::AnnotateErrorWithKeyDescription(
      controller.DescribeKey(controller.GetKey()), action, error, loc));
}

template <typename Controller>
void PerformWriteback(Driver* driver, Controller controller,
                      ReadResult read_result) {
  if (!StorageGeneration::IsDirty(read_result.stamp.generation)) {
    if (!StorageGeneration::IsConditional(read_result.stamp.generation) ||
        read_result.stamp.time > controller.GetTransactionNode()
                                     .transaction()
                                     ->commit_start_time()) {
      // The read was not conditional, or the read timestamp is after the
      // transaction commit timestamp.
      controller.Success(read_result.stamp, read_result.stamp.generation);
      return;
    }
    // This is a conditional read or stale read; but not a dirty read, so
    // reissue the read.
    ReadOptions read_options;
    StorageGeneration orig_generation = std::move(read_result.stamp.generation);
    auto if_not_equal = StorageGeneration::Clean(orig_generation);
    read_options.generation_conditions.if_not_equal = if_not_equal;
    read_options.byte_range = OptionalByteRangeRequest::Stat();
    auto future = driver->Read(controller.GetKey(), std::move(read_options));
    future.Force();
    std::move(future).ExecuteWhenReady(
        [controller, if_not_equal = std::move(if_not_equal),
         orig_generation = std::move(orig_generation)](
            ReadyFuture<ReadResult> future) mutable {
          auto& r = future.result();
          if (!r.ok()) {
            ReportWritebackError(controller, "reading", r.status());
          } else if (r->aborted() || r->stamp.generation == if_not_equal) {
            controller.Success(std::move(r->stamp), orig_generation);
          } else {
            controller.Retry(r->stamp.time);
          }
        });
    return;
  }

  // This is a dirty entry, so attempt a conditional write if the generation
  // matches.
  WriteOptions write_options;
  assert(!read_result.aborted());
  StorageGeneration orig_generation = std::move(read_result.stamp.generation);
  write_options.generation_conditions.if_equal =
      StorageGeneration::Clean(orig_generation);
  auto future = driver->Write(controller.GetKey(),
                              std::move(read_result).optional_value(),
                              std::move(write_options));
  future.Force();
  std::move(future).ExecuteWhenReady(
      [controller, orig_generation = std::move(orig_generation)](
          ReadyFuture<TimestampedStorageGeneration> future) mutable {
        auto& r = future.result();
        if (!r.ok()) {
          ReportWritebackError(controller, "writing", r.status());
        } else if (StorageGeneration::IsUnknown(r->generation)) {
          controller.Retry(r->time);
        } else {
          controller.Success(std::move(*r), orig_generation);
        }
      });
}

void StartWriteback(ReadModifyWriteEntry& entry,
                    absl::Time staleness_bound = absl::InfinitePast());

void DeletedEntryDone(DeleteRangeEntry& dr_entry, bool error, size_t count = 1);

void EntryDone(SinglePhaseMutation& single_phase_mutation, bool error,
               size_t count = 1);

/// Checks the data structure invariants of an entry as well as entries it
/// supersedes.
///
/// This is for debugging/testing and only used if
/// `TENSORSTORE_INTERNAL_KVSTORETORE_TRANSACTION_DEBUG` is defined.
[[maybe_unused]] void CheckInvariants(ReadModifyWriteEntry* entry) {
  do {
    assert(!(entry->flags_.load(std::memory_order_relaxed) &
             ReadModifyWriteEntry::kDeleted));
    if (entry->prev_) {
      assert(entry->prev_->single_phase_mutation().phase_number_ <=
             entry->single_phase_mutation().phase_number_);
      assert(entry->prev_->key_ == entry->key_);
    }
    entry = entry->prev_;
  } while (entry);
}

/// Checks the invariants of all entries in all phases.
///
/// This is for debugging/testing and only used if
/// `TENSORSTORE_INTERNAL_KVSTORETORE_TRANSACTION_DEBUG` is defined.
[[maybe_unused]] void CheckInvariants(MultiPhaseMutation& multi_phase,
                                      bool commit_started) {
  absl::btree_map<size_t, size_t> phase_entry_count;
  for (auto* single_phase_mutation = &multi_phase.phases_;;) {
    if (single_phase_mutation != &multi_phase.phases_) {
      assert(single_phase_mutation->phase_number_ >
             single_phase_mutation->prev_->phase_number_);
    }
    for (MutationEntry *
             tree_entry = single_phase_mutation->entries_.begin().to_pointer(),
            *tree_next;
         tree_entry; tree_entry = tree_next) {
      ++phase_entry_count[tree_entry->single_phase_mutation().phase_number_];
      if (commit_started) {
        assert(&tree_entry->single_phase_mutation() == single_phase_mutation);
      } else {
        assert(&tree_entry->single_phase_mutation() == single_phase_mutation ||
               single_phase_mutation == multi_phase.phases_.prev_);
      }
      tree_next =
          MutationEntryTree::Traverse(*tree_entry, MutationEntryTree::kRight);
      if (tree_next) {
        assert(tree_next->key_ > tree_entry->key_);
        if (tree_entry->entry_type() != kReadModifyWrite) {
          [[maybe_unused]] auto* dr_entry =
              static_cast<DeleteRangeEntry*>(tree_entry);
          assert(KeyRange::CompareExclusiveMaxAndKey(dr_entry->exclusive_max_,
                                                     tree_next->key_) <= 0);
        }
      }
      if (tree_entry->entry_type() == kReadModifyWrite) {
        auto* rmw_entry = static_cast<ReadModifyWriteEntry*>(tree_entry);
        CheckInvariants(rmw_entry);
      } else {
        auto* dr_entry = static_cast<DeleteRangeEntry*>(tree_entry);
        if (dr_entry->entry_type() == kDeleteRangePlaceholder) {
          // This is a placeholder entry only generated from splitting a
          // prior-phase `DeleteRangeEntry`.
          --phase_entry_count[tree_entry->single_phase_mutation()
                                  .phase_number_];
          assert(dr_entry->superseded_.empty());
        }
        assert(KeyRange::CompareKeyAndExclusiveMax(
                   dr_entry->key_, dr_entry->exclusive_max_) < 0);
        for (ReadModifyWriteEntryTree::iterator
                 entry = dr_entry->superseded_.begin(),
                 next;
             entry != dr_entry->superseded_.end(); entry = next) {
          next = std::next(entry);
          if (next) {
            assert(next->key_ > entry->key_);
          }
          assert(entry->entry_type() == kReadModifyWrite);
          assert(&entry->single_phase_mutation() ==
                 &dr_entry->single_phase_mutation());
          assert(entry->key_ >= dr_entry->key_);
          assert(KeyRange::CompareKeyAndExclusiveMax(
                     entry->key_, dr_entry->exclusive_max_) < 0);
          assert(entry->flags_.load(std::memory_order_relaxed) &
                 ReadModifyWriteEntry::kDeleted);
          if (entry->prev_) {
            CheckInvariants(entry->prev_);
          }
        }
      }
    }
    single_phase_mutation = single_phase_mutation->next_;
    if (single_phase_mutation == &multi_phase.phases_) break;
  }

  for (auto* single_phase_mutation = &multi_phase.phases_;
       single_phase_mutation->next_ != &multi_phase.phases_;
       single_phase_mutation = single_phase_mutation->next_) {
    if (single_phase_mutation->phase_number_ <
        multi_phase.GetTransactionNode().phase()) {
      assert(single_phase_mutation->entries_.empty());
    }
  }
}

#ifdef TENSORSTORE_INTERNAL_KVSTORETORE_TRANSACTION_DEBUG
inline void DebugCheckInvariants(MultiPhaseMutation& multi_phase,
                                 bool commit_started) {
  CheckInvariants(multi_phase, commit_started);
}
class DebugCheckInvariantsInDestructor {
 public:
  explicit DebugCheckInvariantsInDestructor(MultiPhaseMutation& multi_phase,
                                            bool commit_started)
      : multi_phase_(multi_phase), commit_started_(commit_started) {}

  ~DebugCheckInvariantsInDestructor() {
    CheckInvariants(multi_phase_, commit_started_);
  }

 private:
  MultiPhaseMutation& multi_phase_;
  bool commit_started_;
};
#else
inline void DebugCheckInvariants(MultiPhaseMutation& multi_phase,
                                 bool commit_started) {}
class DebugCheckInvariantsInDestructor {
 public:
  explicit DebugCheckInvariantsInDestructor(MultiPhaseMutation& multi_phase,
                                            bool commit_started) {}
};
#endif

void DestroyReadModifyWriteSequence(ReadModifyWriteEntry* entry) {
  if (auto* next_rmw = entry->next_read_modify_write()) {
    next_rmw->prev_ = nullptr;
  }
  auto& multi_phase = entry->multi_phase();
  while (true) {
    auto* prev = entry->prev_;
    multi_phase.FreeReadModifyWriteEntry(entry);
    if (!prev) break;
    entry = prev;
  }
}

auto CompareToEntry(MutationEntry& e) {
  return [&e](MutationEntry& other) {
    return internal::CompareResultAsWeakOrdering(e.key_.compare(other.key_));
  };
}

void InsertIntoPriorPhase(MutationEntry* entry) {
  if (entry->entry_type() == kDeleteRangePlaceholder) {
    // `entry` is a placeholder entry due to a split of a DeleteRangeEntry
    // already added to the prior phase.
    delete static_cast<DeleteRangeEntry*>(entry);
    return;
  }
  entry->single_phase_mutation().entries_.FindOrInsert(
      CompareToEntry(*entry), [entry] { return entry; });
}

DeleteRangeEntry* MakeDeleteRangeEntry(
    MutationEntryType entry_type,
    SinglePhaseMutation& assigned_single_phase_mutation, KeyRange&& range) {
  auto* entry = new DeleteRangeEntry;
  entry->key_ = std::move(range.inclusive_min);
  entry->exclusive_max_ = std::move(range.exclusive_max);
  entry->single_phase_mutation_ = {&assigned_single_phase_mutation,
                                   static_cast<uintptr_t>(entry_type)};
  return entry;
}

DeleteRangeEntry* InsertDeleteRangeEntry(
    MutationEntryType entry_type,
    SinglePhaseMutation& insert_single_phase_mutation,
    SinglePhaseMutation& assigned_single_phase_mutation, KeyRange&& range,
    MutationEntryTree::InsertPosition position) {
  assert(entry_type == kDeleteRange || entry_type == kDeleteRangePlaceholder);
  auto* entry = MakeDeleteRangeEntry(entry_type, assigned_single_phase_mutation,
                                     std::move(range));
  insert_single_phase_mutation.entries_.Insert(position, *entry);
  return entry;
}

ReadModifyWriteEntry* MakeReadModifyWriteEntry(
    SinglePhaseMutation& assigned_single_phase_mutation, std::string&& key) {
  auto* entry = assigned_single_phase_mutation.multi_phase_
                    ->AllocateReadModifyWriteEntry();
  entry->key_ = std::move(key);
  entry->single_phase_mutation_ = {&assigned_single_phase_mutation, 0};
  return entry;
}

SinglePhaseMutation& GetCurrentSinglePhaseMutation(
    MultiPhaseMutation& multi_phase) {
  size_t phase = multi_phase.GetTransactionNode().transaction()->phase();
  SinglePhaseMutation* single_phase_mutation;
  if (multi_phase.phases_.phase_number_ ==
      internal::TransactionState::kInvalidPhase) {
    // No existing phase node.  Must be the first call to this function.
    single_phase_mutation = &multi_phase.phases_;
    single_phase_mutation->phase_number_ = phase;
  } else {
    single_phase_mutation = multi_phase.phases_.prev_;
    assert(single_phase_mutation->phase_number_ <= phase);
    if (single_phase_mutation->phase_number_ != phase) {
      // Phase changed since the last operation on this key-value store.  Create
      // a new SinglePhaseMutation.
      auto* new_single_phase_mutation = new SinglePhaseMutation;
      std::swap(new_single_phase_mutation->entries_,
                single_phase_mutation->entries_);
      new_single_phase_mutation->next_ = &multi_phase.phases_;
      new_single_phase_mutation->prev_ = single_phase_mutation;
      new_single_phase_mutation->phase_number_ = phase;
      new_single_phase_mutation->prev_->next_ = new_single_phase_mutation;
      new_single_phase_mutation->next_->prev_ = new_single_phase_mutation;
      new_single_phase_mutation->multi_phase_ = &multi_phase;
      single_phase_mutation = new_single_phase_mutation;
    }
  }
  return *single_phase_mutation;
}

struct Controller {
  ReadModifyWriteEntry* entry_;
  internal::TransactionState::Node& GetTransactionNode() {
    return entry_->multi_phase().GetTransactionNode();
  }
  std::string DescribeKey(std::string_view key) {
    return entry_->multi_phase().DescribeKey(key);
  }
  const Key& GetKey() { return entry_->key_; }
  void Success(TimestampedStorageGeneration new_stamp,
               const StorageGeneration& orig_generation) {
    if (auto* dr_entry = static_cast<DeleteRangeEntry*>(entry_->next_)) {
      DeletedEntryDone(*dr_entry, /*error=*/false);
      return;
    }
    WritebackSuccess(*entry_, std::move(new_stamp), orig_generation);
    EntryDone(entry_->single_phase_mutation(), /*error=*/false);
  }
  void Error(absl::Status error) {
    auto* dr_entry = static_cast<DeleteRangeEntry*>(entry_->next_);
    auto& single_phase_mutation = entry_->single_phase_mutation();
    entry_->multi_phase().RecordEntryWritebackError(*entry_, std::move(error));
    if (dr_entry) {
      DeletedEntryDone(*dr_entry, /*error=*/true);
    } else {
      EntryDone(single_phase_mutation, /*error=*/true);
    }
  }
  void Retry(absl::Time time) {
    if (entry_->flags_.load(std::memory_order_relaxed) &
        ReadModifyWriteEntry::kNonRetryable) {
      Error(kvstore::Driver::AnnotateErrorWithKeyDescription(
          DescribeKey(GetKey()), "writing",
          absl::AbortedError("Generation mismatch")));
      return;
    }
    kvstore_transaction_retries.Increment();
    StartWriteback(*entry_, time);
  }
};

void ReceiveWritebackCommon(ReadModifyWriteEntry& entry,
                            ReadResult& read_result) {
  TENSORSTORE_KVSTORE_DEBUG_LOG(
      entry, "ReceiveWritebackCommon: state=", read_result.state,
      ", stamp=", read_result.stamp);
  // Update the flags based on the `ReadResult` provided for writeback.
  auto flags = (entry.flags_.load(std::memory_order_relaxed) &
                ~(ReadModifyWriteEntry::kTransitivelyUnconditional |
                  ReadModifyWriteEntry::kTransitivelyDirty)) |
               ReadModifyWriteEntry::kWritebackProvided;
  if (!StorageGeneration::IsConditional(read_result.stamp.generation)) {
    flags |= ReadModifyWriteEntry::kTransitivelyUnconditional;
  }
  if (read_result.state != ReadResult::kUnspecified) {
    flags |= ReadModifyWriteEntry::kTransitivelyDirty;
  }
  entry.flags_.store(flags, std::memory_order_relaxed);
}

void StartWriteback(ReadModifyWriteEntry& entry, absl::Time staleness_bound) {
  TENSORSTORE_KVSTORE_DEBUG_LOG(
      entry, "StartWriteback: staleness_bound=", staleness_bound);
  // First mark all previous entries as not having yet provided a writeback
  // during the current writeback sequence.
  for (auto* e = &entry;;) {
    e->flags_.fetch_and(~(ReadModifyWriteEntry::kWritebackProvided |
                          ReadModifyWriteEntry::kTransitivelyDirty),
                        std::memory_order_relaxed);
    e = e->prev_;
    if (!e) break;
  }

  ReadModifyWriteSource::WritebackOptions writeback_options;
  writeback_options.staleness_bound = staleness_bound;
  writeback_options.writeback_mode =
      (entry.flags_.load(std::memory_order_relaxed) &
       ReadModifyWriteEntry::kDeleted)
          ? ReadModifyWriteSource::kValidateOnly
          : ReadModifyWriteSource::kNormalWriteback;
  if ((!entry.prev_ && !(entry.flags_.load(std::memory_order_relaxed) &
                         ReadModifyWriteEntry::kDeleted)) ||
      !entry.multi_phase()
           .GetTransactionNode()
           .transaction()
           ->commit_started()) {
    // Fast path:
    //
    // (a) This entry sequence consists of just a single entry, and is
    //     not a deleted entry superseded by a `DeleteRange` operation; or
    //
    // (b) the transaction is not being commited, and this writeback is just to
    //     satisfy a read request, e.g. of an entire zarr_sharding_indexed
    //     shard.
    //
    // We don't need to track any state in the `WritebackReceiver` beyond the
    // `entry` itself, and can just forward the writeback result from
    // `ReadModifyWriteSource::KvsWriteback` directly to
    // `MultiPhaseMutation::Writeback`.
    struct WritebackReceiverImpl {
      ReadModifyWriteEntry* entry_;
      void set_error(absl::Status error) {
        ReportWritebackError(Controller{entry_}, "writing", error);
      }
      void set_cancel() { ABSL_UNREACHABLE(); }  // COV_NF_LINE
      void set_value(ReadResult read_result) {
        ReceiveWritebackCommon(*entry_, read_result);
        entry_->multi_phase().Writeback(*entry_, *entry_,
                                        std::move(read_result));
      }
    };
    entry.source_->KvsWriteback(std::move(writeback_options),
                                WritebackReceiverImpl{&entry});
    return;
  }

  // Slow path: The entry sequence consists of more than one entry, or is a
  // deleted entry superseded by a `DeleteRange` operation.  We first call
  // `ReadModifyWriteSource::KvsWriteback` on the last entry in the sequence.
  // In many cases this will in turn lead to read requests that lead to
  // writeback requests on all prior entries in the sequence.  Note that this
  // code path also works for the "fast path" above.
  //
  // However, if a read-modify-write operation in the sequence is
  // unconditional, then requesting its writeback value will not lead to a read
  // request, and therefore will not lead to a writeback request on the prior
  // entry.  We may need to issue additional writeback requests on those
  // "skipped" entries in order to give their writeback implementations a chance
  // to validate any constraints on the existing read value (e.g. a "consistent
  // read" constraint, or a metadata consistency constraint).  These "skipped"
  // entries cannot affect the writeback value, but they can result in an error,
  // or add a generation constraint to the writeback.

  struct SequenceWritebackReceiverImpl {
    // Heap-allocated state used for both the initial writeback request to the
    // last entry in the sequence and for any additional writeback requests to
    // "skipped" prior entries.
    struct State {
      // Entry in the sequence to which most recent writeback request was
      // issued.
      ReadModifyWriteEntry* entry;
      // Staleness bound for the original `StartWriteback` request.  This is
      // used for all writeback requests to "skipped" entries as well.
      absl::Time staleness_bound;
      // Current writeback value, as of when the writeback request was *issued*
      // to `entry`.  If `!entry->next_read_modify_write()`, then this is just
      // default constructed.
      //
      // Note that `read_result.state` and `read_result.value` are determined
      // solely from the initial writeback request to
      // `GetLastReadModifyWriteEntry()`.  However, `read_result.stamp` may be
      // affected by "skipped" entries.
      ReadResult read_result;

      // Entry from which `read_result` was obtained.
      ReadModifyWriteEntry* source_entry = nullptr;

      // Returns the last (i.e. most recent) entry in the sequence.  This is the
      // entry to which the initial writeback request is issued.
      ReadModifyWriteEntry* GetLastReadModifyWriteEntry() {
        auto* e = entry;
        while (auto* next = e->next_read_modify_write()) e = next;
        return e;
      }
    };

    std::unique_ptr<State> state_;
    void set_error(absl::Status error) {
      ReportWritebackError(Controller{state_->GetLastReadModifyWriteEntry()},
                           "writing", error);
    }
    void set_cancel() { ABSL_UNREACHABLE(); }  // COV_NF_LINE
    void set_value(ReadResult read_result) {
      auto& entry = *state_->entry;
      ReceiveWritebackCommon(entry, read_result);
      if (!state_->entry->next_ &&
          !(state_->entry->flags_.load(std::memory_order_relaxed) &
            ReadModifyWriteEntry::kDeleted)) {
        // `state_->entry` is the last entry in the sequence and not superseded
        // by a `DeleteRange` operation.  This writeback result is for the
        // initial writeback request.  Overwrite the existing (default
        // constructed) `state->read_result` with this initial writeback
        // request.
        state_->read_result = std::move(read_result);
        state_->source_entry = &entry;
      } else {
        // This branch covers two possible cases in which we update only
        // `state_->read_result.stamp` but leave `state_->read_result.state` and
        // `state_->read_result.value` unchanged:
        //
        // 1. If `state_->entry->next_ != nullptr`, then `state_->entry` is a
        //    "skipped" entry, which implies that `state_->read_result` must be
        //    unconditional, since a conditional writeback result is only
        //    possible by a sequence of reads/writebacks all the way to the
        //    beginning of the sequence.
        //
        // 2. If `state_->entry_.flags & ReadModifyWriteEntry::kDeleted`, then
        //    this is necessarily the initial writeback request (since
        //    `kDeleted` can only be set on the last entry in a sequence).
        //    Therefore, `state_->read_result` is in its default-constructed
        //    state, and we leave `state=kUnspecified` because we don't want to
        //    actually perform a writeback for a key that will subsequently be
        //    deleted.
        assert(!StorageGeneration::IsConditional(
            state_->read_result.stamp.generation));
        if (state_->read_result.state == ReadResult::kUnspecified) {
          TENSORSTORE_KVSTORE_DEBUG_LOG(
              entry,
              "Replacing: existing_result state=", state_->read_result.state,
              ", stamp=", state_->read_result.stamp,
              ", new_result state=", read_result.state,
              ", stamp=", read_result.stamp);
          state_->read_result = std::move(read_result);
          state_->source_entry = &entry;
        } else {
          if (!StorageGeneration::IsUnknown(read_result.stamp.generation) ||
              state_->read_result.stamp.time == absl::InfinitePast()) {
            state_->read_result.stamp.time = read_result.stamp.time;
          }
          TENSORSTORE_KVSTORE_DEBUG_LOG(entry, "Conditioning: existing_stamp=",
                                        state_->read_result.stamp.generation,
                                        ", new_stamp=", read_result.stamp);
          state_->read_result.stamp.generation = StorageGeneration::Condition(
              state_->read_result.stamp.generation,
              std::move(read_result.stamp.generation));
        }
      }
      if (entry.flags_.load(std::memory_order_relaxed) &
          ReadModifyWriteEntry::kTransitivelyUnconditional) {
        // The writeback is still unconditional.  There may still be "skipped"
        // entries that need a writeback request.

        // If the writeback result is "unmodified", the real writeback result,
        // if any, will come from a "skipped" entry.
        const bool unmodified =
            state_->read_result.state == ReadResult::kUnspecified;

        // Finds the first prior superseded entry for which writeback must
        // still be requested as part of the current writeback sequence.
        //
        // This ensures entries superseded by an unconditional writeback are
        // still given a chance to validate any constraints on the existing
        // read value and return an error if constraints are violated, even
        // though they do not affect the value that will be written back.
        auto GetPrevSupersededEntryToWriteback =
            [&](ReadModifyWriteEntry* entry) -> ReadModifyWriteEntry* {
          while (true) {
            entry = entry->prev_;
            if (!entry) return nullptr;

            if (unmodified) {
              // Entry needs to be validated, or has already been validated but
              // may provide a modified writeback result.
              if (auto flags = entry->flags_.load(std::memory_order_relaxed);
                  !(flags & ReadModifyWriteEntry::kWritebackProvided) ||
                  (flags & ReadModifyWriteEntry::kTransitivelyDirty)) {
                return entry;
              }
            } else {
              // We don't need to request writeback of `entry` if it is known
              // that its constraints are not violated.  There are two cases in
              // which this is known:
              //
              // 1. `entry` already provided a writeback in the current
              // writeback
              //    sequence (e.g. because the `ReadModifyWriteSource` of
              //    `entry->next_` requested a read).
              //
              // 2. `entry` provided a writeback in the current or a prior
              //    writeback sequence, and is known to be unconditional.  In
              //    this case, it is not affected by an updated read result.
              if (!(entry->flags_.load(std::memory_order_relaxed) &
                    (ReadModifyWriteEntry::kWritebackProvided |
                     ReadModifyWriteEntry::kTransitivelyUnconditional))) {
                return entry;
              }
            }
          }
        };

        if (auto* prev = GetPrevSupersededEntryToWriteback(&entry)) {
          // Issue a writeback request to the first "skipped" entry that we
          // found.
          state_->entry = prev;
          TENSORSTORE_KVSTORE_DEBUG_LOG(*prev,
                                        "Continuing writeback validate only");
          ReadModifyWriteSource::WritebackOptions writeback_options;
          writeback_options.staleness_bound = state_->staleness_bound;
          writeback_options.writeback_mode =
              unmodified ? ReadModifyWriteSource::kNormalWriteback
                         : ReadModifyWriteSource::kValidateOnly;
          prev->source_->KvsWriteback(std::move(writeback_options),
                                      std::move(*this));
          return;
        }
      }
      // No remaining "skipped" entries.  Forward the combined writeback result
      // to `MultiPhaseMutation::Writeback`.
      auto* last_entry = state_->GetLastReadModifyWriteEntry();
      if (last_entry->next_) {
        // This entry is superseded by a `DeleteRangeEntry`.  Ensure that no
        // value is actually written back.
        state_->read_result.state = ReadResult::kUnspecified;
      }
      TENSORSTORE_KVSTORE_DEBUG_LOG(*last_entry,
                                    "No remaining skipped entries, forwarding "
                                    "to MultiPhaseMutation::Writeback: ",
                                    state_->read_result.stamp);
      last_entry->multi_phase().Writeback(
          *last_entry,
          state_->source_entry ? *state_->source_entry : *last_entry,
          std::move(state_->read_result));
    }
  };
  auto state = std::unique_ptr<SequenceWritebackReceiverImpl::State>(
      new SequenceWritebackReceiverImpl::State{&entry, staleness_bound});
  if (entry.flags_.load(std::memory_order_relaxed) &
      ReadModifyWriteEntry::kDeleted) {
    // Mark the value as deleted, to avoid the `unmodified` condition above
    // resulting in unnecessary reads.  This will be overwritten back to
    // `ReadResult::kUnspecified` before calling
    // `MultiPhaseMutation::Writeback`.
    state->read_result.state = ReadResult::kMissing;
  }
  entry.source_->KvsWriteback(std::move(writeback_options),
                              SequenceWritebackReceiverImpl{std::move(state)});
}

void HandleDeleteRangeDone(DeleteRangeEntry& dr_entry) {
  const bool error = dr_entry.remaining_entries_.HasError();
  if (error) {
    WritebackError(dr_entry);
  } else {
    WritebackSuccess(dr_entry);
  }
  EntryDone(dr_entry.single_phase_mutation(), error);
}

void DeletedEntryDone(DeleteRangeEntry& dr_entry, bool error, size_t count) {
  if (error) dr_entry.remaining_entries_.SetError();
  if (!dr_entry.remaining_entries_.DecrementCount(count)) return;
  if (dr_entry.remaining_entries_.HasError()) {
    HandleDeleteRangeDone(dr_entry);
    return;
  }

  dr_entry.multi_phase().WritebackDelete(dr_entry);
}

std::string DescribeEntry(MutationEntry& entry) {
  return tensorstore::StrCat(
      entry.entry_type() == kReadModifyWrite ? "read/write " : "delete ",
      entry.multi_phase().DescribeKey(entry.key_));
}

void EntryDone(SinglePhaseMutation& single_phase_mutation, bool error,
               size_t count) {
  auto& multi_phase = *single_phase_mutation.multi_phase_;
  if (error) single_phase_mutation.remaining_entries_.SetError();
  if (!single_phase_mutation.remaining_entries_.DecrementCount(count)) {
    return;
  }
  multi_phase.AllEntriesDone(single_phase_mutation);
}

absl::Status ApplyByteRange(ReadResult& read_result,
                            OptionalByteRangeRequest byte_range) {
  if (read_result.has_value() && !byte_range.IsFull()) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto resolved_byte_range,
                                 byte_range.Validate(read_result.value.size()));
    read_result.value =
        internal::GetSubCord(read_result.value, resolved_byte_range);
  }
  return absl::OkStatus();
}

template <bool ReadFromPrev>
void RequestWritebackForRead(
    ReadModifyWriteEntry* rmw_entry,
    ReadModifyWriteTarget::ReadModifyWriteReadOptions options,
    ReadModifyWriteTarget::ReadReceiver receiver) {
  struct ReadReceiverImpl {
    ReadModifyWriteEntry* entry_;
    ReadModifyWriteTarget::ReadReceiver receiver_;
    OptionalByteRangeRequest byte_range_;
    void set_cancel() { execution::set_cancel(receiver_); }
    void set_value(ReadResult read_result) {
      {
        assert(!StorageGeneration::IsUnknown(read_result.stamp.generation));
        absl::MutexLock lock(&entry_->mutex());
        auto* req_entry = ReadFromPrev ? entry_->prev_ : entry_;
        ReceiveWritebackCommon(*req_entry, read_result);
        if constexpr (ReadFromPrev) {
          entry_->flags_.fetch_or(
              (req_entry->flags_.load(std::memory_order_relaxed) &
               ReadModifyWriteEntry::kTransitivelyUnconditional),
              std::memory_order_relaxed);
        }
      }
      TENSORSTORE_RETURN_IF_ERROR(
          ApplyByteRange(read_result, byte_range_),
          execution::set_error(receiver_, std::move(_)));
      execution::set_value(receiver_, std::move(read_result));
    }
    void set_error(absl::Status error) {
      execution::set_error(receiver_, std::move(error));
    }
  };

  ReadModifyWriteSource::WritebackOptions writeback_options;
  writeback_options.generation_conditions.if_not_equal =
      std::move(options.generation_conditions.if_not_equal);
  writeback_options.staleness_bound = options.staleness_bound;
  switch (options.read_mode) {
    case ReadModifyWriteTarget::kNormalRead:
      writeback_options.writeback_mode =
          ReadModifyWriteSource::kSpecifyUnchangedWriteback;
      break;
    case ReadModifyWriteTarget::kValueDiscarded:
      writeback_options.writeback_mode = ReadModifyWriteSource::kValueDiscarded;
      break;
    case ReadModifyWriteTarget::kValueDiscardedSpecifyUnchanged:
      writeback_options.writeback_mode =
          ReadModifyWriteSource::kValueDiscardedSpecifyUnchanged;
      break;
  }
  if (options.byte_range.IsStat()) {
    writeback_options.writeback_mode =
        ReadModifyWriteSource::kValueDiscardedSpecifyUnchanged;
  }
  if (rmw_entry->flags_.load(std::memory_order_relaxed) &
      ReadModifyWriteEntry::kSupportsByteRangeReads) {
    writeback_options.byte_range = std::exchange(options.byte_range, {});
  }
  auto* req_entry = ReadFromPrev ? rmw_entry->prev_ : rmw_entry;
  TENSORSTORE_KVSTORE_DEBUG_LOG(
      *req_entry, "Requesting writeback for read, writeback_mode=",
      static_cast<int>(writeback_options.writeback_mode));
  req_entry->source_->KvsWriteback(
      std::move(writeback_options),
      ReadReceiverImpl{rmw_entry, std::move(receiver), options.byte_range});
}

}  // namespace

void ReadModifyWriteEntry::KvsRead(
    ReadModifyWriteTarget::ReadModifyWriteReadOptions options,
    ReadModifyWriteTarget::ReadReceiver receiver) {
  if (flags_.load(std::memory_order_relaxed) &
      ReadModifyWriteEntry::kPrevDeleted) {
    TENSORSTORE_KVSTORE_DEBUG_LOG(*this, "Requesting read (prev deleted)");
    StorageGeneration generation;
    generation.MarkDirty(StorageGeneration::kDeletionMutationId);
    execution::set_value(
        receiver,
        ReadResult::Missing({std::move(generation), absl::InfiniteFuture()}));
  } else if (prev_) {
    RequestWritebackForRead</*ReadFromPrev=*/true>(this, std::move(options),
                                                   std::move(receiver));
  } else {
    TENSORSTORE_KVSTORE_DEBUG_LOG(
        *this, "Requesting read (no prev), byte_range=", options.byte_range,
        ", if_not_equal=", options.generation_conditions.if_not_equal);
    multi_phase().Read(this->key_, std::move(options), std::move(receiver));
  }
}

bool ReadModifyWriteEntry::KvsReadsCommitted() {
  return prev_ == nullptr &&
         !(flags_.load(std::memory_order_relaxed) &
           ReadModifyWriteEntry::kPrevDeleted) &&
         multi_phase().MultiPhaseReadsCommitted();
}

void ReadModifyWriteEntry::EnsureRevoked() {
  if (flags_.fetch_or(kRevoked, std::memory_order_relaxed) & kRevoked) return;
  source_->KvsRevoke();
}

void DestroyPhaseEntries(SinglePhaseMutation& single_phase_mutation) {
  auto& multi_phase = *single_phase_mutation.multi_phase_;
  for (MutationEntryTree::iterator
           tree_entry = single_phase_mutation.entries_.begin(),
           tree_next;
       tree_entry != single_phase_mutation.entries_.end();
       tree_entry = tree_next) {
    tree_next = std::next(tree_entry);
    single_phase_mutation.entries_.Remove(*tree_entry);
    if (tree_entry->entry_type() == kReadModifyWrite) {
      DestroyReadModifyWriteSequence(
          static_cast<ReadModifyWriteEntry*>(&*tree_entry));
    } else {
      auto& dr_entry = static_cast<DeleteRangeEntry&>(*tree_entry);
      for (ReadModifyWriteEntryTree::iterator
               entry = dr_entry.superseded_.begin(),
               next;
           entry != dr_entry.superseded_.end(); entry = next) {
        next = std::next(entry);
        dr_entry.superseded_.Remove(*entry);
        DestroyReadModifyWriteSequence(entry.to_pointer());
      }
      delete &dr_entry;
    }
  }
  if (&single_phase_mutation != &multi_phase.phases_) {
    single_phase_mutation.prev_->next_ = single_phase_mutation.next_;
    single_phase_mutation.next_->prev_ = single_phase_mutation.prev_;
    delete &single_phase_mutation;
  }
}

namespace {
void InvalidateReadStateGoingBackward(ReadModifyWriteEntry* entry) {
  do {
    entry->source_->KvsInvalidateReadState();
    entry = entry->prev_;
  } while (entry);
}
}  // namespace

void InvalidateReadState(SinglePhaseMutation& single_phase_mutation) {
  for (auto& entry : single_phase_mutation.entries_) {
    if (entry.entry_type() == kReadModifyWrite) {
      InvalidateReadStateGoingBackward(
          static_cast<ReadModifyWriteEntry*>(&entry));
    } else {
      for (auto& deleted_entry :
           static_cast<DeleteRangeEntry&>(entry).superseded_) {
        InvalidateReadStateGoingBackward(&deleted_entry);
      }
    }
  }
}

void WritebackSuccess(ReadModifyWriteEntry& entry,
                      TimestampedStorageGeneration new_stamp,
                      const StorageGeneration& orig_generation) {
  assert(!entry.next_read_modify_write());
  for (ReadModifyWriteEntry* e = &entry;;) {
    e->source_->KvsWritebackSuccess(new_stamp, orig_generation);
    e = e->prev_;
    if (!e) break;
  }
}

void WritebackError(ReadModifyWriteEntry& entry) {
  assert(!entry.next_read_modify_write());
  if (entry.flags_.fetch_or(ReadModifyWriteEntry::kError,
                            std::memory_order_relaxed) &
      ReadModifyWriteEntry::kError) {
    return;
  }
  for (ReadModifyWriteEntry* e = &entry;;) {
    e->source_->KvsWritebackError();
    e = e->prev_;
    if (!e) break;
  }
}

void WritebackError(DeleteRangeEntry& entry) {
  for (auto& e : entry.superseded_) {
    WritebackError(e);
  }
}

void WritebackSuccess(DeleteRangeEntry& entry) {
  for (auto& e : entry.superseded_) {
    WritebackSuccess(e,
                     TimestampedStorageGeneration{StorageGeneration::Unknown(),
                                                  absl::InfiniteFuture()},
                     StorageGeneration::Unknown());
  }
}

void WritebackError(MutationEntry& entry) {
  if (entry.entry_type() == kReadModifyWrite) {
    WritebackError(static_cast<ReadModifyWriteEntry&>(entry));
  } else {
    WritebackError(static_cast<DeleteRangeEntry&>(entry));
  }
}

void WritebackError(SinglePhaseMutation& single_phase_mutation) {
  for (auto& entry : single_phase_mutation.entries_) {
    WritebackError(entry);
  }
}

MultiPhaseMutation::MultiPhaseMutation() {
  phases_.next_ = phases_.prev_ = &phases_;
  phases_.phase_number_ = internal::TransactionState::kInvalidPhase;
  phases_.multi_phase_ = this;
}

SinglePhaseMutation& MultiPhaseMutation::GetCommittingPhase() {
  auto* single_phase_mutation = &phases_;
  auto initial_phase_number = single_phase_mutation->phase_number_;
  // If no entries have been added, the initial phase number will equal
  // `kInvalidPhase`.
  if (initial_phase_number != this->GetTransactionNode().phase() &&
      initial_phase_number != internal::TransactionState::kInvalidPhase) {
    single_phase_mutation = single_phase_mutation->next_;
    assert(single_phase_mutation->phase_number_ ==
           this->GetTransactionNode().phase());
  }
  return *single_phase_mutation;
}

void MultiPhaseMutation::AllEntriesDone(
    SinglePhaseMutation& single_phase_mutation) {
  size_t next_phase = 0;
  if (single_phase_mutation.next_ != &this->phases_) {
    next_phase = single_phase_mutation.next_->phase_number_;
  }
  DestroyPhaseEntries(single_phase_mutation);
  this->PhaseCommitDone(next_phase);
}

namespace {
void InvalidateReadStateGoingForward(ReadModifyWriteEntry* entry) {
  auto& single_phase_mutation = entry->single_phase_mutation();
  do {
    entry->source_->KvsInvalidateReadState();
    entry->flags_.fetch_and(~ReadModifyWriteEntry::kTransitivelyUnconditional,
                            std::memory_order_relaxed);
    entry = entry->next_read_modify_write();
  } while (entry &&
           (&entry->single_phase_mutation() == &single_phase_mutation));
}

void WritebackPhase(
    SinglePhaseMutation& single_phase_mutation, absl::Time staleness_bound,
    absl::FunctionRef<bool(ReadModifyWriteEntry& entry)> predicate) {
  assert(single_phase_mutation.remaining_entries_.IsDone());
  size_t entry_count = 0;
  for (auto& entry : single_phase_mutation.entries_) {
    if (entry.entry_type() == kReadModifyWrite) {
      auto& rmw_entry = static_cast<ReadModifyWriteEntry&>(entry);
      if (auto* next = static_cast<ReadModifyWriteEntry*>(rmw_entry.next_)) {
        // Disconnect from next phase.
        assert(next->entry_type() == kReadModifyWrite);
        assert(&next->single_phase_mutation() != &single_phase_mutation);
        next->prev_ = nullptr;
        InvalidateReadStateGoingForward(next);
        rmw_entry.next_ = nullptr;
      }
      if (predicate(rmw_entry)) {
        ++entry_count;
        StartWriteback(rmw_entry, staleness_bound);
      }
    } else {
      auto& dr_entry = static_cast<DeleteRangeEntry&>(entry);
      assert(dr_entry.remaining_entries_.IsDone());
      ++entry_count;
      size_t deleted_entry_count = 0;
      for (auto& deleted_entry : dr_entry.superseded_) {
        auto& rmw_entry = static_cast<ReadModifyWriteEntry&>(deleted_entry);
        rmw_entry.next_ = &dr_entry;
        if (predicate(rmw_entry)) {
          ++deleted_entry_count;
          StartWriteback(static_cast<ReadModifyWriteEntry&>(deleted_entry),
                         staleness_bound);
        }
      }
      DeletedEntryDone(dr_entry, /*error=*/false, -deleted_entry_count);
    }
  }
  EntryDone(single_phase_mutation, /*error=*/false, -entry_count);
}
}  // namespace

void MultiPhaseMutation::CommitNextPhase() {
  size_t cur_phase_number = GetTransactionNode().phase();
  DebugCheckInvariants(*this, false);
  {
    DebugCheckInvariantsInDestructor debug_check(*this, true);
    if (cur_phase_number == 0) {
      if (phases_.next_ != &phases_) {
        // Multiple phases

        auto* last_phase = phases_.prev_;
        for (MutationEntryTree::iterator entry = last_phase->entries_.begin(),
                                         next;
             entry != last_phase->entries_.end(); entry = next) {
          // Save next entry pointer since we may remove this entry below.
          next = std::next(entry);
          if (&entry->single_phase_mutation() != last_phase) {
            last_phase->entries_.Remove(*entry);
            InsertIntoPriorPhase(entry.to_pointer());
          }
        }
      }
      if (cur_phase_number != phases_.phase_number_) {
        this->PhaseCommitDone(phases_.phase_number_);
        return;
      }
    }
  }

  auto& single_phase_mutation = GetCommittingPhase();
  WritebackPhase(single_phase_mutation, absl::InfinitePast(),
                 [](ReadModifyWriteEntry& entry) { return true; });
}

void MultiPhaseMutation::AbortRemainingPhases() {
  for (auto* single_phase_mutation = &phases_;;) {
    auto* next = single_phase_mutation->next_;
    DestroyPhaseEntries(*single_phase_mutation);
    if (next == &phases_) break;
    single_phase_mutation = next;
  }
}

namespace {
// Find either an existing `ReadModifyWriteEntry` with the same key, or an
// existing `DeleteRangeEntry` that contains `key`.
internal::intrusive_red_black_tree::FindResult<MutationEntry>
FindExistingEntryCoveringKey(SinglePhaseMutation& single_phase_mutation,
                             std::string_view key) {
  return single_phase_mutation.entries_.Find(
      [key](MutationEntry& existing_entry) {
        auto c = key.compare(existing_entry.key_);
        if (c <= 0) return internal::CompareResultAsWeakOrdering(c);
        if (existing_entry.entry_type() == kReadModifyWrite) {
          return absl::weak_ordering::greater;
        }
        return KeyRange::CompareKeyAndExclusiveMax(
                   key, static_cast<DeleteRangeEntry&>(existing_entry)
                            .exclusive_max_) < 0
                   ? absl::weak_ordering::equivalent
                   : absl::weak_ordering::greater;
      });
}
}  // namespace

MultiPhaseMutation::ReadModifyWriteStatus MultiPhaseMutation::ReadModifyWrite(
    size_t& phase, Key key, ReadModifyWriteSource& source) {
  DebugCheckInvariantsInDestructor debug_check(*this, false);
#ifndef NDEBUG
  mutex().AssertHeld();
#endif
  auto& single_phase_mutation = GetCurrentSinglePhaseMutation(*this);
  phase = single_phase_mutation.phase_number_;
  auto* entry = MakeReadModifyWriteEntry(single_phase_mutation, std::move(key));
  TENSORSTORE_KVSTORE_DEBUG_LOG(*entry, "ReadModifyWrite: add");
  entry->source_ = &source;
  entry->source_->KvsSetTarget(*entry);

  // We need to insert `entry` into the interval map for
  // `single_phase_mutation`.  This may involve marking an existing
  // `ReadModifyWriteEntry` for the same key as superseded, or splitting an
  // existing `DeleteRangeEntry` that contains `key`.

  auto find_result =
      FindExistingEntryCoveringKey(single_phase_mutation, entry->key_);
  if (!find_result.found) {
    // No existing `ReadModifyWriteEntry` or `DeleteRangeEntry` covering `key`
    // was found.
    const bool was_empty = single_phase_mutation.entries_.empty();
    single_phase_mutation.entries_.Insert(find_result.insert_position(),
                                          *entry);
    return was_empty ? ReadModifyWriteStatus::kAddedFirst
                     : ReadModifyWriteStatus::kAddedSubsequent;
  }

  // Existing `ReadModifyWriteEntry` or `DeleteRangeEntry` covering `key` was
  // found.
  single_phase_mutation.entries_.Replace(*find_result.node, *entry);
  if (find_result.node->entry_type() == kReadModifyWrite) {
    // New entry supersedes existing entry.
    auto* existing_entry = static_cast<ReadModifyWriteEntry*>(find_result.node);
    assert(existing_entry->key_ == entry->key_);
    if (&existing_entry->single_phase_mutation() != &single_phase_mutation) {
      InsertIntoPriorPhase(existing_entry);
    }
    existing_entry->EnsureRevoked();
    assert(existing_entry->next_ == nullptr);
    entry->prev_ = existing_entry;
    entry->flags_.store(existing_entry->flags_.load(std::memory_order_relaxed) &
                            ReadModifyWriteEntry::kNonRetryable,
                        std::memory_order_relaxed);
    existing_entry->next_ = entry;
    return ReadModifyWriteStatus::kExisting;
  }

  // `DeleteRangeEntry` contains `key`.  It needs to be split into a
  // "before" range and an "after" range.
  auto* existing_entry = static_cast<DeleteRangeEntry*>(find_result.node);
  assert(existing_entry->key_ <= entry->key_);
  assert(KeyRange::CompareKeyAndExclusiveMax(
             entry->key_, existing_entry->exclusive_max_) < 0);
  entry->flags_.fetch_or((ReadModifyWriteEntry::kPrevDeleted |
                          ReadModifyWriteEntry::kTransitivelyUnconditional),
                         std::memory_order_relaxed);
  if (&existing_entry->single_phase_mutation() != &single_phase_mutation) {
    // The existing `DeleteRangeEntry` is for a previous phase.  We must
    // move it into the interval tree for that phase, and add placeholder
    // `DeleteRangeEntry` nodes to current phase.
    if (existing_entry->key_ != entry->key_) {
      // "Left" interval is non-empty.
      InsertDeleteRangeEntry(kDeleteRangePlaceholder, single_phase_mutation,
                             existing_entry->single_phase_mutation(),
                             KeyRange{existing_entry->key_, entry->key_},
                             {entry, MutationEntryTree::kLeft});
    }
    if (auto successor = KeyRange::Successor(entry->key_);
        successor != existing_entry->exclusive_max_) {
      // "Right" interval is non-empty.
      InsertDeleteRangeEntry(
          kDeleteRangePlaceholder, single_phase_mutation,
          existing_entry->single_phase_mutation(),
          KeyRange{std::move(successor), existing_entry->exclusive_max_},
          {entry, MutationEntryTree::kRight});
    }
    InsertIntoPriorPhase(existing_entry);
    return ReadModifyWriteStatus::kExisting;
  }

  // The existing `DeleteRangeEntry` is for the current phase.  We must
  // split its `superseded_` tree of `ReadModifyWriteEntry` nodes.
  auto split_result = existing_entry->superseded_.FindSplit(
      [key = std::string_view(entry->key_)](MutationEntry& e) {
        return internal::CompareResultAsWeakOrdering(key.compare(e.key_));
      });
  if (split_result.center) {
    split_result.center->flags_.fetch_and(~ReadModifyWriteEntry::kDeleted,
                                          std::memory_order_relaxed);
    entry->prev_ = split_result.center;
    split_result.center->next_ = entry;
  }
  if (existing_entry->key_ != entry->key_) {
    // "Left" interval is non-empty.
    auto* dr_entry = InsertDeleteRangeEntry(
        kDeleteRange, single_phase_mutation,
        existing_entry->single_phase_mutation(),
        KeyRange{std::move(existing_entry->key_), entry->key_},
        {entry, MutationEntryTree::kLeft});
    dr_entry->superseded_ = std::move(split_result.trees[0]);
  } else {
    assert(split_result.trees[0].empty());
  }
  existing_entry->key_ = KeyRange::Successor(entry->key_);
  if (existing_entry->key_ != existing_entry->exclusive_max_) {
    // "Right" interval is non-empty.  Reuse the existing entry for the
    // right interval.
    single_phase_mutation.entries_.Insert({entry, MutationEntryTree::kRight},
                                          *existing_entry);
    existing_entry->superseded_ = std::move(split_result.trees[1]);
  } else {
    assert(split_result.trees[1].empty());
    delete existing_entry;
  }
  return ReadModifyWriteStatus::kExisting;
}

void MultiPhaseMutation::DeleteRange(KeyRange range) {
#ifndef NDEBUG
  mutex().AssertHeld();
#endif
  if (range.empty()) return;
  DebugCheckInvariantsInDestructor debug_check(*this, false);
  auto& single_phase_mutation = GetCurrentSinglePhaseMutation(*this);

  // Find the first existing entry that intersects or is after `range`.  We
  // iterate forwards starting from this entry to find all existing entries that
  // intersect `range`.
  auto find_result =
      single_phase_mutation.entries_.FindBound<MutationEntryTree::kLeft>(
          [&](MutationEntry& existing_entry) {
            if (existing_entry.entry_type() == kReadModifyWrite) {
              return existing_entry.key_ < range.inclusive_min;
            } else {
              return KeyRange::CompareExclusiveMaxAndKey(
                         static_cast<DeleteRangeEntry&>(existing_entry)
                             .exclusive_max_,
                         range.inclusive_min) <= 0;
            }
          });

  // New entry containing `range`.  Will be set to an existing
  // `DeleteRangeEntry` that can be extended to include `range`, if any.
  DeleteRangeEntry* new_entry = nullptr;

  // Tree of `ReadModifyWriteEntry` nodes that will be superseded by the
  // `DeleteRangeEntry` that will contain `range`.
  ReadModifyWriteEntryTree superseded;

  // Temporary tree node that will be replaced with the actual
  // `DeleteRangeEntry` node containing `range`.
  DeleteRangeEntry insert_placeholder;
  single_phase_mutation.entries_.Insert(find_result.insert_position(),
                                        insert_placeholder);
  for (MutationEntry *existing_entry = find_result.found_node(), *next;
       existing_entry; existing_entry = next) {
    if (KeyRange::CompareKeyAndExclusiveMax(existing_entry->key_,
                                            range.exclusive_max) >= 0) {
      break;
    }
    next =
        MutationEntryTree::Traverse(*existing_entry, MutationEntryTree::kRight);
    single_phase_mutation.entries_.Remove(*existing_entry);
    if (existing_entry->entry_type() == kReadModifyWrite) {
      auto* existing_rmw_entry =
          static_cast<ReadModifyWriteEntry*>(existing_entry);
      existing_rmw_entry->EnsureRevoked();
      if (&existing_rmw_entry->single_phase_mutation() !=
          &single_phase_mutation) {
        // Existing `ReadModifyWriteEntry` is for a prior phase.  Just move it
        // into the interval tree for that phase.  We don't mark it as
        // superseded since it is not superseded in its own phase.
        InsertIntoPriorPhase(existing_entry);
      } else {
        // Existing `ReadModifyWriteEntry` is for the current phase.  Mark it as
        // superseded and add it to the `superseded` tree.
        existing_rmw_entry->flags_.fetch_or(ReadModifyWriteEntry::kDeleted,
                                            std::memory_order_relaxed);
        [[maybe_unused]] bool inserted =
            superseded
                .FindOrInsert(CompareToEntry(*existing_rmw_entry),
                              [=] { return existing_rmw_entry; })
                .second;
        assert(inserted);
      }
    } else {
      auto* existing_dr_entry = static_cast<DeleteRangeEntry*>(existing_entry);
      if (&existing_dr_entry->single_phase_mutation() !=
          &single_phase_mutation) {
        // Intersecting entry is for a prior phase.  We must move it into the
        // interval tree for that phase, and add placeholder entries to the
        // current phase for the begin/end portion of its range not covered by
        // `range`.
        if (KeyRange::CompareExclusiveMax(
                range.exclusive_max, existing_dr_entry->exclusive_max_) < 0) {
          InsertDeleteRangeEntry(
              kDeleteRangePlaceholder, single_phase_mutation,
              existing_dr_entry->single_phase_mutation(),
              KeyRange{range.exclusive_max, existing_dr_entry->exclusive_max_},
              {&insert_placeholder, MutationEntryTree::kRight});
        }
        if (existing_dr_entry->key_ < range.inclusive_min) {
          InsertDeleteRangeEntry(
              kDeleteRangePlaceholder, single_phase_mutation,
              existing_dr_entry->single_phase_mutation(),
              KeyRange{existing_dr_entry->key_, range.inclusive_min},
              {&insert_placeholder, MutationEntryTree::kLeft});
        }
        InsertIntoPriorPhase(existing_dr_entry);
      } else {
        // Intersecting entry is for the current phase.  Merge it into the new
        // entry.
        superseded = ReadModifyWriteEntryTree::Join(
            superseded, existing_dr_entry->superseded_);
        if (!new_entry) {
          new_entry = existing_dr_entry;
        } else {
          new_entry->exclusive_max_ =
              std::move(existing_dr_entry->exclusive_max_);
          delete existing_dr_entry;
        }
      }
    }
  }
  if (new_entry) {
    // Extend `new_entry` to include `range`.
    if (range.inclusive_min < new_entry->key_) {
      new_entry->key_ = std::move(range.inclusive_min);
    }
    if (KeyRange::CompareExclusiveMax(range.exclusive_max,
                                      new_entry->exclusive_max_) > 0) {
      new_entry->exclusive_max_ = std::move(range.exclusive_max);
    }
  } else {
    new_entry = MakeDeleteRangeEntry(kDeleteRange, single_phase_mutation,
                                     std::move(range));
  }
  new_entry->superseded_ = std::move(superseded);
  single_phase_mutation.entries_.Replace(insert_placeholder, *new_entry);
}

namespace {

/// `TransactionState::Node` type used to represent a
/// `ReadViaExistingTransaction` operation.
class ReadViaExistingTransactionNode : public internal::TransactionState::Node,
                                       public ReadModifyWriteSource {
 public:
  ReadViaExistingTransactionNode()
      :  // No associated data.
        internal::TransactionState::Node(nullptr) {}

  // Implementation of `TransactionState::Node` requirements:

  void PrepareForCommit() override {
    // Ensure `this` is not destroyed before `Commit` is called if
    // `WritebackSuccess` or `WritebackError` is triggered synchronously from
    // a transaction node on which `Commit` happens to be called first.
    intrusive_ptr_increment(this);
    this->PrepareDone();
    this->ReadyForCommit();
  }

  void Commit() override { intrusive_ptr_decrement(this); }

  void Abort() override { AbortDone(); }

  // Implementation of `ReadModifyWriteSource` interface:

  void KvsSetTarget(ReadModifyWriteTarget& target) override {
    target_ = &target;
    static_cast<ReadModifyWriteEntry&>(target).flags_.fetch_or(
        ReadModifyWriteEntry::kNonRetryable |
            ReadModifyWriteEntry::kSupportsByteRangeReads,
        std::memory_order_relaxed);
  }
  void KvsInvalidateReadState() override {}
  void KvsWriteback(
      ReadModifyWriteSource::WritebackOptions options,
      ReadModifyWriteSource::WritebackReceiver receiver) override {
    ReadModifyWriteTarget::ReadModifyWriteReadOptions read_options;
    static_cast<kvstore::TransactionalReadOptions&>(read_options) = options;
    if (options.writeback_mode == ReadModifyWriteSource::kValueDiscarded ||
        options.writeback_mode ==
            ReadModifyWriteSource::kValueDiscardedSpecifyUnchanged) {
      read_options.read_mode =
          options.writeback_mode == ReadModifyWriteSource::kValueDiscarded
              ? ReadModifyWriteTarget::kValueDiscarded
              : ReadModifyWriteTarget::kValueDiscardedSpecifyUnchanged;
      target_->KvsRead(std::move(read_options), std::move(receiver));
      return;
    }

    if (options.writeback_mode !=
        ReadModifyWriteSource::kSpecifyUnchangedWriteback) {
      // If `expected_stamp_.generation` is clean, then there are no prior
      // modifications.  In that case writeback can proceed immediately
      // without a prior read request.
      TimestampedStorageGeneration expected_stamp;
      {
        absl::MutexLock lock(&mutex_);
        expected_stamp = expected_stamp_;
      }
      if (StorageGeneration::IsUnknown(expected_stamp.generation)) {
        // No repeatable_read validation required.
        execution::set_value(
            receiver, ReadResult::Unspecified(
                          TimestampedStorageGeneration::Unconditional()));
        return;
      }
      if (!StorageGeneration::IsInnerLayerDirty(expected_stamp.generation) &&
          expected_stamp.time >= read_options.staleness_bound) {
        // Nothing to write back, just need to verify generation.
        execution::set_value(
            receiver, ReadResult::Unspecified(std::move(expected_stamp)));
        return;
      }

      read_options.byte_range = OptionalByteRangeRequest::Stat();
    }

    // Read request is needed to verify/update existing value.

    struct ReadReceiverImpl {
      ReadViaExistingTransactionNode& node_;
      ReadModifyWriteSource::WritebackReceiver receiver_;
      void set_value(ReadResult read_result) {
        bool mismatch;
        {
          absl::MutexLock lock(&node_.mutex_);
          mismatch = !StorageGeneration::EqualOrUnspecified(
              read_result.stamp.generation, node_.expected_stamp_.generation);
        }
        if (mismatch) {
          auto error = node_.AnnotateError(GetGenerationMismatchError());
          node_.SetError(std::move(error));
          execution::set_error(receiver_, error);
          return;
        }
        execution::set_value(receiver_, std::move(read_result));
      }
      void set_cancel() { execution::set_cancel(receiver_); }
      void set_error(absl::Status error) {
        execution::set_error(receiver_, std::move(error));
      }
    };
    target_->KvsRead(std::move(read_options),
                     ReadReceiverImpl{*this, std::move(receiver)});
  }
  void KvsWritebackError() override { this->CommitDone(); }
  void KvsRevoke() override {}
  void KvsWritebackSuccess(TimestampedStorageGeneration new_stamp,
                           const StorageGeneration& orig_generation) override {
    this->CommitDone();
  }

  static absl::Status GetGenerationMismatchError() {
    return absl::AbortedError(
        "Generation mismatch in repeatable_read transaction");
  }

  absl::Status AnnotateError(const absl::Status& error,
                             SourceLocation loc = SourceLocation::current()) {
    auto* rmw_entry = static_cast<ReadModifyWriteEntry*>(target_);
    return kvstore::Driver::AnnotateErrorWithKeyDescription(
        rmw_entry->multi_phase().DescribeKey(rmw_entry->key_), "reading", error,
        loc);
  }

  void IssueReadRequest(kvstore::TransactionalReadOptions&& options,
                        ReadModifyWriteTarget::ReadReceiver&& receiver) {
    struct InitialReadReceiverImpl {
      internal::OpenTransactionNodePtr<ReadViaExistingTransactionNode>
          read_node_;
      ReadModifyWriteTarget::ReadReceiver receiver_;

      void set_value(ReadResult read_result) {
        if (read_node_->transaction()->mode() & repeatable_read) {
          absl::MutexLock lock(&read_node_->mutex_);
          if (!StorageGeneration::IsUnknown(read_result.stamp.generation) &&
              !StorageGeneration::EqualOrUnspecified(
                  read_result.stamp.generation,
                  read_node_->expected_stamp_.generation)) {
            auto error = GetGenerationMismatchError();
            read_node_->SetError(read_node_->AnnotateError(error));
            execution::set_error(receiver_, std::move(error));
            return;
          }
          read_node_->expected_stamp_ = read_result.stamp;
        }
        execution::set_value(receiver_, std::move(read_result));
      }
      void set_cancel() { execution::set_cancel(receiver_); }
      void set_error(absl::Status error) {
        execution::set_error(receiver_, std::move(error));
      }
    };
    this->target_->KvsRead(
        ReadModifyWriteTarget::ReadModifyWriteReadOptions{std::move(options)},
        InitialReadReceiverImpl{
            internal::OpenTransactionNodePtr<ReadViaExistingTransactionNode>(
                this),
            std::move(receiver)});
  }

  absl::Mutex mutex_;

  /// Expected read generation.
  ///
  /// If not equal to `StorageGeneration::Unknown()`, `Commit` will fail if
  /// this does not match.
  TimestampedStorageGeneration expected_stamp_;

  ReadModifyWriteTarget* target_;
};

template <typename BaseReceiver>
struct IfEqualCheckingReadReceiver {
  BaseReceiver receiver_;
  StorageGeneration if_equal_;

  void set_value(ReadResult read_result) {
    if (read_result.has_value() &&
        !StorageGeneration::EqualOrUnspecified(read_result.stamp.generation,
                                               if_equal_)) {
      read_result = ReadResult::Unspecified(std::move(read_result.stamp));
    }
    execution::set_value(receiver_, std::move(read_result));
  }
  void set_cancel() { execution::set_cancel(receiver_); }
  void set_error(absl::Status error) {
    execution::set_error(receiver_, std::move(error));
  }
};
}  // namespace

Future<ReadResult> MultiPhaseMutation::ReadImpl(
    internal::OpenTransactionNodePtr<> node, kvstore::Driver* driver, Key&& key,
    kvstore::ReadOptions&& options, absl::FunctionRef<void()> unlock) {
  auto& single_phase_mutation = GetCurrentSinglePhaseMutation(*this);

  bool repeatable_read_mode =
      static_cast<bool>(node->transaction()->mode() & repeatable_read);

  auto promise_future_pair = PromiseFuturePair<ReadResult>::Make();
  // Not using structured bindings due to C++17 structured binding lambda
  // capture limitation.
  auto& promise = promise_future_pair.promise;
  auto& future = promise_future_pair.future;
  auto get_read_receiver = [&]() -> ReadModifyWriteTarget::ReadReceiver {
    if (!StorageGeneration::IsUnknown(options.generation_conditions.if_equal)) {
      return IfEqualCheckingReadReceiver<Promise<kvstore::ReadResult>>{
          std::move(promise),
          std::move(options.generation_conditions.if_equal)};
    }
    return std::move(promise);
  };

  kvstore::TransactionalReadOptions transactional_read_options;
  transactional_read_options.byte_range = options.byte_range;
  transactional_read_options.generation_conditions.if_not_equal =
      std::move(options.generation_conditions.if_not_equal);
  transactional_read_options.staleness_bound = options.staleness_bound;
  transactional_read_options.batch = std::move(options.batch);

  auto find_result = FindExistingEntryCoveringKey(single_phase_mutation, key);
  if (find_result.found) {
    if (find_result.node->entry_type() == kReadModifyWrite) {
      auto* rmw_entry = static_cast<ReadModifyWriteEntry*>(find_result.node);
      if (typeid(*rmw_entry->source_) ==
          typeid(ReadViaExistingTransactionNode)) {
        // Compared to `RequestWritebackForRead`, if this is the first entry in
        // the chain, this will handle byte ranges efficiently.
        unlock();
        static_cast<ReadViaExistingTransactionNode*>(rmw_entry->source_)
            ->IssueReadRequest(std::move(transactional_read_options),
                               get_read_receiver());
        return std::move(future);
      }
      if (!repeatable_read_mode) {
        rmw_entry->EnsureRevoked();
        unlock();
        RequestWritebackForRead</*ReadFromPrev=*/false>(
            rmw_entry,
            ReadModifyWriteTarget::ReadModifyWriteReadOptions{
                std::move(transactional_read_options)},
            get_read_receiver());
        return std::move(future);
      }
    } else {
      // DeleteRange entry
      unlock();
      promise.SetResult(ReadResult::Missing(
          {StorageGeneration::Dirty(StorageGeneration::Unknown(),
                                    StorageGeneration::kDeletionMutationId),
           absl::InfiniteFuture()}));
      return std::move(future);
    }
  } else {
    if (!repeatable_read_mode) {
      unlock();
      this->Read(key, {std::move(transactional_read_options)},
                 get_read_receiver());
      return std::move(future);
    }
  }

  // It is sufficient to use `WeakTransactionNodePtr`, since the transaction
  // is already kept open by the `transaction` pointer.
  internal::WeakTransactionNodePtr<ReadViaExistingTransactionNode> read_node;
  read_node.reset(new ReadViaExistingTransactionNode);
  size_t phase;
  auto rmw_status = this->ReadModifyWrite(phase, std::move(key), *read_node);
  unlock();
  TENSORSTORE_RETURN_IF_ERROR(this->ValidateReadModifyWriteStatus(rmw_status));
  read_node->SetTransaction(*node->transaction());
  read_node->SetPhase(phase);
  TENSORSTORE_RETURN_IF_ERROR(read_node->Register());
  read_node->IssueReadRequest(std::move(transactional_read_options),
                              get_read_receiver());
  return std::move(future);
}

namespace {
struct ListOperationState
    : public internal::FlowSenderOperationState<kvstore::ListEntry> {
  using Base = internal::FlowSenderOperationState<kvstore::ListEntry>;

  static void Start(MultiPhaseMutation& multi_phase,
                    internal::OpenTransactionNodePtr<> node,
                    kvstore::ListOptions&& options,
                    kvstore::ListReceiver&& receiver) {
    auto [modified_keys, ranges] =
        GetModifiedKeysAndRangesToQuery(multi_phase, options.range);
    if (modified_keys.empty() && ranges.size() == 1) {
      // Single range with no modified keys.  Just perform normal list.
      options.range = std::move(ranges.front().range);
      multi_phase.ListUnderlying(std::move(options), std::move(receiver));
      return;
    }

    auto state = internal::MakeIntrusivePtr<ListOperationState>(
        std::move(receiver), std::move(modified_keys), std::move(node),
        options.strip_prefix_length);

    for (auto& range : ranges) {
      state->QueryExistingRange(multi_phase, std::move(range), options);
    }

    for (size_t i = 0, num_keys = state->modified_keys.size(); i < num_keys;
         ++i) {
      state->QueryModifiedKey(i, options);
    }
  }

  struct RangeToQuery {
    KeyRange range;
    size_t key_begin_index, key_end_index;
  };

  static std::pair<std::vector<ReadModifyWriteEntry*>,
                   std::vector<RangeToQuery>>
  GetModifiedKeysAndRangesToQuery(MultiPhaseMutation& multi_phase,
                                  const KeyRange& range) {
    std::vector<ReadModifyWriteEntry*> modified_keys;
    std::vector<RangeToQuery> ranges;

    absl::MutexLock lock(&multi_phase.mutex());
    auto& single_phase_mutation = GetCurrentSinglePhaseMutation(multi_phase);

    // Find the first existing entry that intersects or is after `range`.  We
    // iterate forwards starting from this entry to find all existing entries
    // that intersect `range`.
    auto find_result =
        single_phase_mutation.entries_.FindBound<MutationEntryTree::kLeft>(
            [&](MutationEntry& existing_entry) {
              if (existing_entry.entry_type() == kReadModifyWrite) {
                return existing_entry.key_ < range.inclusive_min;
              } else {
                return KeyRange::CompareExclusiveMaxAndKey(
                           static_cast<DeleteRangeEntry&>(existing_entry)
                               .exclusive_max_,
                           range.inclusive_min) <= 0;
              }
            });

    std::string cur_range_begin = range.inclusive_min;
    size_t cur_range_key_offset = 0;
    bool no_more_ranges = false;

    for (MutationEntry *existing_entry = find_result.found_node(), *next;
         existing_entry; existing_entry = next) {
      if (KeyRange::CompareKeyAndExclusiveMax(existing_entry->key_,
                                              range.exclusive_max) >= 0) {
        break;
      }
      next = MutationEntryTree::Traverse(*existing_entry,
                                         MutationEntryTree::kRight);
      if (existing_entry->entry_type() == kReadModifyWrite) {
        // Add to list of keys
        auto* rmw_entry = static_cast<ReadModifyWriteEntry*>(existing_entry);
        rmw_entry->EnsureRevoked();
        modified_keys.push_back(rmw_entry);
      } else {
        auto* dr_entry = static_cast<DeleteRangeEntry*>(existing_entry);
        // End current range due to `DeleteRangeEntry`.
        if (existing_entry->key_ != cur_range_begin) {
          ranges.push_back(RangeToQuery{
              KeyRange(std::move(cur_range_begin), existing_entry->key_),
              cur_range_key_offset, modified_keys.size()});
          cur_range_key_offset = modified_keys.size();
        }
        cur_range_begin = dr_entry->exclusive_max_;
        if (cur_range_begin.empty()) {
          // Delete range has no upper bound.
          no_more_ranges = true;
          break;
        }
      }
    }

    if (!no_more_ranges && KeyRange::CompareKeyAndExclusiveMax(
                               cur_range_begin, range.exclusive_max) < 0) {
      ranges.push_back(RangeToQuery{
          KeyRange(std::move(cur_range_begin), std::move(range.exclusive_max)),
          cur_range_key_offset, modified_keys.size()});
    }
    return {std::move(modified_keys), std::move(ranges)};
  }

  ListOperationState(kvstore::ListReceiver&& base_receiver,
                     std::vector<ReadModifyWriteEntry*>&& modified_keys,
                     internal::OpenTransactionNodePtr<> transaction_node,
                     size_t strip_prefix_length)
      : Base(std::move(base_receiver)),
        transaction_node(std::move(transaction_node)),
        strip_prefix_length(strip_prefix_length),
        modified_keys(std::move(modified_keys)),
        modified_keys_info(this->modified_keys.size()) {
    for (auto& info : modified_keys_info) {
      info.store(kModifiedKeyUnknown, std::memory_order_relaxed);
    }
  }

  void QueryExistingRange(MultiPhaseMutation& multi_phase, RangeToQuery&& range,
                          const kvstore::ListOptions& options) {
    kvstore::ListOptions sub_range_options;
    sub_range_options.range = std::move(range.range);
    sub_range_options.strip_prefix_length = options.strip_prefix_length;
    sub_range_options.staleness_bound = options.staleness_bound;
    multi_phase.ListUnderlying(
        std::move(sub_range_options),
        ExistingRangeListReceiver{
            internal::IntrusivePtr<ListOperationState>(this),
            range.key_begin_index, range.key_end_index});
  }

  struct ExistingRangeListReceiver {
    internal::IntrusivePtr<ListOperationState> state;
    size_t key_begin, key_end;
    FutureCallbackRegistration cancel_registration;

    void set_starting(AnyCancelReceiver cancel) {
      cancel_registration =
          state->promise.ExecuteWhenNotNeeded(std::move(cancel));
    }
    void set_stopping() { cancel_registration(); }
    void set_done() {}
    void set_error(absl::Status error) { state->SetError(std::move(error)); }
    void set_value(kvstore::ListEntry entry) {
      if (key_begin != key_end) {
        auto& modified_keys = state->modified_keys;

        struct GetKey {
          size_t strip_prefix_length;

          std::string_view operator()(std::string_view key) const {
            return key;
          }
          std::string_view operator()(ReadModifyWriteEntry* entry) const {
            std::string_view key = entry->key_;
            key.remove_prefix(std::min(strip_prefix_length, key.size()));
            return key;
          }
        };

        const GetKey get_key{state->strip_prefix_length};
        auto it = std::lower_bound(
            modified_keys.begin() + key_begin, modified_keys.begin() + key_end,
            std::string_view(entry.key),
            [=](auto a, auto b) { return get_key(a) < get_key(b); });
        if (it != modified_keys.end() && get_key(*it) == entry.key) {
          auto& info = state->modified_keys_info[it - modified_keys.begin()];
          int64_t expected = kModifiedKeyUnknown;
          // Set the existing key state if the modified key state is
          // unknown.
          info.compare_exchange_strong(expected,
                                       entry.size >= 0 ? entry.size : -1);
          return;
        }
      }
      execution::set_value(state->shared_receiver->receiver, std::move(entry));
    }
  };

  void QueryModifiedKey(size_t i, const kvstore::ListOptions& options) {
    auto* entry = modified_keys[i];
    ReadModifyWriteSource::WritebackOptions writeback_options;
    writeback_options.writeback_mode = ReadModifyWriteSource::kValueDiscarded;
    writeback_options.staleness_bound = options.staleness_bound;
    writeback_options.byte_range = OptionalByteRangeRequest::Stat();
    entry->source_->KvsWriteback(
        std::move(writeback_options),
        ModifiedKeyReadReceiver{
            internal::IntrusivePtr<ListOperationState>(this), i});
  }

  struct ModifiedKeyReadReceiver {
    internal::IntrusivePtr<ListOperationState> state;
    size_t key_index;

    void set_value(ReadResult read_result) {
      if (read_result.state == ReadResult::kUnspecified) {
        return;
      }
      state->modified_keys_info[key_index].store(
          (read_result.state == ReadResult::kValue) ? kModifiedKeyPresent
                                                    : kModifiedKeyDeleted);
    }
    void set_error(absl::Status error) { state->SetError(std::move(error)); }
    void set_cancel() { ABSL_UNREACHABLE(); }
  };

  ~ListOperationState() {
    // Emit modified keys that are not deleted.
    size_t num_keys = modified_keys.size();
    for (size_t i = 0; i < num_keys; ++i) {
      int64_t info = modified_keys_info[i].load(std::memory_order_relaxed);
      if (info < kModifiedKeyPresent) continue;
      std::string_view key;
      key = modified_keys[i]->key_;
      key.remove_prefix(std::min(key.size(), strip_prefix_length));
      execution::set_value(
          shared_receiver->receiver,
          kvstore::ListEntry{std::string(key),
                             std::max(info, static_cast<int64_t>(-1))});
    }
  }

  internal::OpenTransactionNodePtr<> transaction_node;
  size_t strip_prefix_length;
  std::vector<ReadModifyWriteEntry*> modified_keys;

  // Indicates modification status and/or existing size for each entry in
  // `modified_keys`. Values are described below.
  std::vector<std::atomic<int64_t>> modified_keys_info;

  // == -1  -> existing key is present with unknown size
  // >= 0   -> existing key is present with known size
  constexpr static int64_t kModifiedKeyPresent = -2;
  constexpr static int64_t kModifiedKeyDeleted = -3;
  constexpr static int64_t kModifiedKeyUnknown = -4;
};
}  // namespace

void MultiPhaseMutation::ListImpl(internal::OpenTransactionNodePtr<> node,
                                  kvstore::ListOptions&& options,
                                  kvstore::ListReceiver&& receiver) {
  ListOperationState::Start(*this, std::move(node), std::move(options),
                            std::move(receiver));
}

std::string MultiPhaseMutation::DescribeFirstEntry() {
  assert(!phases_.prev_->entries_.empty());
  return DescribeEntry(*phases_.prev_->entries_.begin());
}

ReadModifyWriteEntry* MultiPhaseMutation::AllocateReadModifyWriteEntry() {
  return new ReadModifyWriteEntry;
}
void MultiPhaseMutation::FreeReadModifyWriteEntry(ReadModifyWriteEntry* entry) {
  delete entry;
}
void ReadDirectly(Driver* driver, std::string_view key,
                  ReadModifyWriteTarget::ReadModifyWriteReadOptions&& options,
                  ReadModifyWriteTarget::ReadReceiver&& receiver) {
  if (options.read_mode == ReadModifyWriteTarget::kValueDiscarded) {
    execution::set_value(
        receiver,
        ReadResult::Unspecified(TimestampedStorageGeneration::Unconditional()));
    return;
  }
  ReadOptions kvstore_options;
  kvstore_options.staleness_bound = options.staleness_bound;
  kvstore_options.generation_conditions.if_not_equal =
      std::move(options.generation_conditions.if_not_equal);
  if (options.read_mode ==
      ReadModifyWriteTarget::kValueDiscardedSpecifyUnchanged) {
    kvstore_options.byte_range = OptionalByteRangeRequest::Stat();
  } else {
    kvstore_options.byte_range = options.byte_range;
  }
  kvstore_options.batch = std::move(options.batch);
  execution::submit(driver->Read(std::string(key), std::move(kvstore_options)),
                    std::move(receiver));
}

void WritebackDirectly(Driver* driver, ReadModifyWriteEntry& entry,
                       ReadResult&& read_result) {
  assert(read_result.stamp.time != absl::InfinitePast());
  PerformWriteback(driver, Controller{&entry}, std::move(read_result));
}

void WritebackDirectly(Driver* driver, DeleteRangeEntry& entry) {
  auto future = driver->DeleteRange(KeyRange{entry.key_, entry.exclusive_max_});
  future.Force();
  std::move(future).ExecuteWhenReady([&entry](ReadyFuture<const void> future) {
    auto& r = future.result();
    if (!r.ok()) {
      entry.multi_phase().GetTransactionNode().SetError(r.status());
      entry.remaining_entries_.SetError();
    }
    HandleDeleteRangeDone(entry);
  });
}

void MultiPhaseMutation::RecordEntryWritebackError(ReadModifyWriteEntry& entry,
                                                   absl::Status error) {
  this->GetTransactionNode().SetError(std::move(error));
  WritebackError(entry);
}

void AtomicMultiPhaseMutationBase::RetryAtomicWriteback(
    absl::Time staleness_bound) {
  auto& single_phase_mutation = GetCommittingPhase();
  WritebackPhase(
      single_phase_mutation, staleness_bound, [&](ReadModifyWriteEntry& entry) {
        return static_cast<ReadModifyWriteEntryWithStamp&>(entry).IsOutOfDate(
            staleness_bound);
      });
}

ReadModifyWriteEntry* AtomicMultiPhaseMutation::AllocateReadModifyWriteEntry() {
  return new BufferedReadModifyWriteEntry;
}
void AtomicMultiPhaseMutation::FreeReadModifyWriteEntry(
    ReadModifyWriteEntry* entry) {
  delete static_cast<BufferedReadModifyWriteEntry*>(entry);
}

void AtomicMultiPhaseMutationBase::AtomicWritebackReady(
    ReadModifyWriteEntry& entry) {
  if (auto* dr_entry = static_cast<DeleteRangeEntry*>(entry.next_)) {
    DeletedEntryDone(*dr_entry, /*error=*/false);
  } else {
    EntryDone(entry.single_phase_mutation(), /*error=*/false);
  }
}

void AtomicMultiPhaseMutation::Writeback(ReadModifyWriteEntry& entry,
                                         ReadModifyWriteEntry& source_entry,
                                         ReadResult&& read_result) {
  assert(read_result.stamp.time != absl::InfinitePast());
  auto& buffered = static_cast<BufferedReadModifyWriteEntry&>(entry);
  buffered.stamp() = std::move(read_result.stamp);
  buffered.value_state_ = read_result.state;
  buffered.value_ = std::move(read_result.value);
  AtomicWritebackReady(entry);
}

void AtomicMultiPhaseMutationBase::WritebackDelete(DeleteRangeEntry& entry) {
  EntryDone(entry.single_phase_mutation(), /*error=*/false);
}

void AtomicMultiPhaseMutationBase::AtomicCommitWritebackSuccess() {
  for (auto& entry : GetCommittingPhase().entries_) {
    if (entry.entry_type() == kReadModifyWrite) {
      auto& rmw_entry = static_cast<ReadModifyWriteEntryWithStamp&>(entry);
      internal_kvstore::WritebackSuccess(rmw_entry, std::move(rmw_entry.stamp_),
                                         rmw_entry.orig_generation_);
    } else {
      auto& dr_entry = static_cast<DeleteRangeEntry&>(entry);
      internal_kvstore::WritebackSuccess(dr_entry);
    }
  }
}

void AtomicMultiPhaseMutationBase::RevokeAllEntries() {
  assert(phases_.next_ == &phases_);
  for (auto& entry : phases_.entries_) {
    if (entry.entry_type() != kReadModifyWrite) continue;
    auto& rmw_entry = static_cast<ReadModifyWriteEntry&>(entry);
    rmw_entry.EnsureRevoked();
  }
}

absl::Status AtomicMultiPhaseMutationBase::ValidateReadModifyWriteStatus(
    ReadModifyWriteStatus rmw_status) {
  return absl::OkStatus();
}

namespace {
absl::Status GetNonAtomicReadModifyWriteError(
    NonAtomicTransactionNode& node,
    MultiPhaseMutation::ReadModifyWriteStatus modify_status) {
  if (!node.transaction()->atomic()) {
    return absl::OkStatus();
  }
  using ReadModifyWriteStatus = MultiPhaseMutation::ReadModifyWriteStatus;
  if (modify_status == ReadModifyWriteStatus::kAddedFirst) {
    return node.MarkAsTerminal();
  }
  if (modify_status == ReadModifyWriteStatus::kAddedSubsequent) {
    absl::MutexLock lock(&node.mutex_);
    auto& single_phase_mutation = *node.phases_.prev_;
    // Even though we have released and then re-acquired the mutex, there
    // must still be at least two entries, since the number of entries can
    // only decrease due to a `DeleteRange` operation, which is not
    // supported for an atomic transaction.
    MutationEntry* e0 = single_phase_mutation.entries_.begin().to_pointer();
    assert(e0);
    MutationEntry* e1 =
        MutationEntryTree::Traverse(*e0, MutationEntryTree::kRight);
    assert(e1);
    auto error = internal::TransactionState::Node::GetAtomicError(
        DescribeEntry(*e0), DescribeEntry(*e1));
    node.transaction()->RequestAbort(error);
    return error;
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status NonAtomicTransactionNode::ValidateReadModifyWriteStatus(
    ReadModifyWriteStatus rmw_status) {
  return GetNonAtomicReadModifyWriteError(*this, rmw_status);
}

namespace {
/// `TransactionState::Node` type used to represent a
/// `WriteViaExistingTransaction` operation.
class WriteViaExistingTransactionNode : public internal::TransactionState::Node,
                                        public ReadModifyWriteSource {
 public:
  WriteViaExistingTransactionNode()
      :  // No associated data.
        internal::TransactionState::Node(nullptr) {}

  // Implementation of `TransactionState::Node` requirements:

  void PrepareForCommit() override {
    // Ensure `this` is not destroyed before `Commit` is called if
    // `WritebackSuccess` or `WritebackError` is triggered synchronously from
    // a transaction node on which `Commit` happens to be called first.
    intrusive_ptr_increment(this);
    this->PrepareDone();
    this->ReadyForCommit();
  }

  void Commit() override { intrusive_ptr_decrement(this); }

  void Abort() override { AbortDone(); }

  // Implementation of `ReadModifyWriteSource` interface:

  void KvsSetTarget(ReadModifyWriteTarget& target) override {
    target_ = &target;
    static_cast<ReadModifyWriteEntry&>(target).flags_.fetch_or(
        ReadModifyWriteEntry::kNonRetryable |
            ReadModifyWriteEntry::kSupportsByteRangeReads,
        std::memory_order_relaxed);
  }
  void KvsInvalidateReadState() override {}
  void KvsWriteback(
      ReadModifyWriteSource::WritebackOptions options,
      ReadModifyWriteSource::WritebackReceiver receiver) override {
    if (!fail_transaction_on_mismatch_ &&
        options.writeback_mode == kValidateOnly && !promise_.result_needed()) {
      // Writeback never results in an error, and the promise result is ignored.
      execution::set_value(receiver, kvstore::ReadResult{});
      return;
    }
    UniqueWriterLock lock(mutex_);
    if (!StorageGeneration::IsConditional(read_result_.stamp.generation) ||
        (fail_transaction_on_mismatch_ &&
         !this->transaction()->commit_started())) {
      // Either:
      //
      // (a) Writeback is unconditional. Therefore, the existing read state does
      //     not need to be requested.
      //
      // (b) The condition is "assumed" to be true, and the transaction is not
      //     being committed. Don't validate yet.
      auto read_result = read_result_;
      lock.unlock();

      if (options.generation_conditions.Matches(read_result.stamp.generation)) {
        TENSORSTORE_RETURN_IF_ERROR(
            ApplyByteRange(read_result, options.byte_range),
            execution::set_error(receiver, std::move(_)));
      } else {
        read_result.state = ReadResult::State::kUnspecified;
        read_result.value.Clear();
      }
      execution::set_value(receiver, std::move(read_result));
      return;
    }
    // Writeback is conditional.  A read request must be performed in order to
    // determine an up-to-date writeback value (which may be required by a
    // subsequent read-modify-write operation layered on top of this
    // operation).
    ReadModifyWriteTarget::ReadModifyWriteReadOptions read_options;
    read_options.generation_conditions.if_not_equal = GetBaseGeneration();
    read_options.staleness_bound = options.staleness_bound;
    read_options.byte_range = options.byte_range;
    lock.unlock();
    struct ReadReceiverImpl {
      WriteViaExistingTransactionNode& source_;
      ReadModifyWriteSource::WritebackReceiver receiver_;
      StorageGeneration if_not_equal_;
      OptionalByteRangeRequest byte_range_;
      void set_value(ReadResult read_result) {
        {
          UniqueWriterLock lock(source_.mutex_);
          auto base_generation = source_.GetBaseGeneration();
          // Check if the new read generation matches the condition specified in
          // the read request.
          if (read_result.stamp.generation == base_generation ||
              // As a special case, if the user specified
              // `if_equal=StorageGeneration::NoValue()`, then match based on
              // the `state` rather than the exact generation, since it is valid
              // for a missing value to have a generation other than
              // `StorageGeneration::NoValue()`.  For example,
              // `Uint64ShardedKeyValueStore` uses the generation of the shard
              // even for missing keys.
              (source_.if_equal_no_value_ &&
               read_result.state == ReadResult::kMissing)) {
            // Read generation matches, store the updated stamp.  Normally this
            // will just store an updated time, but in the `if_equal_no_value_`
            // case, this may also store an updated generation.
            source_.read_result_.stamp = std::move(read_result.stamp);

            if (source_.modified_) {
              source_.read_result_.stamp.generation.MarkDirty(
                  source_.mutation_id_);
            }

            read_result = source_.read_result_;

            lock.unlock();

            if (!StorageGeneration::IsUnknown(if_not_equal_) &&
                read_result.stamp.generation == if_not_equal_) {
              read_result.value.Clear();
              read_result.state = ReadResult::kUnspecified;
            } else {
              TENSORSTORE_RETURN_IF_ERROR(
                  ApplyByteRange(read_result, byte_range_),
                  execution::set_error(receiver_, std::move(_)));
            }
          } else {
            // Read generation does not match.  Since the constraint was
            // violated, just provide the existing read value as the writeback
            // value.  The fact that `source_.read_result_.stamp.generation` is
            // not marked dirty indicates to the transaction machinery that
            // writeback is not actually necessary.
            if (source_.fail_transaction_on_mismatch_) {
              lock.unlock();
              auto error = absl::AbortedError("Generation mismatch");
              source_.SetError(error);
              execution::set_error(receiver_, std::move(error));
              return;
            }

            if (!read_result.has_value() || byte_range_.IsFull()) {
              // TODO(jbms): Once `kNonRetryable` is respected by atomic kvstore
              // implementations, this caching can be removed.
              source_.read_result_ = read_result;
              source_.if_equal_no_value_ = false;
              source_.modified_ = false;
            }

            if (read_result.stamp.generation == if_not_equal_) {
              read_result.value.Clear();
              read_result.state = ReadResult::kUnspecified;
            }
          }
        }
        execution::set_value(receiver_, std::move(read_result));
      }
      void set_cancel() { execution::set_cancel(receiver_); }
      void set_error(absl::Status error) {
        execution::set_error(receiver_, std::move(error));
      }
    };
    target_->KvsRead(
        std::move(read_options),
        ReadReceiverImpl{*this, std::move(receiver),
                         std::move(options.generation_conditions.if_not_equal),
                         options.byte_range});
  }
  void KvsWritebackError() override { this->CommitDone(); }
  void KvsRevoke() override {}
  void KvsWritebackSuccess(TimestampedStorageGeneration new_stamp,
                           const StorageGeneration& orig_generation) override {
    // No need to lock `mutex_` because no concurrent access can occur while
    // this method is called.
    if (!modified_) {
      new_stamp = TimestampedStorageGeneration{};
    } else if (!orig_generation.LastMutatedBy(mutation_id_)) {
      new_stamp.generation = StorageGeneration::Invalid();
    }
    promise_.SetResult(std::move(new_stamp));
    this->CommitDone();
  }

  // Get the generation on which the cached value in `read_result_` is
  // conditioned.
  //
  // `mutex_` must be held.
  StorageGeneration GetBaseGeneration() {
    auto& generation = read_result_.stamp.generation;
    return modified_ ? StorageGeneration::StripTag(generation) : generation;
  }

  /// Promise that will be marked ready when the write operation is committed
  /// (or fails).
  Promise<TimestampedStorageGeneration> promise_;

  // Mutation id for this write operation.
  StorageGeneration::MutationId mutation_id_;

  // Guards `read_result_`, `if_equal_no_value_`, and `modified_`.
  absl::Mutex mutex_;

  /// Either the original write request, or the most recent read result.
  ///
  /// If the `if_equal` constraint is found to be violated, then the new read
  /// result is stored here, because the result of "writeback" will actually
  /// just be the existing read result (since the requested conditional write
  /// will have no effect).
  ReadResult read_result_;

  /// If `true`, `if_equal=StorageGeneration::NoValue()` was specified, and it
  /// has not yet been found to have been violated (`read_result_` still
  /// contains the original write value).
  bool if_equal_no_value_;

  /// Indicates that `read_result_` corresponds to the original write request,
  /// as opposed to a cached read result due to a non-matching `if_equal`
  /// condition.
  bool modified_;

  // If `true`, a generation mismatch results in an error being set on the
  // transaction (preventing it from committing successfully).  If `false`, a
  // generation mismatch only results in an error being set on `promise_`.
  bool fail_transaction_on_mismatch_;

  ReadModifyWriteTarget* target_;
};
}  // namespace

Future<TimestampedStorageGeneration> WriteViaExistingTransaction(
    Driver* driver, internal::OpenTransactionPtr& transaction, size_t& phase,
    Key key, std::optional<Value> value, WriteOptions options,
    bool fail_transaction_on_mismatch, StorageGeneration* out_generation) {
  TimestampedStorageGeneration stamp;
  if (StorageGeneration::IsUnknown(options.generation_conditions.if_equal)) {
    stamp.time = absl::InfiniteFuture();
  } else {
    stamp.time = absl::Time();
  }
  bool if_equal_no_value =
      StorageGeneration::IsNoValue(options.generation_conditions.if_equal);
  auto mutation_id = (value || !StorageGeneration::IsUnknown(
                                   options.generation_conditions.if_equal))
                         ? StorageGeneration::AllocateMutationId()
                         // Consistent mutation id for unconditional deletes
                         // isn't necessary but is helpful for unit tests.
                         : StorageGeneration::kDeletionMutationId;
  stamp.generation = std::move(options.generation_conditions.if_equal);
  stamp.generation.MarkDirty(mutation_id);
  if (out_generation) {
    *out_generation = stamp.generation;
  }

  auto [promise, future] =
      PromiseFuturePair<TimestampedStorageGeneration>::Make();
  using Node = WriteViaExistingTransactionNode;
  internal::WeakTransactionNodePtr<Node> node;
  node.reset(new Node);
  node->promise_ = promise;
  node->mutation_id_ = mutation_id;
  node->fail_transaction_on_mismatch_ = fail_transaction_on_mismatch;
  node->modified_ = true;
  node->read_result_ =
      value ? ReadResult::Value(*std::move(value), std::move(stamp))
            : ReadResult::Missing(std::move(stamp));

  node->if_equal_no_value_ = if_equal_no_value;
  TENSORSTORE_RETURN_IF_ERROR(
      driver->ReadModifyWrite(transaction, phase, std::move(key), *node));
  node->SetTransaction(*transaction);
  node->SetPhase(phase);
  TENSORSTORE_RETURN_IF_ERROR(node->Register());
  LinkError(std::move(promise), transaction->future());
  return std::move(future);
}

Future<TimestampedStorageGeneration> WriteViaTransaction(
    Driver* driver, Key key, std::optional<Value> value, WriteOptions options) {
  internal::OpenTransactionPtr transaction;
  size_t phase;
  return WriteViaExistingTransaction(driver, transaction, phase, std::move(key),
                                     std::move(value), std::move(options),
                                     /*fail_transaction_on_mismatch=*/false,
                                     /*out_generation=*/nullptr);
}

}  // namespace internal_kvstore

namespace kvstore {

absl::Status Driver::ReadModifyWrite(internal::OpenTransactionPtr& transaction,
                                     size_t& phase, Key key,
                                     ReadModifyWriteSource& source) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node,
      internal_kvstore::GetTransactionNode<
          internal_kvstore::NonAtomicTransactionNode>(this, transaction));
  internal_kvstore::MultiPhaseMutation::ReadModifyWriteStatus rmw_status;
  {
    absl::MutexLock lock(&node->mutex_);
    rmw_status = node->ReadModifyWrite(phase, std::move(key), source);
  }
  return internal_kvstore::GetNonAtomicReadModifyWriteError(*node, rmw_status);
}

absl::Status Driver::TransactionalDeleteRange(
    const internal::OpenTransactionPtr& transaction, KeyRange range) {
  if (range.empty()) return absl::OkStatus();
  if (transaction && transaction->atomic()) {
    auto error = absl::InvalidArgumentError(
        tensorstore::StrCat("Cannot delete range starting at ",
                            this->DescribeKey(range.inclusive_min),
                            " as single atomic transaction"));
    transaction->RequestAbort(error);
    return error;
  }
  return internal_kvstore::AddDeleteRange<
      internal_kvstore::NonAtomicTransactionNode>(this, transaction,
                                                  std::move(range));
}

Future<ReadResult> Driver::TransactionalRead(
    const internal::OpenTransactionPtr& transaction, Key key,
    ReadOptions options) {
  return internal_kvstore::TransactionalReadImpl<
      internal_kvstore::NonAtomicTransactionNode>(
      this, transaction, std::move(key), std::move(options));
}

void Driver::TransactionalListImpl(
    const internal::OpenTransactionPtr& transaction, ListOptions options,
    ListReceiver receiver) {
  internal_kvstore::TransactionalListImpl<
      internal_kvstore::NonAtomicTransactionNode>(
      this, transaction, std::move(options), std::move(receiver));
}

}  // namespace kvstore
}  // namespace tensorstore
