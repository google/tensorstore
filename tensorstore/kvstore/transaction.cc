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

#include "absl/base/optimization.h"
#include "absl/container/btree_map.h"
#include "absl/functional/function_ref.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/util/execution/future_sender.h"  // IWYU pragma: keep

namespace tensorstore {
namespace internal_kvstore {

namespace {

auto& kvstore_transaction_retries = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/transaction_retries",
    "Count of kvstore transaction retries");

template <typename Controller>
void ReportWritebackError(Controller controller, std::string_view action,
                          const absl::Status& error) {
  controller.Error(kvstore::Driver::AnnotateErrorWithKeyDescription(
      controller.DescribeKey(controller.GetKey()), action, error));
}

template <typename Controller>
void PerformWriteback(Driver* driver, Controller controller,
                      ReadResult read_result) {
  if (!StorageGeneration::IsDirty(read_result.stamp.generation)) {
    // The read is not dirty.
    if (!StorageGeneration::IsConditional(read_result.stamp.generation) ||
        read_result.stamp.time > controller.GetTransactionNode()
                                     .transaction()
                                     ->commit_start_time()) {
      // The read was not conditional, or the read timestamp is after the
      // transaction commit timestamp.
      controller.Success(std::move(read_result.stamp));
      return;
    }
    // This is a conditional read or stale read; but not a dirty read, so
    // reissue the read.
    ReadOptions read_options;
    read_options.if_not_equal =
        StorageGeneration::Clean(std::move(read_result.stamp.generation));
    read_options.byte_range = {0, 0};
    auto future = driver->Read(controller.GetKey(), std::move(read_options));
    future.Force();
    std::move(future).ExecuteWhenReady(
        [controller](ReadyFuture<ReadResult> future) mutable {
          auto& r = future.result();
          if (!r.ok()) {
            ReportWritebackError(controller, "reading", r.status());
          } else if (r->aborted()) {
            controller.Success(std::move(r->stamp));
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
  write_options.if_equal =
      StorageGeneration::Clean(std::move(read_result.stamp.generation));
  auto future = driver->Write(controller.GetKey(),
                              std::move(read_result).optional_value(),
                              std::move(write_options));
  future.Force();
  std::move(future).ExecuteWhenReady(
      [controller](ReadyFuture<TimestampedStorageGeneration> future) mutable {
        auto& r = future.result();
        if (!r.ok()) {
          ReportWritebackError(controller, "writing", r.status());
        } else if (StorageGeneration::IsUnknown(r->generation)) {
          controller.Retry(r->time);
        } else {
          controller.Success(std::move(*r));
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
    assert(!(entry->flags_ & ReadModifyWriteEntry::kDeleted));
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
          assert(entry->flags_ & ReadModifyWriteEntry::kDeleted);
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
  return [&e](MutationEntry& other) { return e.key_.compare(other.key_); };
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

SinglePhaseMutation& GetCommittingPhase(MultiPhaseMutation& multi_phase) {
  auto* single_phase_mutation = &multi_phase.phases_;
  if (single_phase_mutation->phase_number_ !=
      multi_phase.GetTransactionNode().phase()) {
    single_phase_mutation = single_phase_mutation->next_;
    assert(single_phase_mutation->phase_number_ ==
           multi_phase.GetTransactionNode().phase());
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
  void Success(TimestampedStorageGeneration new_stamp) {
    if (auto* dr_entry = static_cast<DeleteRangeEntry*>(entry_->next_)) {
      DeletedEntryDone(*dr_entry, /*error=*/false);
      return;
    }
    WritebackSuccess(*entry_, std::move(new_stamp));
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
    kvstore_transaction_retries.Increment();
    StartWriteback(*entry_, time);
  }
};

void ReceiveWritebackCommon(ReadModifyWriteEntry& entry,
                            StorageGeneration& generation) {
  TENSORSTORE_KVSTORE_DEBUG_LOG(
      entry, "ReceiveWritebackCommon: generation=", generation);
  // Set `kTransitivelyUnconditional` and `kDirty` bits from `generation`.
  auto flags =
      (entry.flags_ & ~(ReadModifyWriteEntry::kTransitivelyUnconditional |
                        ReadModifyWriteEntry::kDirty)) |
      ReadModifyWriteEntry::kWritebackProvided;
  if (!StorageGeneration::IsConditional(generation)) {
    flags |= ReadModifyWriteEntry::kTransitivelyUnconditional;
  }
  if (generation.ClearNewlyDirty()) {
    flags |= ReadModifyWriteEntry::kDirty;
  }
  entry.flags_ = flags;
}

void StartWriteback(ReadModifyWriteEntry& entry, absl::Time staleness_bound) {
  TENSORSTORE_KVSTORE_DEBUG_LOG(
      entry, "StartWriteback: staleness_bound=", staleness_bound);
  // First mark all previous entries as not having yet provided a writeback
  // during the current writeback sequence.
  for (auto* e = &entry;;) {
    e->flags_ &= ~ReadModifyWriteEntry::kWritebackProvided;
    e = e->prev_;
    if (!e) break;
  }

  ReadModifyWriteSource::WritebackOptions writeback_options;
  writeback_options.staleness_bound = staleness_bound;
  writeback_options.writeback_mode =
      (entry.flags_ & ReadModifyWriteEntry::kDeleted)
          ? ReadModifyWriteSource::kValidateOnly
          : ReadModifyWriteSource::kNormalWriteback;
  if (!entry.prev_ && !(entry.flags_ & ReadModifyWriteEntry::kDeleted)) {
    // Fast path: This entry sequence consists of just a single entry, and is
    // not a deleted entry superseded by a `DeleteRange` operation.  We don't
    // need to track any state in the `WritebackReceiver` beyond the `entry`
    // itself, and can just forward the writeback result from
    // `ReadModifyWriteSource::KvsWriteback` directly to
    // `MultiPhaseMutation::Writeback`.
    struct WritebackReceiverImpl {
      ReadModifyWriteEntry* entry_;
      void set_error(absl::Status error) {
        ReportWritebackError(Controller{entry_}, "writing", error);
      }
      void set_cancel() { ABSL_UNREACHABLE(); }  // COV_NF_LINE
      void set_value(ReadResult read_result) {
        ReceiveWritebackCommon(*entry_, read_result.stamp.generation);
        entry_->multi_phase().Writeback(*entry_, std::move(read_result));
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
  // However, if a read-modify-write operations in the sequence is
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
      // Current writeback value, as when the writeback request was *issued* to
      // `entry`.  If `!entry->next_read_modify_write()`, then this is just
      // default constructed.
      //
      // Note that `read_result.state` and `read_result.value` are determined
      // solely from the initial writeback request to
      // `GetLastReadModifyWriteEntry()`.  However, `read_result.stamp` may be
      // affected by "skipped" entries.
      ReadResult read_result;

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
      ReceiveWritebackCommon(entry, read_result.stamp.generation);
      if (!state_->entry->next_ &&
          !(state_->entry->flags_ & ReadModifyWriteEntry::kDeleted)) {
        // `state_->entry` is the last entry in the sequence and not superseded
        // by a `DeleteRange` operation.  This writeback result is for the
        // initial writeback request.  Overwrite the existing (default
        // constructed) `state->read_result` with this initial writeback
        // request.
        state_->read_result = std::move(read_result);
      } else {
        // This branch covers two possible cases in which we update only
        // `state_->read_result.stamp` but leave `state_->read_result.state` and
        // `state_->read_result.value` unchanged:
        //
        // 1. If `state_->entry->next_ == nullptr`, then `state_->entry` is a
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
        //    actually perform a writeback for a deleted key.
        assert(!StorageGeneration::IsConditional(
            state_->read_result.stamp.generation));
        state_->read_result.stamp.time = read_result.stamp.time;
        TENSORSTORE_KVSTORE_DEBUG_LOG(entry, "Conditioning: existing_stamp=",
                                      state_->read_result.stamp.generation,
                                      ", new_stamp=", read_result.stamp);
        state_->read_result.stamp.generation = StorageGeneration::Condition(
            state_->read_result.stamp.generation,
            std::move(read_result.stamp.generation));
      }
      if (entry.flags_ & ReadModifyWriteEntry::kTransitivelyUnconditional) {
        // The writeback is still unconditional.  There may still be "skipped"
        // entries that need a writeback request.

        // Finds the first prior superseded entry for which writeback must
        // still be requested as part of the current writeback sequence.
        //
        // This ensures entries superseded by an unconditional writeback are
        // still given a chance to validate any constraints on the existing
        // read value and return an error if constraints are violated, even
        // though they do not affect the value that will be written back.
        constexpr auto GetPrevSupersededEntryToWriteback =
            [](ReadModifyWriteEntry* entry) -> ReadModifyWriteEntry* {
          while (true) {
            entry = entry->prev_;
            if (!entry) return nullptr;
            // We don't need to request writeback of `entry` if it is known that
            // its constraints are not violated.  There are two cases in which
            // this is known:
            //
            // 1. `entry` already provided a writeback in the current writeback
            // sequence
            //    (e.g. because the `ReadModifyWriteSource` of `entry->next_`
            //    requested a read).
            //
            // 2. `entry` provided a writeback in the current or a prior
            // writeback
            //    sequence, and is known to be unconditional.  In this case, it
            //    is not affected by an updated read result.
            if (!(entry->flags_ &
                  (ReadModifyWriteEntry::kWritebackProvided |
                   ReadModifyWriteEntry::kTransitivelyUnconditional))) {
              return entry;
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
              ReadModifyWriteSource::kValidateOnly;
          prev->source_->KvsWriteback(std::move(writeback_options),
                                      std::move(*this));
          return;
        }
      }
      // No remaining "skipped" entries.  Forward the combined writeback result
      // to `MultiPhaseMutation::Writeback`.
      auto* last_entry = state_->GetLastReadModifyWriteEntry();
      last_entry->multi_phase().Writeback(*last_entry,
                                          std::move(state_->read_result));
    }
  };
  entry.source_->KvsWriteback(
      std::move(writeback_options),
      SequenceWritebackReceiverImpl{
          std::unique_ptr<SequenceWritebackReceiverImpl::State>(
              new SequenceWritebackReceiverImpl::State{&entry,
                                                       staleness_bound})});
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

  dr_entry.multi_phase().Writeback(dr_entry);
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

}  // namespace

void ReadModifyWriteEntry::KvsRead(
    ReadModifyWriteTarget::TransactionalReadOptions options,
    ReadModifyWriteTarget::ReadReceiver receiver) {
  struct ReadReceiverImpl {
    ReadModifyWriteEntry* entry_;
    ReadModifyWriteTarget::ReadReceiver receiver_;
    void set_cancel() { execution::set_cancel(receiver_); }
    void set_value(ReadResult read_result) {
      {
        assert(!StorageGeneration::IsUnknown(read_result.stamp.generation));
        absl::MutexLock lock(&entry_->mutex());
        ReceiveWritebackCommon(*entry_->prev_, read_result.stamp.generation);
        entry_->flags_ |= (entry_->prev_->flags_ &
                           ReadModifyWriteEntry::kTransitivelyUnconditional);
      }
      execution::set_value(receiver_, std::move(read_result));
    }
    void set_error(absl::Status error) {
      execution::set_error(receiver_, std::move(error));
    }
  };
  if (flags_ & ReadModifyWriteEntry::kPrevDeleted) {
    execution::set_value(
        receiver,
        ReadResult{ReadResult::kMissing,
                   {},
                   {StorageGeneration::Dirty(StorageGeneration::Unknown()),
                    absl::InfiniteFuture()}});
  } else if (prev_) {
    TENSORSTORE_KVSTORE_DEBUG_LOG(*prev_, "Requesting writeback for read");
    ReadModifyWriteSource::WritebackOptions writeback_options;
    writeback_options.if_not_equal = std::move(options.if_not_equal);
    writeback_options.staleness_bound = options.staleness_bound;
    writeback_options.writeback_mode =
        ReadModifyWriteSource::kSpecifyUnchangedWriteback;
    this->prev_->source_->KvsWriteback(
        std::move(writeback_options),
        ReadReceiverImpl{this, std::move(receiver)});
  } else {
    multi_phase().Read(*this, std::move(options), std::move(receiver));
  }
}

bool ReadModifyWriteEntry::KvsReadsCommitted() {
  return prev_ == nullptr && !(flags_ & ReadModifyWriteEntry::kPrevDeleted) &&
         multi_phase().MultiPhaseReadsCommitted();
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
                      TimestampedStorageGeneration new_stamp) {
  assert(!entry.next_read_modify_write());
  for (ReadModifyWriteEntry* e = &entry;;) {
    e->source_->KvsWritebackSuccess(new_stamp);
    bool dirty = static_cast<bool>(e->flags_ & ReadModifyWriteEntry::kDirty);
    e = e->prev_;
    if (!e) break;
    if (dirty || !(e->flags_ & ReadModifyWriteEntry::kWritebackProvided)) {
      // Predecessor entries must not assume that their own writeback output is
      // the new committed state.
      new_stamp.generation = StorageGeneration::Unknown();
      new_stamp.time = absl::InfiniteFuture();
    }
  }
}

void WritebackError(ReadModifyWriteEntry& entry) {
  assert(!entry.next_read_modify_write());
  if (entry.flags_ & ReadModifyWriteEntry::kError) return;
  entry.flags_ |= ReadModifyWriteEntry::kError;
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
                                                  absl::InfiniteFuture()});
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
    entry->flags_ &= ~ReadModifyWriteEntry::kTransitivelyUnconditional;
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

  auto& single_phase_mutation = GetCommittingPhase(*this);
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

MultiPhaseMutation::ReadModifyWriteStatus MultiPhaseMutation::ReadModifyWrite(
    size_t& phase, Key key, ReadModifyWriteSource& source) {
  DebugCheckInvariantsInDestructor debug_check(*this, false);
#ifndef NDEBUG
  mutex().AssertHeld();
#endif
  auto& single_phase_mutation = GetCurrentSinglePhaseMutation(*this);
  phase = single_phase_mutation.phase_number_;
  auto* entry = MakeReadModifyWriteEntry(single_phase_mutation, std::move(key));
  entry->source_ = &source;
  entry->source_->KvsSetTarget(*entry);

  // We need to insert `entry` into the interval map for
  // `single_phase_mutation`.  This may involve marking an existing
  // `ReadModifyWriteEntry` for the same key as superseded, or splitting an
  // existing `DeleteRangeEntry` that contains `key`.

  // Find either an existing `ReadModifyWriteEntry` with the same key, or an
  // existing `DeleteRangeEntry` that contains `key`.
  auto find_result = single_phase_mutation.entries_.Find(
      [key = std::string_view(entry->key_)](MutationEntry& existing_entry) {
        auto c = key.compare(existing_entry.key_);
        if (c <= 0) return c;
        if (existing_entry.entry_type() == kReadModifyWrite) return 1;
        return KeyRange::CompareKeyAndExclusiveMax(
                   key, static_cast<DeleteRangeEntry&>(existing_entry)
                            .exclusive_max_) < 0
                   ? 0
                   : 1;
      });
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
    existing_entry->source_->KvsRevoke();
    assert(existing_entry->next_ == nullptr);
    entry->prev_ = existing_entry;
    existing_entry->next_ = entry;
    return ReadModifyWriteStatus::kExisting;
  }

  // `DeleteRangeEntry` contains `key`.  It needs to be split into a
  // "before" range and an "after" range.
  auto* existing_entry = static_cast<DeleteRangeEntry*>(find_result.node);
  assert(existing_entry->key_ <= entry->key_);
  assert(KeyRange::CompareKeyAndExclusiveMax(
             entry->key_, existing_entry->exclusive_max_) < 0);
  entry->flags_ |= (ReadModifyWriteEntry::kPrevDeleted |
                    ReadModifyWriteEntry::kTransitivelyUnconditional);
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
        return key.compare(e.key_);
      });
  if (split_result.center) {
    split_result.center->flags_ &= ~ReadModifyWriteEntry::kDeleted;
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
    // "Right" interval is non-empty.  Re-use the existing entry for the
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
      existing_rmw_entry->source_->KvsRevoke();
      if (&existing_rmw_entry->single_phase_mutation() !=
          &single_phase_mutation) {
        // Existing `ReadModifyWriteEntry` is for a prior phase.  Just move it
        // into the interval tree for that phase.  We don't mark it as
        // superseded since it is not superseded in its own phase.
        InsertIntoPriorPhase(existing_entry);
      } else {
        // Existing `ReadModifyWriteEntry` is for the current phase.  Mark it as
        // superseded and add it to the `superseded` tree.
        existing_rmw_entry->flags_ |= ReadModifyWriteEntry::kDeleted;
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
void ReadDirectly(Driver* driver, ReadModifyWriteEntry& entry,
                  ReadModifyWriteTarget::TransactionalReadOptions&& options,
                  ReadModifyWriteTarget::ReadReceiver&& receiver) {
  ReadOptions kvstore_options;
  kvstore_options.staleness_bound = options.staleness_bound;
  kvstore_options.if_not_equal = std::move(options.if_not_equal);
  execution::submit(driver->Read(entry.key_, std::move(kvstore_options)),
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

ReadModifyWriteEntry* AtomicMultiPhaseMutation::AllocateReadModifyWriteEntry() {
  return new BufferedReadModifyWriteEntry;
}
void AtomicMultiPhaseMutation::FreeReadModifyWriteEntry(
    ReadModifyWriteEntry* entry) {
  delete static_cast<BufferedReadModifyWriteEntry*>(entry);
}

namespace {

void AtomicWritebackReady(
    AtomicMultiPhaseMutation::BufferedReadModifyWriteEntry& entry) {
  if (auto* dr_entry = static_cast<DeleteRangeEntry*>(entry.next_)) {
    DeletedEntryDone(*dr_entry, /*error=*/false);
  } else {
    EntryDone(entry.single_phase_mutation(), /*error=*/false);
  }
}

}  // namespace

void AtomicMultiPhaseMutation::Writeback(ReadModifyWriteEntry& entry,
                                         ReadResult&& read_result) {
  assert(read_result.stamp.time != absl::InfinitePast());
  auto& buffered = static_cast<BufferedReadModifyWriteEntry&>(entry);
  buffered.read_result_ = std::move(read_result);
  AtomicWritebackReady(buffered);
}

void AtomicMultiPhaseMutation::Writeback(DeleteRangeEntry& entry) {
  EntryDone(entry.single_phase_mutation(), /*error=*/false);
}

void RetryAtomicWriteback(SinglePhaseMutation& single_phase_mutation,
                          absl::Time staleness_bound) {
  using BufferedReadModifyWriteEntry =
      AtomicMultiPhaseMutation::BufferedReadModifyWriteEntry;

  WritebackPhase(
      single_phase_mutation, staleness_bound, [&](ReadModifyWriteEntry& entry) {
        return static_cast<BufferedReadModifyWriteEntry&>(entry).IsOutOfDate(
            staleness_bound);
      });
}

void AtomicCommitWritebackSuccess(SinglePhaseMutation& single_phase_mutation) {
  using BufferedReadModifyWriteEntry =
      AtomicMultiPhaseMutation::BufferedReadModifyWriteEntry;
  for (auto& entry : single_phase_mutation.entries_) {
    if (entry.entry_type() == kReadModifyWrite) {
      auto& rmw_entry = static_cast<BufferedReadModifyWriteEntry&>(entry);
      WritebackSuccess(rmw_entry, std::move(rmw_entry.read_result_.stamp));
    } else {
      auto& dr_entry = static_cast<DeleteRangeEntry&>(entry);
      WritebackSuccess(dr_entry);
    }
  }
}

void AtomicMultiPhaseMutation::RevokeAllEntries() {
  assert(phases_.next_ == &phases_);
  for (auto& entry : phases_.entries_) {
    if (entry.entry_type() != kReadModifyWrite) continue;
    auto& rmw_entry = static_cast<ReadModifyWriteEntry&>(entry);
    rmw_entry.source_->KvsRevoke();
  }
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
  }
  void KvsInvalidateReadState() override {}
  void KvsWriteback(
      ReadModifyWriteSource::WritebackOptions options,
      ReadModifyWriteSource::WritebackReceiver receiver) override {
    ReadModifyWriteTarget::TransactionalReadOptions read_options = options;

    if (options.writeback_mode !=
        ReadModifyWriteSource::kSpecifyUnchangedWriteback) {
      // If `expected_stamp_.generation` is clean, then there are no prior
      // modifications.  In that case writeback can proceed immediately without
      // a prior read request.
      TimestampedStorageGeneration expected_stamp;
      {
        absl::MutexLock lock(&mutex_);
        expected_stamp = expected_stamp_;
      }
      if (StorageGeneration::IsClean(expected_stamp.generation) &&
          expected_stamp.time >= read_options.staleness_bound) {
        // Nothing to write back, just need to verify generation.
        execution::set_value(receiver, ReadResult{ReadResult::kUnspecified,
                                                  {},
                                                  std::move(expected_stamp)});
        return;
      }
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
          execution::set_error(receiver_,
                               absl::AbortedError("Generation mismatch"));
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
  void KvsWritebackSuccess(TimestampedStorageGeneration new_stamp) override {
    this->CommitDone();
  }

  absl::Mutex mutex_;

  /// Expected read generation.
  ///
  /// If not equal to `StorageGeneration::Unknown()`, `Commit` will fail if
  /// this does not match.
  TimestampedStorageGeneration expected_stamp_;

  ReadModifyWriteTarget* target_;
};
}  // namespace

Future<ReadResult> ReadViaExistingTransaction(
    Driver* driver, internal::OpenTransactionPtr& transaction, size_t& phase,
    Key key, kvstore::TransactionalReadOptions options) {
  auto [promise, future] = PromiseFuturePair<ReadResult>::Make();
  using Node = ReadViaExistingTransactionNode;
  // It is sufficient to use `WeakTransactionNodePtr`, since the transaction is
  // already kept open by the `transaction` pointer.
  internal::WeakTransactionNodePtr<Node> node;
  node.reset(new Node);
  TENSORSTORE_RETURN_IF_ERROR(
      driver->ReadModifyWrite(transaction, phase, std::move(key), *node));
  node->SetTransaction(*transaction);
  node->SetPhase(phase);
  TENSORSTORE_RETURN_IF_ERROR(node->Register());
  struct InitialReadReceiverImpl {
    internal::OpenTransactionNodePtr<Node> node_;
    Promise<ReadResult> promise_;

    void set_value(ReadResult read_result) {
      {
        absl::MutexLock lock(&node_->mutex_);
        node_->expected_stamp_ = read_result.stamp;
      }
      promise_.SetResult(std::move(read_result));
    }
    void set_cancel() {}
    void set_error(absl::Status error) { promise_.SetResult(std::move(error)); }
  };
  node->target_->KvsRead(std::move(options),
                         InitialReadReceiverImpl{
                             internal::OpenTransactionNodePtr<Node>(node.get()),
                             std::move(promise)});
  return std::move(future);
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
  }
  void KvsInvalidateReadState() override {}
  void KvsWriteback(
      ReadModifyWriteSource::WritebackOptions options,
      ReadModifyWriteSource::WritebackReceiver receiver) override {
    if (!StorageGeneration::IsConditional(read_result_.stamp.generation)) {
      // Writeback is unconditional.  Therefore, the existing read state does
      // not need to be requested.
      execution::set_value(receiver, read_result_);
      return;
    }
    // Writeback is conditional.  A read request must be performed in order to
    // determine an up-to-date writeback value (which may be required by a
    // subsequent read-modify-write operation layered on top of this operation).
    ReadModifyWriteTarget::TransactionalReadOptions read_options;
    read_options.if_not_equal =
        StorageGeneration::Clean(read_result_.stamp.generation);
    read_options.staleness_bound = options.staleness_bound;
    struct ReadReceiverImpl {
      WriteViaExistingTransactionNode& source_;
      ReadModifyWriteSource::WritebackReceiver receiver_;
      void set_value(ReadResult read_result) {
        auto& existing_generation = source_.read_result_.stamp.generation;
        auto clean_generation = StorageGeneration::Clean(existing_generation);
        // Check if the new read generation matches the condition specified in
        // the read request.
        if (read_result.stamp.generation == clean_generation ||
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
          source_.read_result_.stamp.generation.MarkDirty();
        } else {
          // Read generation does not match.  Since the constraint was violated,
          // just provide the existing read value as the writeback value.  The
          // fact that `source_.read_result_.stamp.generation` is not marked
          // dirty indicates to the transaction machinery that writeback is not
          // actually necessary.
          assert(
              !StorageGeneration::IsNewlyDirty(read_result.stamp.generation));
          source_.read_result_ = std::move(read_result);
          source_.if_equal_no_value_ = false;
        }
        execution::set_value(receiver_, source_.read_result_);
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
  void KvsWritebackSuccess(TimestampedStorageGeneration new_stamp) override {
    if (!StorageGeneration::IsNewlyDirty(read_result_.stamp.generation)) {
      new_stamp = TimestampedStorageGeneration{};
    } else if (new_stamp.time == absl::InfiniteFuture()) {
      new_stamp.generation = StorageGeneration::Invalid();
    }
    promise_.SetResult(std::move(new_stamp));
    this->CommitDone();
  }

  /// Promise that will be marked ready when the write operation is committed
  /// (or fails).
  Promise<TimestampedStorageGeneration> promise_;

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

  ReadModifyWriteTarget* target_;
};
}  // namespace

Future<TimestampedStorageGeneration> WriteViaExistingTransaction(
    Driver* driver, internal::OpenTransactionPtr& transaction, size_t& phase,
    Key key, std::optional<Value> value, WriteOptions options) {
  ReadResult read_result;
  if (value) {
    read_result.state = ReadResult::kValue;
    read_result.value = std::move(*value);
  } else {
    read_result.state = ReadResult::kMissing;
  }
  if (StorageGeneration::IsUnknown(options.if_equal)) {
    read_result.stamp.time = absl::InfiniteFuture();
  } else {
    assert(StorageGeneration::IsClean(options.if_equal));
    read_result.stamp.time = absl::Time();
  }
  bool if_equal_no_value = StorageGeneration::IsNoValue(options.if_equal);
  read_result.stamp.generation = std::move(options.if_equal);
  read_result.stamp.generation.MarkDirty();
  auto [promise, future] =
      PromiseFuturePair<TimestampedStorageGeneration>::Make();
  using Node = WriteViaExistingTransactionNode;
  internal::WeakTransactionNodePtr<Node> node;
  node.reset(new Node);
  node->promise_ = promise;
  node->read_result_ = std::move(read_result);
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
                                     std::move(value), std::move(options));
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

}  // namespace kvstore
}  // namespace tensorstore
