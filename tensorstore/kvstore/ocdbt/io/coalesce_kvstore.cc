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

#include "tensorstore/kvstore/ocdbt/io/coalesce_kvstore.h"

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/thread/schedule_at.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");

absl::Cord DeepCopyCord(const absl::Cord& cord) {
  // If the Cord is flat, skipping the CordBuilder improves performance.
  if (std::optional<absl::string_view> flat = cord.TryFlat();
      flat.has_value()) {
    return absl::Cord(*flat);
  }
  internal::FlatCordBuilder builder(cord.size(), false);
  for (absl::string_view s : cord.Chunks()) {
    builder.Append(s);
  }
  return std::move(builder).Build();
}

absl::Cord MaybeDeepCopyCord(absl::Cord cord) {
  if (cord.EstimatedMemoryUsage() > (cord.size() * 1.2)) {
    return DeepCopyCord(cord);
  }
  return cord;
}

struct PendingRead : public internal::AtomicReferenceCount<PendingRead> {
  kvstore::Key key;

  struct Op {
    kvstore::ReadOptions options;
    Promise<kvstore::ReadResult> promise;
  };
  std::vector<Op> pending_ops;
};

struct PendingReadEq {
  using is_transparent = void;

  inline bool operator()(const PendingRead& a, const PendingRead& b) const {
    return a.key == b.key;
  }

  inline bool operator()(std::string_view a, std::string_view b) const {
    return a == b;
  }
  inline bool operator()(const PendingRead& a, std::string_view b) const {
    return a.key == b;
  }
  inline bool operator()(std::string_view a, const PendingRead& b) const {
    return a == b.key;
  }

  inline bool operator()(std::string_view a,
                         const internal::IntrusivePtr<PendingRead>& b) const {
    return b == nullptr ? false : PendingReadEq{}(a, *b);
  }

  inline bool operator()(const internal::IntrusivePtr<PendingRead>& a,
                         std::string_view b) const {
    return a == nullptr ? false : PendingReadEq{}(*a, b);
  }

  inline bool operator()(const internal::IntrusivePtr<PendingRead>& a,
                         const internal::IntrusivePtr<PendingRead>& b) const {
    return a->key == b->key;
  }
};

struct PendingReadHash {
  using is_transparent = void;

  size_t operator()(std::string_view k) const { return absl::HashOf(k); }
  size_t operator()(const internal::IntrusivePtr<PendingRead>& k) const {
    return absl::HashOf(k->key);
  }
};

class CoalesceKvStoreDriver final : public kvstore::Driver {
 public:
  explicit CoalesceKvStoreDriver(kvstore::DriverPtr base, size_t threshold,
                                 size_t merged_threshold,
                                 absl::Duration interval, Executor executor)
      : base_(std::move(base)),
        threshold_(threshold),
        merged_threshold_(merged_threshold),
        interval_(interval),
        thread_pool_executor_(std::move(executor)) {}

  ~CoalesceKvStoreDriver() override = default;

  Future<ReadResult> Read(Key key, ReadOptions options = {}) override;

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override {
    return base_->Write(std::move(key), std::move(value), std::move(options));
  }

  absl::Status ReadModifyWrite(internal::OpenTransactionPtr& transaction,
                               size_t& phase, Key key,
                               ReadModifyWriteSource& source) override {
    return base_->ReadModifyWrite(transaction, phase, std::move(key), source);
  }

  absl::Status TransactionalDeleteRange(
      const internal::OpenTransactionPtr& transaction,
      KeyRange range) override {
    return base_->TransactionalDeleteRange(transaction, std::move(range));
  }

  Future<const void> DeleteRange(KeyRange range) override {
    return base_->DeleteRange(std::move(range));
  }

  void ListImpl(ListOptions options, ListReceiver receiver) override {
    return base_->ListImpl(std::move(options), std::move(receiver));
  }

  std::string DescribeKey(std::string_view key) override {
    return base_->DescribeKey(key);
  }

  Result<kvstore::DriverSpecPtr> GetBoundSpec() const override {
    return base_->GetBoundSpec();
  }

  kvstore::SupportedFeatures GetSupportedFeatures(
      const KeyRange& key_range) const final {
    return base_->GetSupportedFeatures(key_range);
  }

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const override {
    return base_->GarbageCollectionVisit(visitor);
  }

  void StartNextRead(internal::IntrusivePtr<PendingRead> state_ptr);

 private:
  kvstore::DriverPtr base_;
  size_t threshold_;
  size_t merged_threshold_;
  absl::Duration interval_;
  Executor thread_pool_executor_;

  absl::Mutex mu_;
  absl::flat_hash_set<internal::IntrusivePtr<PendingRead>, PendingReadHash,
                      PendingReadEq>
      pending_ ABSL_GUARDED_BY(mu_);
};

Future<kvstore::ReadResult> CoalesceKvStoreDriver::Read(Key key,
                                                        ReadOptions options) {
  internal::IntrusivePtr<PendingRead> state_ptr;
  {
    absl::MutexLock l(&mu_);
    auto it = pending_.find(std::string_view(key));
    if (it != pending_.end()) {
      /// This key is already "reserved" by a PendingRead object, so enqueue
      /// the read for later.
      auto& state = *it;
      auto op = PromiseFuturePair<ReadResult>::Make();
      state->pending_ops.emplace_back(
          PendingRead::Op{std::move(options), std::move(op.promise)});
      return std::move(op.future);
    } else {
      /// This key is unowned, so mark it as "reserved" and start a read.
      state_ptr = internal::MakeIntrusivePtr<PendingRead>();
      state_ptr->key = key;
      bool inserted;
      std::tie(it, inserted) = pending_.insert(state_ptr);

      if (interval_ != absl::ZeroDuration()) {
        // interval based read, add current read to pending queue and schedule
        // the next read
        internal::ScheduleAt(
            absl::Now() + interval_,
            [self = internal::IntrusivePtr<CoalesceKvStoreDriver>(this),
             state = std::move(state_ptr)] {
              auto& executor = self->thread_pool_executor_;
              executor([self = std::move(self), state = std::move(state)] {
                self->StartNextRead(std::move(state));
              });
            });

        auto& state = *it;
        auto op = PromiseFuturePair<ReadResult>::Make();
        state->pending_ops.emplace_back(
            PendingRead::Op{std::move(options), std::move(op.promise)});
        return std::move(op.future);
      }
    }
  }

  // non-interval based trigger
  auto future = base_->Read(key, std::move(options));
  future.ExecuteWhenReady(
      [self = internal::IntrusivePtr<CoalesceKvStoreDriver>(this),
       state = std::move(state_ptr)](ReadyFuture<ReadResult>) {
        auto& executor = self->thread_pool_executor_;
        executor([self = std::move(self), state = std::move(state)] {
          self->StartNextRead(std::move(state));
        });
      });

  return future;
}

struct MergeValue {
  kvstore::ReadOptions options;

  struct Entry {
    OptionalByteRangeRequest byte_range;
    Promise<kvstore::ReadResult> promise;
  };
  std::vector<Entry> subreads;
};

void OnReadComplete(MergeValue merge_values,
                    ReadyFuture<kvstore::ReadResult> ready) {
  // If there is no value, or there is a single subread, then forward the
  // ReadResult to all subreads.
  if (!ready.result().ok() || !ready.value().has_value() ||
      merge_values.subreads.size() == 1) {
    for (const auto& e : merge_values.subreads) {
      e.promise.SetResult(ready.result());
    }
  } else {
    /// Otherwise extract the desired range and return that.
    kvstore::ReadResult result = ready.value();
    absl::Cord value = std::move(result.value);

    for (const auto& e : merge_values.subreads) {
      size_t request_start, request_size;
      if (e.byte_range.inclusive_min < 0) {
        request_start = value.size() + e.byte_range.inclusive_min;
      } else {
        request_start = e.byte_range.inclusive_min -
                        merge_values.options.byte_range.inclusive_min;
      }
      if (e.byte_range.exclusive_max == -1) {
        request_size = std::numeric_limits<size_t>::max();
      } else {
        request_size = e.byte_range.exclusive_max - e.byte_range.inclusive_min;
      }
      result.value =
          MaybeDeepCopyCord(value.Subcord(request_start, request_size));
      e.promise.SetResult(result);
    }
  }
}

void CoalesceKvStoreDriver::StartNextRead(
    internal::IntrusivePtr<PendingRead> state_ptr) {
  std::vector<PendingRead::Op> pending;
  {
    absl::MutexLock l(&mu_);
    if (state_ptr->pending_ops.empty()) {
      // No buffered reads for this key; remove the "reservation" and exit.
      pending_.erase(state_ptr->key);
      return;
    } else {
      std::swap(pending, state_ptr->pending_ops);
    }
  }

  if (interval_ != absl::ZeroDuration()) {
    // schedule the next read
    internal::ScheduleAt(
        absl::Now() + interval_,
        [self = internal::IntrusivePtr<CoalesceKvStoreDriver>(this),
         state = state_ptr] {
          auto& executor = self->thread_pool_executor_;
          executor([self = std::move(self), state = std::move(state)] {
            self->StartNextRead(std::move(state));
          });
        });
  }

  // Order the pending reads by the kvstore::ReadOption fields such that the
  // byte_ranges of compatible option sequences are seen in order.
  std::sort(pending.begin(), pending.end(), [](const auto& a, const auto& b) {
    return std::tie(a.options.generation_conditions.if_equal.value,
                    a.options.generation_conditions.if_not_equal.value,
                    a.options.byte_range.inclusive_min,
                    a.options.byte_range.exclusive_max) <
           std::tie(b.options.generation_conditions.if_equal.value,
                    b.options.generation_conditions.if_not_equal.value,
                    b.options.byte_range.inclusive_min,
                    b.options.byte_range.exclusive_max);
  });

  kvstore::Key key = state_ptr->key;

  MergeValue merged;
  const auto& first_pending = pending.front();
  merged.options = first_pending.options;
  // Add to queue.
  merged.subreads.emplace_back(
      MergeValue::Entry{std::move(first_pending.options.byte_range),
                        std::move(first_pending.promise)});

  for (size_t i = 1; i < pending.size(); ++i) {
    auto& e = pending[i];
    if (e.options.generation_conditions.if_equal !=
            merged.options.generation_conditions.if_equal ||
        e.options.generation_conditions.if_not_equal !=
            merged.options.generation_conditions.if_not_equal ||
        // Don't merge suffix length byte requests with non-suffix-length byte
        // requests.
        (e.options.byte_range.inclusive_min < 0) !=
            (merged.options.byte_range.inclusive_min < 0)) {
      // The options differ from the prior options, so issue the pending
      // request and start another.
      assert(!merged.subreads.empty());
      auto f = base_->Read(key, merged.options);
      f.ExecuteWhenReady(
          [merged = std::move(merged)](ReadyFuture<kvstore::ReadResult> ready) {
            OnReadComplete(std::move(merged), std::move(ready));
          });
      merged = MergeValue{};
      merged.options = e.options;
    } else if (merged.options.byte_range.exclusive_max != -1 &&
               ((e.options.byte_range.inclusive_min -
                     merged.options.byte_range.exclusive_max >
                 threshold_) ||
                (merged_threshold_ > 0 &&
                 merged.options.byte_range.size() > merged_threshold_))) {
      // The distance from the end of the prior read to the beginning of the
      // next read exceeds threshold_ or the total merged_size exceeds
      // merged_threshold_, so issue the pending request and start
      // another.
      assert(!merged.subreads.empty());
      auto f = base_->Read(key, merged.options);
      f.ExecuteWhenReady(
          [merged = std::move(merged)](ReadyFuture<kvstore::ReadResult> ready) {
            OnReadComplete(std::move(merged), std::move(ready));
          });
      merged = MergeValue{};
      merged.options = e.options;
    } else {
      // Pick latest staleness bounds
      merged.options.staleness_bound =
          std::max(merged.options.staleness_bound, e.options.staleness_bound);
      // Merge byte_ranges
      merged.options.byte_range.inclusive_min =
          std::min(merged.options.byte_range.inclusive_min,
                   e.options.byte_range.inclusive_min);
      if (merged.options.byte_range.exclusive_max != -1) {
        if (e.options.byte_range.exclusive_max != -1) {
          merged.options.byte_range.exclusive_max =
              std::max(merged.options.byte_range.exclusive_max,
                       e.options.byte_range.exclusive_max);
        } else {
          merged.options.byte_range.exclusive_max = -1;
        }
      }
    }

    // Add to queue.
    merged.subreads.emplace_back(MergeValue::Entry{
        std::move(e.options.byte_range), std::move(e.promise)});
  }

  // Issue final request. This request will trigger additional reads via
  // StartNextRead.
  assert(!merged.subreads.empty());
  auto f = base_->Read(key, merged.options);
  f.ExecuteWhenReady(
      [self = internal::IntrusivePtr<CoalesceKvStoreDriver>(this),
       merged = std::move(merged),
       state = std::move(state_ptr)](ReadyFuture<kvstore::ReadResult> ready) {
        auto& executor = self->thread_pool_executor_;
        executor([self = std::move(self), merged = std::move(merged),
                  state = std::move(state), ready = std::move(ready)] {
          OnReadComplete(std::move(merged), std::move(ready));
          if (self->interval_ == absl::ZeroDuration()) {
            self->StartNextRead(std::move(state));
          }
        });
      });
}

}  // namespace

kvstore::DriverPtr MakeCoalesceKvStoreDriver(kvstore::DriverPtr base,
                                             size_t threshold,
                                             size_t merged_threshold,
                                             absl::Duration interval,
                                             Executor executor) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Coalescing reads with threshold: " << threshold
      << ", merged_threshold: " << merged_threshold
      << ", interval: " << interval;
  return internal::MakeIntrusivePtr<CoalesceKvStoreDriver>(
      std::move(base), threshold, merged_threshold, interval,
      std::move(executor));
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
