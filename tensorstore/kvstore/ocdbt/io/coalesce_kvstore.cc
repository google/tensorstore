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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/debug_log.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {

absl::Cord DeepCopyCord(const absl::Cord& cord) {
  // If the Cord is flat, skipping the CordBuilder improves performance.
  if (std::optional<absl::string_view> flat = cord.TryFlat();
      flat.has_value()) {
    return absl::Cord(*flat);
  }
  internal::FlatCordBuilder builder(cord.size());
  size_t offset = 0;
  for (absl::string_view s : cord.Chunks()) {
    std::memcpy(builder.data() + offset, s.data(), s.size());
    offset += s.size();
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
  explicit CoalesceKvStoreDriver(kvstore::DriverPtr base, size_t threshold)
      : base_(std::move(base)), threshold_(threshold) {}

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

  void ListImpl(ListOptions options,
                AnyFlowReceiver<absl::Status, Key> receiver) override {
    return base_->ListImpl(std::move(options), std::move(receiver));
  }

  void EncodeCacheKey(std::string* out) const override {
    return base_->EncodeCacheKey(out);
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
      pending_.insert(state_ptr);
    }
  }

  auto future = base_->Read(key, std::move(options));
  future.ExecuteWhenReady(
      [self = internal::IntrusivePtr<CoalesceKvStoreDriver>(this),
       state = std::move(state_ptr)](ReadyFuture<ReadResult>) {
        self->StartNextRead(std::move(state));
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
      auto request_start = e.byte_range.inclusive_min -
                           merge_values.options.byte_range.inclusive_min;
      auto request_size = e.byte_range.exclusive_max.value_or(
                              std::numeric_limits<uint64_t>::max()) -
                          e.byte_range.inclusive_min;
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

  // Order the pending reads by the kvstore::ReadOption fields such that the
  // byte_ranges of compatible option sequences are seen in order.
  std::sort(pending.begin(), pending.end(), [](const auto& a, const auto& b) {
    return std::tie(a.options.if_equal.value, a.options.if_not_equal.value,
                    a.options.byte_range.inclusive_min,
                    a.options.byte_range.exclusive_max) <
           std::tie(b.options.if_equal.value, b.options.if_not_equal.value,
                    b.options.byte_range.inclusive_min,
                    b.options.byte_range.exclusive_max);
  });

  kvstore::Key key = state_ptr->key;

  MergeValue merged;
  merged.options = pending.front().options;

  for (auto& e : pending) {
    if (e.options.if_equal != merged.options.if_equal ||
        e.options.if_not_equal != merged.options.if_not_equal) {
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
    } else if (merged.options.byte_range.exclusive_max.has_value() &&
               e.options.byte_range.inclusive_min >
                   (merged.options.byte_range.exclusive_max.value() +
                    threshold_)) {
      // The distance from the end of the prior read to the beginning of the
      // next read exceeds threshold_, so issue the pending request and start
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
      if (merged.options.byte_range.exclusive_max.has_value()) {
        if (e.options.byte_range.exclusive_max.has_value()) {
          merged.options.byte_range.exclusive_max =
              std::max(merged.options.byte_range.exclusive_max.value(),
                       e.options.byte_range.exclusive_max.value());
        } else {
          merged.options.byte_range.exclusive_max = std::nullopt;
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
        OnReadComplete(std::move(merged), std::move(ready));
        self->StartNextRead(std::move(state));
      });
}

}  // namespace

kvstore::DriverPtr MakeCoalesceKvStoreDriver(kvstore::DriverPtr base,
                                             size_t threshold) {
  ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG)
      << "Coalescing reads with threshold " << threshold;
  return internal::MakeIntrusivePtr<CoalesceKvStoreDriver>(std::move(base),
                                                           threshold);
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
