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

#ifndef TENSORSTORE_INTERNAL_CACHE_KVS_BACKED_CACHE_H_
#define TENSORSTORE_INTERNAL_CACHE_KVS_BACKED_CACHE_H_

/// \file
///
/// Integrates `AsyncCache` with `kvstore::Driver`.

#include <stddef.h>

#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_modify_write.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/future_sender.h"  // IWYU pragma: keep
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// KvsBackedCache metric increment functions.
void KvsBackedCache_IncrementReadUnchangedMetric();
void KvsBackedCache_IncrementReadChangedMetric();
void KvsBackedCache_IncrementReadErrorMetric();

/// Base class that integrates an `AsyncCache` with a `kvstore::Driver`.
///
/// Each cache entry is assumed to correspond one-to-one with a key in a
/// `kvstore::Driver`, defined by the `GetKeyValueStoreKey` method.
///
/// To use this class, define a `Derived` class that inherits from
/// `KvsBackedCache<Parent>`, where `Parent` is the desired base class.  The
/// derived class is responsible for defining:
///
/// 1. `DoDecode`, which decodes an `std::optional<absl::Cord>` read from
///    the `kvstore::Driver` into an `std::shared_ptr<const ReadData>` object
///    (see `DoDecode`);
///
/// 2. (if writing is supported) `DoEncode`, which encodes an
///    `std::shared_ptr<const ReadData>` object into an
///    `std::optional<absl::Cord>` to write it back to the
///    `kvstore::Driver`.
///
/// 3. overrides `GetKeyValueStoreKey` if necessary.
///
/// This class takes care of reading from and writing to the
/// `kvstore::Driver`, and handling the timestamps and `StorageGeneration`
/// values.
///
/// \tparam Parent Parent class, must inherit from (or equal) `AsyncCache`.
template <typename Derived, typename Parent>
class KvsBackedCache : public Parent {
  static_assert(std::is_base_of_v<AsyncCache, Parent>);

 public:
  /// Constructs a `KvsBackedCache`.
  ///
  /// \param kvstore The `kvstore::Driver` to use.  If `nullptr`,
  ///     `SetKvStoreDriver` must be called before any read or write operations
  ///     are performed.
  /// \param args Arguments to forward to the `Parent` constructor.
  template <typename... U>
  explicit KvsBackedCache(kvstore::DriverPtr kvstore_driver, U&&... args)
      : Parent(std::forward<U>(args)...) {
    SetKvStoreDriver(std::move(kvstore_driver));
  }

  class TransactionNode;

  class Entry : public Parent::Entry {
   public:
    using OwningCache = KvsBackedCache;

    /// Defines the mapping from a cache entry to a kvstore key.
    ///
    /// By default the cache entry key is used, but derived classes may override
    /// this behavior.
    virtual std::string GetKeyValueStoreKey() {
      return std::string{this->key()};
    }

    template <typename EntryOrNode>
    struct DecodeReceiverImpl {
      EntryOrNode* self_;
      TimestampedStorageGeneration stamp_;
      void set_error(absl::Status error) {
        self_->ReadError(
            GetOwningEntry(*self_).AnnotateError(error,
                                                 /*reading=*/true));
      }
      void set_cancel() { set_error(absl::CancelledError("")); }
      void set_value(std::shared_ptr<const void> data) {
        AsyncCache::ReadState read_state;
        read_state.stamp = std::move(stamp_);
        read_state.data = std::move(data);
        self_->ReadSuccess(std::move(read_state));
      }
    };

    template <typename EntryOrNode>
    struct ReadReceiverImpl {
      EntryOrNode* entry_or_node_;
      std::shared_ptr<const void> existing_read_data_;
      void set_value(kvstore::ReadResult read_result) {
        if (read_result.aborted()) {
          ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
              << *entry_or_node_
              << "Value has not changed, stamp=" << read_result.stamp;
          KvsBackedCache_IncrementReadUnchangedMetric();
          // Value has not changed.
          entry_or_node_->ReadSuccess(AsyncCache::ReadState{
              std::move(existing_read_data_), std::move(read_result.stamp)});
          return;
        }
        ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
            << *entry_or_node_ << "DoDecode: " << read_result.stamp;
        KvsBackedCache_IncrementReadChangedMetric();
        GetOwningEntry(*entry_or_node_)
            .DoDecode(std::move(read_result).optional_value(),
                      DecodeReceiverImpl<EntryOrNode>{
                          entry_or_node_, std::move(read_result.stamp)});
      }
      void set_error(absl::Status error) {
        KvsBackedCache_IncrementReadErrorMetric();
        entry_or_node_->ReadError(GetOwningEntry(*entry_or_node_)
                                      .AnnotateError(error, /*reading=*/true));
      }
      void set_cancel() { ABSL_UNREACHABLE(); }  // COV_NF_LINE
    };

    /// Implements reading for the `AsyncCache` interface.
    ///
    /// Reads from the `kvstore::Driver` and invokes `DoDecode` with the result.
    ///
    /// If an error occurs, calls `ReadError` directly without invoking
    /// `DoDecode`.
    void DoRead(AsyncCache::AsyncCacheReadRequest request) final {
      kvstore::ReadOptions kvstore_options;
      kvstore_options.staleness_bound = request.staleness_bound;
      auto read_state = AsyncCache::ReadLock<void>(*this).read_state();
      kvstore_options.generation_conditions.if_not_equal =
          std::move(read_state.stamp.generation);
      kvstore_options.batch = request.batch;
      auto& cache = GetOwningCache(*this);
      auto future = cache.kvstore_driver_->Read(this->GetKeyValueStoreKey(),
                                                std::move(kvstore_options));
      execution::submit(
          std::move(future),
          ReadReceiverImpl<Entry>{this, std::move(read_state.data)});
    }

    using DecodeReceiver =
        AnyReceiver<absl::Status,
                    std::shared_ptr<const typename Derived::ReadData>>;

    /// Decodes a value from the kvstore into a `ReadData` object.
    ///
    /// The derived class implementation should use a separate executor to
    /// complete any expensive computations.
    virtual void DoDecode(std::optional<absl::Cord> value,
                          DecodeReceiver receiver) = 0;

    using EncodeReceiver = AnyReceiver<absl::Status, std::optional<absl::Cord>>;

    /// Encodes a `ReadData` object into a value to write back to the
    /// kvstore.
    ///
    /// The derived class implementation should synchronously perform any
    /// operations that require the lock be held , and then use a separate
    /// executor for any expensive computations.
    virtual void DoEncode(
        std::shared_ptr<const typename Derived::ReadData> read_data,
        EncodeReceiver receiver) {
      ABSL_UNREACHABLE();  // COV_NF_LINE
    }

    absl::Status AnnotateError(const absl::Status& error, bool reading) {
      return GetOwningCache(*this).kvstore_driver_->AnnotateError(
          this->GetKeyValueStoreKey(), reading ? "reading" : "writing", error);
    }
  };

  class TransactionNode : public Parent::TransactionNode,
                          public kvstore::ReadModifyWriteSource {
   public:
    using OwningCache = KvsBackedCache;
    using Parent::TransactionNode::TransactionNode;

    absl::Status DoInitialize(
        internal::OpenTransactionPtr& transaction) override {
      TENSORSTORE_RETURN_IF_ERROR(
          Parent::TransactionNode::DoInitialize(transaction));
      size_t phase;
      TENSORSTORE_RETURN_IF_ERROR(
          GetOwningCache(*this).kvstore_driver()->ReadModifyWrite(
              transaction, phase, GetOwningEntry(*this).GetKeyValueStoreKey(),
              std::ref(*this)));
      this->SetPhase(phase);
      if (this->target_->KvsReadsCommitted()) {
        this->SetReadsCommitted();
      }
      return absl::OkStatus();
    }

    void DoRead(AsyncCache::AsyncCacheReadRequest request) final {
      auto read_state = AsyncCache::ReadLock<void>(*this).read_state();
      kvstore::TransactionalReadOptions kvstore_options;
      kvstore_options.generation_conditions.if_not_equal =
          std::move(read_state.stamp.generation);
      kvstore_options.staleness_bound = request.staleness_bound;
      kvstore_options.batch = request.batch;
      target_->KvsRead(
          std::move(kvstore_options),
          typename Entry::template ReadReceiverImpl<TransactionNode>{
              this, std::move(read_state.data)});
    }

    using ReadModifyWriteSource = kvstore::ReadModifyWriteSource;
    using ReadModifyWriteTarget = kvstore::ReadModifyWriteTarget;

    // Implementation of the `ReadModifyWriteSource` interface:

    void KvsSetTarget(ReadModifyWriteTarget& target) override {
      target_ = &target;
    }

    void KvsInvalidateReadState() override {
      if (this->target_->KvsReadsCommitted()) {
        this->SetReadsCommitted();
      }
      this->InvalidateReadState();
    }

    void KvsWriteback(
        ReadModifyWriteSource::WritebackOptions options,
        ReadModifyWriteSource::WritebackReceiver receiver) override {
      ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
          << *this << "KvsWriteback: if_not_equal="
          << options.generation_conditions.if_not_equal
          << ", staleness_bound=" << options.staleness_bound
          << ", mode=" << options.writeback_mode;
      auto read_state = AsyncCache::ReadLock<void>(*this).read_state();
      if (!StorageGeneration::IsUnknown(
              options.generation_conditions.if_not_equal) &&
          options.generation_conditions.if_not_equal ==
              read_state.stamp.generation &&
          read_state.stamp.time >= options.staleness_bound) {
        ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
            << *this << "KvsWriteback: skipping because condition is satisfied";
        return execution::set_value(receiver, kvstore::ReadResult::Unspecified(
                                                  std::move(read_state.stamp)));
      }
      if (!StorageGeneration::IsUnknown(require_repeatable_read_) &&
          read_state.stamp.time < options.staleness_bound) {
        // Read required to validate repeatable read.
        auto read_future = this->Read({options.staleness_bound});
        read_future.Force();
        read_future.ExecuteWhenReady(
            [this, options = std::move(options),
             receiver =
                 std::move(receiver)](ReadyFuture<const void> future) mutable {
              this->KvsWriteback(std::move(options), std::move(receiver));
            });
        return;
      }
      struct EncodeReceiverImpl {
        TransactionNode* self_;
        TimestampedStorageGeneration update_stamp_;
        ReadModifyWriteSource::WritebackReceiver receiver_;
        void set_error(absl::Status error) {
          error = GetOwningEntry(*self_).AnnotateError(std::move(error),
                                                       /*reading=*/false);
          execution::set_error(receiver_, std::move(error));
        }
        void set_cancel() { ABSL_UNREACHABLE(); }  // COV_NF_LINE
        void set_value(std::optional<absl::Cord> value) {
          kvstore::ReadResult read_result =
              value ? kvstore::ReadResult::Value(std::move(*value),
                                                 std::move(update_stamp_))
                    : kvstore::ReadResult::Missing(std::move(update_stamp_));
          execution::set_value(receiver_, std::move(read_result));
        }
      };
      struct ApplyReceiverImpl {
        TransactionNode* self_;
        StorageGeneration if_not_equal_;
        ReadModifyWriteSource::WritebackMode writeback_mode_;
        ReadModifyWriteSource::WritebackReceiver receiver_;
        void set_error(absl::Status error) {
          execution::set_error(receiver_, std::move(error));
        }
        void set_cancel() { ABSL_UNREACHABLE(); }  // COV_NF_LINE
        void set_value(AsyncCache::ReadState update) {
          if (!StorageGeneration::IsUnknown(self_->require_repeatable_read_)) {
            if (!StorageGeneration::IsConditional(update.stamp.generation)) {
              update.stamp.generation = StorageGeneration::Condition(
                  update.stamp.generation, self_->require_repeatable_read_);
              auto read_stamp = AsyncCache::ReadLock<void>(*self_).stamp();
              if (!StorageGeneration::IsUnknown(read_stamp.generation) &&
                  read_stamp.generation != self_->require_repeatable_read_) {
                execution::set_error(receiver_, GetGenerationMismatchError());
                return;
              }
              update.stamp.time = read_stamp.time;
            } else if (!StorageGeneration::IsConditionalOn(
                           update.stamp.generation,
                           self_->require_repeatable_read_)) {
              execution::set_error(receiver_, GetGenerationMismatchError());
              return;
            }
          }
          if (!StorageGeneration::NotEqualOrUnspecified(update.stamp.generation,
                                                        if_not_equal_)) {
            return execution::set_value(
                receiver_,
                kvstore::ReadResult::Unspecified(std::move(update.stamp)));
          }
          if (!StorageGeneration::IsInnerLayerDirty(update.stamp.generation) &&
              writeback_mode_ !=
                  ReadModifyWriteSource::kSpecifyUnchangedWriteback) {
            ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
                << *self_ << "DoApply: if_not_equal=" << if_not_equal_
                << ", mode=" << writeback_mode_
                << ", unmodified: " << update.stamp;
            if (StorageGeneration::IsUnknown(update.stamp.generation)) {
              self_->new_data_ = std::nullopt;
            } else {
              self_->new_data_ = std::move(update.data);
            }
            return execution::set_value(
                receiver_,
                kvstore::ReadResult::Unspecified(std::move(update.stamp)));
          }
          ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
              << *self_ << "DoApply: if_not_equal=" << if_not_equal_
              << ", mode=" << writeback_mode_ << ", encoding: " << update.stamp
              << ", commit_started=" << self_->transaction()->commit_started();
          self_->new_data_ = update.data;
          ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
              << *self_ << "DoEncode";
          auto update_data =
              std::static_pointer_cast<const typename Derived::ReadData>(
                  std::move(update.data));
          GetOwningEntry(*self_).DoEncode(
              std::move(update_data),
              EncodeReceiverImpl{self_, std::move(update.stamp),
                                 std::move(receiver_)});
        }
      };
      AsyncCache::TransactionNode::ApplyOptions apply_options;
      apply_options.staleness_bound = options.staleness_bound;
      switch (options.writeback_mode) {
        case ReadModifyWriteSource::kValidateOnly:
          apply_options.apply_mode =
              AsyncCache::TransactionNode::ApplyOptions::kValidateOnly;
          break;
        case ReadModifyWriteSource::kSpecifyUnchangedWriteback:
          apply_options.apply_mode =
              AsyncCache::TransactionNode::ApplyOptions::kSpecifyUnchanged;
          break;
        case ReadModifyWriteSource::kNormalWriteback:
          apply_options.apply_mode =
              AsyncCache::TransactionNode::ApplyOptions::kNormal;
          break;
      }
      this->DoApply(
          std::move(apply_options),
          ApplyReceiverImpl{
              this, std::move(options.generation_conditions.if_not_equal),
              options.writeback_mode, std::move(receiver)});
    }

    void KvsWritebackSuccess(TimestampedStorageGeneration new_stamp) override {
      if (new_data_) {
        this->WritebackSuccess(
            AsyncCache::ReadState{std::move(*new_data_), std::move(new_stamp)});
      } else {
        // Unmodified.
        this->WritebackSuccess(AsyncCache::ReadState{});
      }
    }
    void KvsWritebackError() override { this->WritebackError(); }

    void KvsRevoke() override { this->Revoke(); }

    // Must be called with `mutex()` held.
    virtual absl::Status RequireRepeatableRead(
        const StorageGeneration& generation) {
      this->DebugAssertMutexHeld();
      if (!StorageGeneration::IsUnknown(require_repeatable_read_)) {
        if (require_repeatable_read_ != generation) {
          return GetOwningEntry(*this).AnnotateError(
              GetGenerationMismatchError(),
              /*reading=*/true);
        }
      } else {
        require_repeatable_read_ = generation;
      }
      return absl::OkStatus();
    }

    static absl::Status GetGenerationMismatchError() {
      return absl::AbortedError("Generation mismatch");
    }

   private:
    friend class KvsBackedCache;

    // Target to which this `ReadModifyWriteSource` is bound.
    ReadModifyWriteTarget* target_;

    // New data for the cache if the writeback completes successfully.
    std::optional<std::shared_ptr<const void>> new_data_;

    // If not `StorageGeneration::Unknown()`, requires that the prior generation
    // match this generation when the transaction is committed.
    StorageGeneration require_repeatable_read_;
  };

  /// Returns the associated `kvstore::Driver`.
  kvstore::Driver* kvstore_driver() { return kvstore_driver_.get(); }

  /// Sets the `kvstore::Driver`.  The caller is responsible for ensuring there
  /// are no concurrent read or write operations.
  void SetKvStoreDriver(kvstore::DriverPtr driver) {
    if (driver) {
      this->SetBatchNestingDepth(driver->BatchNestingDepth() + 1);
    }
    kvstore_driver_ = std::move(driver);
  }

  kvstore::DriverPtr kvstore_driver_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_KVS_BACKED_CACHE_H_
