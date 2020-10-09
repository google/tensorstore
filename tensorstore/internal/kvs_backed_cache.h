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

#ifndef TENSORSTORE_INTERNAL_KVS_BACKED_CACHE_H_
#define TENSORSTORE_INTERNAL_KVS_BACKED_CACHE_H_

/// \file
///
/// Integrates `AsyncCache` with `KeyValueStore`.

#include <type_traits>

#include "absl/time/time.h"
#include "tensorstore/internal/async_cache.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal {

/// Base class that integrates an `AsyncCache` with a
/// `KeyValueStore`.
///
/// Each cache entry is assumed to correspond one-to-one with a key in a
/// `KeyValueStore`, defined by the `GetKeyValueStoreKey` method.
///
/// To use this class, define a `Derived` class that inherits from
/// `KvsBackedCache<Parent>`, where `Parent` is the desired base class.  The
/// derived class is responsible for defining:
///
/// 1. `DoDecode`, which decodes an `std::optional<absl::Cord>` read from the
///   `KeyValueStore` into an `std::shared_ptr<const ReadData>` object (see
///   `DoDecode`);
///
/// 2. (if writing is supported) `DoEncode`, which encodes an
///    `std::shared_ptr<const ReadData>` object into an
///    `std::optional<absl::Cord>` to write it back to the `KeyValueStore`.
///
/// 3. overrides `GetKeyValueStoreKey` if necessary.
///
/// This class takes care of reading from and writing to the `KeyValueStore`,
/// and handling the timestamps and `StorageGeneration` values.
///
/// \tparam Parent Parent class, must inherit from (or equal) `AsyncCache`.
template <typename Derived, typename Parent>
class KvsBackedCache : public Parent {
  static_assert(std::is_base_of_v<AsyncCache, Parent>);

 public:
  /// Constructs a `KvsBackedCache`.
  ///
  /// \param kvstore The `KeyValueStore` to use.  If `nullptr`,
  ///     `SetKeyValueStore` must be called before any read or write operations
  ///     are performed.
  /// \param args Arguments to forward to the `Parent` constructor.
  template <typename... U>
  explicit KvsBackedCache(KeyValueStore::Ptr kvstore, U&&... args)
      : Parent(std::forward<U>(args)...), kvstore_(std::move(kvstore)) {}

  class TransactionNode;

  class Entry : public Parent::Entry {
   public:
    using Cache = KvsBackedCache;

    /// Defines the mapping from a cache entry to a `KeyValueStore` key.
    ///
    /// By default the cache entry key is used, but derived classes may override
    /// this behavior.
    virtual std::string GetKeyValueStoreKey() {
      return std::string{this->key()};
    }

    template <typename EntryOrNode>
    struct ReadReceiverImpl {
      EntryOrNode* entry_or_node_;
      std::shared_ptr<const void> existing_read_data_;
      void set_value(KeyValueStore::ReadResult read_result) {
        if (read_result.aborted()) {
          TENSORSTORE_ASYNC_CACHE_DEBUG_LOG(
              *entry_or_node_,
              "Value has not changed, stamp=", read_result.stamp);
          // Value has not changed.
          entry_or_node_->ReadSuccess(AsyncCache::ReadState{
              std::move(existing_read_data_), std::move(read_result.stamp)});
          return;
        }
        TENSORSTORE_ASYNC_CACHE_DEBUG_LOG(*entry_or_node_,
                                          "DoDecode: ", read_result.stamp);
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
        GetOwningEntry(*entry_or_node_)
            .DoDecode(std::move(read_result).optional_value(),
                      DecodeReceiverImpl{entry_or_node_,
                                         std::move(read_result.stamp)});
      }
      void set_error(absl::Status error) {
        entry_or_node_->ReadError(GetOwningEntry(*entry_or_node_)
                                      .AnnotateError(error, /*reading=*/true));
      }
      void set_cancel() { TENSORSTORE_UNREACHABLE; }
    };

    /// Implements reading for the `AsyncCache` interface.
    ///
    /// Reads from the `KeyValueStore` and invokes `DoDecode` with the result.
    ///
    /// If an error occurs, calls `ReadError` directly without invoking
    /// `DoDecode`.
    void DoRead(absl::Time staleness_bound) final {
      KeyValueStore::ReadOptions options;
      options.staleness_bound = staleness_bound;
      auto read_state = AsyncCache::ReadLock<void>(*this).read_state();
      options.if_not_equal = std::move(read_state.stamp.generation);
      auto& cache = GetOwningCache(*this);
      auto future =
          cache.kvstore_->Read(this->GetKeyValueStoreKey(), std::move(options));
      execution::submit(
          std::move(future),
          ReadReceiverImpl<Entry>{this, std::move(read_state.data)});
    }

    using DecodeReceiver =
        AnyReceiver<absl::Status,
                    std::shared_ptr<const typename Derived::ReadData>>;

    /// Decodes a value from the `KeyValueStore` into a `ReadData` object.
    ///
    /// The derived class implementation should use a separate executor to
    /// complete any expensive computations.
    virtual void DoDecode(std::optional<absl::Cord> value,
                          DecodeReceiver receiver) = 0;

    using EncodeReceiver = AnyReceiver<absl::Status, std::optional<absl::Cord>>;

    /// Encodes a `ReadData` object into a value to write back to the
    /// `KeyValueStore`.
    ///
    /// The derived class implementation should synchronously perform any
    /// operations that require the lock be held , and then use a separate
    /// executor for any expensive computations.
    virtual void DoEncode(
        std::shared_ptr<const typename Derived::ReadData> read_data,
        UniqueWriterLock<AsyncCache::TransactionNode> lock,
        EncodeReceiver receiver) {
      TENSORSTORE_UNREACHABLE;
    }

    absl::Status AnnotateError(const absl::Status& error, bool reading) {
      return GetOwningCache(*this).kvstore_->AnnotateError(
          this->GetKeyValueStoreKey(), reading ? "reading" : "writing", error);
    }
  };

  class TransactionNode : public Parent::TransactionNode,
                          public KeyValueStore::ReadModifyWriteSource {
   public:
    using Cache = KvsBackedCache;
    using Parent::TransactionNode::TransactionNode;

    absl::Status DoInitialize(
        internal::OpenTransactionPtr& transaction) override {
      TENSORSTORE_RETURN_IF_ERROR(
          Parent::TransactionNode::DoInitialize(transaction));
      size_t phase;
      TENSORSTORE_RETURN_IF_ERROR(
          GetOwningCache(*this).kvstore()->ReadModifyWrite(
              transaction, phase, GetOwningEntry(*this).GetKeyValueStoreKey(),
              std::ref(*this)));
      this->SetPhase(phase);
      if (this->target_->KvsReadsCommitted()) {
        this->SetReadsCommitted();
      }
      return absl::OkStatus();
    }

    void DoRead(absl::Time staleness_bound) final {
      auto read_state = AsyncCache::ReadLock<void>(*this).read_state();
      target_->KvsRead(
          {std::move(read_state.stamp.generation), staleness_bound},
          typename Entry::template ReadReceiverImpl<TransactionNode>{
              this, std::move(read_state.data)});
    }

    using ReadModifyWriteSource = KeyValueStore::ReadModifyWriteSource;
    using ReadModifyWriteTarget = KeyValueStore::ReadModifyWriteTarget;

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
      TENSORSTORE_ASYNC_CACHE_DEBUG_LOG(
          *this, "KvsWriteback: if_not_equal=", options.if_not_equal,
          ", staleness_bound=", options.staleness_bound,
          ", mode=", options.writeback_mode);
      auto read_state = AsyncCache::ReadLock<void>(*this).read_state();
      if (!StorageGeneration::IsUnknown(options.if_not_equal) &&
          options.if_not_equal == read_state.stamp.generation &&
          read_state.stamp.time >= options.staleness_bound) {
        TENSORSTORE_ASYNC_CACHE_DEBUG_LOG(
            *this, "KvsWriteback: skipping because condition is satisfied");
        KeyValueStore::ReadResult read_result;
        read_result.stamp = std::move(read_state.stamp);
        return execution::set_value(receiver, std::move(read_result));
      }
      struct EncodeReceiverImpl {
        TransactionNode* self_;
        AsyncCache::ReadState update_;
        ReadModifyWriteSource::WritebackReceiver receiver_;
        void set_error(absl::Status error) {
          error = GetOwningEntry(*self_).AnnotateError(std::move(error),
                                                       /*reading=*/false);
          execution::set_error(receiver_, std::move(error));
        }
        void set_cancel() { TENSORSTORE_UNREACHABLE; }
        void set_value(std::optional<absl::Cord> value) {
          KeyValueStore::ReadResult read_result;
          read_result.stamp = std::move(update_.stamp);
          if (value) {
            read_result.state = KeyValueStore::ReadResult::kValue;
            read_result.value = std::move(*value);
          } else {
            read_result.state = KeyValueStore::ReadResult::kMissing;
          }

          // FIXME: only save if committing, also could do this inside
          // ApplyReceiverImpl
          self_->new_data_ = std::move(update_.data);
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
        void set_cancel() { TENSORSTORE_UNREACHABLE; }
        void set_value(AsyncCache::ReadState update,
                       UniqueWriterLock<AsyncCache::TransactionNode> lock) {
          if (!StorageGeneration::NotEqualOrUnspecified(update.stamp.generation,
                                                        if_not_equal_)) {
            lock.unlock();
            return execution::set_cancel(receiver_);
          }
          if (!StorageGeneration::IsInnerLayerDirty(update.stamp.generation) &&
              writeback_mode_ !=
                  ReadModifyWriteSource::kSpecifyUnchangedWriteback) {
            lock.unlock();
            if (self_->transaction()->commit_started()) {
              self_->new_data_ = std::move(update.data);
            }
            return execution::set_value(receiver_, std::move(update.stamp));
          }
          TENSORSTORE_ASYNC_CACHE_DEBUG_LOG(*self_, "DoEncode");
          auto update_data =
              std::static_pointer_cast<const typename Derived::ReadData>(
                  update.data);
          GetOwningEntry(*self_).DoEncode(
              std::move(update_data), std::move(lock),
              EncodeReceiverImpl{self_, std::move(update),
                                 std::move(receiver_)});
        }
      };
      AsyncCache::TransactionNode::ApplyOptions apply_options;
      apply_options.staleness_bound = options.staleness_bound;
      apply_options.validate_only =
          options.writeback_mode == ReadModifyWriteSource::kValidateOnly;
      this->DoApply(
          std::move(apply_options),
          ApplyReceiverImpl{this, std::move(options.if_not_equal),
                            options.writeback_mode, std::move(receiver)});
    }

    void KvsWritebackSuccess(TimestampedStorageGeneration new_stamp) override {
      return this->WritebackSuccess(
          AsyncCache::ReadState{std::move(new_data_), std::move(new_stamp)});
    }
    void KvsWritebackError() override { this->WritebackError(); }

    void KvsRevoke() override { this->Revoke(); }

   private:
    friend class KvsBackedCache;

    // Target to which this `ReadModifyWriteSource` is bound.
    ReadModifyWriteTarget* target_;
    std::shared_ptr<const void> new_data_;
  };

  /// Returns the associated `KeyValueStore`.
  KeyValueStore* kvstore() { return kvstore_.get(); }

  /// Sets the KeyValueStore.  The caller is responsible for ensuring there are
  /// no concurrent read or write operations.
  void SetKeyValueStore(KeyValueStore::Ptr kvstore) {
    kvstore_ = std::move(kvstore);
  }

  KeyValueStore::Ptr kvstore_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_KVS_BACKED_CACHE_H_
